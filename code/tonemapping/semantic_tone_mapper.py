"""
Semantic-adjustable tone mapping for 16-bit linear input.
This module provides a learnable per-class tone curve, GT supervision,
and LUT export for mobile ISP deployment.
"""

import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def _inv_softplus(x):
    return torch.log(torch.expm1(x))


class SemanticToneMapper(nn.Module):
    """
    Tone mapper with semantic class conditioning.

    Inputs:
        linear_16: (B,C,H,W) or (C,H,W), 16-bit linear values (uint16 or float).
        seg_map:   (B,H,W) or (H,W), integer class ids.
        exposure_ev: scalar or (B,), exposure value (EV).
    Output:
        tone mapped image in [0,1], shape matches input (B,C,H,W) or (C,H,W).
    """

    def __init__(
        self,
        num_classes,
        init_gain=1.0,
        init_gamma=2.2,
        init_white=4.0,
        bias_range=0.30,
        max_input=8.0,
        eps=1e-6,
    ):
        super(SemanticToneMapper, self).__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")

        self.num_classes = int(num_classes)
        self.bias_range = float(bias_range)
        self.max_input = float(max_input)
        self.eps = float(eps)

        # Use log-domain or softplus to keep parameters in valid ranges.
        self.log_gain = nn.Parameter(
            torch.log(torch.ones(self.num_classes) * float(init_gain))
        )
        self.log_gamma = nn.Parameter(
            torch.log(torch.ones(self.num_classes) * float(init_gamma))
        )
        init_white = max(float(init_white), 1.0 + 1e-3)
        self.raw_white = nn.Parameter(
            _inv_softplus(torch.ones(self.num_classes) * (init_white - 1.0))
        )
        self.raw_bias = nn.Parameter(torch.zeros(self.num_classes))

    def _params(self):
        gain = torch.exp(self.log_gain)
        gamma = torch.exp(self.log_gamma) + self.eps
        white = 1.0 + F.softplus(self.raw_white)
        bias = self.bias_range * torch.tanh(self.raw_bias)
        return gain, gamma, white, bias

    def get_class_params(self):
        gain, gamma, white, bias = self._params()
        return {
            "gain": gain.detach().cpu().numpy(),
            "gamma": gamma.detach().cpu().numpy(),
            "white": white.detach().cpu().numpy(),
            "bias": bias.detach().cpu().numpy(),
        }

    def set_class_params(self, class_idx, gain=None, gamma=None, white=None, bias=None):
        if class_idx < 0 or class_idx >= self.num_classes:
            raise ValueError("class_idx out of range")
        if gain is not None:
            self.log_gain.data[class_idx] = math.log(float(gain))
        if gamma is not None:
            self.log_gamma.data[class_idx] = math.log(float(gamma))
        if white is not None:
            w = max(float(white), 1.0 + 1e-3)
            self.raw_white.data[class_idx] = _inv_softplus(
                torch.tensor(w - 1.0, device=self.raw_white.device)
            )
        if bias is not None:
            if self.bias_range <= 0:
                raise ValueError("bias_range must be > 0 to set bias")
            b = float(bias)
            b = max(-self.bias_range, min(self.bias_range, b))
            self.raw_bias.data[class_idx] = math.atanh(b / self.bias_range)

    def _ensure_tensor(self, x, device, dtype):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(x, device=device, dtype=dtype)

    def _exposure_scale(self, exposure_ev, ref_tensor):
        if exposure_ev is None:
            return 1.0
        ev = self._ensure_tensor(exposure_ev, ref_tensor.device, ref_tensor.dtype)
        if ev.dim() == 0:
            return torch.pow(torch.tensor(2.0, device=ev.device, dtype=ev.dtype), ev)
        if ev.dim() == 1:
            return torch.pow(torch.tensor(2.0, device=ev.device, dtype=ev.dtype), ev).view(
                -1, 1, 1, 1
            )
        if ev.dim() == 2 and ev.size(1) == 1:
            return torch.pow(torch.tensor(2.0, device=ev.device, dtype=ev.dtype), ev).view(
                -1, 1, 1, 1
            )
        if ev.dim() == 4:
            return torch.pow(torch.tensor(2.0, device=ev.device, dtype=ev.dtype), ev)
        raise ValueError("Unsupported exposure_ev shape: {}".format(ev.size()))

    def _prepare_inputs(self, linear_16, seg_map):
        if not torch.is_tensor(linear_16):
            linear_16 = torch.from_numpy(linear_16)
        if not torch.is_tensor(seg_map):
            seg_map = torch.from_numpy(seg_map)

        if linear_16.dim() == 3:
            linear_16 = linear_16.unsqueeze(0)
        if linear_16.dim() != 4:
            raise ValueError("linear_16 must be (B,C,H,W) or (C,H,W)")

        if seg_map.dim() == 2:
            seg_map = seg_map.unsqueeze(0)
        if seg_map.dim() == 4 and seg_map.size(1) == 1:
            seg_map = seg_map[:, 0]
        if seg_map.dim() != 3:
            raise ValueError("seg_map must be (B,H,W) or (H,W)")

        if seg_map.max().item() >= self.num_classes:
            raise ValueError("seg_map contains class id >= num_classes")

        linear_16 = linear_16.float()
        seg_map = seg_map.long().to(device=linear_16.device)
        return linear_16, seg_map

    def _tone_curve(self, x, gain, gamma, white, bias):
        x = x * gain
        x = torch.clamp(x, 0.0, self.max_input)

        # Reinhard curve with controllable white point.
        y = (x * (1.0 + x / (white * white))) / (1.0 + x)
        y = torch.clamp(y, 0.0, 1.0)

        # Bias to shift mid-tones, then gamma.
        y = torch.clamp(y + bias, 0.0, 1.0)
        y = torch.pow(y + self.eps, 1.0 / gamma)
        return y

    def forward(self, linear_16, seg_map, exposure_ev=None, normalize_input=True):
        input_was_unbatched = linear_16.dim() == 3

        linear_16, seg_map = self._prepare_inputs(linear_16, seg_map)
        device = self.log_gain.device
        if linear_16.device != device:
            linear_16 = linear_16.to(device)
        if seg_map.device != device:
            seg_map = seg_map.to(device)
        if normalize_input:
            linear_16 = linear_16 / 65535.0

        scale = self._exposure_scale(exposure_ev, linear_16)
        linear_16 = linear_16 * scale

        gain, gamma, white, bias = self._params()

        gain_map = gain[seg_map].unsqueeze(1)
        gamma_map = gamma[seg_map].unsqueeze(1)
        white_map = white[seg_map].unsqueeze(1)
        bias_map = bias[seg_map].unsqueeze(1)

        out = self._tone_curve(linear_16, gain_map, gamma_map, white_map, bias_map)
        if input_was_unbatched:
            out = out.squeeze(0)
        return out

    @torch.no_grad()
    def export_lut(self, num_points=1024, device="cpu"):
        """
        Export per-class 1D LUT for fast ISP deployment.
        Returns: numpy array [num_classes, num_points] in [0,1].
        """
        device = torch.device(device)
        gain, gamma, white, bias = self._params()
        gain = gain.to(device)
        gamma = gamma.to(device)
        white = white.to(device)
        bias = bias.to(device)

        x = torch.linspace(0.0, 1.0, num_points, device=device)
        x = x.unsqueeze(0).repeat(self.num_classes, 1)

        y = self._tone_curve(
            x,
            gain.unsqueeze(1),
            gamma.unsqueeze(1),
            white.unsqueeze(1),
            bias.unsqueeze(1),
        )
        return y.cpu().numpy()

    @torch.no_grad()
    def export_quantized_lut(self, num_points=1024, bit_depth=16, device="cpu"):
        lut = self.export_lut(num_points=num_points, device=device)
        max_val = float((1 << bit_depth) - 1)
        lut_q = np.clip(lut * max_val + 0.5, 0, max_val).astype(np.uint16)
        return lut_q


class ToneMappingDataset(Dataset):
    """
    Dataset for semantic tone mapping.
    Manifest CSV columns:
        linear_path, seg_path, gt_path, exposure_ev
    Paths can be absolute or relative to root.
    """

    def __init__(self, manifest_csv, root=None):
        self.root = root
        self.items = []
        with open(manifest_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "linear_path" not in row or "seg_path" not in row or "gt_path" not in row:
                    raise ValueError("Manifest must include linear_path, seg_path, gt_path")
                ev = row.get("exposure_ev", "0")
                self.items.append(
                    {
                        "linear_path": row["linear_path"],
                        "seg_path": row["seg_path"],
                        "gt_path": row["gt_path"],
                        "exposure_ev": float(ev) if ev is not None else 0.0,
                    }
                )

    def _resolve(self, path):
        if self.root is None:
            return path
        return os.path.join(self.root, path)

    def _read_image(self, path, keep_channels=False):
        try:
            import cv2

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError("cv2.imread failed")
            if img.ndim == 2:
                if keep_channels:
                    img = img[:, :, None]
            else:
                if img.shape[2] == 3:
                    img = img[:, :, ::-1]
                elif img.shape[2] == 4:
                    img = img[:, :, :3][:, :, ::-1]
            return img
        except Exception:
            from PIL import Image

            img = Image.open(path)
            arr = np.array(img)
            if arr.ndim == 2:
                if keep_channels:
                    arr = arr[:, :, None]
            return arr

    def _read_linear(self, path):
        arr = self._read_image(path, keep_channels=True)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return arr

    def _read_seg(self, path):
        arr = self._read_image(path, keep_channels=False)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr

    def _read_gt(self, path):
        arr = self._read_image(path, keep_channels=True)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        elif arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return arr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        linear = self._read_linear(self._resolve(item["linear_path"]))
        seg = self._read_seg(self._resolve(item["seg_path"]))
        gt = self._read_gt(self._resolve(item["gt_path"]))

        linear = torch.from_numpy(linear).permute(2, 0, 1).float()
        seg = torch.from_numpy(seg).long()
        gt = torch.from_numpy(gt).permute(2, 0, 1).float()
        ev = torch.tensor(item["exposure_ev"]).float()

        return linear, seg, gt, ev


def tone_mapping_loss(pred, gt, use_smooth_l1=False):
    if use_smooth_l1:
        return F.smooth_l1_loss(pred, gt)
    return F.l1_loss(pred, gt)
