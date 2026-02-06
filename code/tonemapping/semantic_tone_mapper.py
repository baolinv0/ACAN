"""
Semantic-adjustable tone mapping for 16-bit linear input.
This module provides global + local adjustments, learnable per-class curves,
GT supervision, and LUT export for mobile ISP deployment.
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


def _safe_atanh(x, eps=1e-6):
    x = max(-1.0 + eps, min(1.0 - eps, float(x)))
    return math.atanh(x)


def _box_filter(x, window):
    if window <= 1:
        return x
    return F.avg_pool2d(x, window, stride=1, padding=window // 2)


class SemanticToneMapper(nn.Module):
    """
    Tone mapper with semantic class conditioning.

    Inputs:
        linear_16: (B,C,H,W) or (C,H,W), 16-bit linear values (uint16 or float).
        seg_map:   (B,H,W) or (H,W), integer class ids.
        exposure_ev: scalar or (B,), exposure value (EV).
        adjustments: optional dict for active tuning (global/class/local overrides).
    Output:
        tone mapped image in [0,1], shape matches input (B,C,H,W) or (C,H,W).
    """

    def __init__(
        self,
        num_classes,
        init_gain=1.0,
        init_gamma=2.2,
        init_white=4.0,
        init_global_gain=1.0,
        init_global_gamma=1.0,
        init_global_white=1.0,
        bias_range=0.30,
        global_bias_range=None,
        max_input=8.0,
        local_window=9,
        local_gain_range=0.8,
        local_bias_range=0.15,
        local_enable=True,
        local_method="box",
        guided_eps=1e-3,
        bilateral_sigma_spatial=None,
        bilateral_sigma_range=0.1,
        eps=1e-6,
    ):
        super(SemanticToneMapper, self).__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")

        self.num_classes = int(num_classes)
        self.bias_range = float(bias_range)
        if global_bias_range is None:
            global_bias_range = bias_range
        self.global_bias_range = float(global_bias_range)
        self.max_input = float(max_input)
        self.local_window = int(local_window)
        self.local_gain_range = float(local_gain_range)
        self.local_bias_range = float(local_bias_range)
        self.local_enable = bool(local_enable)
        self.local_method = self._normalize_local_method(local_method)
        self.guided_eps = float(guided_eps)
        self.bilateral_sigma_spatial = bilateral_sigma_spatial
        self.bilateral_sigma_range = float(bilateral_sigma_range)
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

        # Global adjustments (image-level).
        self.log_global_gain = nn.Parameter(torch.log(torch.tensor(float(init_global_gain))))
        self.log_global_gamma = nn.Parameter(torch.log(torch.tensor(float(init_global_gamma))))
        init_global_white = max(float(init_global_white), 1.0 + 1e-3)
        self.raw_global_white = nn.Parameter(
            _inv_softplus(torch.tensor(init_global_white - 1.0))
        )
        self.raw_global_bias = nn.Parameter(torch.tensor(0.0))

        # Local adjustment strengths (per class).
        self.raw_local_gain = nn.Parameter(torch.zeros(self.num_classes))
        self.raw_local_bias = nn.Parameter(torch.zeros(self.num_classes))

    def _params(self):
        gain = torch.exp(self.log_gain)
        gamma = torch.exp(self.log_gamma) + self.eps
        white = 1.0 + F.softplus(self.raw_white)
        bias = self.bias_range * torch.tanh(self.raw_bias)
        return gain, gamma, white, bias

    def _global_params(self):
        gain = torch.exp(self.log_global_gain)
        gamma = torch.exp(self.log_global_gamma) + self.eps
        white = 1.0 + F.softplus(self.raw_global_white)
        bias = self.global_bias_range * torch.tanh(self.raw_global_bias)
        return gain, gamma, white, bias

    def _local_params(self):
        gain = self.local_gain_range * torch.tanh(self.raw_local_gain)
        bias = self.local_bias_range * torch.tanh(self.raw_local_bias)
        return gain, bias

    def get_class_params(self):
        gain, gamma, white, bias = self._params()
        return {
            "gain": gain.detach().cpu().numpy(),
            "gamma": gamma.detach().cpu().numpy(),
            "white": white.detach().cpu().numpy(),
            "bias": bias.detach().cpu().numpy(),
        }

    def get_adjustment_params(self):
        gain, gamma, white, bias = self._params()
        g_gain, g_gamma, g_white, g_bias = self._global_params()
        l_gain, l_bias = self._local_params()
        return {
            "class": {
                "gain": gain.detach().cpu().numpy(),
                "gamma": gamma.detach().cpu().numpy(),
                "white": white.detach().cpu().numpy(),
                "bias": bias.detach().cpu().numpy(),
            },
            "global": {
                "gain": float(g_gain.detach().cpu().item()),
                "gamma": float(g_gamma.detach().cpu().item()),
                "white": float(g_white.detach().cpu().item()),
                "bias": float(g_bias.detach().cpu().item()),
            },
            "local": {
                "gain_strength": l_gain.detach().cpu().numpy(),
                "bias_strength": l_bias.detach().cpu().numpy(),
                "window": int(self.local_window),
                "method": self._normalize_local_method(self.local_method),
                "guided_eps": float(self.guided_eps),
                "bilateral_sigma_spatial": self.bilateral_sigma_spatial,
                "bilateral_sigma_range": float(self.bilateral_sigma_range),
                "enable": bool(self.local_enable),
            },
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
            self.raw_bias.data[class_idx] = _safe_atanh(b / self.bias_range)

    def set_global_params(self, gain=None, gamma=None, white=None, bias=None):
        if gain is not None:
            self.log_global_gain.data = torch.log(
                torch.tensor(float(gain), device=self.log_global_gain.device)
            )
        if gamma is not None:
            self.log_global_gamma.data = torch.log(
                torch.tensor(float(gamma), device=self.log_global_gamma.device)
            )
        if white is not None:
            w = max(float(white), 1.0 + 1e-3)
            self.raw_global_white.data = _inv_softplus(
                torch.tensor(w - 1.0, device=self.raw_global_white.device)
            )
        if bias is not None:
            if self.global_bias_range <= 0:
                raise ValueError("global_bias_range must be > 0 to set bias")
            b = float(bias)
            b = max(-self.global_bias_range, min(self.global_bias_range, b))
            self.raw_global_bias.data = _safe_atanh(b / self.global_bias_range)

    def set_local_params(self, class_idx=None, gain_strength=None, bias_strength=None):
        if class_idx is None:
            indices = range(self.num_classes)
        else:
            if class_idx < 0 or class_idx >= self.num_classes:
                raise ValueError("class_idx out of range")
            indices = [class_idx]

        if gain_strength is not None:
            g = float(gain_strength)
            g = max(-self.local_gain_range, min(self.local_gain_range, g))
            raw_g = (
                _safe_atanh(g / self.local_gain_range) if self.local_gain_range > 0 else 0.0
            )
            for idx in indices:
                self.raw_local_gain.data[idx] = raw_g

        if bias_strength is not None:
            b = float(bias_strength)
            b = max(-self.local_bias_range, min(self.local_bias_range, b))
            raw_b = (
                _safe_atanh(b / self.local_bias_range) if self.local_bias_range > 0 else 0.0
            )
            for idx in indices:
                self.raw_local_bias.data[idx] = raw_b

    def set_class_params_all(self, gain=None, gamma=None, white=None, bias=None):
        device = self.log_gain.device
        dtype = self.log_gain.dtype
        if gain is not None:
            gain_t = self._coerce_class_param(gain, device, dtype)
            self.log_gain.data = torch.log(gain_t)
        if gamma is not None:
            gamma_t = self._coerce_class_param(gamma, device, dtype)
            self.log_gamma.data = torch.log(gamma_t)
        if white is not None:
            white_t = self._coerce_class_param(white, device, dtype)
            white_t = torch.clamp(white_t, min=1.0 + 1e-3)
            self.raw_white.data = _inv_softplus(white_t - 1.0)
        if bias is not None:
            if self.bias_range <= 0:
                raise ValueError("bias_range must be > 0 to set bias")
            bias_t = self._coerce_class_param(bias, device, dtype)
            bias_t = torch.clamp(bias_t, -self.bias_range, self.bias_range)
            raw = []
            for b in bias_t.tolist():
                raw.append(_safe_atanh(b / self.bias_range))
            self.raw_bias.data = torch.tensor(raw, device=device, dtype=self.raw_bias.dtype)

    def set_local_params_all(self, gain_strength=None, bias_strength=None):
        device = self.raw_local_gain.device
        dtype = self.raw_local_gain.dtype
        if gain_strength is not None:
            gain_t = self._coerce_class_param(gain_strength, device, dtype)
            gain_t = torch.clamp(gain_t, -self.local_gain_range, self.local_gain_range)
            raw = []
            for g in gain_t.tolist():
                if self.local_gain_range > 0:
                    raw.append(_safe_atanh(g / self.local_gain_range))
                else:
                    raw.append(0.0)
            self.raw_local_gain.data = torch.tensor(raw, device=device, dtype=self.raw_local_gain.dtype)
        if bias_strength is not None:
            bias_t = self._coerce_class_param(bias_strength, device, dtype)
            bias_t = torch.clamp(bias_t, -self.local_bias_range, self.local_bias_range)
            raw = []
            for b in bias_t.tolist():
                if self.local_bias_range > 0:
                    raw.append(_safe_atanh(b / self.local_bias_range))
                else:
                    raw.append(0.0)
            self.raw_local_bias.data = torch.tensor(raw, device=device, dtype=self.raw_local_bias.dtype)

    def set_adjustment_params(self, params):
        if not params:
            return
        if "global" in params and params["global"] is not None:
            g = params["global"]
            self.set_global_params(
                gain=g.get("gain"),
                gamma=g.get("gamma"),
                white=g.get("white"),
                bias=g.get("bias"),
            )
        if "class" in params and params["class"] is not None:
            c = params["class"]
            self.set_class_params_all(
                gain=c.get("gain"),
                gamma=c.get("gamma"),
                white=c.get("white"),
                bias=c.get("bias"),
            )
        if "local" in params and params["local"] is not None:
            l = params["local"]
            self.set_local_params_all(
                gain_strength=l.get("gain_strength"),
                bias_strength=l.get("bias_strength"),
            )
            if "window" in l and l["window"] is not None:
                self.local_window = self._normalize_local_window(l["window"])
            if "enable" in l and l["enable"] is not None:
                self.local_enable = bool(l["enable"])
            if "method" in l and l["method"] is not None:
                self.local_method = self._normalize_local_method(l["method"])
            if "guided_eps" in l and l["guided_eps"] is not None:
                self.guided_eps = float(l["guided_eps"])
            if "bilateral_sigma_spatial" in l and l["bilateral_sigma_spatial"] is not None:
                self.bilateral_sigma_spatial = float(l["bilateral_sigma_spatial"])
            if "bilateral_sigma_range" in l and l["bilateral_sigma_range"] is not None:
                self.bilateral_sigma_range = float(l["bilateral_sigma_range"])

        if "local_method" in params and params["local_method"] is not None:
            self.local_method = self._normalize_local_method(params["local_method"])
        if "guided_eps" in params and params["guided_eps"] is not None:
            self.guided_eps = float(params["guided_eps"])
        if "bilateral_sigma_spatial" in params and params["bilateral_sigma_spatial"] is not None:
            self.bilateral_sigma_spatial = float(params["bilateral_sigma_spatial"])
        if "bilateral_sigma_range" in params and params["bilateral_sigma_range"] is not None:
            self.bilateral_sigma_range = float(params["bilateral_sigma_range"])

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

    def _normalize_local_window(self, local_window):
        if local_window is None:
            return self.local_window
        window = int(local_window)
        if window < 1:
            return 1
        if window % 2 == 0:
            window += 1
        return window

    def _normalize_local_method(self, method):
        if method is None:
            method = getattr(self, "local_method", "box")
        method = str(method).lower()
        if method not in ("box", "guided", "bilateral"):
            raise ValueError("local_method must be one of: box, guided, bilateral")
        return method

    def _guided_filter(self, guidance, inp, window, eps):
        mean_g = _box_filter(guidance, window)
        mean_p = _box_filter(inp, window)
        corr_g = _box_filter(guidance * guidance, window)
        corr_gp = _box_filter(guidance * inp, window)
        var_g = corr_g - mean_g * mean_g
        cov_gp = corr_gp - mean_g * mean_p
        a = cov_gp / (var_g + eps)
        b = mean_p - a * mean_g
        mean_a = _box_filter(a, window)
        mean_b = _box_filter(b, window)
        return mean_a * guidance + mean_b

    def _bilateral_filter(self, inp, window, sigma_spatial, sigma_range):
        if window <= 1:
            return inp
        radius = window // 2
        if sigma_spatial is None or sigma_spatial <= 0:
            sigma_spatial = max(1.0, float(window) / 3.0)
        if sigma_range is None or sigma_range <= 0:
            sigma_range = 1e-3

        device = inp.device
        dtype = inp.dtype
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        try:
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        except TypeError:
            yy, xx = torch.meshgrid(coords, coords)
        spatial = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma_spatial * sigma_spatial))
        spatial = spatial.reshape(1, -1, 1, 1)

        pad = [radius, radius, radius, radius]
        padded = F.pad(inp, pad, mode="reflect")
        patches = F.unfold(padded, kernel_size=window)
        b, k, hw = patches.shape
        patches = patches.view(b, k, inp.size(2), inp.size(3))
        center = inp
        range_w = torch.exp(-((patches - center) ** 2) / (2.0 * sigma_range * sigma_range))
        weights = spatial * range_w
        weighted = (weights * patches).sum(dim=1, keepdim=True)
        norm = weights.sum(dim=1, keepdim=True) + self.eps
        return weighted / norm

    def _compute_local_mean(
        self, luminance, window, method, guided_eps, bilateral_sigma_spatial, bilateral_sigma_range
    ):
        if method == "box":
            return _box_filter(luminance, window)
        if method == "guided":
            return self._guided_filter(luminance, luminance, window, guided_eps)
        if method == "bilateral":
            return self._bilateral_filter(
                luminance, window, bilateral_sigma_spatial, bilateral_sigma_range
            )
        raise ValueError("Unsupported local method: {}".format(method))

    def _local_adjust(
        self,
        linear_16,
        seg_map,
        local_gain_strength,
        local_bias_strength,
        window,
        method,
        guided_eps,
        bilateral_sigma_spatial,
        bilateral_sigma_range,
    ):
        if window <= 1:
            return None, None
        luminance = linear_16.mean(dim=1, keepdim=True)
        local_mean = self._compute_local_mean(
            luminance, window, method, guided_eps, bilateral_sigma_spatial, bilateral_sigma_range
        )
        global_mean = luminance.mean(dim=(2, 3), keepdim=True)
        delta = (local_mean - global_mean) / (global_mean + self.eps)

        gain_strength = local_gain_strength[seg_map].unsqueeze(1)
        bias_strength = local_bias_strength[seg_map].unsqueeze(1)

        local_gain = 1.0 + gain_strength * delta
        local_bias = bias_strength * delta
        local_gain = torch.clamp(local_gain, 0.25, 4.0)
        local_bias = torch.clamp(local_bias, -self.local_bias_range, self.local_bias_range)
        return local_gain, local_bias

    def _coerce_class_param(self, value, device, dtype):
        t = self._ensure_tensor(value, device, dtype).flatten()
        if t.numel() == 1:
            t = t.repeat(self.num_classes)
        if t.numel() != self.num_classes:
            raise ValueError("Class parameter must have num_classes elements")
        return t

    def _coerce_global_param(self, value, device, dtype):
        t = self._ensure_tensor(value, device, dtype).flatten()
        if t.numel() != 1:
            raise ValueError("Global parameter must be a scalar")
        return t.squeeze(0)

    def _apply_adjustments(
        self,
        gain,
        gamma,
        white,
        bias,
        g_gain,
        g_gamma,
        g_white,
        g_bias,
        l_gain,
        l_bias,
        use_local,
        local_window,
        local_method,
        guided_eps,
        bilateral_sigma_spatial,
        bilateral_sigma_range,
        adjustments,
        device,
        dtype,
    ):
        if not adjustments:
            return (
                gain,
                gamma,
                white,
                bias,
                g_gain,
                g_gamma,
                g_white,
                g_bias,
                l_gain,
                l_bias,
                use_local,
                local_window,
                local_method,
                guided_eps,
                bilateral_sigma_spatial,
                bilateral_sigma_range,
            )

        if "global" in adjustments and adjustments["global"] is not None:
            g_adj = adjustments["global"]
            if "gain" in g_adj and g_adj["gain"] is not None:
                g_gain = self._coerce_global_param(g_adj["gain"], device, dtype)
            if "gamma" in g_adj and g_adj["gamma"] is not None:
                g_gamma = self._coerce_global_param(g_adj["gamma"], device, dtype)
            if "white" in g_adj and g_adj["white"] is not None:
                g_white = self._coerce_global_param(g_adj["white"], device, dtype)
            if "bias" in g_adj and g_adj["bias"] is not None:
                g_bias = self._coerce_global_param(g_adj["bias"], device, dtype)

        if "class" in adjustments and adjustments["class"] is not None:
            c_adj = adjustments["class"]
            if "gain" in c_adj and c_adj["gain"] is not None:
                gain = self._coerce_class_param(c_adj["gain"], device, dtype)
            if "gamma" in c_adj and c_adj["gamma"] is not None:
                gamma = self._coerce_class_param(c_adj["gamma"], device, dtype)
            if "white" in c_adj and c_adj["white"] is not None:
                white = self._coerce_class_param(c_adj["white"], device, dtype)
            if "bias" in c_adj and c_adj["bias"] is not None:
                bias = self._coerce_class_param(c_adj["bias"], device, dtype)

        if "local" in adjustments and adjustments["local"] is not None:
            l_adj = adjustments["local"]
            if "gain_strength" in l_adj and l_adj["gain_strength"] is not None:
                l_gain = self._coerce_class_param(l_adj["gain_strength"], device, dtype)
            if "bias_strength" in l_adj and l_adj["bias_strength"] is not None:
                l_bias = self._coerce_class_param(l_adj["bias_strength"], device, dtype)
            if "enable" in l_adj and l_adj["enable"] is not None:
                use_local = bool(l_adj["enable"])
            if "window" in l_adj and l_adj["window"] is not None:
                local_window = self._normalize_local_window(l_adj["window"])
            if "method" in l_adj and l_adj["method"] is not None:
                local_method = self._normalize_local_method(l_adj["method"])
            if "guided_eps" in l_adj and l_adj["guided_eps"] is not None:
                guided_eps = float(l_adj["guided_eps"])
            if "bilateral_sigma_spatial" in l_adj and l_adj["bilateral_sigma_spatial"] is not None:
                bilateral_sigma_spatial = float(l_adj["bilateral_sigma_spatial"])
            if "bilateral_sigma_range" in l_adj and l_adj["bilateral_sigma_range"] is not None:
                bilateral_sigma_range = float(l_adj["bilateral_sigma_range"])

        if "local_enable" in adjustments and adjustments["local_enable"] is not None:
            use_local = bool(adjustments["local_enable"])
        if "local_window" in adjustments and adjustments["local_window"] is not None:
            local_window = self._normalize_local_window(adjustments["local_window"])
        if "local_method" in adjustments and adjustments["local_method"] is not None:
            local_method = self._normalize_local_method(adjustments["local_method"])
        if "guided_eps" in adjustments and adjustments["guided_eps"] is not None:
            guided_eps = float(adjustments["guided_eps"])
        if "bilateral_sigma_spatial" in adjustments and adjustments["bilateral_sigma_spatial"] is not None:
            bilateral_sigma_spatial = float(adjustments["bilateral_sigma_spatial"])
        if "bilateral_sigma_range" in adjustments and adjustments["bilateral_sigma_range"] is not None:
            bilateral_sigma_range = float(adjustments["bilateral_sigma_range"])

        return (
            gain,
            gamma,
            white,
            bias,
            g_gain,
            g_gamma,
            g_white,
            g_bias,
            l_gain,
            l_bias,
            use_local,
            local_window,
            local_method,
            guided_eps,
            bilateral_sigma_spatial,
            bilateral_sigma_range,
        )

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

    def forward(
        self,
        linear_16,
        seg_map,
        exposure_ev=None,
        normalize_input=True,
        use_local=None,
        local_window=None,
        local_method=None,
        guided_eps=None,
        bilateral_sigma_spatial=None,
        bilateral_sigma_range=None,
        adjustments=None,
    ):
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
        g_gain, g_gamma, g_white, g_bias = self._global_params()
        l_gain, l_bias = self._local_params()

        if use_local is None:
            use_local = self.local_enable
        local_window = self._normalize_local_window(local_window)
        local_method = self._normalize_local_method(local_method)
        if guided_eps is None:
            guided_eps = self.guided_eps
        if bilateral_sigma_spatial is None:
            bilateral_sigma_spatial = self.bilateral_sigma_spatial
        if bilateral_sigma_range is None:
            bilateral_sigma_range = self.bilateral_sigma_range

        (
            gain,
            gamma,
            white,
            bias,
            g_gain,
            g_gamma,
            g_white,
            g_bias,
            l_gain,
            l_bias,
            use_local,
            local_window,
            local_method,
            guided_eps,
            bilateral_sigma_spatial,
            bilateral_sigma_range,
        ) = self._apply_adjustments(
            gain,
            gamma,
            white,
            bias,
            g_gain,
            g_gamma,
            g_white,
            g_bias,
            l_gain,
            l_bias,
            use_local,
            local_window,
            local_method,
            guided_eps,
            bilateral_sigma_spatial,
            bilateral_sigma_range,
            adjustments,
            linear_16.device,
            linear_16.dtype,
        )

        gain_map = gain[seg_map].unsqueeze(1) * g_gain
        gamma_map = gamma[seg_map].unsqueeze(1) * g_gamma
        white_map = white[seg_map].unsqueeze(1) * g_white
        bias_map = bias[seg_map].unsqueeze(1) + g_bias

        if use_local and local_window > 1:
            local_gain, local_bias = self._local_adjust(
                linear_16,
                seg_map,
                l_gain,
                l_bias,
                local_window,
                local_method,
                guided_eps,
                bilateral_sigma_spatial,
                bilateral_sigma_range,
            )
            if local_gain is not None:
                gain_map = gain_map * local_gain
                bias_map = bias_map + local_bias

        out = self._tone_curve(linear_16, gain_map, gamma_map, white_map, bias_map)
        if input_was_unbatched:
            out = out.squeeze(0)
        return out

    @torch.no_grad()
    def export_lut(self, num_points=1024, device="cpu"):
        """
        Export per-class 1D LUT for fast ISP deployment.
        Global adjustments are applied, local adjustments are ignored.
        Returns: numpy array [num_classes, num_points] in [0,1].
        """
        device = torch.device(device)
        gain, gamma, white, bias = self._params()
        g_gain, g_gamma, g_white, g_bias = self._global_params()
        gain = (gain * g_gain).to(device)
        gamma = (gamma * g_gamma).to(device)
        white = (white * g_white).to(device)
        bias = (bias + g_bias).to(device)

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
