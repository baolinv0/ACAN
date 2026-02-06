"""
HDRNet-style semantic tone mapper with bilateral grid slicing.
Provides semantic controllable blending between curve-based mapping and HDRNet output.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .semantic_tone_mapper import SemanticToneMapper
except ImportError:
    from semantic_tone_mapper import SemanticToneMapper


def _luminance(x):
    if x.size(1) == 1:
        return x
    if x.size(1) >= 3:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return x.mean(dim=1, keepdim=True)


def _safe_logit(x, eps=1e-6):
    x = max(eps, min(1.0 - eps, float(x)))
    return math.log(x / (1.0 - x))


def _safe_atanh(x, eps=1e-6):
    x = max(-1.0 + eps, min(1.0 - eps, float(x)))
    return math.atanh(x)


class HDRNetLocal(nn.Module):
    """
    Lightweight HDRNet-style bilateral grid for local tone mapping.
    Produces per-pixel affine coefficients (3x4) and applies them to RGB.
    """

    def __init__(
        self,
        num_classes,
        grid_depth=8,
        grid_height=16,
        grid_width=16,
        coeffs=12,
        coeff_scale_range=0.5,
        coeff_bias_range=0.25,
        guide_bias_range=0.25,
        embedding_dim=8,
        hidden=32,
        guide_hidden=16,
        use_exposure=True,
        eps=1e-6,
    ):
        super(HDRNetLocal, self).__init__()
        self.num_classes = int(num_classes)
        self.grid_depth = int(grid_depth)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.coeffs = int(coeffs)
        self.coeff_scale_range = float(coeff_scale_range)
        self.coeff_bias_range = float(coeff_bias_range)
        self.guide_bias_range = float(guide_bias_range)
        self.embedding_dim = int(embedding_dim)
        self.use_exposure = bool(use_exposure)
        self.eps = float(eps)
        if self.coeffs != 12:
            raise ValueError("coeffs must be 12 for RGB affine (3x4)")

        in_ch = 1 + self.embedding_dim + (1 if self.use_exposure else 0)
        self.seg_embed = nn.Embedding(self.num_classes, self.embedding_dim)

        self.lowres_net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.coeffs * self.grid_depth, kernel_size=1),
        )

        self.guide_net = nn.Sequential(
            nn.Conv2d(in_ch, guide_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(guide_hidden, 1, kernel_size=1),
        )
        self.guide_act = nn.Sigmoid()

        self.raw_coeff_scale = nn.Parameter(torch.zeros(self.num_classes, self.coeffs))
        self.raw_coeff_bias = nn.Parameter(torch.zeros(self.num_classes, self.coeffs))
        self.raw_guide_bias = nn.Parameter(torch.zeros(self.num_classes))

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

    def _ev_map(self, exposure_ev, ref_tensor, height, width):
        if not self.use_exposure:
            return None
        if exposure_ev is None:
            return torch.zeros(
                ref_tensor.size(0), 1, height, width, device=ref_tensor.device, dtype=ref_tensor.dtype
            )
        ev = self._ensure_tensor(exposure_ev, ref_tensor.device, ref_tensor.dtype)
        if ev.dim() == 0:
            ev = ev.view(1, 1, 1, 1)
        if ev.dim() == 1:
            ev = ev.view(-1, 1, 1, 1)
        if ev.dim() == 2 and ev.size(1) == 1:
            ev = ev.view(-1, 1, 1, 1)
        if ev.dim() != 4:
            raise ValueError("Unsupported exposure_ev shape: {}".format(ev.size()))
        return ev.expand(-1, 1, height, width)

    def _build_features(self, luma, seg_map, exposure_ev):
        seg_emb = self.seg_embed(seg_map).permute(0, 3, 1, 2)
        features = [luma, seg_emb]
        ev_map = self._ev_map(exposure_ev, luma, luma.size(2), luma.size(3))
        if ev_map is not None:
            features.append(ev_map)
        return torch.cat(features, dim=1)

    def _coerce_class_param(self, value, device, dtype):
        t = torch.as_tensor(value, device=device, dtype=dtype).flatten()
        if t.numel() == 1:
            t = t.repeat(self.num_classes)
        if t.numel() != self.num_classes:
            raise ValueError("Parameter must have num_classes elements")
        return t

    def _coerce_coeff_param(self, value, device, dtype):
        t = torch.as_tensor(value, device=device, dtype=dtype)
        if t.numel() == 1:
            t = t.view(1, 1).repeat(self.num_classes, self.coeffs)
        elif t.dim() == 1 and t.numel() == self.num_classes:
            t = t.view(self.num_classes, 1).repeat(1, self.coeffs)
        elif t.dim() == 2 and t.size(0) == self.num_classes and t.size(1) == self.coeffs:
            pass
        else:
            raise ValueError("Coeff param must be scalar, [num_classes], or [num_classes, coeffs]")
        return t

    def _semantic_coeff_params(self):
        scale = 1.0 + self.coeff_scale_range * torch.tanh(self.raw_coeff_scale)
        bias = self.coeff_bias_range * torch.tanh(self.raw_coeff_bias)
        return scale, bias

    def _semantic_guide_bias(self):
        return self.guide_bias_range * torch.tanh(self.raw_guide_bias)

    def get_semantic_params(self):
        scale, bias = self._semantic_coeff_params()
        guide_bias = self._semantic_guide_bias()
        return {
            "coeff_scale": scale.detach().cpu().numpy(),
            "coeff_bias": bias.detach().cpu().numpy(),
            "guide_bias": guide_bias.detach().cpu().numpy(),
        }

    def set_semantic_params(self, coeff_scale=None, coeff_bias=None, guide_bias=None):
        device = self.raw_coeff_scale.device
        dtype = self.raw_coeff_scale.dtype
        if coeff_scale is not None:
            scale = self._coerce_coeff_param(coeff_scale, device, dtype)
            if self.coeff_scale_range > 0:
                scale = torch.clamp(
                    scale, 1.0 - self.coeff_scale_range, 1.0 + self.coeff_scale_range
                )
                raw = (scale - 1.0) / self.coeff_scale_range
                raw = torch.clamp(raw, -0.999, 0.999)
                self.raw_coeff_scale.data = torch.atanh(raw)
        if coeff_bias is not None:
            bias = self._coerce_coeff_param(coeff_bias, device, dtype)
            if self.coeff_bias_range > 0:
                bias = torch.clamp(bias, -self.coeff_bias_range, self.coeff_bias_range)
                raw = bias / self.coeff_bias_range
                raw = torch.clamp(raw, -0.999, 0.999)
                self.raw_coeff_bias.data = torch.atanh(raw)
        if guide_bias is not None:
            bias = self._coerce_class_param(guide_bias, device, dtype)
            if self.guide_bias_range > 0:
                bias = torch.clamp(bias, -self.guide_bias_range, self.guide_bias_range)
                raw = bias / self.guide_bias_range
                raw = torch.clamp(raw, -0.999, 0.999)
                self.raw_guide_bias.data = torch.atanh(raw)

    def _slice_coeffs(self, coeff_grid, guide):
        b, c, d, gh, gw = coeff_grid.shape
        _, _, h, w = guide.shape
        device = guide.device
        dtype = guide.dtype

        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        try:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        except TypeError:
            grid_y, grid_x = torch.meshgrid(ys, xs)
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(1)
        grid_xy = grid_xy.repeat(b, 1, 1, 1, 1)

        guide_z = guide.permute(0, 2, 3, 1).unsqueeze(1) * 2.0 - 1.0
        grid = torch.cat([grid_xy, guide_z], dim=-1)

        coeff = F.grid_sample(coeff_grid, grid, mode="bilinear", align_corners=True)
        return coeff.squeeze(2)

    def _resolve_semantic_params(self, semantic_adjustments, device, dtype):
        scale, bias = self._semantic_coeff_params()
        guide_bias = self._semantic_guide_bias()

        if semantic_adjustments is not None:
            if "coeff_scale" in semantic_adjustments and semantic_adjustments["coeff_scale"] is not None:
                scale = self._coerce_coeff_param(
                    semantic_adjustments["coeff_scale"], device, dtype
                )
            if "coeff_bias" in semantic_adjustments and semantic_adjustments["coeff_bias"] is not None:
                bias = self._coerce_coeff_param(
                    semantic_adjustments["coeff_bias"], device, dtype
                )
            if "guide_bias" in semantic_adjustments and semantic_adjustments["guide_bias"] is not None:
                guide_bias = self._coerce_class_param(
                    semantic_adjustments["guide_bias"], device, dtype
                )
        return scale, bias, guide_bias

    def _apply_coeffs(self, inp, coeff_map):
        b, c, h, w = inp.shape
        if c == 1:
            rgb = inp.repeat(1, 3, 1, 1)
        elif c >= 3:
            rgb = inp[:, 0:3]
        else:
            rgb = inp.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        coeff = coeff_map.view(b, 3, 4, h, w)
        r = coeff[:, 0, 0] * rgb[:, 0] + coeff[:, 0, 1] * rgb[:, 1] + coeff[:, 0, 2] * rgb[:, 2]
        r = r + coeff[:, 0, 3]
        g = coeff[:, 1, 0] * rgb[:, 0] + coeff[:, 1, 1] * rgb[:, 1] + coeff[:, 1, 2] * rgb[:, 2]
        g = g + coeff[:, 1, 3]
        b_out = coeff[:, 2, 0] * rgb[:, 0] + coeff[:, 2, 1] * rgb[:, 1] + coeff[:, 2, 2] * rgb[:, 2]
        b_out = b_out + coeff[:, 2, 3]

        out = torch.stack([r, g, b_out], dim=1)
        out = torch.clamp(out, 0.0, 1.0)
        if c == 1:
            return out[:, 0:1]
        return out

    def forward(
        self,
        linear_16,
        seg_map,
        exposure_ev=None,
        normalize_input=True,
        semantic_adjustments=None,
    ):
        input_was_unbatched = linear_16.dim() == 3

        linear_16, seg_map = self._prepare_inputs(linear_16, seg_map)
        device = next(self.parameters()).device
        if linear_16.device != device:
            linear_16 = linear_16.to(device)
        if seg_map.device != device:
            seg_map = seg_map.to(device)
        if normalize_input:
            linear_16 = linear_16 / 65535.0

        scale = self._exposure_scale(exposure_ev, linear_16)
        linear_16 = linear_16 * scale

        luma = _luminance(linear_16)
        guide_in = self._build_features(luma, seg_map, exposure_ev)
        guide_logits = self.guide_net(guide_in)
        scale, bias, guide_bias = self._resolve_semantic_params(
            semantic_adjustments, linear_16.device, linear_16.dtype
        )
        guide_logits = guide_logits + guide_bias[seg_map].unsqueeze(1)

        luma_low = F.adaptive_avg_pool2d(luma, (self.grid_height, self.grid_width))
        seg_low = F.interpolate(
            seg_map.unsqueeze(1).float(),
            size=(self.grid_height, self.grid_width),
            mode="nearest",
        ).long()
        seg_low = seg_low[:, 0]
        low_in = self._build_features(luma_low, seg_low, exposure_ev)
        coeff = self.lowres_net(low_in)
        coeff = coeff.view(
            coeff.size(0), self.coeffs, self.grid_depth, self.grid_height, self.grid_width
        )

        guide_for_slice = self.guide_act(guide_logits)
        coeff_map = self._slice_coeffs(coeff, guide_for_slice)
        scale_map = scale[seg_map].permute(0, 3, 1, 2)
        bias_map = bias[seg_map].permute(0, 3, 1, 2)
        coeff_map = coeff_map * scale_map + bias_map
        out = self._apply_coeffs(linear_16, coeff_map)

        if input_was_unbatched:
            out = out.squeeze(0)
        return out


class SemanticHDRNetToneMapper(nn.Module):
    """
    Semantic HDRNet tone mapper with per-class mix control.
    Output = (1 - mix) * base_curve + mix * hdrnet_output
    """

    def __init__(self, num_classes, base_kwargs=None, hdrnet_kwargs=None, mix_init=0.5, base_use_local=False):
        super(SemanticHDRNetToneMapper, self).__init__()
        base_kwargs = {} if base_kwargs is None else dict(base_kwargs)
        if "local_enable" not in base_kwargs:
            base_kwargs["local_enable"] = bool(base_use_local)

        hdrnet_kwargs = {} if hdrnet_kwargs is None else dict(hdrnet_kwargs)

        self.base = SemanticToneMapper(num_classes=num_classes, **base_kwargs)
        self.hdrnet = HDRNetLocal(num_classes=num_classes, **hdrnet_kwargs)

        self.raw_mix = nn.Parameter(torch.zeros(int(num_classes)))
        self.set_mix_params(mix_init)

    def _coerce_class_param(self, value, device, dtype):
        t = torch.as_tensor(value, device=device, dtype=dtype).flatten()
        if t.numel() == 1:
            t = t.repeat(self.raw_mix.numel())
        if t.numel() != self.raw_mix.numel():
            raise ValueError("mix must have num_classes elements")
        return t

    def set_mix_params(self, mix):
        if mix is None:
            return
        device = self.raw_mix.device
        dtype = self.raw_mix.dtype
        mix_t = self._coerce_class_param(mix, device, dtype)
        mix_t = torch.clamp(mix_t, 0.0, 1.0)
        raw = []
        for v in mix_t.tolist():
            raw.append(_safe_logit(v))
        self.raw_mix.data = torch.tensor(raw, device=device, dtype=self.raw_mix.dtype)

    def get_adjustment_params(self):
        mix = torch.sigmoid(self.raw_mix).detach().cpu().numpy()
        return {
            "base": self.base.get_adjustment_params(),
            "mix": mix,
            "hdrnet": {
                "grid_depth": int(self.hdrnet.grid_depth),
                "grid_height": int(self.hdrnet.grid_height),
                "grid_width": int(self.hdrnet.grid_width),
                "coeffs": int(self.hdrnet.coeffs),
            },
            "hdrnet_semantic": self.hdrnet.get_semantic_params(),
        }

    def set_adjustment_params(self, params):
        if not params:
            return
        if "base" in params and params["base"] is not None:
            self.base.set_adjustment_params(params["base"])
        if "mix" in params and params["mix"] is not None:
            self.set_mix_params(params["mix"])
        if "hdrnet_semantic" in params and params["hdrnet_semantic"] is not None:
            self.hdrnet.set_semantic_params(**params["hdrnet_semantic"])

    def export_lut(self, num_points=1024, device="cpu"):
        return self.base.export_lut(num_points=num_points, device=device)

    def export_quantized_lut(self, num_points=1024, bit_depth=16, device="cpu"):
        return self.base.export_quantized_lut(
            num_points=num_points, bit_depth=bit_depth, device=device
        )

    def forward(
        self,
        linear_16,
        seg_map,
        exposure_ev=None,
        normalize_input=True,
        base_adjustments=None,
        mix_adjustments=None,
        hdrnet_adjustments=None,
        adjustments=None,
    ):
        input_was_unbatched = linear_16.dim() == 3
        linear_16, seg_map = self.base._prepare_inputs(linear_16, seg_map)
        device = self.raw_mix.device
        if linear_16.device != device:
            linear_16 = linear_16.to(device)
        if seg_map.device != device:
            seg_map = seg_map.to(device)

        if adjustments is not None:
            if "base" in adjustments:
                base_adjustments = adjustments.get("base")
            if "mix" in adjustments:
                mix_adjustments = adjustments.get("mix")
            if "hdrnet_semantic" in adjustments:
                hdrnet_adjustments = adjustments.get("hdrnet_semantic")

        base_out = self.base(
            linear_16,
            seg_map,
            exposure_ev=exposure_ev,
            normalize_input=normalize_input,
            adjustments=base_adjustments,
        )
        hdr_out = self.hdrnet(
            linear_16,
            seg_map,
            exposure_ev=exposure_ev,
            normalize_input=normalize_input,
            semantic_adjustments=hdrnet_adjustments,
        )

        mix = torch.sigmoid(self.raw_mix)
        if mix_adjustments is not None:
            mix = self._coerce_class_param(mix_adjustments, linear_16.device, linear_16.dtype)
            mix = torch.clamp(mix, 0.0, 1.0)

        mix_map = mix[seg_map].unsqueeze(1)
        out = base_out * (1.0 - mix_map) + hdr_out * mix_map
        if input_was_unbatched:
            out = out.squeeze(0)
        return out
