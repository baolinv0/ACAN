"""
Person brightness analysis and active adjustment prediction.
Provides feature extraction, target fitting, and a small predictor model.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_luminance(img):
    if img.ndim == 2:
        return img
    if img.shape[0] == 1:
        return img[0]
    if img.shape[0] >= 3:
        return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    return img.mean(axis=0)


def _normalize_linear(linear_16):
    if linear_16.dtype == np.uint16:
        return linear_16.astype(np.float32) / 65535.0
    linear_16 = linear_16.astype(np.float32)
    max_val = float(linear_16.max()) if linear_16.size > 0 else 1.0
    if max_val > 2.0:
        return linear_16 / 65535.0
    return linear_16


def _masked_values(luma, mask):
    if mask is None:
        return luma.reshape(-1)
    values = luma[mask]
    return values.reshape(-1)


def _safe_percentile(values, q):
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def compute_stats(luma, mask=None):
    values = _masked_values(luma, mask)
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p5": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        }
    mean = float(values.mean())
    std = float(values.std())
    p5 = _safe_percentile(values, 5)
    p50 = _safe_percentile(values, 50)
    p95 = _safe_percentile(values, 95)
    return {"mean": mean, "std": std, "p5": p5, "p50": p50, "p95": p95}


def compute_histogram(luma, bins=16, mask=None):
    values = _masked_values(luma, mask)
    if values.size == 0:
        return np.zeros((bins,), dtype=np.float32)
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32)
    hist = hist / max(1.0, float(hist.sum()))
    return hist


def compute_quantile_curve(values, quantiles=None, degree=2):
    if quantiles is None:
        quantiles = [5, 25, 50, 75, 95]
    if values.size == 0:
        return np.zeros((len(quantiles),), dtype=np.float32), np.zeros((degree + 1,), dtype=np.float32)
    q_vals = np.percentile(values, quantiles).astype(np.float32)
    q_x = np.array(quantiles, dtype=np.float32) / 100.0
    try:
        coeff = np.polyfit(q_x, q_vals, degree).astype(np.float32)
    except Exception:
        coeff = np.zeros((degree + 1,), dtype=np.float32)
    return q_vals, coeff


class HistogramProjector(object):
    def __init__(self, dim=8):
        self.dim = int(dim)
        self.mean = None
        self.components = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError("HistogramProjector expects 2D input with non-zero features")
        if self.dim > x.shape[1]:
            self.dim = x.shape[1]
        self.mean = x.mean(axis=0, keepdims=True)
        x0 = x - self.mean
        u, s, vt = np.linalg.svd(x0, full_matrices=False)
        self.components = vt[: self.dim]
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        x0 = x - self.mean
        return np.dot(x0, self.components.T)

    def to_dict(self):
        return {"dim": self.dim, "mean": self.mean.tolist(), "components": self.components.tolist()}

    @classmethod
    def from_dict(cls, data):
        obj = cls(dim=int(data["dim"]))
        obj.mean = np.array(data["mean"], dtype=np.float32)
        obj.components = np.array(data["components"], dtype=np.float32)
        return obj


def compute_features(
    linear_16,
    seg_map,
    exposure_ev,
    person_class,
    bins=16,
    quantiles=None,
    quantile_degree=2,
    hist_projector=None,
    include_raw_hist=True,
    return_parts=False,
):
    linear = _normalize_linear(linear_16)
    luma = compute_luminance(linear)
    person_mask = seg_map == int(person_class)

    global_stats = compute_stats(luma, None)
    person_stats = compute_stats(luma, person_mask)
    global_hist = compute_histogram(luma, bins=bins, mask=None)
    person_hist = compute_histogram(luma, bins=bins, mask=person_mask)

    person_values = _masked_values(luma, person_mask)
    quantile_vals, quantile_coeff = compute_quantile_curve(
        person_values, quantiles=quantiles, degree=quantile_degree
    )

    contrast_global = global_stats["p95"] - global_stats["p5"]
    contrast_person = person_stats["p95"] - person_stats["p5"]

    ev = float(exposure_ev) if exposure_ev is not None else 0.0

    features = [
        global_stats["mean"],
        global_stats["std"],
        global_stats["p5"],
        global_stats["p50"],
        global_stats["p95"],
        person_stats["mean"],
        person_stats["std"],
        person_stats["p5"],
        person_stats["p50"],
        person_stats["p95"],
        contrast_global,
        contrast_person,
        ev,
    ]
    features = np.array(features, dtype=np.float32)
    hist_vec = np.concatenate([global_hist, person_hist], axis=0)
    if hist_projector is not None:
        hist_proj = hist_projector.transform(hist_vec[None, :])[0]
        if include_raw_hist:
            hist_vec = np.concatenate([hist_vec, hist_proj], axis=0)
        else:
            hist_vec = hist_proj

    features = np.concatenate([features, quantile_vals, quantile_coeff, hist_vec], axis=0)
    if return_parts:
        return {
            "features": features,
            "hist": hist_vec,
            "quantile_vals": quantile_vals,
            "quantile_coeff": quantile_coeff,
        }
    return features


def feature_names(bins=16, quantiles=None, quantile_degree=2, hist_proj_dim=None, include_raw_hist=True):
    if quantiles is None:
        quantiles = [5, 25, 50, 75, 95]
    base = [
        "global_mean",
        "global_std",
        "global_p5",
        "global_p50",
        "global_p95",
        "person_mean",
        "person_std",
        "person_p5",
        "person_p50",
        "person_p95",
        "global_contrast",
        "person_contrast",
        "exposure_ev",
    ]
    base += ["person_q{}".format(q) for q in quantiles]
    base += ["person_q_poly_{}".format(i) for i in range(quantile_degree + 1)]
    hist_names = ["global_hist_{}".format(i) for i in range(bins)]
    hist_names += ["person_hist_{}".format(i) for i in range(bins)]
    if hist_proj_dim is not None and not include_raw_hist:
        hist_names = ["hist_proj_{}".format(i) for i in range(hist_proj_dim)]
    elif hist_proj_dim is not None and include_raw_hist:
        hist_names += ["hist_proj_{}".format(i) for i in range(hist_proj_dim)]
    base += hist_names
    return base


def fit_person_linear_adjustment(
    linear_16,
    gt,
    seg_map,
    person_class,
    min_pixels=200,
    gain_range=(0.25, 4.0),
    bias_range=(-0.3, 0.3),
):
    linear = _normalize_linear(linear_16)
    luma_in = compute_luminance(linear)
    luma_gt = compute_luminance(gt)
    person_mask = seg_map == int(person_class)

    x = luma_in[person_mask].reshape(-1)
    y = luma_gt[person_mask].reshape(-1)
    if x.size < min_pixels:
        return None

    A = np.stack([x, np.ones_like(x)], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    gain = float(coeff[0])
    bias = float(coeff[1])
    gain = max(gain_range[0], min(gain_range[1], gain))
    bias = max(bias_range[0], min(bias_range[1], bias))
    return gain, bias


def pearsonr(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum() * (b * b).sum())) + eps
    return float((a * b).sum() / denom)


class Standardizer(object):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std = np.maximum(self.std, 1e-6)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.mean = np.array(data["mean"], dtype=np.float32)
        obj.std = np.array(data["std"], dtype=np.float32)
        return obj


def augment_features(x, noise_std=0.01, dropout_prob=0.1):
    if noise_std > 0:
        x = x + np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    if dropout_prob > 0:
        mask = np.random.rand(*x.shape) > dropout_prob
        x = x * mask
    return x


def augment_features_torch(x, noise_std=0.01, dropout_prob=0.1):
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    if dropout_prob > 0:
        mask = (torch.rand_like(x) > dropout_prob).float()
        x = x * mask
    return x


def nt_xent_loss(z1, z2, temperature=0.1, eps=1e-8):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / max(temperature, eps)

    diag = torch.eye(2 * batch, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -1e9)

    labels = torch.arange(2 * batch, device=z.device)
    labels = (labels + batch) % (2 * batch)
    loss = F.cross_entropy(sim, labels)
    return loss


class PersonAdjustmentModel(nn.Module):
    def __init__(self, in_dim, hidden=64, embed_dim=64, out_dim=2, scene_classes=None, proj_dim=None):
        super(PersonAdjustmentModel, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.reg_head = nn.Linear(embed_dim, out_dim)
        self.scene_head = None
        self.proj_head = None
        if scene_classes is not None:
            self.scene_head = nn.Linear(embed_dim, int(scene_classes))
        if proj_dim is not None:
            self.proj_head = nn.Sequential(
                nn.Linear(embed_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim),
            )

    def forward(self, x, return_dict=False):
        emb = self.trunk(x)
        pred = self.reg_head(emb)
        if not return_dict:
            return pred
        out = {"pred": pred, "embedding": emb}
        if self.scene_head is not None:
            out["scene_logits"] = self.scene_head(emb)
        if self.proj_head is not None:
            out["proj"] = self.proj_head(emb)
        return out


class PersonAdjustmentPredictor(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super(PersonAdjustmentPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_predictor(
    features,
    targets,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    hidden=64,
    embed_dim=64,
    batch_size=256,
    scene_labels=None,
    scene_weight=0.2,
    contrastive_weight=0.0,
    contrastive_temperature=0.1,
    contrastive_proj_dim=32,
    augment_noise_std=0.01,
    augment_dropout_prob=0.1,
    device="cpu",
    val_ratio=0.1,
):
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if features.size == 0:
        raise ValueError("No features to train on")

    num_samples = features.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(num_samples * (1.0 - val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < num_samples else indices[:0]

    standardizer = Standardizer().fit(features[train_idx])
    x_train = standardizer.transform(features[train_idx])
    y_train = targets[train_idx]
    x_val = standardizer.transform(features[val_idx]) if val_idx.size > 0 else None
    y_val = targets[val_idx] if val_idx.size > 0 else None

    scene_train = None
    scene_val = None
    scene_classes = None
    if scene_labels is not None:
        scene_labels = np.asarray(scene_labels, dtype=np.int64)
        if scene_labels.shape[0] != features.shape[0]:
            raise ValueError("scene_labels length mismatch")
        scene_train = scene_labels[train_idx]
        scene_val = scene_labels[val_idx] if val_idx.size > 0 else None
        scene_classes = int(scene_labels.max()) + 1

    device = torch.device(device)
    use_contrastive = contrastive_weight > 0.0
    proj_dim = contrastive_proj_dim if use_contrastive else None
    model = PersonAdjustmentModel(
        in_dim=features.shape[1],
        hidden=hidden,
        embed_dim=embed_dim,
        out_dim=targets.shape[1],
        scene_classes=scene_classes,
        proj_dim=proj_dim,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    scene_train_t = torch.from_numpy(scene_train).to(device) if scene_train is not None else None

    if batch_size is None or batch_size <= 0:
        batch_size = x_train_t.size(0)

    num_batches = int(math.ceil(x_train_t.size(0) / float(batch_size)))

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_train_t.size(0), device=device)
        epoch_loss = 0.0
        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = x_train_t[idx]
            yb = y_train_t[idx]
            sb = scene_train_t[idx] if scene_train_t is not None else None

            out = model(xb, return_dict=True)
            loss = loss_fn(out["pred"], yb)

            if sb is not None and out.get("scene_logits") is not None:
                loss = loss + scene_weight * F.cross_entropy(out["scene_logits"], sb)

            if use_contrastive and out.get("proj") is not None:
                xb1 = augment_features_torch(xb, augment_noise_std, augment_dropout_prob)
                xb2 = augment_features_torch(xb, augment_noise_std, augment_dropout_prob)
                out1 = model(xb1, return_dict=True)
                out2 = model(xb2, return_dict=True)
                loss = loss + contrastive_weight * nt_xent_loss(
                    out1["proj"], out2["proj"], temperature=contrastive_temperature
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            msg = "Epoch {}/{} loss {:.6f}".format(epoch + 1, epochs, epoch_loss / num_batches)
            if x_val is not None and x_val.size > 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(torch.from_numpy(x_val).to(device))
                    val_loss = loss_fn(val_pred, torch.from_numpy(y_val).to(device))
                    msg += " val {:.6f}".format(val_loss.item())
            print(msg)

    return model, standardizer


def save_predictor(model, standardizer, save_dir, config=None, hist_projector=None):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "predictor.pth"))
    with open(os.path.join(save_dir, "standardizer.json"), "w") as f:
        json.dump(standardizer.to_dict(), f, indent=2)
    if hist_projector is not None:
        with open(os.path.join(save_dir, "hist_projector.json"), "w") as f:
            json.dump(hist_projector.to_dict(), f, indent=2)
    if config is not None:
        with open(os.path.join(save_dir, "predictor_config.json"), "w") as f:
            json.dump(config, f, indent=2)


def load_predictor(model_path, standardizer_path, in_dim, hidden=64, embed_dim=64, out_dim=2, device="cpu"):
    device = torch.device(device)
    model = PersonAdjustmentPredictor(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with open(standardizer_path, "r") as f:
        standardizer = Standardizer.from_dict(json.load(f))
    return model, standardizer


def load_predictor_bundle(load_dir, device="cpu"):
    model_path = os.path.join(load_dir, "predictor.pth")
    standardizer_path = os.path.join(load_dir, "standardizer.json")
    config_path = os.path.join(load_dir, "predictor_config.json")
    hist_proj_path = os.path.join(load_dir, "hist_projector.json")

    config = None
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    if config is None:
        raise ValueError("predictor_config.json not found in {}".format(load_dir))

    device = torch.device(device)
    model_type = config.get("model", "predictor")
    if model_type == "advanced":
        model = PersonAdjustmentModel(
            in_dim=int(config["in_dim"]),
            hidden=int(config.get("hidden", 64)),
            embed_dim=int(config.get("embed_dim", 64)),
            out_dim=int(config.get("out_dim", 2)),
            scene_classes=config.get("scene_classes"),
            proj_dim=config.get("proj_dim"),
        )
    else:
        model = PersonAdjustmentPredictor(
            in_dim=int(config["in_dim"]),
            hidden=int(config.get("hidden", 64)),
            out_dim=int(config.get("out_dim", 2)),
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with open(standardizer_path, "r") as f:
        standardizer = Standardizer.from_dict(json.load(f))

    hist_projector = None
    if os.path.isfile(hist_proj_path):
        with open(hist_proj_path, "r") as f:
            hist_projector = HistogramProjector.from_dict(json.load(f))

    return model, standardizer, hist_projector, config


def build_person_adjustments(num_classes, person_class, gain, bias):
    gains = np.ones((num_classes,), dtype=np.float32)
    biases = np.zeros((num_classes,), dtype=np.float32)
    gains[int(person_class)] = float(gain)
    biases[int(person_class)] = float(bias)
    return {"class": {"gain": gains, "bias": biases}}


def build_hdrnet_semantic_adjustments(
    num_classes,
    person_class,
    coeff_scale,
    coeff_bias,
    guide_bias,
    coeffs=12,
):
    coeff_scale_map = np.ones((num_classes, coeffs), dtype=np.float32)
    coeff_bias_map = np.zeros((num_classes, coeffs), dtype=np.float32)
    coeff_scale_map[int(person_class)] = float(coeff_scale)
    coeff_bias_map[int(person_class)] = float(coeff_bias)
    guide_bias_map = np.zeros((num_classes,), dtype=np.float32)
    guide_bias_map[int(person_class)] = float(guide_bias)
    return {
        "coeff_scale": coeff_scale_map,
        "coeff_bias": coeff_bias_map,
        "guide_bias": guide_bias_map,
    }


def build_hdrnet_mix_adjustments(num_classes, person_class, mix):
    mix_map = np.zeros((num_classes,), dtype=np.float32)
    mix_map[int(person_class)] = float(mix)
    return mix_map


def decode_predictions(
    pred,
    gain_range=(0.25, 4.0),
    bias_range=(-0.3, 0.3),
    coeff_scale_range=0.5,
    coeff_bias_range=0.25,
    guide_bias_range=0.25,
):
    gain = float(pred[0])
    bias = float(pred[1])
    gain = max(gain_range[0], min(gain_range[1], gain))
    bias = max(bias_range[0], min(bias_range[1], bias))

    hdrnet = None
    if pred.shape[0] >= 6:
        mix = 1.0 / (1.0 + np.exp(-float(pred[2])))
        guide_bias = float(pred[3])
        coeff_scale = float(pred[4])
        coeff_bias = float(pred[5])

        guide_bias = max(-guide_bias_range, min(guide_bias_range, guide_bias))
        coeff_scale = max(1.0 - coeff_scale_range, min(1.0 + coeff_scale_range, coeff_scale))
        coeff_bias = max(-coeff_bias_range, min(coeff_bias_range, coeff_bias))
        hdrnet = {
            "mix": mix,
            "guide_bias": guide_bias,
            "coeff_scale": coeff_scale,
            "coeff_bias": coeff_bias,
        }
    return gain, bias, hdrnet


def derive_hdrnet_targets(
    gain,
    bias,
    gain_range=(0.25, 4.0),
    bias_range=(-0.3, 0.3),
    coeff_scale_range=0.5,
    coeff_bias_range=0.25,
    guide_bias_range=0.25,
    mix_slope=2.0,
):
    gain = float(gain)
    bias = float(bias)
    gain_span = max(1e-6, max(gain_range[1] - 1.0, 1.0 - gain_range[0]))
    gain_norm = (gain - 1.0) / gain_span
    gain_norm = max(-1.0, min(1.0, gain_norm))

    bias_span = max(abs(bias_range[0]), abs(bias_range[1]), 1e-6)
    bias_norm = bias / bias_span
    bias_norm = max(-1.0, min(1.0, bias_norm))

    mix = 1.0 / (1.0 + math.exp(-mix_slope * gain_norm))
    coeff_scale = 1.0 + coeff_scale_range * gain_norm
    coeff_bias = coeff_bias_range * bias_norm
    guide_bias = guide_bias_range * bias_norm
    mix_logit = math.log(max(1e-6, min(1.0 - 1e-6, mix)) / max(1e-6, 1.0 - mix))
    return mix_logit, guide_bias, coeff_scale, coeff_bias


def predict_person_adjustments(
    model,
    standardizer,
    linear_16,
    seg_map,
    exposure_ev,
    person_class,
    num_classes,
    bins=16,
    quantiles=None,
    quantile_degree=2,
    hist_projector=None,
    include_raw_hist=True,
    device="cpu",
):
    features = compute_features(
        linear_16,
        seg_map,
        exposure_ev,
        person_class,
        bins=bins,
        quantiles=quantiles,
        quantile_degree=quantile_degree,
        hist_projector=hist_projector,
        include_raw_hist=include_raw_hist,
    )
    x = standardizer.transform(features[None, :])
    x_t = torch.from_numpy(x).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x_t).cpu().numpy()[0]
    gain, bias, _ = decode_predictions(pred)
    return build_person_adjustments(num_classes, person_class, gain, bias)


def predict_person_hdrnet_adjustments(
    model,
    standardizer,
    linear_16,
    seg_map,
    exposure_ev,
    person_class,
    num_classes,
    bins=16,
    quantiles=None,
    quantile_degree=2,
    hist_projector=None,
    include_raw_hist=True,
    coeffs=12,
    device="cpu",
):
    features = compute_features(
        linear_16,
        seg_map,
        exposure_ev,
        person_class,
        bins=bins,
        quantiles=quantiles,
        quantile_degree=quantile_degree,
        hist_projector=hist_projector,
        include_raw_hist=include_raw_hist,
    )
    x = standardizer.transform(features[None, :])
    x_t = torch.from_numpy(x).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x_t).cpu().numpy()[0]

    gain, bias, hdrnet = decode_predictions(pred)
    adjustments = {
        "base": build_person_adjustments(num_classes, person_class, gain, bias),
    }
    if hdrnet is not None:
        adjustments["mix"] = build_hdrnet_mix_adjustments(num_classes, person_class, hdrnet["mix"])
        adjustments["hdrnet_semantic"] = build_hdrnet_semantic_adjustments(
            num_classes,
            person_class,
            hdrnet["coeff_scale"],
            hdrnet["coeff_bias"],
            hdrnet["guide_bias"],
            coeffs=coeffs,
        )
    return adjustments
