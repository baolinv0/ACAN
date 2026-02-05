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


def compute_features(
    linear_16,
    seg_map,
    exposure_ev,
    person_class,
    bins=16,
):
    linear = _normalize_linear(linear_16)
    luma = compute_luminance(linear)
    person_mask = seg_map == int(person_class)

    global_stats = compute_stats(luma, None)
    person_stats = compute_stats(luma, person_mask)
    global_hist = compute_histogram(luma, bins=bins, mask=None)
    person_hist = compute_histogram(luma, bins=bins, mask=person_mask)

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
    features = np.concatenate([features, global_hist, person_hist], axis=0)
    return features


def feature_names(bins=16):
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
    base += ["global_hist_{}".format(i) for i in range(bins)]
    base += ["person_hist_{}".format(i) for i in range(bins)]
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

    device = torch.device(device)
    model = PersonAdjustmentPredictor(in_dim=features.shape[1], hidden=hidden, out_dim=targets.shape[1])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)

    for epoch in range(epochs):
        model.train()
        pred = model(x_train_t)
        loss = loss_fn(pred, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            msg = "Epoch {}/{} loss {:.6f}".format(epoch + 1, epochs, loss.item())
            if x_val is not None and x_val.size > 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(torch.from_numpy(x_val).to(device))
                    val_loss = loss_fn(val_pred, torch.from_numpy(y_val).to(device))
                    msg += " val {:.6f}".format(val_loss.item())
            print(msg)

    return model, standardizer


def save_predictor(model, standardizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "predictor.pth"))
    with open(os.path.join(save_dir, "standardizer.json"), "w") as f:
        json.dump(standardizer.to_dict(), f, indent=2)


def load_predictor(model_path, standardizer_path, in_dim, hidden=64, device="cpu"):
    device = torch.device(device)
    model = PersonAdjustmentPredictor(in_dim=in_dim, hidden=hidden, out_dim=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with open(standardizer_path, "r") as f:
        standardizer = Standardizer.from_dict(json.load(f))
    return model, standardizer


def build_person_adjustments(num_classes, person_class, gain, bias):
    gains = np.ones((num_classes,), dtype=np.float32)
    biases = np.zeros((num_classes,), dtype=np.float32)
    gains[int(person_class)] = float(gain)
    biases[int(person_class)] = float(bias)
    return {"class": {"gain": gains, "bias": biases}}


def predict_person_adjustments(
    model,
    standardizer,
    linear_16,
    seg_map,
    exposure_ev,
    person_class,
    num_classes,
    bins=16,
    device="cpu",
):
    features = compute_features(linear_16, seg_map, exposure_ev, person_class, bins=bins)
    x = standardizer.transform(features[None, :])
    x_t = torch.from_numpy(x).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x_t).cpu().numpy()[0]
    gain, bias = float(pred[0]), float(pred[1])
    return build_person_adjustments(num_classes, person_class, gain, bias)
