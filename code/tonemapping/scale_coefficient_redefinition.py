"""
Redefine semantic scale coefficients by image types to ease prediction.
"""

import json

import numpy as np


def normalize_adjustments(adjustments):
    if adjustments is None:
        return {}
    if isinstance(adjustments, str):
        return json.loads(adjustments)
    return adjustments


def get_by_path(data, path):
    cur = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def extract_scale_coeffs(
    adjustments,
    scale_key,
    num_classes,
    coeffs=12,
    reduce="mean",
):
    data = normalize_adjustments(adjustments)
    value = get_by_path(data, scale_key)
    if value is None:
        if scale_key.endswith("gain") or scale_key.endswith("coeff_scale"):
            return np.ones((num_classes,), dtype=np.float32)
        return np.zeros((num_classes,), dtype=np.float32)

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.full((num_classes,), float(arr), dtype=np.float32)

    if arr.ndim == 1:
        if arr.shape[0] >= num_classes:
            return arr[:num_classes].astype(np.float32)
        out = np.ones((num_classes,), dtype=np.float32) * float(arr.mean())
        out[: arr.shape[0]] = arr
        return out

    if arr.ndim == 2:
        if arr.shape[0] < num_classes:
            pad = np.tile(arr.mean(axis=1, keepdims=True), (1, arr.shape[1]))
            arr = np.concatenate([arr, pad], axis=0)
        arr = arr[:num_classes]
        if reduce == "flatten":
            return arr.reshape(-1).astype(np.float32)
        return arr.mean(axis=1).astype(np.float32)

    return np.zeros((num_classes,), dtype=np.float32)


def compute_seg_distribution(seg_map, num_classes):
    seg_flat = seg_map.reshape(-1).astype(np.int64)
    counts = np.bincount(seg_flat, minlength=num_classes).astype(np.float32)
    denom = max(1.0, float(counts.sum()))
    return counts / denom


def compute_luminance(img):
    if img.ndim == 2:
        return img
    if img.shape[0] == 1:
        return img[0]
    if img.shape[0] >= 3:
        return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    return img.mean(axis=0)


def normalize_linear(linear_16):
    if linear_16.dtype == np.uint16:
        return linear_16.astype(np.float32) / 65535.0
    linear_16 = linear_16.astype(np.float32)
    max_val = float(linear_16.max()) if linear_16.size > 0 else 1.0
    if max_val > 2.0:
        return linear_16 / 65535.0
    return linear_16


def compute_histogram(luma, bins=16):
    values = np.clip(luma.reshape(-1), 0.0, 1.0)
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32)
    hist = hist / max(1.0, float(hist.sum()))
    return hist


def compute_image_features(linear_16, hist_bins=16):
    linear = normalize_linear(linear_16)
    luma = compute_luminance(linear)
    mean = float(luma.mean())
    std = float(luma.std())
    p5 = float(np.percentile(luma, 5))
    p50 = float(np.percentile(luma, 50))
    p95 = float(np.percentile(luma, 95))
    contrast = p95 - p5
    hist = compute_histogram(luma, bins=hist_bins)
    return np.concatenate(
        [np.array([mean, std, p5, p50, p95, contrast], dtype=np.float32), hist], axis=0
    )


def image_feature_names(hist_bins=16):
    names = [
        "luma_mean",
        "luma_std",
        "luma_p5",
        "luma_p50",
        "luma_p95",
        "luma_contrast",
    ]
    names += ["luma_hist_{}".format(i) for i in range(hist_bins)]
    return names


def build_feature_names(num_classes, hist_bins=16, include_exposure=True):
    names = image_feature_names(hist_bins=hist_bins)
    names += ["seg_ratio_{}".format(i) for i in range(num_classes)]
    if include_exposure:
        names.append("exposure_ev")
    return names


def standardize_features(x):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (x - mean) / std, mean, std


def fisher_score(features, labels):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    n, d = features.shape
    if n == 0:
        return np.zeros((d,), dtype=np.float32)
    classes = np.unique(labels)
    overall_mean = features.mean(axis=0)
    sb = np.zeros((d,), dtype=np.float32)
    sw = np.zeros((d,), dtype=np.float32)
    for c in classes:
        mask = labels == c
        if not np.any(mask):
            continue
        x_c = features[mask]
        mean_c = x_c.mean(axis=0)
        sb += float(mask.sum()) * (mean_c - overall_mean) ** 2
        sw += ((x_c - mean_c) ** 2).sum(axis=0)
    scores = sb / np.maximum(sw, 1e-6)
    return scores


def nearest_centroid_accuracy(features, labels):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.shape[0] == 0:
        return 0.0, np.zeros((0, features.shape[1]), dtype=np.float32)
    classes = np.unique(labels)
    centroids = []
    for c in classes:
        centroids.append(features[labels == c].mean(axis=0))
    centroids = np.stack(centroids, axis=0)
    dists = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    pred = classes[dists.argmin(axis=1)]
    acc = float((pred == labels).mean())
    return acc, centroids


def kmeans_numpy(x, k, iters=20, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, x.shape[1]), dtype=np.float32)
    k = min(int(k), n)
    indices = rng.choice(n, size=k, replace=False)
    centers = x[indices]
    for _ in range(iters):
        dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centers = []
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers.append(x[mask].mean(axis=0))
            else:
                new_centers.append(centers[j])
        centers = np.stack(new_centers, axis=0)
    return labels, centers


class ScaleCoefficientRedefiner(object):
    def __init__(self, num_types, residual_mode="delta", seed=0):
        self.num_types = int(num_types)
        self.residual_mode = str(residual_mode)
        self.seed = int(seed)
        self.labels = None
        self.centers = None
        self.coeff_mean = None
        self.coeff_std = None

    def fit(self, features, coeffs):
        labels, centers = kmeans_numpy(features, self.num_types, iters=25, seed=self.seed)
        self.labels = labels
        self.centers = centers
        coeffs = np.asarray(coeffs, dtype=np.float32)
        num_types = int(labels.max()) + 1 if labels.size > 0 else 0
        means = []
        stds = []
        for i in range(num_types):
            mask = labels == i
            if np.any(mask):
                means.append(coeffs[mask].mean(axis=0))
                stds.append(coeffs[mask].std(axis=0))
            else:
                means.append(np.zeros((coeffs.shape[1],), dtype=np.float32))
                stds.append(np.ones((coeffs.shape[1],), dtype=np.float32))
        self.coeff_mean = np.stack(means, axis=0)
        self.coeff_std = np.stack(stds, axis=0)
        return self

    def transform(self, coeffs, labels=None):
        coeffs = np.asarray(coeffs, dtype=np.float32)
        if labels is None:
            labels = self.labels
        residuals = []
        for idx, coeff in enumerate(coeffs):
            t = int(labels[idx])
            mean = self.coeff_mean[t]
            std = np.maximum(self.coeff_std[t], 1e-6)
            if self.residual_mode == "ratio":
                residuals.append(coeff / np.maximum(mean, 1e-6))
            elif self.residual_mode == "zscore":
                residuals.append((coeff - mean) / std)
            else:
                residuals.append(coeff - mean)
        return np.stack(residuals, axis=0)

    def inverse(self, residuals, labels=None):
        residuals = np.asarray(residuals, dtype=np.float32)
        if labels is None:
            labels = self.labels
        coeffs = []
        for idx, resid in enumerate(residuals):
            t = int(labels[idx])
            mean = self.coeff_mean[t]
            std = np.maximum(self.coeff_std[t], 1e-6)
            if self.residual_mode == "ratio":
                coeffs.append(resid * mean)
            elif self.residual_mode == "zscore":
                coeffs.append(resid * std + mean)
            else:
                coeffs.append(resid + mean)
        return np.stack(coeffs, axis=0)

    def summary(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=np.float32)
        residuals = self.transform(coeffs)
        orig_std = coeffs.std(axis=0).mean()
        resid_std = residuals.std(axis=0).mean()
        reduction = resid_std / max(1e-6, orig_std)
        return {
            "orig_std_mean": float(orig_std),
            "resid_std_mean": float(resid_std),
            "reduction_ratio": float(reduction),
        }
