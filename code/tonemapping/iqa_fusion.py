"""
Fusion IQA with DE-IQA, Q-Insight, MDIQA adapters and pairwise ranking.
"""

import csv
import importlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def load_image(path, to_rgb=True):
    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("cv2.imread failed")
        if img.ndim == 2:
            img = img[:, :, None]
        if img.shape[2] == 3 and to_rgb:
            img = img[:, :, ::-1]
        elif img.shape[2] == 4:
            img = img[:, :, :3]
            if to_rgb:
                img = img[:, :, ::-1]
        return img
    except Exception:
        from PIL import Image

        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return arr


def to_float_image(img):
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def compute_luminance(img):
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    if img.shape[2] >= 3:
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return img.mean(axis=2)


def compute_basic_stats(luma):
    p5 = float(np.percentile(luma, 5))
    p50 = float(np.percentile(luma, 50))
    p95 = float(np.percentile(luma, 95))
    mean = float(luma.mean())
    std = float(luma.std())
    contrast = p95 - p5
    return [mean, std, p5, p50, p95, contrast]


def compute_histogram(luma, bins=16):
    values = np.clip(luma.reshape(-1), 0.0, 1.0)
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32)
    hist = hist / max(1.0, float(hist.sum()))
    return hist


class IQAAdapter(object):
    def score(self, image):
        raise NotImplementedError()

    def confidence(self, image):
        return None


class ZScoreCalibrator(object):
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        self.mean = float(scores.mean()) if scores.size > 0 else 0.0
        self.std = float(scores.std()) if scores.size > 0 else 1.0
        if self.std < 1e-6:
            self.std = 1.0
        return self

    def transform(self, score):
        return (float(score) - self.mean) / self.std

    def to_dict(self):
        return {"type": "zscore", "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.mean = float(data.get("mean", 0.0))
        obj.std = float(data.get("std", 1.0))
        if obj.std < 1e-6:
            obj.std = 1.0
        return obj


class MinMaxCalibrator(object):
    def __init__(self):
        self.min = 0.0
        self.max = 1.0

    def fit(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            self.min = 0.0
            self.max = 1.0
        else:
            self.min = float(scores.min())
            self.max = float(scores.max())
            if abs(self.max - self.min) < 1e-6:
                self.max = self.min + 1.0
        return self

    def transform(self, score):
        return (float(score) - self.min) / (self.max - self.min)

    def to_dict(self):
        return {"type": "minmax", "min": self.min, "max": self.max}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.min = float(data.get("min", 0.0))
        obj.max = float(data.get("max", 1.0))
        if abs(obj.max - obj.min) < 1e-6:
            obj.max = obj.min + 1.0
        return obj


class DynamicIQAAdapter(IQAAdapter):
    def __init__(self, module, class_name, init_kwargs=None, score_fn="score", device="cpu"):
        init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
        mod = importlib.import_module(module)
        cls = getattr(mod, class_name)
        self.model = cls(**init_kwargs)
        self.score_fn = score_fn
        if hasattr(self.model, "to"):
            self.model.to(device)
        self.device = device

    def score(self, image):
        fn = getattr(self.model, self.score_fn)
        return float(fn(image))


class CallableIQAAdapter(IQAAdapter):
    def __init__(self, score_fn, conf_fn=None):
        self._score_fn = score_fn
        self._conf_fn = conf_fn

    def score(self, image):
        return float(self._score_fn(image))

    def confidence(self, image):
        if self._conf_fn is None:
            return None
        return float(self._conf_fn(image))


class QInsightAdapter(IQAAdapter):
    """
    Example adapter for Q-Insight.
    Provide module/class and score_fn matching the repo implementation.
    """

    def __init__(
        self,
        module,
        class_name=None,
        init_kwargs=None,
        score_fn=None,
        preprocess_fn=None,
        device="cpu",
    ):
        init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
        mod = importlib.import_module(module)
        if class_name:
            cls = getattr(mod, class_name)
            self.model = cls(**init_kwargs)
        elif hasattr(mod, "build_model"):
            self.model = mod.build_model(**init_kwargs)
        else:
            raise ValueError("Q-Insight adapter requires class_name or build_model()")
        if hasattr(self.model, "to"):
            self.model.to(device)
        self.device = device
        self.score_fn = score_fn
        self.preprocess_fn = preprocess_fn

    def score(self, image):
        if self.preprocess_fn is not None:
            image = self.preprocess_fn(image)
        if self.score_fn and hasattr(self.model, self.score_fn):
            return float(getattr(self.model, self.score_fn)(image))
        if hasattr(self.model, "score"):
            return float(self.model.score(image))
        if hasattr(self.model, "predict"):
            return float(self.model.predict(image))
        if callable(self.model):
            return float(self.model(image))
        raise ValueError("Q-Insight model does not expose score method")


class MDIQAAdapter(IQAAdapter):
    """
    Example adapter for MDIQA.
    Provide module/class and score_fn matching the repo implementation.
    """

    def __init__(
        self,
        module,
        class_name=None,
        init_kwargs=None,
        score_fn=None,
        preprocess_fn=None,
        device="cpu",
    ):
        init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
        mod = importlib.import_module(module)
        if class_name:
            cls = getattr(mod, class_name)
            self.model = cls(**init_kwargs)
        elif hasattr(mod, "build_model"):
            self.model = mod.build_model(**init_kwargs)
        else:
            raise ValueError("MDIQA adapter requires class_name or build_model()")
        if hasattr(self.model, "to"):
            self.model.to(device)
        self.device = device
        self.score_fn = score_fn
        self.preprocess_fn = preprocess_fn

    def score(self, image):
        if self.preprocess_fn is not None:
            image = self.preprocess_fn(image)
        if self.score_fn and hasattr(self.model, self.score_fn):
            return float(getattr(self.model, self.score_fn)(image))
        if hasattr(self.model, "score"):
            return float(self.model.score(image))
        if hasattr(self.model, "predict"):
            return float(self.model.predict(image))
        if callable(self.model):
            return float(self.model(image))
        raise ValueError("MDIQA model does not expose score method")


def _load_callable(path):
    if not path:
        return None
    if ":" not in path:
        raise ValueError("Callable path must be module:function")
    module, func = path.split(":", 1)
    return getattr(importlib.import_module(module), func)


class IQAFeatureExtractor(object):
    def __init__(
        self,
        adapters,
        include_hist=True,
        hist_bins=16,
        include_confidence=True,
        calibrators=None,
        reliability_weights=None,
        weight_mode="multiply",
    ):
        self.adapters = adapters
        self.include_hist = bool(include_hist)
        self.hist_bins = int(hist_bins)
        self.include_confidence = bool(include_confidence)
        self.calibrators = calibrators
        self.reliability_weights = reliability_weights
        self.weight_mode = weight_mode

    def feature_dim(self):
        num_models = len(self.adapters)
        dims = num_models
        if self.include_confidence:
            dims += num_models
        dims += int(num_models * (num_models - 1) / 2)
        dims += 6
        if self.include_hist:
            dims += self.hist_bins
        if self.weight_mode in ("feature", "both"):
            dims += num_models
        return dims

    def extract(self, img):
        img = to_float_image(img)
        luma = compute_luminance(img)
        stats = compute_basic_stats(luma)
        hist = compute_histogram(luma, bins=self.hist_bins) if self.include_hist else None

        scores = []
        confs = []
        for adapter in self.adapters:
            scores.append(adapter.score(img))
            conf = adapter.confidence(img)
            confs.append(0.0 if conf is None else float(conf))

        if self.calibrators is not None:
            scores = [
                cal.transform(score) if cal is not None else float(score)
                for score, cal in zip(scores, self.calibrators)
            ]

        weights = None
        if self.reliability_weights is not None:
            weights = [float(w) for w in self.reliability_weights]
            if self.weight_mode in ("multiply", "both"):
                scores = [s * w for s, w in zip(scores, weights)]

        diffs = []
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                diffs.append(abs(scores[i] - scores[j]))

        feats = []
        feats += scores
        if self.include_confidence:
            feats += confs
        feats += diffs
        feats += stats
        if hist is not None:
            feats += hist.tolist()
        if weights is not None and self.weight_mode in ("feature", "both"):
            feats += weights
        return np.asarray(feats, dtype=np.float32)


class FusionIQAModel(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super(FusionIQAModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PairwiseIQADataset(Dataset):
    def __init__(self, csv_path, feature_extractor, cache=True):
        self.rows = []
        self.feature_extractor = feature_extractor
        self.cache = {} if cache else None
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def _get_feature(self, path):
        if self.cache is not None and path in self.cache:
            return self.cache[path]
        img = load_image(path)
        feat = self.feature_extractor.extract(img)
        if self.cache is not None:
            self.cache[path] = feat
        return feat

    def __getitem__(self, idx):
        row = self.rows[idx]
        path_a = row["image_a"]
        path_b = row["image_b"]
        label = row.get("label", "1")
        label = float(label)
        label = 1.0 if label >= 0.5 else -1.0
        feat_a = self._get_feature(path_a)
        feat_b = self._get_feature(path_b)
        return feat_a, feat_b, label


def accuracy_to_weight(acc, alpha=4.0, min_weight=0.1, max_weight=2.0):
    w = float(acc)
    w = min(1.0, max(0.0, w))
    w = np.exp(alpha * (w - 0.5))
    w = min(max_weight, max(min_weight, w))
    return float(w)


def _collect_unique_images(csv_path):
    images = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            images[row["image_a"]] = True
            images[row["image_b"]] = True
    return list(images.keys())


def _collect_pair_rows(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "1")
            label = float(label)
            label = 1.0 if label >= 0.5 else -1.0
            rows.append((row["image_a"], row["image_b"], label))
    return rows


def fit_calibrators_and_weights(
    train_csv,
    adapters,
    calibrate_mode="zscore",
    reliability_alpha=4.0,
    min_weight=0.1,
    max_weight=2.0,
):
    if calibrate_mode not in ("zscore", "minmax", "none"):
        raise ValueError("calibrate_mode must be zscore, minmax, or none")

    images = _collect_unique_images(train_csv)
    score_cache = [{} for _ in adapters]

    for path in images:
        img = load_image(path)
        for idx, adapter in enumerate(adapters):
            score_cache[idx][path] = adapter.score(img)

    calibrators = []
    if calibrate_mode == "none":
        calibrators = [None for _ in adapters]
    else:
        for idx, adapter in enumerate(adapters):
            scores = list(score_cache[idx].values())
            if calibrate_mode == "minmax":
                calibrators.append(MinMaxCalibrator().fit(scores))
            else:
                calibrators.append(ZScoreCalibrator().fit(scores))

    rows = _collect_pair_rows(train_csv)
    accuracies = []
    for idx, adapter in enumerate(adapters):
        correct = 0
        total = 0
        for path_a, path_b, label in rows:
            s_a = score_cache[idx][path_a]
            s_b = score_cache[idx][path_b]
            cal = calibrators[idx]
            if cal is not None:
                s_a = cal.transform(s_a)
                s_b = cal.transform(s_b)
            pred = 1.0 if (s_a - s_b) >= 0 else -1.0
            if pred == label:
                correct += 1
            total += 1
        acc = float(correct) / max(1.0, float(total))
        accuracies.append(acc)

    weights = [
        accuracy_to_weight(acc, alpha=reliability_alpha, min_weight=min_weight, max_weight=max_weight)
        for acc in accuracies
    ]
    return calibrators, weights, accuracies


def ranknet_loss(score_a, score_b, label):
    diff = score_a - score_b
    return F.softplus(-label * diff).mean()


def pairwise_accuracy(score_a, score_b, label):
    pred = torch.sign(score_a - score_b)
    correct = (pred == label).float().mean().item()
    return correct


def train_fusion(
    train_csv,
    val_csv,
    adapters,
    include_hist=True,
    hist_bins=16,
    include_confidence=True,
    calibrate_mode="zscore",
    reliability_alpha=4.0,
    min_weight=0.1,
    max_weight=2.0,
    weight_mode="multiply",
    hidden=64,
    batch_size=64,
    epochs=10,
    lr=1e-3,
    device="cpu",
):
    calibrators, weights, accuracies = fit_calibrators_and_weights(
        train_csv,
        adapters,
        calibrate_mode=calibrate_mode,
        reliability_alpha=reliability_alpha,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    feature_extractor = IQAFeatureExtractor(
        adapters,
        include_hist=include_hist,
        hist_bins=hist_bins,
        include_confidence=include_confidence,
        calibrators=calibrators,
        reliability_weights=weights,
        weight_mode=weight_mode,
    )
    train_set = PairwiseIQADataset(train_csv, feature_extractor, cache=True)
    val_set = PairwiseIQADataset(val_csv, feature_extractor, cache=True) if val_csv else None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0) if val_set else None

    model = FusionIQAModel(in_dim=feature_extractor.feature_dim(), hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for feat_a, feat_b, label in train_loader:
            feat_a = feat_a.to(device)
            feat_b = feat_b.to(device)
            label = label.to(device)
            score_a = model(feat_a)
            score_b = model(feat_b)
            loss = ranknet_loss(score_a, score_b, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        msg = "Epoch {}/{} loss {:.6f}".format(epoch + 1, epochs, loss_sum / len(train_loader))

        if val_loader is not None:
            model.eval()
            accs = []
            with torch.no_grad():
                for feat_a, feat_b, label in val_loader:
                    feat_a = feat_a.to(device)
                    feat_b = feat_b.to(device)
                    label = label.to(device)
                    score_a = model(feat_a)
                    score_b = model(feat_b)
                    accs.append(pairwise_accuracy(score_a, score_b, label))
            msg += " val_acc {:.4f}".format(float(np.mean(accs)))
        print(msg)

    config = {
        "feature_dim": feature_extractor.feature_dim(),
        "hidden": hidden,
        "include_hist": include_hist,
        "hist_bins": hist_bins,
        "include_confidence": include_confidence,
        "calibrate_mode": calibrate_mode,
        "reliability_alpha": reliability_alpha,
        "reliability_min_weight": min_weight,
        "reliability_max_weight": max_weight,
        "weight_mode": weight_mode,
        "reliability_weights": weights,
        "reliability_acc": accuracies,
        "calibration": [c.to_dict() if c is not None else None for c in calibrators],
    }
    return model, config


def save_fusion_model(model, config, adapters_config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "fusion_iqa.pth"))
    with open(os.path.join(save_dir, "fusion_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(save_dir, "adapters.json"), "w") as f:
        json.dump(adapters_config, f, indent=2)


def load_fusion_model(save_dir, adapters, device="cpu"):
    with open(os.path.join(save_dir, "fusion_config.json"), "r") as f:
        config = json.load(f)
    model = FusionIQAModel(in_dim=int(config["feature_dim"]), hidden=int(config["hidden"]))
    model.load_state_dict(torch.load(os.path.join(save_dir, "fusion_iqa.pth"), map_location=device))
    model.to(device)
    model.eval()

    calibrators = []
    calib_data = config.get("calibration")
    if calib_data:
        for item in calib_data:
            if item is None:
                calibrators.append(None)
            elif item.get("type") == "minmax":
                calibrators.append(MinMaxCalibrator.from_dict(item))
            else:
                calibrators.append(ZScoreCalibrator.from_dict(item))
    else:
        calibrators = None

    feature_extractor = IQAFeatureExtractor(
        adapters,
        include_hist=bool(config.get("include_hist", True)),
        hist_bins=int(config.get("hist_bins", 16)),
        include_confidence=bool(config.get("include_confidence", True)),
        calibrators=calibrators,
        reliability_weights=config.get("reliability_weights"),
        weight_mode=config.get("weight_mode", "multiply"),
    )
    return model, feature_extractor


def build_adapters_from_config(adapters_config):
    adapters = []
    for spec in adapters_config:
        spec = dict(spec)
        if spec.get("type") == "dynamic":
            adapters.append(
                DynamicIQAAdapter(
                    module=spec["module"],
                    class_name=spec["class"],
                    init_kwargs=spec.get("init_kwargs"),
                    score_fn=spec.get("score_fn", "score"),
                    device=spec.get("device", "cpu"),
                )
            )
        elif spec.get("type") == "qinsight":
            preprocess_fn = _load_callable(spec.get("preprocess")) if spec.get("preprocess") else None
            adapters.append(
                QInsightAdapter(
                    module=spec["module"],
                    class_name=spec.get("class"),
                    init_kwargs=spec.get("init_kwargs"),
                    score_fn=spec.get("score_fn"),
                    preprocess_fn=preprocess_fn,
                    device=spec.get("device", "cpu"),
                )
            )
        elif spec.get("type") == "mdiqa":
            preprocess_fn = _load_callable(spec.get("preprocess")) if spec.get("preprocess") else None
            adapters.append(
                MDIQAAdapter(
                    module=spec["module"],
                    class_name=spec.get("class"),
                    init_kwargs=spec.get("init_kwargs"),
                    score_fn=spec.get("score_fn"),
                    preprocess_fn=preprocess_fn,
                    device=spec.get("device", "cpu"),
                )
            )
        elif spec.get("type") == "callable":
            raise ValueError("callable adapter must be injected in code")
        else:
            raise ValueError("Unknown adapter type: {}".format(spec.get("type")))
    return adapters


class FusionIQAPipeline(object):
    def __init__(self, model, feature_extractor, device="cpu"):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = torch.device(device)

    def score(self, image_or_path):
        if isinstance(image_or_path, str):
            img = load_image(image_or_path)
        else:
            img = image_or_path
        feat = self.feature_extractor.extract(img)
        feat_t = torch.from_numpy(feat).to(self.device)
        with torch.no_grad():
            score = self.model(feat_t.unsqueeze(0)).cpu().numpy()[0]
        return float(score)

    def score_pair(self, image_a, image_b):
        score_a = self.score(image_a)
        score_b = self.score(image_b)
        return score_a, score_b, (1.0 if score_a >= score_b else -1.0)

    def best_of(self, images):
        scores = []
        for img in images:
            scores.append(self.score(img))
        best_idx = int(np.argmax(scores)) if scores else -1
        return best_idx, scores


def load_fusion_pipeline(save_dir, adapters_config, device="cpu"):
    adapters = build_adapters_from_config(adapters_config["adapters"])
    model, extractor = load_fusion_model(save_dir, adapters, device=device)
    return FusionIQAPipeline(model, extractor, device=device)


def load_fusion_pipeline_from_dir(save_dir, device="cpu"):
    adapters_path = os.path.join(save_dir, "adapters.json")
    if not os.path.isfile(adapters_path):
        raise ValueError("adapters.json not found in {}".format(save_dir))
    with open(adapters_path, "r") as f:
        adapters_config = json.load(f)
    return load_fusion_pipeline(save_dir, adapters_config, device=device)
