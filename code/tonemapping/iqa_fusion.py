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


class IQAFeatureExtractor(object):
    def __init__(self, adapters, include_hist=True, hist_bins=16, include_confidence=True):
        self.adapters = adapters
        self.include_hist = bool(include_hist)
        self.hist_bins = int(hist_bins)
        self.include_confidence = bool(include_confidence)

    def feature_dim(self):
        num_models = len(self.adapters)
        dims = num_models
        if self.include_confidence:
            dims += num_models
        dims += int(num_models * (num_models - 1) / 2)
        dims += 6
        if self.include_hist:
            dims += self.hist_bins
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
    hidden=64,
    batch_size=64,
    epochs=10,
    lr=1e-3,
    device="cpu",
):
    feature_extractor = IQAFeatureExtractor(
        adapters,
        include_hist=include_hist,
        hist_bins=hist_bins,
        include_confidence=include_confidence,
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

    feature_extractor = IQAFeatureExtractor(
        adapters,
        include_hist=bool(config.get("include_hist", True)),
        hist_bins=int(config.get("hist_bins", 16)),
        include_confidence=bool(config.get("include_confidence", True)),
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
        elif spec.get("type") == "callable":
            raise ValueError("callable adapter must be injected in code")
        else:
            raise ValueError("Unknown adapter type: {}".format(spec.get("type")))
    return adapters
