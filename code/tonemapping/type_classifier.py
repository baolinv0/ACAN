"""
Type classifier for online inference based on discriminative features.
"""

import json
import os

import numpy as np

try:
    from .scale_coefficient_redefinition import compute_full_features
except ImportError:
    from scale_coefficient_redefinition import compute_full_features


class TypeClassifier(object):
    def __init__(
        self,
        num_classes,
        hist_bins,
        include_exposure,
        selected_indices,
        mean,
        std,
        centroids,
        feature_names=None,
    ):
        self.num_classes = int(num_classes)
        self.hist_bins = int(hist_bins)
        self.include_exposure = bool(include_exposure)
        self.selected_indices = np.asarray(selected_indices, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, -1)
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.feature_names = feature_names

    def _standardize(self, features):
        return (features - self.mean) / np.maximum(self.std, 1e-6)

    def _select(self, features):
        if self.selected_indices.size == 0:
            return features
        return features[:, self.selected_indices]

    def predict(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        features = self._standardize(features)
        features = self._select(features)
        dists = ((features[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return dists.argmin(axis=1)

    def predict_from_inputs(self, linear_16, seg_map, exposure_ev=None):
        feat = compute_full_features(
            linear_16,
            seg_map,
            num_classes=self.num_classes,
            hist_bins=self.hist_bins,
            exposure_ev=exposure_ev if self.include_exposure else None,
        )
        return int(self.predict(feat)[0])


def load_type_classifier(info_path, centroids_path):
    with open(info_path, "r") as f:
        info = json.load(f)
    centroids = np.load(centroids_path)
    return TypeClassifier(
        num_classes=int(info["num_classes"]),
        hist_bins=int(info["hist_bins"]),
        include_exposure=bool(info.get("include_exposure", True)),
        selected_indices=info["selected_indices"],
        mean=info["mean"],
        std=info["std"],
        centroids=centroids,
        feature_names=info.get("feature_names"),
    )


def load_type_classifier_bundle(output_dir):
    info_path = os.path.join(output_dir, "type_feature_info.json")
    centroids_path = os.path.join(output_dir, "type_centroids.npy")
    if not os.path.isfile(info_path) or not os.path.isfile(centroids_path):
        raise ValueError("Missing type_feature_info.json or type_centroids.npy in {}".format(output_dir))
    return load_type_classifier(info_path, centroids_path)
