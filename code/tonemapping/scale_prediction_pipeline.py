"""
Unified inference pipeline: type classifier + scale predictor.
"""

import json
import os

import numpy as np
import torch

try:
    from .person_adjustment_predictor import load_predictor_bundle
    from .scale_coefficient_redefinition import compute_full_features
    from .type_classifier import load_type_classifier_bundle
except ImportError:
    from person_adjustment_predictor import load_predictor_bundle
    from scale_coefficient_redefinition import compute_full_features
    from type_classifier import load_type_classifier_bundle


def _set_by_path(data, path, value):
    keys = path.split(".")
    cur = data
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value
    return data


class ScalePredictionPipeline(object):
    def __init__(
        self,
        predictor_model,
        predictor_standardizer,
        predictor_config,
        type_classifier,
        redefinition_params=None,
        residual_mode=None,
        scale_key=None,
        coeff_reduce="mean",
        coeffs=12,
        device="cpu",
    ):
        self.model = predictor_model
        self.standardizer = predictor_standardizer
        self.config = predictor_config or {}
        self.type_classifier = type_classifier
        self.redefinition_params = redefinition_params
        self.residual_mode = residual_mode or self.config.get("residual_mode", "delta")
        self.scale_key = scale_key or self.config.get("scale_key", "base.class.gain")
        self.coeff_reduce = coeff_reduce or self.config.get("coeff_reduce", "mean")
        self.coeffs = int(coeffs)
        self.device = torch.device(device)

        self.selected_indices = self.config.get("selected_indices")
        self.num_classes = int(self.config.get("num_classes", 0))
        self.hist_bins = int(self.config.get("hist_bins", 16))

    def _predict_raw(self, features):
        x = self.standardizer.transform(features[None, :])
        x_t = torch.from_numpy(x).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_t).cpu().numpy()[0]
        return pred

    def _apply_selected_indices(self, features):
        if self.selected_indices is None:
            return features
        idx = np.asarray(self.selected_indices, dtype=np.int64)
        return features[idx]

    def _reconstruct_coeffs(self, pred, type_id):
        pred = np.asarray(pred, dtype=np.float32)
        coeffs = pred.copy()
        if self.config.get("target_kind") != "residuals":
            return coeffs

        if self.redefinition_params is None:
            raise ValueError("redefinition_params required for residuals")

        mean = np.asarray(self.redefinition_params.get("coeff_mean"), dtype=np.float32)
        std = np.asarray(self.redefinition_params.get("coeff_std"), dtype=np.float32)
        if mean.ndim == 1:
            mean = mean[None, :]
        if std.ndim == 1:
            std = std[None, :]

        t = int(type_id)
        mean_t = mean[t]
        std_t = std[t] if std.size > 0 else np.ones_like(mean_t)

        if self.residual_mode == "ratio":
            coeffs = coeffs * np.maximum(mean_t, 1e-6)
        elif self.residual_mode == "zscore":
            coeffs = coeffs * np.maximum(std_t, 1e-6) + mean_t
        else:
            coeffs = coeffs + mean_t
        return coeffs

    def _format_coeffs(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=np.float32)
        if self.coeff_reduce == "flatten":
            coeffs = coeffs.reshape(self.num_classes, self.coeffs)
        return coeffs

    def predict(
        self,
        linear_16,
        seg_map,
        exposure_ev=None,
        type_id=None,
        return_adjustments=True,
    ):
        if type_id is None:
            if self.type_classifier is None:
                raise ValueError("type_id or type_classifier required")
            type_id = self.type_classifier.predict_from_inputs(
                linear_16, seg_map, exposure_ev=exposure_ev
            )

        features = compute_full_features(
            linear_16,
            seg_map,
            num_classes=self.num_classes,
            hist_bins=self.hist_bins,
            exposure_ev=exposure_ev,
        )
        features = self._apply_selected_indices(features)
        pred = self._predict_raw(features)

        coeffs = self._reconstruct_coeffs(pred, type_id)
        coeffs = self._format_coeffs(coeffs)

        if not return_adjustments:
            return {
                "type_id": int(type_id),
                "coeffs": coeffs,
                "raw_pred": pred,
            }

        adjustments = _set_by_path({}, self.scale_key, coeffs)
        return {
            "type_id": int(type_id),
            "coeffs": coeffs,
            "adjustments": adjustments,
            "raw_pred": pred,
        }


def load_scale_prediction_pipeline(output_dir, device="cpu"):
    model, standardizer, _, config = load_predictor_bundle(output_dir, device=device)
    type_classifier = None
    try:
        type_classifier = load_type_classifier_bundle(output_dir)
    except Exception:
        type_classifier = None

    redefinition_params = None
    params_path = os.path.join(output_dir, "redefinition_params.json")
    if os.path.isfile(params_path):
        with open(params_path, "r") as f:
            redefinition_params = json.load(f)

    residual_mode = None
    scale_key = None
    coeff_reduce = None
    coeffs = None
    report_path = os.path.join(output_dir, "analysis_report.json")
    if os.path.isfile(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
            residual_mode = report.get("residual_mode")
            scale_key = report.get("scale_key")
            coeff_reduce = report.get("coeff_reduce")
            coeffs = report.get("coeff_dim")
    if config:
        residual_mode = residual_mode or config.get("residual_mode")
        scale_key = scale_key or config.get("scale_key")
        coeff_reduce = coeff_reduce or config.get("coeff_reduce")
    if coeffs is None:
        coeffs = int(config.get("coeffs", 12)) if config else 12

    return ScalePredictionPipeline(
        predictor_model=model,
        predictor_standardizer=standardizer,
        predictor_config=config,
        type_classifier=type_classifier,
        redefinition_params=redefinition_params,
        residual_mode=residual_mode,
        scale_key=scale_key,
        coeff_reduce=coeff_reduce,
        coeffs=coeffs,
        device=device,
    )
