"""
Analyze image types and redefine semantic scale coefficients.
"""

import argparse
import csv
import json
import os

import numpy as np

try:
    from .scale_coefficient_redefinition import (
        ScaleCoefficientRedefiner,
        build_feature_names,
        compute_image_features,
        compute_seg_distribution,
        extract_scale_coeffs,
        fisher_score,
        nearest_centroid_accuracy,
        normalize_adjustments,
        standardize_features,
    )
except ImportError:
    from scale_coefficient_redefinition import (
        ScaleCoefficientRedefiner,
        build_feature_names,
        compute_image_features,
        compute_seg_distribution,
        extract_scale_coeffs,
        fisher_score,
        nearest_centroid_accuracy,
        normalize_adjustments,
        standardize_features,
    )


def _read_image(path, keep_channels=False):
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
        if arr.ndim == 2 and keep_channels:
            arr = arr[:, :, None]
        return arr


def _read_linear(path):
    arr = _read_image(path, keep_channels=True)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def _read_seg(path):
    arr = _read_image(path, keep_channels=False)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def load_rows(csv_path, data_root, image_id_column, adj_column, linear_col, seg_col, exp_col):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get(image_id_column) if image_id_column else None
            if not image_id:
                image_id = row[linear_col]
            linear_path = row[linear_col]
            seg_path = row[seg_col]
            exposure = float(row.get(exp_col, "0") or 0.0)
            adjustments = row[adj_column]
            if data_root:
                linear_path = os.path.join(data_root, linear_path)
                seg_path = os.path.join(data_root, seg_path)
            rows.append(
                {
                    "image_id": image_id,
                    "linear_path": linear_path,
                    "seg_path": seg_path,
                    "exposure": exposure,
                    "adjustments": adjustments,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Redefine semantic scale coefficients")
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--image-id-column", default="image_id")
    parser.add_argument("--adjustments-column", default="adjustments_json")
    parser.add_argument("--linear-column", default="linear_path")
    parser.add_argument("--seg-column", default="seg_path")
    parser.add_argument("--exposure-column", default="exposure_ev")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--scale-key", default="base.class.gain")
    parser.add_argument("--coeffs", type=int, default=12)
    parser.add_argument("--coeff-reduce", choices=["mean", "flatten"], default="mean")
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--num-types", type=int, default=6)
    parser.add_argument("--cluster-use-coeffs", action="store_true")
    parser.add_argument("--residual-mode", choices=["delta", "ratio", "zscore"], default="delta")
    parser.add_argument("--top-k-features", type=int, default=20)
    parser.add_argument("--export-selected-features", action="store_true")
    parser.add_argument("--output-dir", default="scale_redefinition")
    args = parser.parse_args()

    rows = load_rows(
        args.data_csv,
        args.data_root,
        args.image_id_column,
        args.adjustments_column,
        args.linear_column,
        args.seg_column,
        args.exposure_column,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    features = []
    coeffs = []
    for row in rows:
        linear_np = _read_linear(row["linear_path"])
        seg_np = _read_seg(row["seg_path"])
        image_feat = compute_image_features(linear_np, hist_bins=args.hist_bins)
        seg_dist = compute_seg_distribution(seg_np, args.num_classes)
        feat = np.concatenate([image_feat, seg_dist], axis=0)
        if row["exposure"] is not None:
            feat = np.concatenate([feat, np.array([row["exposure"]], dtype=np.float32)], axis=0)
        features.append(feat)

        coeff_vec = extract_scale_coeffs(
            row["adjustments"],
            args.scale_key,
            args.num_classes,
            coeffs=args.coeffs,
            reduce=args.coeff_reduce,
        )
        coeffs.append(coeff_vec)

    features = np.asarray(features, dtype=np.float32)
    coeffs = np.asarray(coeffs, dtype=np.float32)

    if args.cluster_use_coeffs:
        cluster_input = np.concatenate([features, coeffs], axis=1)
    else:
        cluster_input = features

    # standardize clustering input to balance scales
    mean = cluster_input.mean(axis=0, keepdims=True)
    std = cluster_input.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    cluster_input = (cluster_input - mean) / std

    redefiner = ScaleCoefficientRedefiner(
        num_types=args.num_types, residual_mode=args.residual_mode, seed=0
    ).fit(cluster_input, coeffs)
    residuals = redefiner.transform(coeffs)

    feature_names = build_feature_names(
        num_classes=args.num_classes,
        hist_bins=args.hist_bins,
        include_exposure=True,
    )

    features_std, feat_mean, feat_std = standardize_features(features)
    type_acc_full, centroids_full = nearest_centroid_accuracy(features_std, redefiner.labels)
    scores = fisher_score(features_std, redefiner.labels)
    top_k = min(int(args.top_k_features), scores.shape[0]) if scores.size > 0 else 0
    top_idx = np.argsort(scores)[::-1][:top_k] if top_k > 0 else np.array([], dtype=np.int64)
    top_features = [(feature_names[i], float(scores[i])) for i in top_idx]
    selected_features = features_std[:, top_idx] if top_idx.size > 0 else np.zeros((features_std.shape[0], 0))
    type_acc_top, centroids_top = nearest_centroid_accuracy(selected_features, redefiner.labels)

    summary = redefiner.summary(coeffs)
    summary.update(
        {
            "num_samples": int(coeffs.shape[0]),
            "feature_dim": int(features.shape[1]),
            "coeff_dim": int(coeffs.shape[1]),
            "scale_key": args.scale_key,
            "coeff_reduce": args.coeff_reduce,
            "num_types": int(args.num_types),
            "residual_mode": args.residual_mode,
            "cluster_use_coeffs": bool(args.cluster_use_coeffs),
            "type_classifier_accuracy_full": float(type_acc_full),
            "type_classifier_accuracy_topk": float(type_acc_top),
            "type_feature_topk": top_features,
        }
    )

    np.save(os.path.join(args.output_dir, "features.npy"), features)
    np.save(os.path.join(args.output_dir, "coeffs.npy"), coeffs)
    np.save(os.path.join(args.output_dir, "type_ids.npy"), redefiner.labels)
    np.save(os.path.join(args.output_dir, "residuals.npy"), residuals)

    params = {
        "coeff_mean": redefiner.coeff_mean.tolist() if redefiner.coeff_mean is not None else None,
        "coeff_std": redefiner.coeff_std.tolist() if redefiner.coeff_std is not None else None,
    }
    with open(os.path.join(args.output_dir, "redefinition_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(args.output_dir, "analysis_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if args.export_selected_features:
        np.save(os.path.join(args.output_dir, "type_selected_features.npy"), selected_features)
        np.save(os.path.join(args.output_dir, "type_centroids.npy"), centroids_top)
        with open(os.path.join(args.output_dir, "type_feature_info.json"), "w") as f:
            json.dump(
                {
                    "feature_names": feature_names,
                    "selected_indices": top_idx.tolist(),
                    "selected_names": [name for name, _ in top_features],
                    "mean": feat_mean.reshape(-1).tolist(),
                    "std": feat_std.reshape(-1).tolist(),
                },
                f,
                indent=2,
            )

    mapping_path = os.path.join(args.output_dir, "redefined_coeffs.csv")
    with open(mapping_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "type_id", "residual"])
        for idx, row in enumerate(rows):
            writer.writerow([row["image_id"], int(redefiner.labels[idx]), residuals[idx].tolist()])


if __name__ == "__main__":
    main()
