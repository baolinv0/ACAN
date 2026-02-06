"""
Select best adjustments by score, analyze correlations, and train a predictor.
"""

import argparse
import csv
import json
import os

import numpy as np

try:
    from .adjustment_selector import flatten_adjustments, vector_to_adjustments
    from .person_adjustment_predictor import (
        HistogramProjector,
        compute_features,
        compute_histogram,
        compute_luminance,
        compute_stats,
        pearsonr,
        save_predictor,
        train_predictor,
    )
except ImportError:
    from adjustment_selector import flatten_adjustments, vector_to_adjustments
    from person_adjustment_predictor import (
        HistogramProjector,
        compute_features,
        compute_histogram,
        compute_luminance,
        compute_stats,
        pearsonr,
        save_predictor,
        train_predictor,
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


def _compute_scene_features(linear_np, hist_bins=16):
    luma = compute_luminance(linear_np)
    stats = compute_stats(luma, None)
    hist_vec = compute_histogram(luma, bins=hist_bins, mask=None)
    return np.array(
        [
            stats["mean"],
            stats["std"],
            stats["p5"],
            stats["p50"],
            stats["p95"],
            stats["p95"] - stats["p5"],
            *hist_vec.tolist(),
        ],
        dtype=np.float32,
    )


def _kmeans_numpy(x, k, iters=20, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    indices = rng.choice(n, size=min(k, n), replace=False)
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
    return labels


def load_candidates(csv_path, data_root, image_id_column, score_column, adj_column, linear_col, seg_col, exp_col):
    groups = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get(image_id_column) if image_id_column else None
            if not image_id:
                image_id = row[linear_col]
            linear_path = row[linear_col]
            seg_path = row[seg_col]
            exposure = float(row.get(exp_col, "0") or 0.0)
            score = float(row[score_column])
            adjustments = row[adj_column]
            if data_root:
                linear_path = os.path.join(data_root, linear_path)
                seg_path = os.path.join(data_root, seg_path)
            item = {
                "linear_path": linear_path,
                "seg_path": seg_path,
                "exposure": exposure,
                "score": score,
                "adjustments": adjustments,
            }
            groups.setdefault(image_id, []).append(item)
    return groups


def select_best(groups):
    selected = []
    for image_id, items in groups.items():
        items = sorted(items, key=lambda x: x["score"], reverse=True)
        best = dict(items[0])
        best["image_id"] = image_id
        selected.append(best)
    return selected


def analyze_and_prepare(args):
    groups = load_candidates(
        args.candidates_csv,
        args.data_root,
        args.image_id_column,
        args.score_column,
        args.adjustments_column,
        args.linear_column,
        args.seg_column,
        args.exposure_column,
    )
    selected = select_best(groups)

    feats = []
    targets = []
    scores = []
    hist_vecs = []
    scene_feats = []
    rows = []

    for item in selected:
        linear_np = _read_linear(item["linear_path"])
        seg_np = _read_seg(item["seg_path"])
        exposure = item["exposure"]
        scene_feats.append(_compute_scene_features(linear_np, hist_bins=args.hist_bins))

        parts = compute_features(
            linear_np,
            seg_np,
            exposure,
            args.focus_class,
            bins=args.hist_bins,
            quantiles=args.quantiles,
            quantile_degree=args.quantile_degree,
            return_parts=True,
        )
        feats.append(parts["features"])
        hist_vecs.append(parts["hist"])

        vec, names = flatten_adjustments(
            item["adjustments"],
            args.focus_class,
            include_hdrnet=not args.no_hdrnet,
            include_local=not args.no_local,
            coeffs=args.hdrnet_coeffs,
        )
        targets.append(vec)
        scores.append(item["score"])
        rows.append(item)

    feats = np.asarray(feats, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    hist_vecs = np.asarray(hist_vecs, dtype=np.float32)
    scene_feats = np.asarray(scene_feats, dtype=np.float32)

    hist_projector = None
    if args.hist_proj_dim is not None and args.hist_proj_dim > 0:
        hist_projector = HistogramProjector(dim=args.hist_proj_dim).fit(hist_vecs)
        feats = []
        for item in selected:
            linear_np = _read_linear(item["linear_path"])
            seg_np = _read_seg(item["seg_path"])
            exposure = item["exposure"]
            feature = compute_features(
                linear_np,
                seg_np,
                exposure,
                args.focus_class,
                bins=args.hist_bins,
                quantiles=args.quantiles,
                quantile_degree=args.quantile_degree,
                hist_projector=hist_projector,
                include_raw_hist=not args.hist_proj_only,
            )
            feats.append(feature)
        feats = np.asarray(feats, dtype=np.float32)

    scene_labels = None
    if args.scene_clusters is not None and args.scene_clusters > 0:
        scene_labels = _kmeans_numpy(scene_feats, k=args.scene_clusters, iters=25, seed=0)

    return feats, targets, scores, names, rows, hist_projector, scene_labels


def main():
    parser = argparse.ArgumentParser(description="Select best adjustments and train predictor")
    parser.add_argument("--candidates-csv", required=True, help="CSV with candidate adjustments+scores")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--image-id-column", default="image_id")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--adjustments-column", default="adjustments_json")
    parser.add_argument("--linear-column", default="linear_path")
    parser.add_argument("--seg-column", default="seg_path")
    parser.add_argument("--exposure-column", default="exposure_ev")
    parser.add_argument("--focus-class", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output-dir", default="selection_analysis")
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--hist-proj-dim", type=int, default=0)
    parser.add_argument("--hist-proj-only", action="store_true")
    parser.add_argument("--quantiles", default="5,25,50,75,95")
    parser.add_argument("--quantile-degree", type=int, default=2)
    parser.add_argument("--scene-clusters", type=int, default=0)
    parser.add_argument("--no-hdrnet", action="store_true")
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--hdrnet-coeffs", type=int, default=12)
    parser.add_argument("--train-predictor", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--scene-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-proj-dim", type=int, default=32)
    parser.add_argument("--augment-noise-std", type=float, default=0.01)
    parser.add_argument("--augment-dropout-prob", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if isinstance(args.quantiles, str):
        args.quantiles = [int(x) for x in args.quantiles.split(",") if x.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    feats, targets, scores, names, rows, hist_projector, scene_labels = analyze_and_prepare(args)

    np.save(os.path.join(args.output_dir, "features.npy"), feats)
    np.save(os.path.join(args.output_dir, "targets.npy"), targets)
    np.save(os.path.join(args.output_dir, "scores.npy"), scores)

    if hist_projector is not None:
        with open(os.path.join(args.output_dir, "hist_projector.json"), "w") as f:
            json.dump(hist_projector.to_dict(), f, indent=2)
    if scene_labels is not None:
        np.save(os.path.join(args.output_dir, "scene_labels.npy"), scene_labels)

    # Correlation analysis
    correlations = {}
    for j, name in enumerate(names):
        correlations[name] = []
        for i in range(feats.shape[1]):
            correlations[name].append(float(pearsonr(feats[:, i], targets[:, j])))
    corr_summary = {
        name: sorted(
            [(i, correlations[name][i]) for i in range(len(correlations[name]))],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]
        for name in correlations
    }

    report = {
        "num_samples": int(feats.shape[0]),
        "feature_dim": int(feats.shape[1]),
        "target_dim": int(targets.shape[1]),
        "target_names": names,
        "hist_bins": int(args.hist_bins),
        "hist_proj_dim": int(args.hist_proj_dim) if args.hist_proj_dim else 0,
        "hist_proj_only": bool(args.hist_proj_only),
        "quantiles": args.quantiles,
        "quantile_degree": int(args.quantile_degree),
        "corr_top10": corr_summary,
    }
    with open(os.path.join(args.output_dir, "analysis_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if args.train_predictor:
        model, standardizer = train_predictor(
            feats,
            targets,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            embed_dim=args.embed_dim,
            batch_size=args.batch_size,
            scene_labels=scene_labels,
            scene_weight=args.scene_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_proj_dim=args.contrastive_proj_dim,
            augment_noise_std=args.augment_noise_std,
            augment_dropout_prob=args.augment_dropout_prob,
            device=args.device,
        )
        config = {
            "model": "advanced",
            "in_dim": int(feats.shape[1]),
            "hidden": int(args.hidden),
            "embed_dim": int(args.embed_dim),
            "out_dim": int(targets.shape[1]),
            "scene_classes": int(scene_labels.max() + 1) if scene_labels is not None else None,
            "proj_dim": int(args.contrastive_proj_dim) if args.contrastive_weight > 0 else None,
            "target_names": names,
            "focus_class": int(args.focus_class),
            "num_classes": int(args.num_classes),
            "include_hdrnet": bool(not args.no_hdrnet),
            "include_local": bool(not args.no_local),
            "hdrnet_coeffs": int(args.hdrnet_coeffs),
            "hist_bins": int(args.hist_bins),
            "hist_proj_dim": int(args.hist_proj_dim) if args.hist_proj_dim else 0,
            "quantiles": args.quantiles,
            "quantile_degree": int(args.quantile_degree),
        }
        save_predictor(model, standardizer, args.output_dir, config=config, hist_projector=hist_projector)


if __name__ == "__main__":
    main()
