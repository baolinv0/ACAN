"""
Analyze person brightness distribution and train an active adjustment predictor.
"""

import argparse
import json
import os

import numpy as np

try:
    from .person_adjustment_predictor import (
        compute_features,
        compute_histogram,
        compute_luminance,
        compute_stats,
        derive_hdrnet_targets,
        feature_names,
        fit_person_linear_adjustment,
        HistogramProjector,
        pearsonr,
        save_predictor,
        train_predictor,
    )
    from .semantic_tone_mapper import ToneMappingDataset
except ImportError:
    from person_adjustment_predictor import (
        compute_features,
        compute_histogram,
        compute_luminance,
        compute_stats,
        derive_hdrnet_targets,
        feature_names,
        fit_person_linear_adjustment,
        HistogramProjector,
        pearsonr,
        save_predictor,
        train_predictor,
    )
    from semantic_tone_mapper import ToneMappingDataset


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _compute_gt_stats(gt, seg_map, person_class):
    luma_gt = compute_luminance(gt)
    mask = seg_map == int(person_class)
    return compute_stats(luma_gt, mask)


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


def analyze_dataset(args):
    dataset = ToneMappingDataset(args.manifest, root=args.data_root)

    feats = []
    targets = []
    gt_stats = []
    person_counts = []
    hist_vecs = []
    scene_feats = []
    skipped = 0
    for idx in range(len(dataset)):
        linear, seg, gt, ev = dataset[idx]
        linear_np = linear.numpy()
        seg_np = seg.numpy()
        gt_np = gt.numpy()
        person_counts.append(int((seg_np == int(args.person_class)).sum()))
        scene_feats.append(_compute_scene_features(linear_np, hist_bins=args.hist_bins))

        parts = compute_features(
            linear_np,
            seg_np,
            float(ev.item()),
            args.person_class,
            bins=args.hist_bins,
            quantiles=args.quantiles,
            quantile_degree=args.quantile_degree,
            return_parts=True,
        )
        feats.append(parts["features"])
        hist_vecs.append(parts["hist"])

        stats = _compute_gt_stats(gt_np, seg_np, args.person_class)
        gt_stats.append(stats)

        target = fit_person_linear_adjustment(
            linear_np,
            gt_np,
            seg_np,
            args.person_class,
            min_pixels=args.min_person_pixels,
            gain_range=(args.gain_min, args.gain_max),
            bias_range=(args.bias_min, args.bias_max),
        )
        if target is None:
            skipped += 1
            targets.append([np.nan, np.nan])
        else:
            targets.append(list(target))

    feats = np.asarray(feats, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    hist_vecs = np.asarray(hist_vecs, dtype=np.float32)
    scene_feats = np.asarray(scene_feats, dtype=np.float32)

    hist_projector = None
    if args.hist_proj_dim is not None and args.hist_proj_dim > 0:
        hist_projector = HistogramProjector(dim=args.hist_proj_dim).fit(hist_vecs)
        feats = []
        for idx in range(len(dataset)):
            linear, seg, _, ev = dataset[idx]
            linear_np = linear.numpy()
            seg_np = seg.numpy()
            feature = compute_features(
                linear_np,
                seg_np,
                float(ev.item()),
                args.person_class,
                bins=args.hist_bins,
                quantiles=args.quantiles,
                quantile_degree=args.quantile_degree,
                hist_projector=hist_projector,
                include_raw_hist=not args.hist_proj_only,
            )
            feats.append(feature)
        feats = np.asarray(feats, dtype=np.float32)

    valid_mask = np.isfinite(targets[:, 0]) & np.isfinite(targets[:, 1])
    valid_feats = feats[valid_mask]
    valid_targets = targets[valid_mask]
    person_valid = np.array(person_counts) >= int(args.min_person_pixels)

    report = {
        "num_samples": int(len(dataset)),
        "num_valid_targets": int(valid_feats.shape[0]),
        "num_skipped": int(skipped),
        "num_person_valid": int(person_valid.sum()),
        "feature_dim": int(feats.shape[1]),
        "hist_bins": int(args.hist_bins),
        "hist_proj_dim": int(args.hist_proj_dim) if args.hist_proj_dim else 0,
        "hist_proj_only": bool(args.hist_proj_only),
        "quantiles": args.quantiles,
        "quantile_degree": int(args.quantile_degree),
    }

    feat_names = feature_names(
        bins=args.hist_bins,
        quantiles=args.quantiles,
        quantile_degree=args.quantile_degree,
        hist_proj_dim=args.hist_proj_dim,
        include_raw_hist=not args.hist_proj_only,
    )
    gt_means = np.array([s["mean"] for s in gt_stats], dtype=np.float32)
    gt_p50 = np.array([s["p50"] for s in gt_stats], dtype=np.float32)
    gt_p95 = np.array([s["p95"] for s in gt_stats], dtype=np.float32)
    gt_means_valid = gt_means[person_valid]
    gt_p50_valid = gt_p50[person_valid]
    gt_p95_valid = gt_p95[person_valid]

    report["gt_person_stats_mean"] = {
        "mean": float(gt_means_valid.mean()) if gt_means_valid.size > 0 else 0.0,
        "p50": float(gt_p50_valid.mean()) if gt_p50_valid.size > 0 else 0.0,
        "p95": float(gt_p95_valid.mean()) if gt_p95_valid.size > 0 else 0.0,
    }

    correlations = {"gt_mean": [], "gt_p50": [], "gt_p95": [], "gain": [], "bias": []}
    for i, name in enumerate(feat_names):
        correlations["gt_mean"].append((name, pearsonr(feats[person_valid, i], gt_means_valid)))
        correlations["gt_p50"].append((name, pearsonr(feats[person_valid, i], gt_p50_valid)))
        correlations["gt_p95"].append((name, pearsonr(feats[person_valid, i], gt_p95_valid)))
        if valid_feats.shape[0] > 0:
            correlations["gain"].append((name, pearsonr(valid_feats[:, i], valid_targets[:, 0])))
            correlations["bias"].append((name, pearsonr(valid_feats[:, i], valid_targets[:, 1])))

    for key in correlations:
        correlations[key] = sorted(correlations[key], key=lambda x: abs(x[1]), reverse=True)

    report["correlations_top"] = {
        key: correlations[key][: args.top_k] for key in correlations
    }

    scene_labels = None
    if args.scene_clusters is not None and args.scene_clusters > 0:
        scene_labels = _kmeans_numpy(scene_feats, k=args.scene_clusters, iters=25, seed=0)
        report["scene_clusters"] = int(args.scene_clusters)
        report["scene_cluster_counts"] = [
            int((scene_labels == i).sum()) for i in range(int(args.scene_clusters))
        ]

    return feats, targets, report, hist_projector, scene_labels


def build_parser():
    parser = argparse.ArgumentParser(description="Person brightness analysis and predictor")
    parser.add_argument("--manifest", required=True, help="CSV manifest path")
    parser.add_argument("--data-root", default=None, help="Root for relative paths")
    parser.add_argument("--person-class", type=int, required=True)
    parser.add_argument("--output-dir", default="person_brightness")
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--hist-proj-dim", type=int, default=0)
    parser.add_argument("--hist-proj-only", action="store_true")
    parser.add_argument("--quantiles", default="5,25,50,75,95")
    parser.add_argument("--quantile-degree", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-person-pixels", type=int, default=200)
    parser.add_argument("--gain-min", type=float, default=0.25)
    parser.add_argument("--gain-max", type=float, default=4.0)
    parser.add_argument("--bias-min", type=float, default=-0.3)
    parser.add_argument("--bias-max", type=float, default=0.3)
    parser.add_argument("--scene-clusters", type=int, default=0)
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
    parser.add_argument("--hdrnet-link", action="store_true")
    parser.add_argument("--hdrnet-coeff-scale-range", type=float, default=0.5)
    parser.add_argument("--hdrnet-coeff-bias-range", type=float, default=0.25)
    parser.add_argument("--hdrnet-guide-bias-range", type=float, default=0.25)
    parser.add_argument("--hdrnet-mix-slope", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if isinstance(args.quantiles, str):
        args.quantiles = [int(x) for x in args.quantiles.split(",") if x.strip()]

    feats, targets, report, hist_projector, scene_labels = analyze_dataset(args)
    _save_json(os.path.join(args.output_dir, "analysis_report.json"), report)
    np.save(os.path.join(args.output_dir, "features.npy"), feats)
    np.save(os.path.join(args.output_dir, "targets.npy"), targets)
    if hist_projector is not None:
        _save_json(os.path.join(args.output_dir, "hist_projector.json"), hist_projector.to_dict())
    if scene_labels is not None:
        np.save(os.path.join(args.output_dir, "scene_labels.npy"), scene_labels)

    print("Analysis saved to {}".format(args.output_dir))
    print("Valid targets: {}".format(report["num_valid_targets"]))

    if args.train_predictor:
        valid_mask = np.isfinite(targets[:, 0]) & np.isfinite(targets[:, 1])
        feats_valid = feats[valid_mask]
        targets_valid = targets[valid_mask]
        scene_labels_valid = scene_labels[valid_mask] if scene_labels is not None else None
        if args.hdrnet_link:
            hdr_targets = []
            for gain, bias in targets_valid:
                mix_logit, guide_bias, coeff_scale, coeff_bias = derive_hdrnet_targets(
                    gain,
                    bias,
                    gain_range=(args.gain_min, args.gain_max),
                    bias_range=(args.bias_min, args.bias_max),
                    coeff_scale_range=args.hdrnet_coeff_scale_range,
                    coeff_bias_range=args.hdrnet_coeff_bias_range,
                    guide_bias_range=args.hdrnet_guide_bias_range,
                    mix_slope=args.hdrnet_mix_slope,
                )
                hdr_targets.append([mix_logit, guide_bias, coeff_scale, coeff_bias])
            hdr_targets = np.asarray(hdr_targets, dtype=np.float32)
            targets_valid = np.concatenate([targets_valid, hdr_targets], axis=1)
        model, standardizer = train_predictor(
            feats_valid,
            targets_valid,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            embed_dim=args.embed_dim,
            batch_size=args.batch_size,
            scene_labels=scene_labels_valid,
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
            "in_dim": int(feats_valid.shape[1]),
            "hidden": int(args.hidden),
            "embed_dim": int(args.embed_dim),
            "out_dim": int(targets_valid.shape[1]),
            "scene_classes": int(scene_labels_valid.max() + 1) if scene_labels_valid is not None else None,
            "proj_dim": int(args.contrastive_proj_dim) if args.contrastive_weight > 0 else None,
            "hist_bins": int(args.hist_bins),
            "hist_proj_dim": int(args.hist_proj_dim) if args.hist_proj_dim else 0,
            "quantiles": args.quantiles,
            "quantile_degree": int(args.quantile_degree),
            "hdrnet_link": bool(args.hdrnet_link),
        }
        save_predictor(model, standardizer, args.output_dir, config=config, hist_projector=hist_projector)
        print("Predictor saved to {}".format(args.output_dir))
