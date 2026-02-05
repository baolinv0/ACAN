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
        compute_luminance,
        compute_stats,
        feature_names,
        fit_person_linear_adjustment,
        pearsonr,
        save_predictor,
        train_predictor,
    )
    from .semantic_tone_mapper import ToneMappingDataset
except ImportError:
    from person_adjustment_predictor import (
        compute_features,
        compute_luminance,
        compute_stats,
        feature_names,
        fit_person_linear_adjustment,
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


def analyze_dataset(args):
    dataset = ToneMappingDataset(args.manifest, root=args.data_root)

    feats = []
    targets = []
    gt_stats = []
    person_counts = []
    skipped = 0
    for idx in range(len(dataset)):
        linear, seg, gt, ev = dataset[idx]
        linear_np = linear.numpy()
        seg_np = seg.numpy()
        gt_np = gt.numpy()
        person_counts.append(int((seg_np == int(args.person_class)).sum()))

        feature = compute_features(
            linear_np,
            seg_np,
            float(ev.item()),
            args.person_class,
            bins=args.hist_bins,
        )
        feats.append(feature)

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

    valid_mask = np.isfinite(targets[:, 0]) & np.isfinite(targets[:, 1])
    valid_feats = feats[valid_mask]
    valid_targets = targets[valid_mask]
    person_valid = np.array(person_counts) >= int(args.min_person_pixels)

    report = {
        "num_samples": int(len(dataset)),
        "num_valid_targets": int(valid_feats.shape[0]),
        "num_skipped": int(skipped),
        "num_person_valid": int(person_valid.sum()),
    }

    feat_names = feature_names(bins=args.hist_bins)
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

    return feats, targets, report


def build_parser():
    parser = argparse.ArgumentParser(description="Person brightness analysis and predictor")
    parser.add_argument("--manifest", required=True, help="CSV manifest path")
    parser.add_argument("--data-root", default=None, help="Root for relative paths")
    parser.add_argument("--person-class", type=int, required=True)
    parser.add_argument("--output-dir", default="person_brightness")
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-person-pixels", type=int, default=200)
    parser.add_argument("--gain-min", type=float, default=0.25)
    parser.add_argument("--gain-max", type=float, default=4.0)
    parser.add_argument("--bias-min", type=float, default=-0.3)
    parser.add_argument("--bias-max", type=float, default=0.3)
    parser.add_argument("--train-predictor", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    feats, targets, report = analyze_dataset(args)
    _save_json(os.path.join(args.output_dir, "analysis_report.json"), report)
    np.save(os.path.join(args.output_dir, "features.npy"), feats)
    np.save(os.path.join(args.output_dir, "targets.npy"), targets)

    print("Analysis saved to {}".format(args.output_dir))
    print("Valid targets: {}".format(report["num_valid_targets"]))

    if args.train_predictor:
        valid_mask = np.isfinite(targets[:, 0]) & np.isfinite(targets[:, 1])
        feats_valid = feats[valid_mask]
        targets_valid = targets[valid_mask]
        model, standardizer = train_predictor(
            feats_valid,
            targets_valid,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            device=args.device,
        )
        save_predictor(model, standardizer, args.output_dir)
        print("Predictor saved to {}".format(args.output_dir))
