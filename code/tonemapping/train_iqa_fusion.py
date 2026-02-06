import argparse
import json

try:
    from .iqa_fusion import (
        build_adapters_from_config,
        save_fusion_model,
        train_fusion,
    )
except ImportError:
    from iqa_fusion import (
        build_adapters_from_config,
        save_fusion_model,
        train_fusion,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Train fusion IQA with pairwise labels")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--adapters-json", required=True)
    parser.add_argument("--output-dir", default="fusion_iqa")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--no-hist", action="store_true")
    parser.add_argument("--no-confidence", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    with open(args.adapters_json, "r") as f:
        adapters_config = json.load(f)

    adapters = build_adapters_from_config(adapters_config["adapters"])

    model, config = train_fusion(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        adapters=adapters,
        include_hist=not args.no_hist,
        hist_bins=args.hist_bins,
        include_confidence=not args.no_confidence,
        hidden=args.hidden,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    save_fusion_model(model, config, adapters_config, args.output_dir)


if __name__ == "__main__":
    main()
