import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

try:
    from .semantic_tone_mapper import SemanticToneMapper, ToneMappingDataset, tone_mapping_loss
except ImportError:
    from semantic_tone_mapper import SemanticToneMapper, ToneMappingDataset, tone_mapping_loss


def _set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    _set_seed(args.seed)

    dataset = ToneMappingDataset(args.manifest, root=args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = SemanticToneMapper(
        num_classes=args.num_classes,
        init_gain=args.init_gain,
        init_gamma=args.init_gamma,
        init_white=args.init_white,
        bias_range=args.bias_range,
        max_input=args.max_input,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0))

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for it, batch in enumerate(loader):
            linear, seg, gt, ev = batch
            linear = linear.to(device)
            seg = seg.to(device)
            gt = gt.to(device)
            ev = ev.to(device)

            pred = model(linear, seg, exposure_ev=ev, normalize_input=True)
            loss = tone_mapping_loss(pred, gt, use_smooth_l1=args.use_smooth_l1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if args.log_every > 0 and (it + 1) % args.log_every == 0:
                avg = epoch_loss / float(it + 1)
                print(
                    "Epoch {}/{} Iter {}/{} Loss {:.6f}".format(
                        epoch + 1, args.epochs, it + 1, len(loader), avg
                    )
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(1.0, float(len(loader)))
        print(
            "Epoch {}/{} done. Avg Loss {:.6f}. Time {:.2f}s".format(
                epoch + 1, args.epochs, avg_loss, elapsed
            )
        )

        if args.save_path:
            ckpt = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "num_classes": args.num_classes,
            }
            torch.save(ckpt, args.save_path)

    if args.export_lut:
        lut = model.export_quantized_lut(
            num_points=args.lut_points, bit_depth=args.lut_bit_depth, device="cpu"
        )
        lut_path = args.export_lut
        npy_dir = os.path.dirname(lut_path)
        if npy_dir:
            os.makedirs(npy_dir, exist_ok=True)
        import numpy as np

        np.save(lut_path, lut)
        print("Saved LUT to {}".format(lut_path))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train semantic-adjustable tone mapper with GT supervision"
    )
    parser.add_argument("--manifest", required=True, help="CSV manifest path")
    parser.add_argument("--data-root", default=None, help="Root for relative paths")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of semantic classes")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-gain", type=float, default=1.0)
    parser.add_argument("--init-gamma", type=float, default=2.2)
    parser.add_argument("--init-white", type=float, default=4.0)
    parser.add_argument("--bias-range", type=float, default=0.30)
    parser.add_argument("--max-input", type=float, default=8.0)
    parser.add_argument("--use-smooth-l1", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-path", default="semantic_tone_mapper.pth")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--export-lut", default=None, help="Path to save LUT .npy")
    parser.add_argument("--lut-points", type=int, default=1024)
    parser.add_argument("--lut-bit-depth", type=int, default=16)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
