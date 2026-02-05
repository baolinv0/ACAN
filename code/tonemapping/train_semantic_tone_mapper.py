import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

try:
    from .semantic_tone_mapper import SemanticToneMapper, ToneMappingDataset, tone_mapping_loss
    from .hdrnet_tone_mapper import SemanticHDRNetToneMapper
except ImportError:
    from semantic_tone_mapper import SemanticToneMapper, ToneMappingDataset, tone_mapping_loss
    from hdrnet_tone_mapper import SemanticHDRNetToneMapper


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
    base_kwargs = {
        "init_gain": args.init_gain,
        "init_gamma": args.init_gamma,
        "init_white": args.init_white,
        "init_global_gain": args.init_global_gain,
        "init_global_gamma": args.init_global_gamma,
        "init_global_white": args.init_global_white,
        "bias_range": args.bias_range,
        "global_bias_range": args.global_bias_range,
        "max_input": args.max_input,
        "local_window": args.local_window,
        "local_gain_range": args.local_gain_range,
        "local_bias_range": args.local_bias_range,
        "local_enable": not args.disable_local,
        "local_method": args.local_method,
        "guided_eps": args.guided_eps,
        "bilateral_sigma_spatial": args.bilateral_sigma_spatial,
        "bilateral_sigma_range": args.bilateral_sigma_range,
    }

    if args.model == "hdrnet":
        base_kwargs["local_enable"] = bool(args.hdrnet_base_local)
        hdrnet_kwargs = {
            "grid_depth": args.hdrnet_grid_depth,
            "grid_height": args.hdrnet_grid_height,
            "grid_width": args.hdrnet_grid_width,
            "coeffs": args.hdrnet_coeffs,
            "embedding_dim": args.hdrnet_embedding_dim,
            "hidden": args.hdrnet_hidden,
            "guide_hidden": args.hdrnet_guide_hidden,
            "use_exposure": not args.hdrnet_disable_exposure,
        }
        model = SemanticHDRNetToneMapper(
            num_classes=args.num_classes,
            base_kwargs=base_kwargs,
            hdrnet_kwargs=hdrnet_kwargs,
            mix_init=args.hdrnet_mix_init,
            base_use_local=args.hdrnet_base_local,
        )
    else:
        model = SemanticToneMapper(num_classes=args.num_classes, **base_kwargs)
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
    parser.add_argument("--model", choices=["curve", "hdrnet"], default="curve")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-gain", type=float, default=1.0)
    parser.add_argument("--init-gamma", type=float, default=2.2)
    parser.add_argument("--init-white", type=float, default=4.0)
    parser.add_argument("--init-global-gain", type=float, default=1.0)
    parser.add_argument("--init-global-gamma", type=float, default=1.0)
    parser.add_argument("--init-global-white", type=float, default=1.0)
    parser.add_argument("--bias-range", type=float, default=0.30)
    parser.add_argument("--global-bias-range", type=float, default=None)
    parser.add_argument("--max-input", type=float, default=8.0)
    parser.add_argument("--local-window", type=int, default=9)
    parser.add_argument("--local-gain-range", type=float, default=0.8)
    parser.add_argument("--local-bias-range", type=float, default=0.15)
    parser.add_argument("--local-method", default="box", choices=["box", "guided", "bilateral"])
    parser.add_argument("--guided-eps", type=float, default=1e-3)
    parser.add_argument("--bilateral-sigma-spatial", type=float, default=None)
    parser.add_argument("--bilateral-sigma-range", type=float, default=0.1)
    parser.add_argument("--disable-local", action="store_true")
    parser.add_argument("--hdrnet-base-local", action="store_true")
    parser.add_argument("--hdrnet-mix-init", type=float, default=0.5)
    parser.add_argument("--hdrnet-grid-depth", type=int, default=8)
    parser.add_argument("--hdrnet-grid-height", type=int, default=16)
    parser.add_argument("--hdrnet-grid-width", type=int, default=16)
    parser.add_argument("--hdrnet-coeffs", type=int, default=12)
    parser.add_argument("--hdrnet-embedding-dim", type=int, default=8)
    parser.add_argument("--hdrnet-hidden", type=int, default=32)
    parser.add_argument("--hdrnet-guide-hidden", type=int, default=16)
    parser.add_argument("--hdrnet-disable-exposure", action="store_true")
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
