"""
Evaluate a trained Mask-Net checkpoint.

Reports Overall Accuracy (OA), mean IoU (mIoU), and Dice score on the
test split, matching Tables 1–3 of the paper.

Usage
-----
python evaluate.py \\
    --checkpoint checkpoints/masknet_9km_final.pth \\
    --image-paths data/sst_2018.nc                 \\
    --mask-paths  data/gt_2018.nc                  \\
    --resolution-km 9
"""

from __future__ import annotations

import argparse

import torch

from data import DEFAULT_TEST_DATES, build_dataloaders
from masknet import build_masknet
from metrics import evaluate


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Mask-Net.")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--image-paths",   nargs="+", required=True)
    p.add_argument("--mask-paths",    nargs="+", required=True)
    p.add_argument("--resolution-km", type=int, default=9, choices=[9, 18])
    p.add_argument("--batch-size",    type=int, default=8)
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, test_loader, _ = build_dataloaders(
        image_paths   = args.image_paths,
        mask_paths    = args.mask_paths,
        resolution_km = args.resolution_km,
        batch_size    = args.batch_size,
        test_dates    = DEFAULT_TEST_DATES,
    )

    model = build_masknet(resolution_km=args.resolution_km).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded: {args.checkpoint}")

    oa, miou, dice = evaluate(model, test_loader, device)
    print(f"\nTest results ({args.resolution_km} km configuration)")
    print(f"  Overall Accuracy  (OA)  : {oa:.2f} %")
    print(f"  Mean IoU          (mIoU): {miou:.2f} %")
    print(f"  Dice score              : {dice:.2f} %")


if __name__ == "__main__":
    main()
