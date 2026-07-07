"""
Training script for Mask-Net.

Reproduces the protocol of Section 4.1: Adam optimiser, combined
BCE + Dice loss, mixed-precision training, and periodic checkpointing.

Usage
-----
python train.py \\
    --image-paths data/sst_2018.nc \\
    --mask-paths  data/gt_2018.nc  \\
    --resolution-km 9              \\
    --batch-size 8                 \\
    --epochs 300                   \\
    --checkpoint-dir checkpoints/

Run ``python train.py --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import os
import time

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data import DEFAULT_TEST_DATES, build_dataloaders
from masknet import build_masknet
from metrics import evaluate


# ── Loss function ────────────────────────────────────────────────────────────

def loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Combined BCE-with-logits + Dice loss (Eq. 21–22 of the paper)."""
    bce  = nn.BCEWithLogitsLoss()(predictions, targets)
    dice = smp.losses.DiceLoss(mode="binary")(predictions, targets)
    return bce + dice


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Mask-Net on AVHRR SST data.")
    p.add_argument("--image-paths", nargs="+", required=True,
                   help="NetCDF file(s) with variable 'sst' (one per year).")
    p.add_argument("--mask-paths",  nargs="+", required=True,
                   help="NetCDF file(s) with variable 'dc'  (one per year).")
    p.add_argument("--resolution-km", type=int, default=9, choices=[9, 18])
    p.add_argument("--batch-size",    type=int, default=8)
    p.add_argument("--epochs",        type=int, default=300)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--save-every",    type=int, default=5,
                   help="Save a checkpoint every N epochs.")
    p.add_argument("--resume",        default=None,
                   help="Path to a checkpoint (.pth) to resume from.")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data
    train_loader, test_loader, _ = build_dataloaders(
        image_paths   = args.image_paths,
        mask_paths    = args.mask_paths,
        resolution_km = args.resolution_km,
        batch_size    = args.batch_size,
        test_dates    = DEFAULT_TEST_DATES,
    )

    # Model
    model = build_masknet(resolution_km=args.resolution_km).to(device)
    start_epoch = 0
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        # Try to infer start epoch from filename (e.g. "…_epoch_60.pth").
        try:
            start_epoch = int(
                os.path.splitext(args.resume)[0].split("_")[-1]
            )
        except ValueError:
            pass
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    t0 = time.time()
    print(f"Training for {args.epochs} epochs …")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for images, masks in bar:
            images = images.to(device)
            masks  = masks.float().unsqueeze(1).to(device)

            with torch.autocast(device_type=device,
                                enabled=(device == "cuda")):
                loss = loss_fn(model(images), masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt = os.path.join(
                args.checkpoint_dir,
                f"masknet_{args.resolution_km}km_epoch_{epoch + 1}.pth",
            )
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Saved {ckpt}")

    elapsed = (time.time() - t0) / 60
    print(f"\nTraining done in {elapsed:.1f} min.")

    # Final evaluation
    for split, loader in [("Train", train_loader), ("Test", test_loader)]:
        oa, miou, dice = evaluate(model, loader, device)
        print(f"  {split:5s} → OA {oa:.2f}%  mIoU {miou:.2f}%  "
              f"Dice {dice:.2f}%")

    final = os.path.join(
        args.checkpoint_dir,
        f"masknet_{args.resolution_km}km_final.pth",
    )
    torch.save(model.state_dict(), final)
    print(f"Final model saved to {final}")


if __name__ == "__main__":
    main()
