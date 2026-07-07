"""
Segmentation metrics used to evaluate Mask-Net.

Implements Overall Accuracy (OA), mean Intersection-over-Union (mIoU),
and Dice score, as reported in the paper's results tables.

The mIoU is the average of per-class IoU computed from a full
confusion matrix, consistent with the metric used during training.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class SegmentationMetrics:
    """Accumulates a confusion matrix across batches (binary masks).

    Parameters
    ----------
    num_classes : int, default 2
        Number of classes (2 for binary cloud / sea segmentation).
    """

    def __init__(self, num_classes: int = 2) -> None:
        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Accumulate one batch into the confusion matrix.

        Parameters
        ----------
        preds : torch.Tensor   — binary predictions (0 / 1), any shape.
        targets : torch.Tensor — binary ground truth, same shape.
        """
        p = preds.detach().cpu().numpy().astype(int).flatten()
        t = targets.detach().cpu().numpy().astype(int).flatten()
        valid   = (t >= 0) & (t < self.num_classes)
        indices = self.num_classes * t[valid] + p[valid]
        self.cm += np.bincount(
            indices, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def mean_iou(self) -> float:
        """Mean IoU averaged over all classes."""
        inter = np.diag(self.cm)
        union = self.cm.sum(1) + self.cm.sum(0) - inter
        return float(np.nanmean(inter / union))

    def overall_accuracy(self) -> float:
        """Fraction of correctly classified pixels."""
        return float(np.diag(self.cm).sum() / self.cm.sum())

    def reset(self) -> None:
        self.cm[:] = 0


def dice_score(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> float:
    """Compute the Dice / F1 score for binary segmentation."""
    inter = (preds * targets).sum()
    return float(
        (2 * inter + eps) / (preds.sum() + targets.sum() + eps)
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader, device: str
) -> Tuple[float, float, float]:
    """Evaluate a model and return OA, mIoU, Dice (all as percentages).

    Parameters
    ----------
    model  : nn.Module    — trained Mask-Net (or any baseline).
    loader : DataLoader   — yields ``(image, mask)`` batches.
    device : str          — ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    oa, miou, dice : float  (values already multiplied by 100)
    """
    model.eval()
    seg  = SegmentationMetrics(2)
    dice = []

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device).unsqueeze(1).float()
        preds  = (torch.sigmoid(model(images)) > 0.5).float()
        seg.update(preds, masks)
        dice.append(dice_score(preds, masks))

    model.train()
    return (
        seg.overall_accuracy() * 100,
        seg.mean_iou()         * 100,
        float(np.mean(dice))   * 100,
    )
