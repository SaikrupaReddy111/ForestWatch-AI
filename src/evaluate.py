"""Pixel-level evaluation metrics and confusion matrix calculations.

Computes IoU, Dice/F1, Precision, Recall, and Confusion Matrices strictly excluding
nodata/unobserved pixels (ignore_index=255). Supports per-region reporting and
clearly distinguishes synthetic validation from real-world satellite benchmarks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.50,
    ignore_index: int = 255,
) -> Dict[str, Any]:
    """Computes pixel-level binary segmentation metrics, excluding nodata pixels.

    Args:
        logits: Raw network outputs of shape (B, 1, H, W) or (B, H, W).
        targets: Ground truth binary masks in {0, 1, ignore_index}.
        threshold: Sigmoid probability threshold for positive forest class.
        ignore_index: Label index to exclude from all metric calculations (255).

    Returns:
        Dictionary containing IoU, Dice/F1, Precision, Recall, and confusion matrix counts.
    """
    if logits.dim() == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if targets.dim() == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()

    # Mask of valid observation pixels
    valid_mask = (targets != ignore_index)

    if not valid_mask.any():
        return {
            "iou": 0.0,
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "valid_pixels": 0,
            "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        }

    v_preds = preds[valid_mask]
    v_targets = targets[valid_mask]

    tp = int(((v_preds == 1) & (v_targets == 1)).sum().item())
    fp = int(((v_preds == 1) & (v_targets == 0)).sum().item())
    fn = int(((v_preds == 0) & (v_targets == 1)).sum().item())
    tn = int(((v_preds == 0) & (v_targets == 0)).sum().item())


    total_valid = tp + fp + fn + tn

    eps = 1e-6
    iou = float(tp / (tp + fp + fn + eps))
    dice = float((2.0 * tp) / (2.0 * tp + fp + fn + eps))
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "valid_pixels": total_valid,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
    threshold: float = 0.50,
    ignore_index: int = 255,
    is_synthetic: bool = False,
    region_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs full evaluation loop across a DataLoader and aggregates pixel confusion metrics.

    Args:
        model: PyTorch model module.
        dataloader: DataLoader for validation or test split.
        device: 'cuda' or 'cpu'.
        threshold: Decision threshold for positive forest class.
        ignore_index: Ignored nodata label (255).
        is_synthetic: Flag indicating whether metrics derive from synthetic demonstration data.
        region_id: Identifier of the evaluated geographic region.

    Returns:
        Structured dictionary with aggregated evaluation metrics and auditing metadata.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.no_grad():
        for images, targets, tile_ids in dataloader:
            images = images.to(dev)
            targets = targets.to(dev)

            logits = model(images)
            if logits.dim() == 4 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            if targets.dim() == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            valid_mask = (targets != ignore_index)
            if valid_mask.any():
                v_preds = preds[valid_mask]
                v_targets = targets[valid_mask]

                total_tp += int(((v_preds == 1) & (v_targets == 1)).sum().item())
                total_fp += int(((v_preds == 1) & (v_targets == 0)).sum().item())
                total_fn += int(((v_preds == 0) & (v_targets == 1)).sum().item())
                total_tn += int(((v_preds == 0) & (v_targets == 0)).sum().item())

    eps = 1e-6
    iou = float(total_tp / (total_tp + total_fp + total_fn + eps))
    dice = float((2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps))
    precision = float(total_tp / (total_tp + total_fp + eps))
    recall = float(total_tp / (total_tp + total_fn + eps))
    total_valid = total_tp + total_fp + total_fn + total_tn

    disclaimer = (
        "DEMONSTRATION RUN: Metrics derived from synthetic demonstration rasters. "
        "Must not be used as real-world satellite model performance."
        if is_synthetic
        else "Real satellite evaluation on held-out geographic partition."
    )

    return {
        "region_id": region_id or "unknown",
        "is_synthetic": is_synthetic,
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "valid_pixels": total_valid,
        "confusion_matrix": {
            "tp": total_tp,
            "fp": total_fp,
            "tn": total_tn,
            "fn": total_fn,
        },
        "disclaimer": disclaimer,
    }
