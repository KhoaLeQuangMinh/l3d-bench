"""The single scoring protocol for every method.

Primary metrics copy A2ONet's a2onet/metrics.py (hyperiondk115/A2ONet @ 765ada1) line for line:
    pred_fg = stack(pred == c for c in 1..3); gt_fg likewise
    dice = (2 * |pred_fg & gt_fg| + 1e-5) / (|pred_fg| + |gt_fg| + 1e-5)    # pooled over the 3 classes
    iou  = dice / (2 - dice)                                                 # derived, as the papers report
    assd = medpy assd(pred_fg, gt_fg) on the (3, H, W) stack if both are non-empty, else 80
TopoNet and D2GPLand compute the same Dice; TopoNet gives 80 only when the prediction is empty.
Inputs are label maps (0 background, 1 ridge, 2 silhouette, 3 ligament) at 1024 x 1024.

Secondary: per-class Dice (not in the papers) to show which landmark fails.
"""
from __future__ import annotations

import numpy as np
from medpy.metric.binary import assd as medpy_assd

EMPTY_ASSD = 80.0
SMOOTH = 1e-5
CLASS_NAMES = {1: "ridge", 2: "silhouette", 3: "ligament"}


def a2onet_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> tuple[float, float, float]:
    """(dice, iou, assd) for one image, exactly as A2ONet computes them."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    pred_fg = np.stack([pred == c for c in range(1, num_classes)])
    gt_fg = np.stack([gt == c for c in range(1, num_classes)])
    inter = np.logical_and(pred_fg, gt_fg).sum()
    dice = (2 * inter + SMOOTH) / (pred_fg.sum() + gt_fg.sum() + SMOOTH)
    iou = dice / (2 - dice)
    if pred_fg.any() and gt_fg.any():
        dist = float(medpy_assd(pred_fg, gt_fg))
    else:
        dist = EMPTY_ASSD
    return float(dice), float(iou), dist


def per_class_dice(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Dice per landmark; NaN when the class is absent from both prediction and ground truth."""
    out = {}
    for c, name in CLASS_NAMES.items():
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        out[f"dice_{name}"] = float("nan") if denom == 0 else float(2 * np.logical_and(p, g).sum() / denom)
    return out


def score_frame(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    dice, iou, dist = a2onet_metrics(pred, gt)
    penalty = not ((pred > 0).any() and (gt > 0).any())
    row = {"dice": dice, "iou": iou, "assd": dist, "assd_empty_penalty": float(penalty)}
    row.update(per_class_dice(pred, gt))
    return row


def a2onet_dice_torch(pred, gt):
    """Batched GPU version of the A2ONet Dice, used only for checkpoint selection during training.

    pred, gt: (B, H, W) integer label maps. With one class per pixel, |P| = #(pred > 0), |G| = #(gt > 0) and
    |P & G| = #(pred == gt > 0), so this equals a2onet_metrics()[0] exactly (tested)."""
    p = (pred > 0).flatten(1).sum(1).double()
    g = (gt > 0).flatten(1).sum(1).double()
    inter = ((pred == gt) & (gt > 0)).flatten(1).sum(1).double()
    return (2 * inter + SMOOTH) / (p + g + SMOOTH)
