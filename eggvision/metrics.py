"""Binary mask metrics and compact qualitative error visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Return pixel-count and overlap metrics for one binary image.

    Empty-mask convention: if both masks are empty all four scores are 1; if
    only one mask is empty, the corresponding precision/recall is 0. This
    convention is explicit so background-only images cannot silently disappear.
    """
    gt = np.asarray(truth) > 0
    pred = np.asarray(prediction) > 0
    if gt.shape != pred.shape:
        raise ValueError(f"mask shapes differ: ground truth {gt.shape}, prediction {pred.shape}")
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    gt_pixels = tp + fn
    pred_pixels = tp + fp
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    dice_denom = 2 * tp + fp + fn
    dice = (2 * tp / dice_denom) if dice_denom else 1.0
    precision = tp / pred_pixels if pred_pixels else (1.0 if gt_pixels == 0 else 0.0)
    recall = tp / gt_pixels if gt_pixels else (1.0 if pred_pixels == 0 else 0.0)
    if fn > max(1.25 * fp, 0):
        failure = "under-segmentation (false negatives dominate)"
    elif fp > max(1.25 * fn, 0):
        failure = "over-segmentation (false positives dominate)"
    else:
        failure = "mixed/balanced pixel errors"
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "ground_truth_pixels": gt_pixels,
        "predicted_pixels": pred_pixels,
        "failure_mode": failure,
    }


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute unweighted per-image means and pooled pixel metrics."""
    if not rows:
        return {"image_count": 0, "macro": None, "pooled": None, "failure_modes": {}}
    keys = ("iou", "dice", "precision", "recall")
    macro = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    tp = sum(int(row["tp_pixels"]) for row in rows)
    fp = sum(int(row["fp_pixels"]) for row in rows)
    fn = sum(int(row["fn_pixels"]) for row in rows)
    union = tp + fp + fn
    pooled = {
        "iou": tp / union if union else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0),
        "recall": tp / (tp + fn) if tp + fn else (1.0 if tp + fp == 0 else 0.0),
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
    }
    modes: dict[str, int] = {}
    for row in rows:
        mode = str(row["failure_mode"])
        modes[mode] = modes.get(mode, 0) + 1
    return {"image_count": len(rows), "macro": macro, "pooled": pooled, "failure_modes": modes}


def error_visualization(
    image_bgr: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    title: str = "",
    target_label: str = "Ground truth",
) -> np.ndarray:
    """Create a four-panel BGR visualization; TP=green, FP=red, FN=blue."""
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    h, w = image_bgr.shape[:2]
    gt = cv2.resize((np.asarray(truth) > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    pred = cv2.resize((np.asarray(prediction) > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    gt_panel = np.zeros_like(image_bgr)
    gt_panel[gt > 0] = (80, 220, 80)
    pred_panel = np.zeros_like(image_bgr)
    pred_panel[pred > 0] = (0, 180, 255)
    errors = np.zeros_like(image_bgr)
    errors[(gt > 0) & (pred > 0)] = (40, 220, 40)  # true positive: green
    errors[(gt == 0) & (pred > 0)] = (0, 0, 255)   # false positive: red
    errors[(gt > 0) & (pred == 0)] = (255, 80, 0)  # false negative: blue

    panels = [image_bgr.copy(), gt_panel, pred_panel, errors]
    labels = ["Input", target_label[:40], "Prediction", "Errors: TP green / FP red / FN blue"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (w - 1, min(30, h - 1)), (25, 25, 25), -1)
        cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.vstack((np.hstack((panels[0], panels[1])), np.hstack((panels[2], panels[3]))))
    if title:
        canvas = cv2.copyMakeBorder(canvas, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 34, 45))
        cv2.putText(canvas, title[:120], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def write_error_visualization(
    path: str | Path,
    image: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    title: str = "",
    target_label: str = "Ground truth",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = error_visualization(image, truth, prediction, title, target_label=target_label)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to save error visualization: {path}")
