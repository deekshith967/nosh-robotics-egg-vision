#!/usr/bin/env python3
"""Run a data-free synthetic smoke test for the classical baseline.

This controlled fixture validates code execution only. It is not dataset
performance and must never be reported as a real-world result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eggvision.classical import ClassicalConfig, segment_classical  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=ROOT / "experiments" / "logs" / "baseline_smoke.json")
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    height, width = 180, 240
    background = np.full((height, width, 3), 28, dtype=np.uint8)
    truth = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(background, (76, 88), (34, 20), -18, 0, 360, (232, 232, 232), -1)
    cv2.ellipse(truth, (76, 88), (34, 20), -18, 0, 360, 255, -1)
    cv2.ellipse(background, (166, 102), (38, 23), 17, 0, 360, (232, 232, 232), -1)
    cv2.ellipse(truth, (166, 102), (38, 23), 17, 0, 360, 255, -1)
    noise = rng.normal(0, 2.0, background.shape).astype(np.int16)
    sample = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    prediction = segment_classical(sample, ClassicalConfig())
    truth_bool, pred_bool = truth > 0, prediction > 0
    intersection = int(np.logical_and(truth_bool, pred_bool).sum())
    union = int(np.logical_or(truth_bool, pred_bool).sum())
    iou = intersection / union if union else 1.0
    dice = (2 * intersection) / (int(truth_bool.sum()) + int(pred_bool.sum()))
    result = {
        "kind": "synthetic_smoke_test_not_dataset_evaluation",
        "seed": 7,
        "image_size_wh": [width, height],
        "config": ClassicalConfig().to_dict(),
        "synthetic_iou": round(iou, 6),
        "synthetic_dice": round(dice, 6),
        "prediction_foreground_pixels": int(pred_bool.sum()),
        "ground_truth_foreground_pixels": int(truth_bool.sum()),
        "status": "passed" if iou >= 0.85 else "failed",
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
