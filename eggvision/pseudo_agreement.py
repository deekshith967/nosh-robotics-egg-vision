"""Compare the frozen classical config with SAM pseudo-label masks.

The resulting scores measure agreement with a model's output, not accuracy. This
separate report never merges SAM pseudo-labels into the ground-truth evaluation.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classical import ClassicalConfig, segment_classical
from .data import discover_dataset
from .metrics import binary_metrics, summarize_metrics, write_error_visualization


def run_pseudo_agreement(
    dataset_root: str | Path,
    classical_experiment_path: str | Path,
    sam_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Score selected classical predictions against SAM masks on the held-out set.

    The held-out image list and frozen configuration are taken from the completed
    classical experiment. SAM masks are explicitly treated as unverified targets.
    """
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "pseudo_agreement.json"
    try:
        experiment_path = Path(classical_experiment_path)
        manifest_path = Path(sam_manifest_path)
        if not experiment_path.is_file():
            raise FileNotFoundError(f"classical experiment log not found: {experiment_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"SAM manifest not found: {manifest_path}")
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if experiment.get("status") != "completed":
            raise ValueError(f"classical experiment is not completed: {experiment.get('status')}")
        evaluation_rows = experiment.get("evaluation", {}).get("per_image", [])
        if not evaluation_rows:
            raise ValueError("classical experiment has no held-out evaluation image list")

        index = discover_dataset(dataset_root)
        records = {record.key: record for record in index.records}
        sam_outputs = {item.get("image"): item for item in manifest.get("image_outputs", []) if item.get("image")}
        config = ClassicalConfig(**experiment["selected_config"])
        per_image: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        qualitative_dir = output_dir / "qualitative"

        for eval_item in evaluation_rows:
            key = eval_item.get("image")
            record = records.get(key)
            sam_item = sam_outputs.get(key)
            if record is None or sam_item is None:
                skipped.append({"image": str(key), "error": "image or SAM output missing from corresponding manifest"})
                continue
            mask_relative = sam_item.get("union_mask_file")
            if not mask_relative:
                skipped.append({"image": key, "error": "SAM manifest has no union mask"})
                continue
            pseudo_path = (manifest_path.parent / mask_relative).resolve()
            try:
                pseudo_mask = cv2.imread(str(pseudo_path), cv2.IMREAD_GRAYSCALE)
                if pseudo_mask is None:
                    raise ValueError(f"could not read SAM union mask: {pseudo_path}")
                image = record.load_image()
                if pseudo_mask.shape != image.shape[:2]:
                    pseudo_mask = cv2.resize(pseudo_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                prediction = segment_classical(image, config)
                row = {
                    "image": key,
                    "comparison_target": "SAM pseudo-label (not verified ground truth)",
                    "sam_mode": sam_item.get("mode"),
                    **binary_metrics(pseudo_mask, prediction),
                }
                per_image.append(row)
                slug = Path(key).with_suffix("").as_posix().replace("/", "__")
                write_error_visualization(
                    qualitative_dir / f"{slug}_sam_agreement.png",
                    image,
                    pseudo_mask,
                    prediction,
                    title=f"{key} | SAM agreement IoU {row['iou']:.3f} (not GT accuracy)",
                    target_label="SAM pseudo-label (NOT GT)",
                )
            except Exception as exc:
                skipped.append({"image": key, "error": f"{type(exc).__name__}: {exc}"})

        status = "completed" if not skipped else ("completed_with_skips" if per_image else "blocked")
        result = {
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - started, 3),
            "dataset_root": str(Path(dataset_root).expanduser().resolve()),
            "classical_experiment": str(experiment_path),
            "sam_manifest": str(manifest_path),
            "comparison_target": "SAM predicted union masks (pseudo-labels)",
            "interpretation_warning": "These are agreement metrics against unverified SAM pseudo-labels, not accuracy against ground truth.",
            "held_out_expected_images": len(evaluation_rows),
            "compared_images": len(per_image),
            "skipped_images": skipped,
            "summary": summarize_metrics(per_image) if per_image else None,
            "per_image": per_image,
            "outputs": {
                "per_image_csv": "pseudo_agreement_per_image.csv",
                "qualitative_directory": "qualitative/",
            },
        }
        if per_image:
            fields = list(per_image[0])
            with (output_dir / "pseudo_agreement_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(per_image)
    except Exception as exc:
        result = {
            "status": "blocked",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - started, 3),
            "blocker": f"{type(exc).__name__}: {exc}",
            "comparison_target": "SAM pseudo-labels only; never verified ground truth",
            "summary": None,
            "metrics": None,
        }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
