"""Leakage-aware classical segmentation experiment runner."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from . import __version__
from .classical import ClassicalConfig, segment_classical
from .data import DatasetIndex, DatasetRecord, discover_dataset, split_records
from .metrics import binary_metrics, summarize_metrics, write_error_visualization


def load_candidate_grid() -> tuple[list[dict[str, Any]], Path, str]:
    """Load and fingerprint the versioned, checked-in parameter grid."""
    grid_path = Path(__file__).resolve().parents[1] / "configs" / "experiment_grid.json"
    raw = grid_path.read_bytes()
    grid = json.loads(raw)
    candidates = grid.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"candidate grid is empty or malformed: {grid_path}")
    return candidates, grid_path, hashlib.sha256(raw).hexdigest()


def candidate_configs() -> list[ClassicalConfig]:
    """Return the predeclared settings; no evaluation-set tuning."""
    candidates, _, _ = load_candidate_grid()
    return [ClassicalConfig(**item["config"]) for item in candidates]


def _evaluate(records: list[DatasetRecord], config: ClassicalConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        image = record.load_image()
        truth = record.load_mask(image.shape)
        prediction = segment_classical(image, config)
        row = {
            "image": record.key,
            "mask_source": record.mask_spec.source if record.mask_spec else None,
            **binary_metrics(truth, prediction),
        }
        rows.append(row)
    return rows


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "package": __version__,
    }


def run_classical_experiment(
    dataset_root: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    group_regex: str | None = None,
    class_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Tune only on the development set and evaluate the selected config once."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result_path = output_dir / "experiment.json"
    try:
        index: DatasetIndex = discover_dataset(dataset_root, class_ids=class_ids)
        labeled = index.annotated_records
        if len(labeled) < 2:
            raise ValueError(
                f"at least two images with pixel segmentation annotations are required; found {len(labeled)}"
            )
        development, evaluation, split = split_records(
            labeled, development_fraction=0.8, seed=seed, group_regex=group_regex
        )
        if not development or not evaluation:
            raise ValueError("the leakage-safe split produced an empty partition")

        candidate_definitions, grid_path, grid_sha256 = load_candidate_grid()
        development_runs: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidate_definitions):
            config = ClassicalConfig(**candidate["config"])
            rows = _evaluate(development, config)
            summary = summarize_metrics(rows)
            development_runs.append({
                "candidate_id": candidate.get("candidate_id", f"candidate_{candidate_index:02d}"),
                "rationale": candidate.get("rationale", "No rationale supplied in the predeclared grid."),
                "config": asdict(config),
                "development_metrics": summary,
                "selection_metric": "macro_dice",
                "selected": False,
                "decision": "PENDING_SELECTION",
            })
        # A stable first-candidate tie break avoids using held-out evaluation data.
        best_index = max(
            range(len(development_runs)),
            key=lambda index_: (
                development_runs[index_]["development_metrics"]["macro"]["dice"],
                development_runs[index_]["development_metrics"]["macro"]["iou"],
                -index_,
            ),
        )
        best_dice = development_runs[best_index]["development_metrics"]["macro"]["dice"]
        best_iou = development_runs[best_index]["development_metrics"]["macro"]["iou"]
        selected_id = development_runs[best_index]["candidate_id"]
        for candidate_index, item in enumerate(development_runs):
            dice = item["development_metrics"]["macro"]["dice"]
            iou = item["development_metrics"]["macro"]["iou"]
            if candidate_index == best_index:
                item["selected"] = True
                item["decision"] = "KEEP_SELECTED"
                item["selection_rationale"] = (
                    f"Selected on development data by the predeclared ordering: macro Dice={dice:.6f}, "
                    f"macro IoU={iou:.6f}, candidate order={candidate_index}. Evaluation data was not used."
                )
            else:
                item["decision"] = "DISCARD"
                if dice < best_dice:
                    reason = f"macro Dice {dice:.6f} was below selected candidate {selected_id} ({best_dice:.6f})"
                elif iou < best_iou:
                    reason = f"macro Dice tied but macro IoU {iou:.6f} was below selected candidate {selected_id} ({best_iou:.6f})"
                else:
                    reason = f"metrics tied; deterministic candidate order favored {selected_id}"
                item["selection_rationale"] = f"Discarded after development-only selection: {reason}. Evaluation data was not used."
        selected_config = ClassicalConfig(**development_runs[best_index]["config"])

        # The held-out evaluation partition is consulted only after selection.
        evaluation_rows: list[dict[str, Any]] = []
        for record in evaluation:
            image = record.load_image()
            truth = record.load_mask(image.shape)
            prediction = segment_classical(image, selected_config)
            row = {
                "image": record.key,
                "mask_source": record.mask_spec.source if record.mask_spec else None,
                **binary_metrics(truth, prediction),
            }
            evaluation_rows.append(row)
            safe_stem = Path(record.key).with_suffix("").as_posix().replace("/", "__")
            write_error_visualization(
                output_dir / "qualitative" / f"{safe_stem}_errors.png",
                image,
                truth,
                prediction,
                title=f"{record.key} | IoU {row['iou']:.3f} | Dice {row['dice']:.3f}",
            )

        result = {
            "status": "completed",
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - started, 3),
            "dataset_root": str(Path(dataset_root).expanduser().resolve()),
            "environment": _environment(),
            "protocol": {
                "development_fraction": 0.8,
                "evaluation_fraction": 0.2,
                "seed": seed,
                "group_regex": group_regex,
                "candidate_grid_file": str(grid_path.relative_to(Path(__file__).resolve().parents[1])),
                "candidate_grid_sha256": grid_sha256,
                "candidate_count": len(candidate_definitions),
                "leakage_safeguard": "exact decoded-image duplicates are grouped; optional regex groups related captures; groups are never split",
                "tuning_rule": "highest macro development Dice; ties resolved by development IoU then candidate order",
                "evaluation_used_for_selection": False,
                "ground_truth_provenance": "provided dataset pixel annotations only; pretrained model pseudo-labels are not used as verified ground truth",
            },
            "dataset": {
                "image_count": len(index.records),
                "annotated_image_count": len(labeled),
                "annotation_sources": sorted({record.mask_spec.source for record in labeled if record.mask_spec}),
                "warnings": index.warnings,
            },
            "split": split,
            "development_candidate_results": development_runs,
            "selected_config": asdict(selected_config),
            "evaluation": {
                "summary": summarize_metrics(evaluation_rows),
                "per_image": evaluation_rows,
                "worst_images_by_iou": sorted(evaluation_rows, key=lambda row: (row["iou"], row["image"]))[:10],
            },
            "outputs": {
                "per_image_csv": "evaluation_per_image.csv",
                "candidate_csv": "development_candidates.csv",
                "qualitative_directory": "qualitative/",
                "selected_config_json": "selected_config.json",
            },
        }
        _save_csv(output_dir / "evaluation_per_image.csv", evaluation_rows)
        candidate_rows = []
        for item in development_runs:
            candidate_rows.append({
                "candidate_id": item["candidate_id"],
                "selected": item["selected"],
                "decision": item["decision"],
                "rationale": item["rationale"],
                "selection_rationale": item["selection_rationale"],
                "config": json.dumps(item["config"], sort_keys=True),
                **item["development_metrics"]["macro"],
            })
        _save_csv(output_dir / "development_candidates.csv", candidate_rows)
        (output_dir / "selected_config.json").write_text(
            json.dumps({"segmentation_config": asdict(selected_config), "source_run": "experiment.json"}, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        result = {
            "status": "blocked",
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - started, 3),
            "dataset_root": str(Path(dataset_root).expanduser().resolve()),
            "environment": _environment(),
            "blocker": f"{type(exc).__name__}: {exc}",
            "metrics": None,
            "note": "No dataset performance metrics were generated because a required input or precondition was missing.",
        }
    result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return result
