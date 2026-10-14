#!/usr/bin/env python3
"""Record whether assignment inputs and optional compute are available."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _memory_info() -> dict[str, int | None]:
    values: dict[str, int | None] = {"total_bytes": None, "available_bytes": None, "swap_total_bytes": None, "swap_free_bytes": None}
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return values
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, _, remainder = line.partition(":")
        if key not in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            continue
        tokens = remainder.split()
        if tokens:
            mapped = {"MemTotal": "total_bytes", "MemAvailable": "available_bytes", "SwapTotal": "swap_total_bytes", "SwapFree": "swap_free_bytes"}[key]
            values[mapped] = int(tokens[0]) * 1024
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "egg-dataset")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "egg-video.mp4")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "sam_vit_b_01ec64.pth")
    parser.add_argument("--assignment-pdf", type=Path, default=ROOT / "assignment.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "logs" / "input_audit.json")
    args = parser.parse_args()
    repo = args.repo.resolve()
    dataset = args.dataset.expanduser().resolve()
    video = args.video.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    assignment_pdf = args.assignment_pdf.expanduser().resolve()
    image_count = sum(1 for path in dataset.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES) if dataset.is_dir() else 0
    pdfs = sorted(
        str(path.relative_to(repo)) for path in repo.rglob("*.pdf")
        if path.is_file() and (path.name.lower().startswith(("assignment", "brief", "task")))
        and "reports" not in path.parts
    )
    if assignment_pdf.is_file() and assignment_pdf.parent == repo:
        assignment_found = True
    else:
        assignment_found = bool(pdfs)
    modules = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "cv2", "torch", "segment_anything", "reportlab")}
    disk = shutil.disk_usage(repo)
    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_cores": os.cpu_count(),
        "gpu_probe": "nvidia-smi available" if shutil.which("nvidia-smi") else "nvidia-smi not found; no GPU confirmed",
        "resources": {
            "memory": _memory_info(),
            "disk": {"path": str(repo), "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        },
        "python_modules_available": modules,
        "inputs": {
            "assignment_pdf": {"path": str(assignment_pdf), "available": assignment_found, "candidate_pdfs": pdfs},
            "dataset": {"path": str(dataset), "available": dataset.is_dir(), "image_count": image_count},
            "video": {"path": str(video), "available": video.is_file()},
            "sam_checkpoint": {"filename": checkpoint.name, "available": checkpoint.is_file()},
        },
        "limitations": [
            "No dataset-specific metrics can be computed without annotated dataset images.",
            "SAM inference cannot run without the optional dependencies and a local checkpoint.",
            "The supplied video cannot be processed until a local copy is provided.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
