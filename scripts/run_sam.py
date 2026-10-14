#!/usr/bin/env python3
"""Run pretrained Segment Anything inference on every discovered image."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eggvision.sam_inference import run_sam_inference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "egg-dataset")
    parser.add_argument("--checkpoint", type=Path, required=True, help="local SAM checkpoint; do not commit it")
    parser.add_argument("--model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--prompt-mode", choices=("annotations", "automatic"), default="annotations")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "sam")
    args = parser.parse_args()
    try:
        result = run_sam_inference(
            args.data, args.checkpoint, args.output,
            model_type=args.model_type, prompt_mode=args.prompt_mode, device=args.device,
        )
    except Exception as exc:
        attempt_log = ROOT / "experiments" / "logs" / "sam_attempt.json"
        attempt_log.parent.mkdir(parents=True, exist_ok=True)
        attempt_log.write_text(json.dumps({
            "status": "blocked",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(args.data.expanduser().resolve()),
            "checkpoint_filename": args.checkpoint.name,
            "model_type": args.model_type,
            "prompt_mode": args.prompt_mode,
            "blocker": f"{type(exc).__name__}: {exc}",
            "processed_count": 0,
            "metrics": None,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"SAM inference blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: value for key, value in result.items() if key not in {"image_outputs", "skipped_images"}}, indent=2))
    return 0 if result["processed_count"] == result["image_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
