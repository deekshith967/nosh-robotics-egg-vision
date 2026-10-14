#!/usr/bin/env python3
"""Track segmented egg candidates and export an annotated MP4."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eggvision.tracking import track_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="local copy of the supplied video")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "tracking" / "annotated.mp4")
    parser.add_argument("--config", type=Path, help="selected classical segmentation config JSON")
    parser.add_argument("--max-distance", type=float, default=90.0)
    parser.add_argument("--max-age", type=int, default=8)
    parser.add_argument("--min-area-fraction", type=float, default=0.0006)
    parser.add_argument("--max-area-fraction", type=float, default=0.30)
    args = parser.parse_args()
    try:
        result = track_video(
            args.video, args.output, segmentation_config_path=args.config,
            max_distance=args.max_distance, max_age=args.max_age,
            min_area_fraction=args.min_area_fraction,
            max_area_fraction=args.max_area_fraction,
        )
    except Exception as exc:
        attempt_log = ROOT / "experiments" / "logs" / "tracking_attempt.json"
        attempt_log.parent.mkdir(parents=True, exist_ok=True)
        attempt_log.write_text(json.dumps({
            "status": "blocked",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_filename": args.video.name,
            "output_filename": args.output.name,
            "blocker": f"{type(exc).__name__}: {exc}",
            "frames_processed": 0,
            "annotated_mp4_generated": False,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Video tracking blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    compact = {key: value for key, value in result.items() if key != "frames"}
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
