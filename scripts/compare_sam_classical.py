#!/usr/bin/env python3
"""Measure frozen classical-vs-SAM pseudo-label agreement on held-out images."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eggvision.pseudo_agreement import run_pseudo_agreement  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "egg-dataset")
    parser.add_argument("--classical", type=Path, default=ROOT / "runs" / "classical" / "experiment.json")
    parser.add_argument("--sam", type=Path, default=ROOT / "runs" / "sam" / "sam_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "pseudo-agreement")
    args = parser.parse_args()
    result = run_pseudo_agreement(args.data, args.classical, args.sam, args.output)
    if result.get("status") == "blocked":
        attempt_log = ROOT / "experiments" / "logs" / "pseudo_agreement_attempt.json"
        attempt_log.parent.mkdir(parents=True, exist_ok=True)
        attempt_log.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in result.items() if key != "per_image"}
    print(json.dumps(compact, indent=2))
    return 0 if result.get("status") == "completed" else (1 if result.get("status") == "completed_with_skips" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
