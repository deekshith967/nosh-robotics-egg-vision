#!/usr/bin/env python3
"""Run the classical segmentation development/evaluation experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eggvision.experiment import run_classical_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "egg-dataset", help="dataset export directory")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "classical", help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-regex", help="optional regex; capture group (or whole match) defines a capture/sequence group")
    parser.add_argument("--class-ids", nargs="+", type=int, help="optional annotation category IDs to treat as foreground")
    args = parser.parse_args()
    result = run_classical_experiment(
        args.data, args.output, seed=args.seed,
        group_regex=args.group_regex,
        class_ids=set(args.class_ids) if args.class_ids else None,
    )
    if result.get("status") == "blocked":
        attempt_log = ROOT / "experiments" / "logs" / "classical_attempt.json"
        attempt_log.parent.mkdir(parents=True, exist_ok=True)
        attempt_log.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
