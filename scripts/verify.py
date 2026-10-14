#!/usr/bin/env python3
"""Run tests and the synthetic smoke baseline, saving their exact outcomes."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    test_command = [sys.executable, "-m", "pytest", "-q"]
    test = subprocess.run(test_command, cwd=ROOT, capture_output=True, text=True)
    smoke_command = [sys.executable, "scripts/smoke_baseline.py"]
    smoke = subprocess.run(smoke_command, cwd=ROOT, capture_output=True, text=True)
    combined = (test.stdout or "") + (test.stderr or "")
    summary = None
    # Parse pytest's final summary robustly without inventing zero counts.
    final_line = next((line.strip() for line in reversed(combined.splitlines()) if " passed" in line or " failed" in line), None)
    if final_line:
        numbers = {name: 0 for name in ("passed", "failed", "skipped", "errors")}
        for number, noun in re.findall(r"(\d+)\s+(passed|failed|skipped|error|errors)", final_line):
            key = "errors" if noun.startswith("error") else noun
            numbers[key] = int(number)
        summary = {"counts": numbers, "pytest_summary_line": final_line}
    verification = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if test.returncode == 0 and smoke.returncode == 0 else "failed",
        "test_command": " ".join(test_command),
        "test_exit_code": test.returncode,
        "test_summary": summary,
        "test_output_tail": combined[-5000:],
        "smoke_command": " ".join(smoke_command),
        "smoke_exit_code": smoke.returncode,
        "smoke_output": (smoke.stdout or "")[-3000:],
        "smoke_error": (smoke.stderr or "")[-1000:],
    }
    log = ROOT / "experiments" / "logs" / "verification.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    verification["log_path"] = str(log.relative_to(ROOT))
    log.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: verification[key] for key in ("status", "test_summary", "smoke_exit_code", "log_path")}, indent=2))
    return 0 if verification["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
