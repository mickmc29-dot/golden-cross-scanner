"""
Scheduled scan job.

Runs the full universe scan and writes the output to results.json.
This is what GitHub Actions executes on a schedule — it can take
several minutes, which is fine here because nothing is waiting on it.

Run locally with:  python run_scan.py
"""

import json
import math
from datetime import datetime, timezone

from scanner import run_full_scan

OUTPUT_PATH = "results.json"


def sanitize(obj):
    """
    Recursively replace NaN/Infinity with None. Python's json.dump
    writes NaN/Infinity as bare, non-standard tokens that Python's own
    json module can read back but browsers' JSON.parse rejects — so
    without this, one stray NaN anywhere in the data breaks the whole
    dashboard fetch. This is a safety net on top of fixing the known
    source (_pct in scanner.py); it catches any other field that
    might return NaN too.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def main():
    started = datetime.now(timezone.utc)
    print(f"Scan started {started.isoformat()}", flush=True)

    results = run_full_scan()

    finished = datetime.now(timezone.utc)
    elapsed = (finished - started).total_seconds()

    payload = sanitize({
        "last_run": finished.isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(
        f"Scan finished in {elapsed:.0f}s — {len(results)} hits written to {OUTPUT_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
