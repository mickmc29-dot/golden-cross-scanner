"""
Scheduled scan job.

Runs the full universe scan and writes the output to results.json.
This is what GitHub Actions executes on a schedule — it can take
several minutes, which is fine here because nothing is waiting on it.

Run locally with:  python run_scan.py
"""

import json
import math
import sys
from datetime import datetime, timezone

import yfinance as yf

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


def verify_todays_close_available():
    """
    Confirms the most recent daily bar available from yfinance matches
    today's date (or, if run on a weekend/holiday, the most recent
    trading day). Uses SPY as a fast reference check before scanning
    hundreds of tickers — ported from the original script, so a
    scheduled run that fires slightly before data settles (or on a
    market holiday) skips rather than scanning stale data.
    """
    today = datetime.now().date()
    hist = yf.Ticker("SPY").history(period="5d")
    if hist.empty:
        return False, None, today
    latest_date = hist.index[-1].date()
    return latest_date == today, latest_date, today


def main():
    today = datetime.now().date()
    if today.weekday() >= 5:
        print(f"{today} is a weekend — no new market close today. Skipping run.", flush=True)
        return

    print("Checking that today's market close is available...", flush=True)
    is_current, latest_date, today = verify_todays_close_available()
    if not is_current:
        print(
            f"Today's close ({today}) is not yet available from the data source "
            f"(latest available: {latest_date}). This usually means the market "
            f"hasn't closed yet, or today is a market holiday. Skipping run rather "
            f"than scanning on stale data — results.json is left untouched.",
            flush=True,
        )
        return
    print(f"Confirmed: latest close is {latest_date}, matches today. Proceeding.", flush=True)

    started = datetime.now(timezone.utc)
    print(f"Scan started {started.isoformat()}", flush=True)

    results = run_full_scan(today=latest_date)

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
