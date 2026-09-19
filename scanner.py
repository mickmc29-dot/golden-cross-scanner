"""
Golden Cross Scanner — core scan logic.

Imported by run_scan.py (the scheduled job that does the heavy
lifting) and kept separate from app.py so the web server never
runs a full scan inside a request.

This ports the real detection logic from the original
golden_cross_scanner.py (sustained MA50 decline before the turn,
sustained time below MA200 before the cross, the "off recent low"
bottoming path, real fundamental thresholds, and the original
scoring formula) into the split scan/serve architecture, while
keeping two fixes made along the way that the original script
didn't have:
  - RSI uses Wilder's smoothing rather than a plain rolling mean
    (the plain version produced implausible near-single-digit
    readings on ordinary pullbacks)
  - rows with a NaN close (yfinance's placeholder for a
    not-yet-finalized current session) are dropped before any
    calculation, so a stale row can never leak into price/RSI/etc.
"""

import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from tickers import TICKER_UNIVERSE

# =============================================================================
# CONFIG — ported from the original script's thresholds
# =============================================================================
MAX_DEBT_TO_EBITDA = 4.0
MIN_INTEREST_COVERAGE = 2.0
RECENT_CROSS_WINDOW = 2       # "recent" golden cross = crossed within last N trading days
MIN_MA50_DECLINE_DAYS = 10    # MA50 must have declined at least this many consecutive
                              # trading days before turning up into the cross
MIN_DAYS_BELOW_MA200 = 15     # MA50 must have been below MA200 at least this many
                              # consecutive trading days before the cross
MIN_SCORE_THRESHOLD = 6       # only report candidates scoring >= this (max possible is 7)


def compute_rsi(closes, period=14):
    """
    Wilder's RSI — the standard variant used by charting platforms.
    See module docstring: kept from the earlier fix rather than
    reverting to the original script's plain rolling mean.
    """
    if len(closes) < period + 1:
        return None

    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def analyze_technicals(closes, volumes):
    """
    closes, volumes: 1-D numpy arrays, ascending by date, already
    cleaned of NaN rows.

    Returns a dict of technical signal fields, or None if there's
    not enough history (mirrors the original script's
    analyze_technicals(), adapted to work on numpy arrays instead
    of a DataFrame since scan_ticker() already extracts those).
    """
    if len(closes) < 210:
        return None

    ma50 = pd.Series(closes).rolling(window=50).mean().to_numpy()
    ma200 = pd.Series(closes).rolling(window=200).mean().to_numpy()
    vol_avg20 = pd.Series(volumes).rolling(window=20).mean().to_numpy()

    valid = ~np.isnan(ma50) & ~np.isnan(ma200)
    if valid.sum() < RECENT_CROSS_WINDOW + 2:
        return None

    # Trim everything to the region where both MAs are defined, so all
    # arrays below stay index-aligned.
    closes_v = closes[valid]
    volumes_v = volumes[valid]
    ma50_v = ma50[valid]
    ma200_v = ma200[valid]
    vol_avg20_v = vol_avg20[valid]

    # Detect every golden-cross point in the trimmed series (not just
    # within a short lookback window) using numpy array comparisons —
    # deliberately not pandas .shift() on a boolean column, which was
    # the source of the earlier bug (shift() promotes bool to object
    # dtype, and ~ then does bitwise-int negation instead of logical
    # negation, so every day looks like "not previously above").
    above = ma50_v > ma200_v
    prev_above = np.empty_like(above)
    prev_above[0] = False
    prev_above[1:] = above[:-1]
    cross_mask = above & ~prev_above
    cross_idxs = np.where(cross_mask)[0]

    days_since_cross = None
    golden_cross_recent = False
    ma50_decline_days_before_turn = None
    ma50_had_sustained_decline = False
    days_below_ma200_before_cross = None
    had_sustained_below_ma200 = False

    if len(cross_idxs) > 0:
        cross_pos = int(cross_idxs[-1])
        days_since_cross = (len(closes_v) - 1) - cross_pos
        golden_cross_recent = days_since_cross <= RECENT_CROSS_WINDOW

        # Find the MA50 trough that preceded this cross (search back up
        # to 90 trading days), then count the consecutive decline streak
        # ending at that trough.
        lookback_start = max(0, cross_pos - 90)
        window_ma50 = ma50_v[lookback_start:cross_pos + 1]
        if len(window_ma50) > 0:
            trough_pos = lookback_start + int(np.argmin(window_ma50))
            streak = 0
            i = trough_pos
            while i > 0 and ma50_v[i - 1] > ma50_v[i]:
                streak += 1
                i -= 1
            ma50_decline_days_before_turn = streak
            ma50_had_sustained_decline = streak >= MIN_MA50_DECLINE_DAYS

        # Count consecutive trading days MA50 was below MA200 immediately
        # preceding the cross (guards against brief whipsaw dips rather
        # than a genuine sustained downtrend before reversal).
        below_streak = 0
        j = cross_pos - 1
        while j >= 0 and not above[j]:
            below_streak += 1
            j -= 1
        days_below_ma200_before_cross = below_streak
        had_sustained_below_ma200 = below_streak >= MIN_DAYS_BELOW_MA200

    # ma50/ma200 "turning up": latest value above the mean of the
    # preceding 5 (MA50) or 10 (MA200) days.
    ma50_turning_up = False
    if len(ma50_v) >= 6:
        ma50_turning_up = bool(ma50_v[-1] > ma50_v[-6:-1].mean())

    ma200_turning_up = False
    if len(ma200_v) >= 11:
        ma200_turning_up = bool(ma200_v[-1] > ma200_v[-11:-1].mean())

    price = float(closes_v[-1])
    price_above_both_mas = bool(price > ma50_v[-1] and price > ma200_v[-1])

    latest_vol_avg20 = vol_avg20_v[-1]
    volume_confirming = bool(
        volumes_v[-1] > latest_vol_avg20 * 1.2 if not np.isnan(latest_vol_avg20) else False
    )

    # "Bottoming" check: price made a low in the trailing ~60 days and
    # has since risen more than 5% off it.
    trailing = closes_v[-60:]
    low_idx = int(np.argmin(trailing))
    trailing_low = float(trailing[low_idx])
    off_recent_low_pct = round((price / trailing_low - 1) * 100, 1) if trailing_low > 0 else 0.0
    is_off_recent_low = (low_idx != len(trailing) - 1) and off_recent_low_pct > 5

    rsi_raw = compute_rsi(closes_v)

    return {
        "price": round(price, 2),
        "ma50": round(float(ma50_v[-1]), 2),
        "ma200": round(float(ma200_v[-1]), 2),
        "rsi": round(rsi_raw, 1) if rsi_raw is not None else None,
        "golden_cross_recent": golden_cross_recent,
        "days_since_cross": days_since_cross,
        "ma50_decline_days_before_turn": ma50_decline_days_before_turn,
        "ma50_had_sustained_decline": ma50_had_sustained_decline,
        "days_below_ma200_before_cross": days_below_ma200_before_cross,
        "had_sustained_below_ma200": had_sustained_below_ma200,
        "ma50_turning_up": ma50_turning_up,
        "ma200_turning_up": ma200_turning_up,
        "price_above_both_mas": price_above_both_mas,
        "volume_confirming": volume_confirming,
        "off_recent_low_pct": off_recent_low_pct,
        "is_off_recent_low": is_off_recent_low,
    }


def analyze_fundamentals(ticker_obj):
    """
    Pulls fundamental fields from yfinance's .info. Field completeness
    varies by ticker, so values may come back None — every caller here
    treats None as "unknown," not as a red flag on its own.
    """
    try:
        info = ticker_obj.info
    except Exception:
        info = {}

    total_debt = info.get("totalDebt")
    ebitda = info.get("ebitda")
    debt_to_ebitda = None
    if total_debt is not None and ebitda:
        if ebitda > 0:
            debt_to_ebitda = round(total_debt / ebitda, 2)
        # negative EBITDA makes the ratio meaningless — leave as None
        # rather than report a nonsense number.

    interest_coverage = None
    interest_expense = info.get("interestExpense")
    if ebitda and interest_expense:
        interest_coverage = round(ebitda / abs(interest_expense), 2)

    sector = info.get("sector", "Unknown")
    revenue_growth_pct = _pct(info.get("revenueGrowth"))
    earnings_growth_pct = _pct(info.get("earningsGrowth"))
    profit_margin_pct = _pct(info.get("profitMargins"))

    ex_div_ts = info.get("exDividendDate")
    ex_dividend_date = (
        datetime.fromtimestamp(ex_div_ts).strftime("%Y-%m-%d") if ex_div_ts else None
    )
    dividend_amount = info.get("dividendRate")

    next_earnings_date = None
    try:
        cal = ticker_obj.calendar
        earnings_dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if earnings_dates:
            next_earnings_date = min(earnings_dates).strftime("%Y-%m-%d")
    except Exception:
        pass

    return {
        "sector": sector,
        "debt_to_ebitda": debt_to_ebitda,
        "interest_coverage": interest_coverage,
        "revenue_growth_pct": revenue_growth_pct,
        "earnings_growth_pct": earnings_growth_pct,
        "profit_margin_pct": profit_margin_pct,
        "ex_dividend_date": ex_dividend_date,
        "dividend_amount": dividend_amount,
        "next_earnings_date": next_earnings_date,
    }


def passes_fundamental_filter(fund):
    """
    Soft filter: flags concerns rather than hard-excluding, since
    yfinance data is often incomplete. Returns (passes: bool, notes: str).
    """
    notes = []
    if fund["debt_to_ebitda"] is not None and fund["debt_to_ebitda"] > MAX_DEBT_TO_EBITDA:
        notes.append(f"high debt/EBITDA ({fund['debt_to_ebitda']}x)")
    if fund["interest_coverage"] is not None and fund["interest_coverage"] < MIN_INTEREST_COVERAGE:
        notes.append(f"low interest coverage ({fund['interest_coverage']}x)")
    if fund["earnings_growth_pct"] is not None and fund["earnings_growth_pct"] < -10:
        notes.append(f"earnings declining ({fund['earnings_growth_pct']}%)")
    if fund["revenue_growth_pct"] is not None and fund["revenue_growth_pct"] < -10:
        notes.append(f"revenue declining ({fund['revenue_growth_pct']}%)")

    passes = len(notes) == 0
    return passes, ("; ".join(notes) if notes else "no major red flags")


def _pct(v):
    """
    Converts a fraction (0.05) to a percentage (5.0), treating None,
    NaN, and infinity all as "missing" — yfinance sometimes returns
    nan for an unavailable numeric field rather than None, and nan
    silently survives arithmetic (round(nan, 1) is still nan), which
    previously broke JSON output downstream.
    """
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except TypeError:
        return None
    return round(v * 100, 1)


def scan_ticker(symbol, today=None, counts=None):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="400d")

        # Drop any row with a NaN close — yfinance sometimes appends a
        # placeholder row for the current session before that day's
        # data is fully finalized, and a NaN close must never leak into
        # price/RSI/volume calculations downstream.
        hist = hist[hist["Close"].notna()]

        if len(hist) < 210:
            return None

        # Skip tickers whose latest bar isn't today's close (newly
        # listed, delisted, or a data gap for this specific symbol) —
        # keeps every result grounded in the same trading day.
        if today is not None and hist.index[-1].date() != today:
            return None

        if counts is not None:
            counts["had_history"] += 1

        closes = hist["Close"].to_numpy()
        volumes = hist["Volume"].to_numpy()

        tech = analyze_technicals(closes, volumes)
        if tech is None:
            return None

        # Only bother pulling fundamentals (a separate API call) for
        # tickers with an interesting technical setup — either a golden
        # cross with real sustained decline/below-MA200 history behind
        # it, or a bottoming pattern (off a recent low, MA50 turning up)
        # even without a cross yet.
        interesting = (
            (tech["golden_cross_recent"] and tech["ma50_had_sustained_decline"]
             and tech["had_sustained_below_ma200"])
            or (tech["is_off_recent_low"] and tech["ma50_turning_up"])
        )
        if not interesting:
            return None

        if counts is not None:
            counts["interesting"] += 1

        fund = analyze_fundamentals(t)
        fund_ok, fund_notes = passes_fundamental_filter(fund)

        score = (
            int(tech["golden_cross_recent"]) * 3
            + int(fund_ok) * 2
            + int(tech["volume_confirming"])
            + int(tech["ma200_turning_up"])
        )

        return {
            "ticker": symbol,
            "price": tech["price"],
            "ma50": tech["ma50"],
            "ma200": tech["ma200"],
            "rsi": tech["rsi"],
            "days_since_cross": tech["days_since_cross"],
            "golden_cross_recent": tech["golden_cross_recent"],
            "ma50_decline_days_before_turn": tech["ma50_decline_days_before_turn"],
            "ma50_had_sustained_decline": tech["ma50_had_sustained_decline"],
            "days_below_ma200_before_cross": tech["days_below_ma200_before_cross"],
            "had_sustained_below_ma200": tech["had_sustained_below_ma200"],
            "ma50_turning_up": tech["ma50_turning_up"],
            "ma200_turning_up": tech["ma200_turning_up"],
            "price_above_both_mas": tech["price_above_both_mas"],
            "volume_confirming": tech["volume_confirming"],
            "off_recent_low_pct": tech["off_recent_low_pct"],
            "is_off_recent_low": tech["is_off_recent_low"],
            "sector": fund["sector"],
            "debt_to_ebitda": fund["debt_to_ebitda"],
            "interest_coverage": fund["interest_coverage"],
            "revenue_growth_pct": fund["revenue_growth_pct"],
            "earnings_growth_pct": fund["earnings_growth_pct"],
            "profit_margin_pct": fund["profit_margin_pct"],
            "fundamental_flags": fund_notes,
            "passes_fundamental_filter": fund_ok,
            "score": score,
            "ex_dividend_date": fund["ex_dividend_date"],
            "dividend_amount": fund["dividend_amount"],
            "next_earnings_date": fund["next_earnings_date"],
        }
    except Exception as e:
        print(f"Skipping {symbol}: {e}")
        return None


def debug_ticker(symbol, today=None):
    """
    Prints every intermediate value for one ticker — run this to compare
    directly against a known result (e.g. from the original script's
    CSV output) when the two disagree on whether a ticker qualifies.

    `today`, if given, reconstructs what the scan would have seen as of
    that date by truncating the fetched history there — yfinance always
    returns live/current data, so checking "does the latest row match
    `today`" only works when `today` really is today. To debug a PAST
    day, this truncates history[:today] instead, which is what makes
    retroactive debugging possible at all.

    Usage:  python -c "from scanner import debug_ticker; debug_ticker('NEM')"
            python -c "from scanner import debug_ticker; from datetime import date; debug_ticker('NEM', date(2026,9,15))"
    """
    import json as _json

    t = yf.Ticker(symbol)
    hist_raw = t.history(period="400d")
    print(f"--- {symbol} ---")
    print(f"raw rows fetched: {len(hist_raw)}")
    print(f"raw last row date: {hist_raw.index[-1].date()}  close: {hist_raw['Close'].iloc[-1]}")

    hist = hist_raw[hist_raw["Close"].notna()]
    print(f"rows after dropping NaN closes: {len(hist)}")
    if len(hist) != len(hist_raw):
        print(f"  -> dropped {len(hist_raw) - len(hist)} row(s) with NaN close")

    if today is not None:
        # Reconstruct that day by truncating — do NOT compare the live
        # latest row to `today`; that only makes sense if `today` is
        # actually today.
        hist = hist[hist.index.date <= today]
        print(f"truncated to rows on/before {today}: {len(hist)} rows remain")
        if len(hist) == 0:
            print("  -> no data available on/before that date")
            return
        last_date = hist.index[-1].date()
        print(f"reconstructed 'latest' row date: {last_date}")
        if last_date != today:
            print(
                f"  -> {today} wasn't a trading day (or no data exists for it) — "
                f"nearest prior trading day is {last_date}, using that"
            )

    if len(hist) < 210:
        print(f"  -> only {len(hist)} rows, need >= 210 — scan_ticker would return None")
        return

    closes = hist["Close"].to_numpy()
    volumes = hist["Volume"].to_numpy()
    tech = analyze_technicals(closes, volumes)
    if tech is None:
        print("analyze_technicals returned None (not enough valid MA history)")
        return

    print("technicals:")
    print(_json.dumps(tech, indent=2, default=str))

    interesting = (
        (tech["golden_cross_recent"] and tech["ma50_had_sustained_decline"]
         and tech["had_sustained_below_ma200"])
        or (tech["is_off_recent_low"] and tech["ma50_turning_up"])
    )
    print(f"\ninteresting: {interesting}")
    if not interesting:
        print("  -> scan_ticker would return None here (fails the interesting gate)")
        return

    fund = analyze_fundamentals(t)
    fund_ok, fund_notes = passes_fundamental_filter(fund)
    print("\nfundamentals (NOTE: this is TODAY's live fundamentals data, not")
    print("what it would have been on the historical date above — yfinance")
    print("doesn't expose point-in-time fundamentals, only point-in-time price)")
    print(_json.dumps(fund, indent=2, default=str))
    print(f"passes_fundamental_filter: {fund_ok} ({fund_notes})")

    score = (
        int(tech["golden_cross_recent"]) * 3
        + int(fund_ok) * 2
        + int(tech["volume_confirming"])
        + int(tech["ma200_turning_up"])
    )
    print(f"\nfinal score: {score}  (threshold: {MIN_SCORE_THRESHOLD})")
    print(f"would appear in results: {score >= MIN_SCORE_THRESHOLD}")


def run_full_scan(tickers=None, today=None, pause=0.15, progress_every=50):
    """
    Scan the whole universe. Takes several minutes — this is meant to
    run in a scheduled job, never inside a web request. Only candidates
    scoring >= MIN_SCORE_THRESHOLD are returned, matching the original
    script's behavior (this is a real cut, not a display filter — most
    scanned tickers won't make it through).

    Prints a funnel summary at the end (how many tickers had enough
    history, how many were "interesting," how many cleared the score
    threshold) so a zero-result run is distinguishable from a silent
    failure rather than a black box.
    """
    tickers = tickers or TICKER_UNIVERSE
    today = today or datetime.now().date()
    results = []
    counts = {"scanned": 0, "had_history": 0, "interesting": 0, "scored": 0}
    total = len(tickers)

    for i, symbol in enumerate(tickers, start=1):
        counts["scanned"] += 1
        row = scan_ticker(symbol, today=today, counts=counts)
        if row is not None:
            results.append(row)
            counts["scored"] += 1
        if pause:
            time.sleep(pause)
        if progress_every and i % progress_every == 0:
            print(f"  scanned {i}/{total} — {len(results)} hits so far", flush=True)

    print(
        f"Funnel: {counts['scanned']} scanned -> {counts['had_history']} had enough "
        f"history -> {counts['interesting']} were 'interesting' -> {counts['scored']} "
        f"passed score>={MIN_SCORE_THRESHOLD}",
        flush=True,
    )

    results = [r for r in results if r["score"] >= MIN_SCORE_THRESHOLD]
    results.sort(key=lambda r: (-r["score"], r["days_since_cross"] if r["days_since_cross"] is not None else 999))
    return results
