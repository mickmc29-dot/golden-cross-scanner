"""
Golden Cross Scanner — core scan logic.

Imported by run_scan.py (the scheduled job that does the heavy
lifting) and kept separate from app.py so the web server never
runs a full scan inside a request.

MERGE NOTE: scan_ticker() below is a working reference version of
the signals discussed — golden cross detected via numpy array
comparisons (not pandas .shift() on booleans, which caused the
earlier false-positive bug), RSI, volume confirmation, fundamentals,
plus ex-dividend and earnings dates. Swap in your own scoring
weights and fundamental-flag logic where the TODOs are.
"""

import math
import time

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from tickers import TICKER_UNIVERSE


def compute_rsi(closes, period=14):
    """
    Wilder's RSI — the standard variant used by charting platforms.

    Seeds with a simple average over the first `period` changes, then
    applies Wilder's smoothing forward across the whole series. A plain
    mean of the last 14 days (the earlier approach here) overreacts:
    if every one of those days is down, avg_gain hits exactly zero and
    RSI pins to 0 no matter how small the losses were, which produced
    implausible single-digit readings.
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


def find_golden_cross(ma50, ma200, lookback=10):
    """
    Detect an actual crossover event (not just ma50 > ma200), using
    numpy array comparisons rather than pandas .shift() on booleans
    — shift() on bool dtype was the source of the earlier bug where
    every day ma50 was above ma200 got flagged as a "cross."
    """
    above = ma50 > ma200
    cross_idxs = np.where(above[1:] & ~above[:-1])[0] + 1
    if len(cross_idxs) == 0 or (len(above) - 1 - cross_idxs[-1]) > lookback:
        return None
    days_since = len(above) - 1 - cross_idxs[-1]
    return int(days_since)


def scan_ticker(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y")

        # yfinance sometimes appends a placeholder row for the current
        # session before that day's data is fully finalized (all OHLC
        # values NaN) — drop any such rows so a NaN close can never
        # leak into price, RSI, volume checks, etc. downstream.
        hist = hist[hist["Close"].notna()]

        if len(hist) < 200:
            return None

        closes = hist["Close"].to_numpy()
        volumes = hist["Volume"].to_numpy()
        ma50 = pd.Series(closes).rolling(50).mean().to_numpy()
        ma200 = pd.Series(closes).rolling(200).mean().to_numpy()

        valid = ~np.isnan(ma50) & ~np.isnan(ma200)
        ma50_v, ma200_v = ma50[valid], ma200[valid]

        days_since_cross = find_golden_cross(ma50_v, ma200_v)
        if days_since_cross is None:
            return None  # no recent cross — not a candidate

        price = float(closes[-1])
        rsi_raw = compute_rsi(closes)
        rsi = round(rsi_raw, 1) if rsi_raw is not None else None
        ma50_turning_up = bool(ma50_v[-1] > ma50_v[-5])
        ma200_turning_up = bool(ma200_v[-1] > ma200_v[-5])
        avg_vol_20 = np.mean(volumes[-20:])
        volume_confirming = bool(volumes[-1] > avg_vol_20 * 1.2)
        recent_low = float(np.min(closes[-60:]))
        off_recent_low_pct = round((price - recent_low) / recent_low * 100, 1)

        info = t.info
        sector = info.get("sector", "Unknown")

        total_debt = info.get("totalDebt")
        ebitda = info.get("ebitda")
        debt_to_ebitda = None
        if total_debt is not None and ebitda:  # ebitda could be 0 or negative
            if ebitda > 0:
                debt_to_ebitda = round(total_debt / ebitda, 2)
            # a company with negative EBITDA has an undefined/meaningless
            # ratio here (and is arguably a red flag on its own) — leave
            # debt_to_ebitda as None rather than report a nonsense number

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
            cal = t.calendar
            earnings_dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if earnings_dates:
                next_earnings_date = min(earnings_dates).strftime("%Y-%m-%d")
        except Exception:
            pass

        score = _score(
            debt_to_ebitda, revenue_growth_pct, earnings_growth_pct,
            profit_margin_pct, volume_confirming
        )

        return {
            "ticker": symbol,
            "price": round(price, 2),
            "ma50": round(float(ma50_v[-1]), 2),
            "ma200": round(float(ma200_v[-1]), 2),
            "rsi": rsi,
            "days_since_cross": days_since_cross,
            "ma50_turning_up": ma50_turning_up,
            "ma200_turning_up": ma200_turning_up,
            "volume_confirming": volume_confirming,
            "off_recent_low_pct": off_recent_low_pct,
            "sector": sector,
            "debt_to_ebitda": debt_to_ebitda,
            "revenue_growth_pct": revenue_growth_pct,
            "earnings_growth_pct": earnings_growth_pct,
            "profit_margin_pct": profit_margin_pct,
            "fundamental_flags": "no major red flags",  # TODO: your actual flag logic
            "score": score,
            "ex_dividend_date": ex_dividend_date,
            "dividend_amount": dividend_amount,
            "next_earnings_date": next_earnings_date,
        }
    except Exception as e:
        print(f"Skipping {symbol}: {e}")
        return None


def _pct(v):
    """
    Converts a fraction (0.05) to a percentage (5.0), treating None,
    NaN, and infinity all as "missing" — yfinance sometimes returns
    nan for an unavailable numeric field rather than None, and nan
    silently survives arithmetic (round(nan, 1) is still nan), so it
    was slipping through as a real value and later broke JSON output
    downstream (Python's json.dump writes a literal NaN token, which
    is not valid JSON and browsers' JSON.parse rejects outright).
    """
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except TypeError:
        return None
    return round(v * 100, 1)


def _score(debt_to_ebitda, rev_growth, earn_growth, margin, volume_confirming):
    """
    Starting point only — adjust these thresholds and weights to match
    what you actually want to reward. Currently: start at 5, add up to
    4 points across debt, growth, margin, and volume confirmation.
    """
    score = 5

    if debt_to_ebitda is not None:
        if debt_to_ebitda < 1.0:
            score += 1
        elif debt_to_ebitda > 4.0:
            score -= 1  # meaningfully leveraged — a real caution flag

    if rev_growth is not None and earn_growth is not None:
        if rev_growth > 15 and earn_growth > 15:
            score += 1

    if margin is not None and margin > 20:
        score += 1

    if volume_confirming:
        score += 1

    return score


def run_full_scan(tickers=None, pause=0.15, progress_every=50):
    """
    Scan the whole universe. Takes several minutes — this is meant to
    run in a scheduled job, never inside a web request.

    `pause` adds a small delay between tickers so Yahoo doesn't
    rate-limit us partway through ~900 symbols.
    """
    tickers = tickers or TICKER_UNIVERSE
    results = []
    total = len(tickers)

    for i, symbol in enumerate(tickers, start=1):
        row = scan_ticker(symbol)
        if row is not None:
            results.append(row)
        if pause:
            time.sleep(pause)
        if progress_every and i % progress_every == 0:
            print(f"  scanned {i}/{total} — {len(results)} hits so far", flush=True)

    results.sort(key=lambda r: (-r["score"], r["days_since_cross"]))
    return results
