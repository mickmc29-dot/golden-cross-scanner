"""
Golden Cross Scanner — Flask API wrapper

Wraps the scan logic in a /scan endpoint that returns JSON, so the
dashboard can call it on demand instead of waiting on a scheduled
CSV email.

MERGE NOTE: the per-ticker scan logic below is a working reference
implementation covering the same signals as your existing script
(golden cross via numpy — not pandas shift(), which was the source
of the earlier bug — RSI, volume confirmation, fundamental filter,
plus the new ex-dividend and earnings-date fields). Swap in your
actual ticker universe and any scoring refinements you've since
made; the structure (one function per ticker, returning a dict) is
built so you can drop your existing logic straight into
`scan_ticker()` below.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # allows the dashboard HTML (served from anywhere) to call this API

# TODO: replace with your full S&P 500 + Dow + S&P 400 universe
TICKER_UNIVERSE = ["CBOE", "APO", "REGN", "CMG", "KNSL"]


def compute_rsi(closes, period=14):
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
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
        rsi = round(compute_rsi(closes), 1)
        ma50_turning_up = bool(ma50_v[-1] > ma50_v[-5])
        ma200_turning_up = bool(ma200_v[-1] > ma200_v[-5])
        avg_vol_20 = np.mean(volumes[-20:])
        volume_confirming = bool(volumes[-1] > avg_vol_20 * 1.2)
        recent_low = float(np.min(closes[-60:]))
        off_recent_low_pct = round((price - recent_low) / recent_low * 100, 1)

        info = t.info
        sector = info.get("sector", "Unknown")
        debt_to_ebitda = info.get("debtToEbitda")
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
    return round(v * 100, 1) if v is not None else None


def _score(debt_to_ebitda, rev_growth, earn_growth, margin, volume_confirming):
    # TODO: replace with your actual scoring weights
    score = 5
    if debt_to_ebitda is not None and debt_to_ebitda < 1.0:
        score += 1
    if volume_confirming:
        score += 1
    return score


@app.route("/scan")
def scan():
    results = [scan_ticker(sym) for sym in TICKER_UNIVERSE]
    results = [r for r in results if r is not None]
    return jsonify({
        "last_run": datetime.utcnow().isoformat() + "Z",
        "results": results,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
