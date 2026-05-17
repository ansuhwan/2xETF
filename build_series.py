"""Stage 6 helper: emit per-ticker price + SMA series for the chart panel.

Reads data/prices.pkl, writes data/series/<TICKER>.json:
  { dates, open, high, low, close, volume, sma: {"5":[...], "20":[...], ...} }
Each per-ticker file is small (~10–20 KB) so the dashboard fetches lazily.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PRICES_PKL = DATA / "prices.pkl"
OUT_DIR = DATA / "series"

SMA_PERIODS = [5, 20, 60, 120]


def fnum(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return round(float(v), 4)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_pickle(PRICES_PKL)
    tickers = sorted(prices.columns.get_level_values(0).unique())

    written = 0
    skipped = 0
    for t in tickers:
        try:
            df = prices[t]
        except KeyError:
            skipped += 1
            continue
        df = df.dropna(subset=["Close"])
        if df.empty:
            skipped += 1
            continue

        close = df["Close"]
        sma = {p: close.rolling(p).mean() for p in SMA_PERIODS}

        payload = {
            "ticker": t,
            "dates":  [d.strftime("%Y-%m-%d") for d in df.index],
            "open":   [fnum(x) for x in df["Open"]],
            "high":   [fnum(x) for x in df["High"]],
            "low":    [fnum(x) for x in df["Low"]],
            "close":  [fnum(x) for x in df["Close"]],
            "volume": [None if pd.isna(x) else int(x) for x in df["Volume"]],
            "sma":    {str(p): [fnum(x) for x in sma[p]] for p in SMA_PERIODS},
        }
        (OUT_DIR / f"{t}.json").write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        written += 1

    print(f"Wrote {written} series files to {OUT_DIR} (skipped {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
