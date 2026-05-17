"""Fetch monthly close prices for underlyings — used for monthly MA alignment.

We keep daily prices.pkl at 1y (~5MB). Monthly bars require longer history
(MA60 monthly = 5 years). This script fetches monthly-interval closes for
underlyings only and stores them in data/monthly.pkl (tiny — ~150KB).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ETF_LIST = DATA / "etf_list.json"
OUT = DATA / "monthly.pkl"

CHUNK = 40


def download_chunk(tickers: list[str]) -> dict[str, pd.Series]:
    df = yf.download(
        tickers=tickers,
        period="max",
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out: dict[str, pd.Series] = {}
    if df is None or df.empty:
        return out
    if len(tickers) == 1:
        if "Close" in df.columns:
            s = df["Close"].dropna()
            if not s.empty:
                out[tickers[0]] = s
        return out
    for t in tickers:
        try:
            s = df[t]["Close"].dropna()
        except (KeyError, AttributeError):
            continue
        if not s.empty:
            out[t] = s
    return out


def main() -> int:
    payload = json.loads(ETF_LIST.read_text(encoding="utf-8"))
    underlyings = sorted(
        {e["underlying"].strip().upper() for e in payload["etfs"] if e.get("underlying")}
    )
    print(f"Fetching monthly closes for {len(underlyings)} underlyings ({(len(underlyings)+CHUNK-1)//CHUNK} chunks)")

    series: dict[str, pd.Series] = {}
    failed: list[str] = []
    for i in range(0, len(underlyings), CHUNK):
        chunk = underlyings[i : i + CHUNK]
        t0 = time.time()
        try:
            got = download_chunk(chunk)
        except Exception as e:
            print(f"  chunk {i//CHUNK+1} FAIL: {e}", file=sys.stderr)
            failed.extend(chunk)
            continue
        series.update(got)
        missing = [t for t in chunk if t not in got]
        failed.extend(missing)
        print(f"  chunk {i//CHUNK+1}: {len(got)}/{len(chunk)} ok ({time.time()-t0:.1f}s)")

    if not series:
        print("ERROR: no monthly data", file=sys.stderr)
        return 2

    closes = pd.concat(series, axis=1)
    closes.to_pickle(OUT)

    by_len = closes.apply(lambda s: s.dropna().size)
    print(f"\nWrote {OUT} ({closes.shape[0]} months x {closes.shape[1]} tickers)")
    print(f"  ticker count: full 60m+ = {(by_len >= 60).sum()}, 20-60m = {((by_len >= 20) & (by_len < 60)).sum()}, <20m = {(by_len < 20).sum()}")
    if failed:
        print(f"  missing ({len(failed)}): {', '.join(failed[:10])}{' ...' if len(failed) > 10 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
