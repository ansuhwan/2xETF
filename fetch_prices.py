"""Stage 2: Download 1y daily prices for all 2X ETFs and their underlyings.

Reads data/etf_list.json, batch-downloads via yfinance in chunks, retries
failures individually, then writes:
  - data/prices.pkl       : pandas DataFrame, MultiIndex columns (ticker, field)
  - data/prices_meta.json : per-ticker status (rows, first/last date), missing list
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ETF_LIST = DATA / "etf_list.json"
PRICES_PKL = DATA / "prices.pkl"
PRICES_META = DATA / "prices_meta.json"

KST = timezone(timedelta(hours=9))
CHUNK = 40
PERIOD = "1y"


BENCHMARK_ETFS = {
    "SPY":  "Market",     # S&P 500 broad
    "QQQ":  "Nasdaq",     # Tech-heavy
    "XLK":  "Tech",       # Tech sector
    "SOXX": "Semi",       # Semiconductor
    "XBI":  "Bio",        # Biotech
    "ITA":  "Defense",    # Aerospace & Defense
    "XLF":  "Fintech",    # Financials
    "XLE":  "Energy",     # Energy
    "URA":  "Nuclear",    # Uranium / Nuclear
    "LIT":  "EV",         # Lithium / Battery
    "ARKK": "AI",         # Innovation/AI proxy
}


def load_tickers() -> tuple[list[str], dict[str, str]]:
    payload = json.loads(ETF_LIST.read_text(encoding="utf-8"))
    kind: dict[str, str] = {}
    for e in payload["etfs"]:
        t2 = (e.get("ticker_2x") or "").strip().upper()
        und = (e.get("underlying") or "").strip().upper()
        if t2:
            kind[t2] = "2x"
        if und:
            kind.setdefault(und, "underlying")
    for b in BENCHMARK_ETFS:
        kind.setdefault(b, "benchmark")
    return sorted(kind.keys()), kind


def download_chunk(tickers: list[str]) -> dict[str, pd.DataFrame]:
    df = yf.download(
        tickers=tickers,
        period=PERIOD,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    if len(tickers) == 1:
        sub = df.dropna(how="all")
        if not sub.empty:
            out[tickers[0]] = sub
        return out
    for t in tickers:
        try:
            sub = df[t].dropna(how="all")
        except KeyError:
            continue
        if not sub.empty:
            out[t] = sub
    return out


def retry_one(t: str) -> pd.DataFrame | None:
    for i in range(3):
        try:
            df = yf.Ticker(t).history(period=PERIOD, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def main() -> int:
    tickers, kind = load_tickers()
    n2 = sum(1 for k in kind.values() if k == "2x")
    nu = sum(1 for k in kind.values() if k == "underlying")
    print(f"Loading {len(tickers)} unique tickers ({n2} 2X + {nu} underlyings)")

    frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    n_chunks = (len(tickers) + CHUNK - 1) // CHUNK
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        t0 = time.time()
        try:
            got = download_chunk(chunk)
        except Exception as e:
            print(f"  chunk {i // CHUNK + 1}/{n_chunks} FAIL: {e}")
            failed.extend(chunk)
            continue
        frames.update(got)
        missing = [t for t in chunk if t not in got]
        failed.extend(missing)
        print(
            f"  chunk {i // CHUNK + 1:>2}/{n_chunks}: "
            f"{len(got)}/{len(chunk)} ok ({time.time() - t0:.1f}s)"
        )

    if failed:
        print(f"\nRetrying {len(failed)} individually...")
    still_failed: list[str] = []
    for t in failed:
        df = retry_one(t)
        if df is not None and not df.empty:
            frames[t] = df
        else:
            still_failed.append(t)
    if failed:
        print(f"Recovered {len(failed) - len(still_failed)}; {len(still_failed)} still missing")

    if not frames:
        print("ERROR: no price data downloaded", file=sys.stderr)
        return 2

    combined = pd.concat(frames, axis=1, names=["ticker", "field"])
    combined.to_pickle(PRICES_PKL)

    meta = {
        "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "period": PERIOD,
        "total_requested": len(tickers),
        "total_ok": len(frames),
        "missing": still_failed,
        "tickers": {
            t: {
                "kind": kind[t],
                "rows": int(len(df)),
                "first": str(df.index[0].date()) if len(df) else None,
                "last": str(df.index[-1].date()) if len(df) else None,
            }
            for t, df in sorted(frames.items())
        },
    }
    PRICES_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {PRICES_PKL} and {PRICES_META}")
    print(f"OK: {len(frames)}/{len(tickers)} tickers")
    if still_failed:
        print(f"Missing: {', '.join(still_failed[:20])}{' ...' if len(still_failed) > 20 else ''}")
    # Partial failure (e.g. delisted JPNL) is normal — only fail if nothing downloaded.
    return 0


if __name__ == "__main__":
    sys.exit(main())
