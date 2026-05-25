"""Fetch front-month options snapshot for each underlying ticker.

For each underlying, pulls the nearest-expiration option chain and computes:
  - Put/Call volume ratio (today's flow)
  - Put/Call open interest ratio (positioning)
  - ATM implied volatility (annualized %)

Results are useful as leading indicators — heavy call buying often precedes
moves, IV expansion signals expected event-driven volatility.
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
OUT = DATA / "options.json"

KST = timezone(timedelta(hours=9))


def snapshot(ticker_obj: yf.Ticker, spot: float | None) -> dict | None:
    try:
        exps = ticker_obj.options
    except Exception:
        return None
    if not exps:
        return None

    front = exps[0]
    try:
        chain = ticker_obj.option_chain(front)
    except Exception:
        return None

    calls = chain.calls
    puts  = chain.puts
    if (calls is None or calls.empty) and (puts is None or puts.empty):
        return None

    def col_sum(df: pd.DataFrame, col: str) -> int:
        if df is None or col not in df.columns: return 0
        v = df[col].fillna(0).sum()
        return int(v) if pd.notna(v) else 0

    call_vol = col_sum(calls, "volume")
    put_vol  = col_sum(puts,  "volume")
    call_oi  = col_sum(calls, "openInterest")
    put_oi   = col_sum(puts,  "openInterest")

    pc_vol = (put_vol / call_vol) if call_vol > 0 else None
    pc_oi  = (put_oi  / call_oi)  if call_oi  > 0 else None

    # ATM implied volatility = avg of nearest call + put IV
    atm_iv = None
    if spot is not None and spot > 0:
        ivs: list[float] = []
        for df in (calls, puts):
            if df is None or df.empty or "impliedVolatility" not in df.columns:
                continue
            dist = (df["strike"] - spot).abs()
            atm = df.loc[dist.idxmin()]
            iv = atm.get("impliedVolatility")
            if pd.notna(iv) and iv > 0:
                ivs.append(float(iv))
        if ivs:
            atm_iv = round(sum(ivs) / len(ivs) * 100, 1)  # to percent

    return {
        "expiration": front,
        "call_volume": call_vol,
        "put_volume": put_vol,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "pc_ratio_vol": round(pc_vol, 2) if pc_vol is not None else None,
        "pc_ratio_oi":  round(pc_oi, 2)  if pc_oi  is not None else None,
        "atm_iv": atm_iv,
    }


def main() -> int:
    payload = json.loads(ETF_LIST.read_text(encoding="utf-8"))
    underlyings: list[str] = sorted({
        (e.get("underlying") or "").strip().upper()
        for e in payload["etfs"] if e.get("underlying")
    })

    prices = None
    try:
        prices = pd.read_pickle(PRICES_PKL)
    except Exception:
        pass

    out: dict[str, dict] = {}
    print(f"Fetching options for {len(underlyings)} underlyings...")

    for i, t in enumerate(underlyings, 1):
        spot = None
        if prices is not None and t in prices.columns.get_level_values(0):
            try:
                spot = float(prices[t]["Close"].dropna().iloc[-1])
            except Exception:
                spot = None

        snap = None
        try:
            snap = snapshot(yf.Ticker(t), spot)
        except Exception:
            snap = None
        if snap:
            out[t] = snap

        time.sleep(0.25)
        if i % 20 == 0 or i == len(underlyings):
            print(f"  {i:>3}/{len(underlyings)}  ok so far: {len(out)}")
            OUT.write_text(
                json.dumps({
                    "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
                    "tickers": out,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"\nDone: {len(out)}/{len(underlyings)} underlyings have options data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
