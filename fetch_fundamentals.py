"""Fetch fundamental data per underlying ticker.

For each underlying, pulls:
  - Short interest % of float (yfinance .info)
  - Recent insider transactions (yfinance .insider_purchases / .insider_transactions)
  - Recent analyst upgrades/downgrades (yfinance .upgrades_downgrades)

These complement technical/options signals — short interest reveals squeeze
candidates, insider buying is the strongest pre-rally tell (CEO/CFO putting
own money in), and analyst upgrades shift consensus.

Output: data/fundamentals.json
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
OUT = DATA / "fundamentals.json"

KST = timezone(timedelta(hours=9))


def short_interest(info: dict) -> dict:
    """Pull short interest fields from .info dict."""
    spct = info.get("shortPercentOfFloat")
    sratio = info.get("shortRatio")  # days to cover
    sshort = info.get("sharesShort")
    return {
        "short_pct_of_float": round(spct * 100, 2) if isinstance(spct, (int, float)) else None,
        "short_ratio_days":   round(sratio, 2) if isinstance(sratio, (int, float)) else None,
        "shares_short":       int(sshort) if isinstance(sshort, (int, float)) else None,
    }


def insider_summary(t: yf.Ticker) -> dict:
    """Aggregate insider transactions over the past 90 days.

    Returns net dollar value (buy - sell) and counts. Officer/director purchases
    are the strongest signal — they're putting own money in with full info.
    """
    try:
        tx = t.insider_transactions
    except Exception:
        tx = None
    if tx is None or tx.empty:
        return {"insider_net_value_90d": None, "insider_buy_count_90d": 0, "insider_sell_count_90d": 0}

    cutoff = datetime.now() - timedelta(days=90)
    df = tx.copy()
    # Standardize date column
    date_col = None
    for c in ("Start Date", "Date", "Transaction Date", "Last Date"):
        if c in df.columns:
            date_col = c; break
    if date_col is None:
        return {"insider_net_value_90d": None, "insider_buy_count_90d": 0, "insider_sell_count_90d": 0}

    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    except Exception:
        return {"insider_net_value_90d": None, "insider_buy_count_90d": 0, "insider_sell_count_90d": 0}

    df = df[df[date_col] >= cutoff].dropna(subset=[date_col])
    if df.empty:
        return {"insider_net_value_90d": 0.0, "insider_buy_count_90d": 0, "insider_sell_count_90d": 0}

    # yfinance encodes direction in Text column ("Purchase at price..." / "Sale at price...")
    # Transaction column itself is often empty. Grants/gifts (Value=0) are ignored.
    # IMPORTANT: Exclude 10%+ beneficial owners (institutions like Volkswagen→RIVN).
    # Their large block "purchases" are strategic investments / fund flows, NOT
    # insider signals. True insider buying = officers/directors using own money.
    text_col = "Text" if "Text" in df.columns else None
    val_col = "Value" if "Value" in df.columns else None
    pos_col = "Position" if "Position" in df.columns else None

    def is_institutional(position: str) -> bool:
        p = (position or "").lower()
        return "10%" in p or "beneficial owner" in p or "more than" in p

    buy_count = sell_count = 0
    net_value = 0.0
    if text_col is not None and val_col is not None:
        for _, row in df.iterrows():
            text = str(row.get(text_col, "")).lower()
            position = str(row.get(pos_col, "")) if pos_col else ""
            try:
                val = float(row.get(val_col) or 0)
            except Exception:
                val = 0.0
            if val <= 0:
                continue  # skip grants/gifts
            if is_institutional(position):
                continue  # skip 10%+ holders — not a true insider signal
            if "purchase" in text or "buy" in text or "acquisition" in text:
                buy_count += 1; net_value += val
            elif "sale" in text or "sold" in text or "sell" in text or "disposition" in text:
                sell_count += 1; net_value -= val

    return {
        "insider_net_value_90d": round(net_value, 0),
        "insider_buy_count_90d": buy_count,
        "insider_sell_count_90d": sell_count,
    }


def analyst_changes(t: yf.Ticker) -> dict:
    """Count upgrades/downgrades + price target raises/lowers in last 30 days.

    yfinance Action codes: up=upgrade, down=downgrade, main=maintain,
    reit=reiterate, init=initiate. Price target column ('Raises'/'Lowers')
    is more granular than rating change alone.
    """
    try:
        ud = t.upgrades_downgrades
    except Exception:
        ud = None
    if ud is None or ud.empty:
        return {"upgrades_30d": 0, "downgrades_30d": 0, "pt_raises_30d": 0, "pt_lowers_30d": 0}

    cutoff = datetime.now() - timedelta(days=30)
    df = ud.copy()
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index >= pd.Timestamp(cutoff)]
    except Exception:
        return {"upgrades_30d": 0, "downgrades_30d": 0, "pt_raises_30d": 0, "pt_lowers_30d": 0}

    if df.empty:
        return {"upgrades_30d": 0, "downgrades_30d": 0, "pt_raises_30d": 0, "pt_lowers_30d": 0}

    upgrades = downgrades = 0
    if "Action" in df.columns:
        actions = df["Action"].astype(str).str.lower()
        upgrades   = int((actions == "up").sum())
        downgrades = int((actions == "down").sum())

    pt_raises = pt_lowers = 0
    if "priceTargetAction" in df.columns:
        pts = df["priceTargetAction"].astype(str).str.lower()
        pt_raises = int((pts == "raises").sum())
        pt_lowers = int((pts == "lowers").sum())

    return {
        "upgrades_30d": upgrades,
        "downgrades_30d": downgrades,
        "pt_raises_30d": pt_raises,
        "pt_lowers_30d": pt_lowers,
    }


def fetch_one(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        out = {}
        out.update(short_interest(info))
        out.update(insider_summary(t))
        out.update(analyst_changes(t))
        return out
    except Exception:
        return None


def main() -> int:
    payload = json.loads(ETF_LIST.read_text(encoding="utf-8"))
    underlyings: list[str] = sorted({
        (e.get("underlying") or "").strip().upper()
        for e in payload["etfs"] if e.get("underlying")
    })

    out: dict[str, dict] = {}
    print(f"Fetching fundamentals for {len(underlyings)} underlyings...")

    for i, t in enumerate(underlyings, 1):
        fields = fetch_one(t)
        if fields:
            out[t] = fields
        time.sleep(0.3)
        if i % 20 == 0 or i == len(underlyings):
            print(f"  {i:>3}/{len(underlyings)}  ok: {len(out)}")
            OUT.write_text(
                json.dumps({
                    "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
                    "tickers": out,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"\nDone: {len(out)}/{len(underlyings)} underlyings have fundamentals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
