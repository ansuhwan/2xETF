"""Fetch next earnings date + EPS estimate for each underlying ticker.

Reads data/etf_list.json, writes data/earnings.json:
  {
    "updated_at": "...",
    "tickers": {
      "NVDA": {"next_earnings": "2026-05-21", "eps_estimate": 1.77},
      ...
    }
  }

Incremental: re-fetches only tickers whose cached earnings date is in the past
or missing. Polite rate: ~0.2s between calls.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ETF_LIST = DATA / "etf_list.json"
OUT_FILE = DATA / "earnings.json"

KST = timezone(timedelta(hours=9))


def get_next_earnings(t: str) -> dict | None:
    try:
        cal = yf.Ticker(t).calendar
        if not cal or not isinstance(cal, dict):
            return None
        dates = cal.get("Earnings Date")
        if not dates:
            return None
        if isinstance(dates, list):
            d = dates[0]
        else:
            d = dates
        if hasattr(d, "isoformat"):
            d_iso = d.isoformat()[:10]
        else:
            d_iso = str(d)[:10]
        return {
            "next_earnings": d_iso,
            "eps_estimate": cal.get("Earnings Average"),
            "eps_low": cal.get("Earnings Low"),
            "eps_high": cal.get("Earnings High"),
        }
    except Exception:
        return None


def main() -> int:
    payload = json.loads(ETF_LIST.read_text(encoding="utf-8"))
    cached: dict[str, dict | None] = {}
    if OUT_FILE.exists():
        try:
            prev = json.loads(OUT_FILE.read_text(encoding="utf-8"))
            cached = prev.get("tickers", {})
        except Exception:
            cached = {}

    underlyings: set[str] = {
        e["underlying"].strip().upper()
        for e in payload["etfs"]
        if e.get("underlying")
    }

    today = date.today()
    todo: list[str] = []
    for t in sorted(underlyings):
        c = cached.get(t)
        if not c or not c.get("next_earnings"):
            todo.append(t)
            continue
        try:
            ed = date.fromisoformat(c["next_earnings"])
            if ed < today:
                todo.append(t)
        except Exception:
            todo.append(t)

    print(f"Refreshing {len(todo)}/{len(underlyings)} underlyings (cached {len(underlyings) - len(todo)})")

    for i, t in enumerate(todo, 1):
        info = get_next_earnings(t)
        if info is not None:
            cached[t] = info
        else:
            cached[t] = cached.get(t)  # keep previous if any
        time.sleep(0.2)
        if i % 20 == 0 or i == len(todo):
            payload_out = {
                "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
                "tickers": cached,
            }
            OUT_FILE.write_text(
                json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tag = info.get("next_earnings") if info else "-"
            print(f"  {i:>3}/{len(todo)}  last: {t} -> {tag}")

    OUT_FILE.write_text(
        json.dumps(
            {"updated_at": datetime.now(KST).isoformat(timespec="seconds"), "tickers": cached},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    have = sum(1 for v in cached.values() if v and v.get("next_earnings"))
    soon = sum(
        1 for v in cached.values()
        if v and v.get("next_earnings")
        and 0 <= (date.fromisoformat(v["next_earnings"]) - today).days <= 7
    )
    print(f"\n{have}/{len(cached)} with earnings date; {soon} within 7 days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
