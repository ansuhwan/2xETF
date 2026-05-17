"""Build ticker -> Toss product code mapping via Toss's autocomplete API.

Toss's stock URLs use internal product codes (e.g. NAS0221213008 for NVDL),
not tickers. This script queries Toss's autocomplete endpoint once per
unique ticker and writes data/toss_ids.json.

Result is incremental — re-running only fetches missing/unresolved tickers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ETF_LIST = DATA / "etf_list.json"
OUT_FILE = DATA / "toss_ids.json"

API = "https://wts-info-api.tossinvest.com/api/v2/search-all/wts-auto-complete"
HEADERS = {
    "Content-Type": "application/json; charset=UTF-8",
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://www.tossinvest.com",
    "Referer": "https://www.tossinvest.com/",
}


def lookup(ticker: str) -> dict | None:
    body = {"query": ticker, "sections": [{"type": "PRODUCT", "size": 10}]}
    try:
        r = requests.post(API, json=body, headers=HEADERS, timeout=10)
        r.raise_for_status()
        result = r.json().get("result") or []
        if not result:
            return None
        items = (result[0].get("data") or {}).get("items") or []
    except Exception:
        return None

    # Prefer exact symbol match
    for it in items:
        if (it.get("symbol") or "").upper() == ticker.upper():
            return {
                "code": it.get("productCode"),
                "name_kr": it.get("subKeyword"),
                "market": it.get("market"),
            }
    return None


def main() -> int:
    payload = json.loads(ETF_LIST.read_text(encoding="utf-8"))
    cached: dict[str, dict | None] = {}
    if OUT_FILE.exists():
        try:
            cached = json.loads(OUT_FILE.read_text(encoding="utf-8"))
        except Exception:
            cached = {}

    tickers: set[str] = set()
    for e in payload["etfs"]:
        if e.get("ticker_2x"):
            tickers.add(e["ticker_2x"].strip().upper())
        if e.get("underlying"):
            tickers.add(e["underlying"].strip().upper())

    todo = sorted(t for t in tickers if not cached.get(t))
    print(f"Resolving {len(todo)}/{len(tickers)} tickers via Toss autocomplete...")

    for i, t in enumerate(todo, 1):
        info = lookup(t)
        cached[t] = info  # None means not found; will retry next run
        time.sleep(0.15)
        if i % 25 == 0 or i == len(todo):
            OUT_FILE.write_text(
                json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tag = info["code"] if info else "—"
            print(f"  {i:>3}/{len(todo)}  last: {t} -> {tag}")

    OUT_FILE.write_text(
        json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    found = sum(1 for v in cached.values() if v)
    missing = [t for t, v in cached.items() if not v]
    print(f"\n{found}/{len(cached)} mapped; {len(cached) - found} not found in Toss")
    if missing:
        print(f"Missing sample: {missing[:15]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
