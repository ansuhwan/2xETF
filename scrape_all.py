"""Run every issuer scraper, dump the unified ETF list to data/etf_list.json.

Each issuer is wrapped in try/except so one failure does not abort the others.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from scrapers import ALL_SCRAPERS

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data"
OUT_FILE = OUT_DIR / "etf_list.json"

KST = timezone(timedelta(hours=9))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    per_issuer: dict[str, dict] = {}

    for issuer, fn in ALL_SCRAPERS:
        t0 = time.time()
        try:
            recs = fn()
            dur = time.time() - t0
            per_issuer[issuer] = {"count": len(recs), "ms": int(dur * 1000), "ok": True}
            all_records.extend(r.as_dict() for r in recs)
            print(f"[OK]   {issuer:<16} {len(recs):>3} ETFs ({dur:.1f}s)")
        except Exception as e:
            per_issuer[issuer] = {"count": 0, "ok": False, "error": str(e)[:200]}
            print(f"[FAIL] {issuer:<16} {e}", file=sys.stderr)

    # Dedupe by 2x ticker (some issuers may overlap, unlikely but cheap)
    seen: set[str] = set()
    unique: list[dict] = []
    for r in all_records:
        if r["ticker_2x"] in seen:
            continue
        seen.add(r["ticker_2x"])
        unique.append(r)

    payload = {
        "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "total": len(unique),
        "per_issuer": per_issuer,
        "etfs": unique,
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_FILE} ({len(unique)} ETFs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
