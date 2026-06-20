"""Run every issuer scraper, dump the unified ETF list to data/etf_list.json.

각 발행사는 try/except로 격리. 추가로 last-known-good 병합:
  - 어떤 발행사가 0개이거나 직전 대비 급감(<70%)하면 → 직전 정상본을 유지(stale 표시).
  - 일시 장애로 발행사가 통째로 사라지는 것을 구조적으로 방지.
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

DROP_FLOOR = 0.7    # 직전 대비 이 비율 미만이면 급감으로 판단 → 폴백
MIN_PREV_FOR_DROP = 5   # 직전 종목수가 이 미만이면 급감 판정 생략(노이즈 방지)


def load_prev():
    """직전 etf_list.json을 발행사별로 로드. (by_issuer, updated_at)"""
    if not OUT_FILE.exists():
        return {}, None
    try:
        prev = json.loads(OUT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}, None
    by_issuer: dict[str, list[dict]] = {}
    for e in prev.get("etfs", []):
        by_issuer.setdefault(e.get("issuer") or "?", []).append(e)
    return by_issuer, prev.get("updated_at")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prev_by_issuer, prev_updated = load_prev()

    all_records: list[dict] = []
    per_issuer: dict[str, dict] = {}
    stale_issuers: list[str] = []

    for issuer, fn in ALL_SCRAPERS:
        t0 = time.time()
        prev_recs = prev_by_issuer.get(issuer, [])
        prev_count = len(prev_recs)
        try:
            recs = fn()
            dur = time.time() - t0
            count = len(recs)
            # 폴백 판단: 0개 또는 직전 대비 급감
            sharp_drop = (prev_count >= MIN_PREV_FOR_DROP
                          and count < prev_count * DROP_FLOOR)
            if count == 0 or sharp_drop:
                reason = "empty" if count == 0 else f"drop {count}<{prev_count}"
                if prev_recs:
                    # 직전 정상본 유지 (stale 표시)
                    fellback = []
                    for e in prev_recs:
                        e2 = dict(e)
                        e2["stale"] = True
                        e2["last_ok"] = e.get("last_ok") or prev_updated
                        fellback.append(e2)
                    all_records.extend(fellback)
                    per_issuer[issuer] = {"count": prev_count, "scraped": count,
                                          "ok": False, "fellback": True, "reason": reason}
                    stale_issuers.append(issuer)
                    print(f"[STALE] {issuer:<16} {count}→직전 {prev_count}개 유지 ({reason})", file=sys.stderr)
                else:
                    per_issuer[issuer] = {"count": 0, "ok": False, "reason": reason}
                    print(f"[FAIL] {issuer:<16} {count}개, 직전본도 없음 ({reason})", file=sys.stderr)
            else:
                per_issuer[issuer] = {"count": count, "ms": int(dur * 1000), "ok": True}
                all_records.extend(r.as_dict() for r in recs)
                print(f"[OK]   {issuer:<16} {count:>3} ETFs ({dur:.1f}s)")
        except Exception as e:
            # 예외 → 직전 정상본 유지
            if prev_recs:
                fellback = []
                for pe in prev_recs:
                    e2 = dict(pe)
                    e2["stale"] = True
                    e2["last_ok"] = pe.get("last_ok") or prev_updated
                    fellback.append(e2)
                all_records.extend(fellback)
                per_issuer[issuer] = {"count": prev_count, "ok": False,
                                      "fellback": True, "error": str(e)[:200]}
                stale_issuers.append(issuer)
                print(f"[STALE] {issuer:<16} 예외→직전 {prev_count}개 유지: {e}", file=sys.stderr)
            else:
                per_issuer[issuer] = {"count": 0, "ok": False, "error": str(e)[:200]}
                print(f"[FAIL] {issuer:<16} {e}", file=sys.stderr)

    # Dedupe by 2x ticker
    seen: set[str] = set()
    unique: list[dict] = []
    for r in all_records:
        if r["ticker_2x"] in seen:
            continue
        seen.add(r["ticker_2x"])
        unique.append(r)

    # 전체 쓰기 가드: 새 총합이 직전의 90% 미만이면 중단 (최후 방어선)
    prev_total = sum(len(v) for v in prev_by_issuer.values())
    if prev_total and len(unique) < prev_total * 0.9:
        print(f"[ABORT] 총 {len(unique)} < 직전 {prev_total}의 90%. 기존 파일 유지(덮어쓰기 거부).",
              file=sys.stderr)
        return 1

    payload = {
        "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "total": len(unique),
        "stale_issuers": stale_issuers,
        "per_issuer": per_issuer,
        "etfs": unique,
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    msg = f"\nWrote {OUT_FILE} ({len(unique)} ETFs)"
    if stale_issuers:
        msg += f"  ⚠ STALE: {', '.join(stale_issuers)}"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
