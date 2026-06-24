"""Defiance Daily Target 2X Long single-stock ETFs.

Defiance moved to a JS-rendered SPA, so the homepage no longer lists funds as
scrapable links (the old link-parsing scraper silently returned 0). But:

  * the full fund universe is enumerable via the ETF sitemap, and
  * each fund still has a server-rendered page at /{ticker}/ whose <title> reads
    "TICKER | Defiance Daily Target 2X Long {UNDERLYING} ETF".

So we enumerate slugs from etf-sitemap.xml, fetch each fund page, and keep only
the 2X-Long single-stock funds (dropping 2X Short, income, and thematic ETFs).
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup

from .base import ETF, http_get

# Defiance는 ~38종을 운용. 이보다 한참 적으면 사이트 빈응답(silent-0)으로 보고 재시도.
SANE_FLOOR = 10
RETRIES = 3

ISSUER = "Defiance"
SITEMAP = "https://www.defianceetfs.com/etf-sitemap.xml"
FUND_URL = "https://www.defianceetfs.com/{slug}/"

# "... Daily Target 2X Long OKLO ETF" -> underlying token "OKLO".
# Matches Long only; 2X Short / income / thematic titles don't match.
NAME_RE = re.compile(r"Daily\s+Target\s+2X\s+Long\s+(.+?)\s+ETF", re.I)
# Explicit "(NYSE: OKLO)" / "(NASDAQ: MSTR)" in the meta description — authoritative.
DESC_TICK_RE = re.compile(
    r"\((?:NYSE|NASDAQ|NYSE\s*Arca|Cboe|BATS|Amex)[^)]*?:\s*([A-Z][A-Z.\-]{0,5})\)", re.I
)
TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,5}$")
SITEMAP_SLUG_RE = re.compile(r"defianceetfs\.com/([a-z0-9\-]+)/?</loc>", re.I)


def _fund_slugs() -> list[str]:
    """All fund-page slugs from the ETF sitemap (deduped, order-preserving)."""
    xml = http_get(SITEMAP)
    seen: set[str] = set()
    out: list[str] = []
    for s in SITEMAP_SLUG_RE.findall(xml):
        u = s.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _parse_fund(slug: str) -> ETF | None:
    """Fetch one fund page; return an ETF iff it's a 2X-Long single-stock fund."""
    url = FUND_URL.format(slug=slug.lower())
    try:
        html = http_get(url)
    except Exception:
        return None
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""
    m = NAME_RE.search(title)
    if not m:
        return None  # 2X Short / income / thematic — not a single-stock long ETF

    name_tok = m.group(1).strip().upper()
    desc = soup.find("meta", attrs={"name": "description"})
    dtxt = desc.get("content", "") if desc else ""
    dm = DESC_TICK_RE.search(dtxt or "")

    # The fund name embeds the underlying's ticker; prefer the explicit
    # "(EXCHANGE: TICKER)" from the description when the name token isn't a clean
    # ticker (e.g. a multi-word company name).
    if TICKER_RE.match(name_tok):
        underlying = name_tok
    elif dm:
        underlying = dm.group(1).upper()
    else:
        return None

    return ETF(
        ticker_2x=slug.upper(),
        name=title[:120],
        underlying=underlying,
        issuer=ISSUER,
        url=url,
        leverage=2,
    )


def _fetch_once() -> list[ETF]:
    slugs = _fund_slugs()
    out: list[ETF] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for rec in ex.map(_parse_fund, slugs):
            if rec is not None:
                out.append(rec)
    # Dedupe by 2X ticker, keep first.
    seen: set[str] = set()
    uniq: list[ETF] = []
    for r in out:
        if r.ticker_2x not in seen:
            seen.add(r.ticker_2x)
            uniq.append(r)
    return uniq


def fetch() -> list[ETF]:
    """빈응답(silent-0) 방지: 결과가 SANE_FLOOR 미만이면 백오프 재시도."""
    res: list[ETF] = []
    for attempt in range(RETRIES):
        try:
            res = _fetch_once()
        except Exception:
            res = []
        if len(res) >= SANE_FLOOR:
            return res
        if attempt < RETRIES - 1:
            time.sleep(3 * (attempt + 1))  # 3s, 6s 백오프
    return res  # 끝까지 적으면 그대로 반환 (scrape_all 가드가 직전본 유지)


if __name__ == "__main__":
    for e in fetch():
        print(e.as_dict())
