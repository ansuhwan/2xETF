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
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup

from .base import ETF, http_get

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


def fetch() -> list[ETF]:
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


if __name__ == "__main__":
    for e in fetch():
        print(e.as_dict())
