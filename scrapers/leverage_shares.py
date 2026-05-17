"""Leverage Shares 2x Long single-stock ETFs (US site)."""
from __future__ import annotations

import re
from bs4 import BeautifulSoup
from .base import ETF, http_get

URL = "https://leverageshares.com/us/"
ISSUER = "LeverageShares"

# URL pattern: /us/etfs/leverage-shares-2x-long-{underlying}-daily-etf/
SLUG_RE = re.compile(
    r"leverageshares\.com/us/etfs/leverage-shares-2x-long-([a-z0-9\-]+?)-daily-etf/?",
    re.I,
)
# Card text shapes seen on the page:
#   "GLWG 2x Long GLW Daily ETF"
#   "AXPG 2x Long AXP"
#   "GLWG 2x Long GLW NEW"
# Always: TICKER (3-5 caps) + "2x Long" + underlying. Trailing label optional.
CARD_RE = re.compile(r"^\s*([A-Z]{3,5})\s+2x\s+Long\s+([A-Z0-9.\-]+)", re.I)
EXCLUDE_UNDERLYING = {"WORLD", "WORLDSTOCK"}


def fetch() -> list[ETF]:
    html = http_get(URL)
    soup = BeautifulSoup(html, "lxml")
    # group anchors by slug, prefer the card variant that has a ticker prefix
    by_slug: dict[str, tuple[str, str, str]] = {}  # slug -> (ticker, name, href)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = SLUG_RE.search(href)
        if not m:
            continue
        slug = m.group(1).replace("-", "").upper()
        if slug in EXCLUDE_UNDERLYING:
            continue
        text = a.get_text(" ", strip=True)
        cm = CARD_RE.match(text)
        if cm:
            ticker = cm.group(1).upper()
            # Prefer first hit; ignore if we already have a complete record
            if slug not in by_slug:
                by_slug[slug] = (ticker, text[:120], href)
    out = [ETF(ticker_2x=tk, name=nm, underlying=slug, issuer=ISSUER, url=hr)
           for slug, (tk, nm, hr) in by_slug.items()]
    return out


if __name__ == "__main__":
    for e in fetch():
        print(e.as_dict())
