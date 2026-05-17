"""Tradr ETFs 2X Long single-stock ETFs."""
from __future__ import annotations

import re
from bs4 import BeautifulSoup
from .base import ETF, http_get

URL = "https://www.tradretfs.com/"
ISSUER = "Tradr"

NAME_RE = re.compile(r"Tradr\s+2X\s+Long\s+([A-Za-z0-9.\-]+)\s+Daily\s+ETF", re.I)
# Tradr ETF page slug = the 2X ticker (e.g., /aaox?hsLang=en)
SLUG_RE = re.compile(r"tradretfs\.com/([a-z0-9]{2,6})(?:[/?#]|$)", re.I)


def fetch() -> list[ETF]:
    html = http_get(URL)
    soup = BeautifulSoup(html, "lxml")
    out: list[ETF] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = SLUG_RE.search(href)
        if not m:
            continue
        text = a.get_text(" ", strip=True)
        if not NAME_RE.search(text) and a.parent:
            text = a.parent.get_text(" ", strip=True)
        nm = NAME_RE.search(text)
        if not nm:
            continue
        ticker = m.group(1).upper()
        if ticker in seen:
            continue
        underlying = nm.group(1).upper()
        # Skip non-single-stock products (QQQ, SPY, world index, etc.)
        if underlying in {"TRIPLE", "QQQ", "SPY", "WORLD"}:
            continue
        out.append(ETF(ticker_2x=ticker, name=text.strip()[:120],
                       underlying=underlying, issuer=ISSUER, url=href))
        seen.add(ticker)
    return out


if __name__ == "__main__":
    for e in fetch():
        print(e.as_dict())
