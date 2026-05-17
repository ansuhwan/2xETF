"""GraniteShares 2x Long single-stock ETFs."""
from __future__ import annotations

import re
from bs4 import BeautifulSoup
from .base import ETF, http_get

URL = "https://graniteshares.com/institutional/us/en-us/etfs/"
ISSUER = "GraniteShares"

NAME_RE = re.compile(r"GraniteShares\s+2x\s+Long\s+([A-Za-z0-9.\-]+)\s+Daily\s+ETF", re.I)
HREF_RE = re.compile(r"/etfs/([a-z0-9]+)/?$", re.I)


def fetch() -> list[ETF]:
    html = http_get(URL)
    soup = BeautifulSoup(html, "lxml")
    out: list[ETF] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = HREF_RE.search(href)
        if not m:
            continue
        ticker = m.group(1).upper()
        if ticker in seen:
            continue
        text = " ".join(a.stripped_strings) or ""
        # walk up to find a longer text node containing the fund name
        if not NAME_RE.search(text):
            parent_text = " ".join((a.parent.stripped_strings if a.parent else []))
            text = parent_text or text
        nm = NAME_RE.search(text)
        if not nm:
            continue
        underlying = nm.group(1).upper()
        full_url = href if href.startswith("http") else f"https://graniteshares.com{href}"
        out.append(ETF(ticker_2x=ticker, name=text.strip()[:120], underlying=underlying,
                       issuer=ISSUER, url=full_url))
        seen.add(ticker)
    return out


if __name__ == "__main__":
    for e in fetch():
        print(e.as_dict())
