"""Direxion single-stock 2X (Bull) leveraged ETFs.

Direxion's marketing site blocks default crawlers (HTTP 403), so we try the
HTML page first and fall back to a curated list. Refresh this list when
Direxion launches new single-stock ETFs.
"""
from __future__ import annotations

import re
from bs4 import BeautifulSoup
from .base import ETF, http_get

URL = "https://www.direxion.com/single-stock-etfs"
ISSUER = "Direxion"

# Curated list of Direxion's single-stock 2X *Bull* ETFs (long only).
# Source: Direxion fund pages as of 2025-2026 launches.
FALLBACK: dict[str, str] = {
    "TSLL": "TSLA",  # Daily TSLA Bull 2X Shares
    "AAPU": "AAPL",
    "AMZU": "AMZN",
    "GGLL": "GOOGL",
    "MSFU": "MSFT",
    "NVDU": "NVDA",
    "METU": "META",
    "MUU":  "MU",
    "TSMX": "TSM",
    "JPNL": "JPM",   # adjust if Direxion uses a different ticker
}

# Direxion's index/sector leveraged Bull ETFs (long only). Site blocks crawlers,
# and these are a stable, slow-changing set, so curate directly.
# (ticker_2x, underlying, leverage, name)
INDEX_ETFS: list[tuple[str, str, int, str]] = [
    ("TNA",  "IWM",  3, "Daily Small Cap Bull 3X Shares"),
    ("MIDU", "MDY",  3, "Daily Mid Cap Bull 3X Shares"),
    ("SOXL", "SOXX", 3, "Daily Semiconductor Bull 3X Shares"),
    ("TECL", "XLK",  3, "Daily Technology Bull 3X Shares"),
    ("FAS",  "XLF",  3, "Daily Financial Bull 3X Shares"),
    ("LABU", "XBI",  3, "Daily S&P Biotech Bull 3X Shares"),
    ("NAIL", "ITB",  3, "Daily Homebuilders & Supplies Bull 3X Shares"),
    ("RETL", "XRT",  3, "Daily Retail Bull 3X Shares"),
    ("DPST", "KRE",  3, "Daily Regional Banks Bull 3X Shares"),
    ("ERX",  "XLE",  2, "Daily Energy Bull 2X Shares"),
    ("NUGT", "GDX",  2, "Daily Gold Miners Bull 2X Shares"),
    ("GUSH", "XOP",  2, "Daily S&P Oil & Gas E&P Bull 2X Shares"),
    ("JNUG", "GDXJ", 2, "Daily Junior Gold Miners Bull 2X Shares"),
    ("TMF",  "TLT",  3, "Daily 20+ Year Treasury Bull 3X Shares"),
    ("TYD",  "IEF",  3, "Daily 7-10 Year Treasury Bull 3X Shares"),
]

NAME_RE = re.compile(
    r"Daily\s+([A-Z.\-]{2,6})\s+Bull\s+2X\s+Shares",
    re.I,
)
HREF_RE = re.compile(r"/etfs/([a-z]{3,5})/?$", re.I)


def fetch() -> list[ETF]:
    out: list[ETF] = []
    seen: set[str] = set()
    try:
        html = http_get(URL)
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            text = " ".join(a.stripped_strings)
            nm = NAME_RE.search(text)
            if not nm:
                continue
            m = HREF_RE.search(a["href"])
            if not m:
                continue
            ticker = m.group(1).upper()
            if ticker in seen:
                continue
            underlying = nm.group(1).upper()
            out.append(ETF(ticker_2x=ticker, name=text.strip()[:120],
                           underlying=underlying, issuer=ISSUER,
                           url=f"https://www.direxion.com{a['href']}"))
            seen.add(ticker)
    except Exception:
        out = []  # fall through to fallback
    if not out:
        for tk, und in FALLBACK.items():
            out.append(ETF(ticker_2x=tk, name=f"Daily {und} Bull 2X Shares",
                           underlying=und, issuer=ISSUER,
                           url=f"https://www.direxion.com/etfs/{tk.lower()}"))
    # Always add the curated index/sector leveraged ETFs (not on the single-stock page).
    have = {e.ticker_2x for e in out}
    for tk, und, lev, name in INDEX_ETFS:
        if tk in have:
            continue
        out.append(ETF(ticker_2x=tk, name=name, underlying=und, issuer=ISSUER,
                       leverage=lev,
                       url=f"https://www.direxion.com/etfs/{tk.lower()}"))
    return out


if __name__ == "__main__":
    for e in fetch():
        print(e.as_dict())
