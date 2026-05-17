"""REX Shares T-Rex 2X Long single-stock leveraged ETFs."""
from __future__ import annotations

import re
from bs4 import BeautifulSoup
from .base import ETF, http_get

URL = "https://www.rexshares.com/t-rex-leveraged-etfs/"
ISSUER = "REX"

# T-Rex names follow "T-REX 2X Long {Company} Daily Target ETF" or similar.
# Underlying ticker is not always in the fund name → rely on a manual map and
# back-fill with the fund-page URL slug when present.
NAME_RE = re.compile(
    r"(?:T-?Rex\s+)?2X\s+Long\s+(?:Daily\s+Target\s+)?([A-Za-z0-9.\-]+?)(?:\s+(?:Daily|ETF))",
    re.I,
)

# Map known T-Rex 2X tickers → underlying. Fund names list company names, not tickers,
# so this manual table is the source of truth for underlyings.
TICKER_TO_UNDERLYING: dict[str, str] = {
    "MSTU": "MSTR", "NVDX": "NVDA", "TSLT": "TSLA", "BTCL": "BTC-USD", "ETU": "ETH-USD",
    "CCUP": "CRCL", "CRWU": "CRWV", "AAPX": "AAPL", "GOOX": "GOOG", "MSFX": "MSFT",
    "NFLU": "NFLX", "ROBN": "HOOD", "DJTU": "DJT", "RBLU": "RBLX", "GMEU": "GME",
    "SNOU": "SNOW", "SMUP": "SMR", "GLXU": "GLXY", "AFRU": "AFRM", "KTUP": "KTOS",
    "TTDU": "TTD", "BMNU": "BMNR", "SBTU": "SBET", "CIFU": "CIFR", "EOSU": "EOSE",
    "RDWU": "RDW", "FGRU": "FIGR", "APHU": "APH", "PAAU": "PAAS", "SNDU": "SNDK",
    "AXTU": "AXTI",
}

TICKER_RE = re.compile(r"\b([A-Z]{3,5})\b")


def fetch() -> list[ETF]:
    html = http_get(URL)
    soup = BeautifulSoup(html, "lxml")
    out: list[ETF] = []
    seen: set[str] = set()
    # Walk all anchor + text blocks; rely on TICKER_TO_UNDERLYING as the truth source.
    for a in soup.find_all("a", href=True):
        text = " ".join(a.stripped_strings)
        if not text:
            continue
        # Look for a known 2X ticker in the visible text or href
        candidates = set(TICKER_RE.findall(text.upper()))
        href_upper = a["href"].upper()
        for tk in TICKER_TO_UNDERLYING:
            if tk in candidates or f"/{tk.lower()}" in a["href"].lower() or f"={tk}" in href_upper:
                if tk in seen:
                    continue
                out.append(ETF(
                    ticker_2x=tk,
                    name=text.strip()[:120],
                    underlying=TICKER_TO_UNDERLYING[tk],
                    issuer=ISSUER,
                    url=a["href"] if a["href"].startswith("http") else f"https://www.rexshares.com{a['href']}",
                ))
                seen.add(tk)
                break
    # Fallback: if scraping caught nothing (page restructured), still emit the known list
    # so downstream price analysis keeps working.
    if not out:
        for tk, und in TICKER_TO_UNDERLYING.items():
            out.append(ETF(ticker_2x=tk, name=f"T-REX 2X Long {und}", underlying=und,
                           issuer=ISSUER, url=URL))
    return out


if __name__ == "__main__":
    for e in fetch():
        print(e.as_dict())
