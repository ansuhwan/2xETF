"""Hand-curated list of major leveraged sector/index ETFs (3X and 2X).

Single-stock 3X is not permitted by the SEC, so these are all sector/index
based. A few (ERX/NUGT/GUSH/JNUG) used to be 3X but Direxion reduced them
to 2X in 2020 due to extreme volatility — leverage is set per-ETF below.

Each is paired with a matching 1X ETF as its 'underlying' so MA/RSI/sector
signals work consistently.
"""
from __future__ import annotations

from .base import ETF

# (ticker, name, underlying 1X ETF, issuer, leverage)
LEVERAGED_ETFS: list[tuple[str, str, str, str, int]] = [
    # Broad indices — 3X
    ("TQQQ", "ProShares UltraPro QQQ",                "QQQ",  "ProShares",    3),
    ("UPRO", "ProShares UltraPro S&P 500",            "SPY",  "ProShares",    3),
    ("UDOW", "ProShares UltraPro Dow30",              "DIA",  "ProShares",    3),
    ("TNA",  "Direxion Daily Small Cap Bull 3X",      "IWM",  "Direxion",     3),
    ("MIDU", "Direxion Daily Mid Cap Bull 3X",        "MDY",  "Direxion",     3),

    # Tech / Semis — 3X
    ("SOXL", "Direxion Daily Semiconductor Bull 3X",  "SOXX", "Direxion",     3),
    ("TECL", "Direxion Daily Technology Bull 3X",     "XLK",  "Direxion",     3),
    ("FNGU", "MicroSectors FANG+ 3X",                 "FNGS", "MicroSectors", 3),
    ("WEBL", "MicroSectors Solactive Internet 3X",    "",     "MicroSectors", 3),

    # Other sectors — 3X
    ("FAS",  "Direxion Daily Financial Bull 3X",      "XLF",  "Direxion",     3),
    ("LABU", "Direxion Daily S&P Biotech Bull 3X",    "XBI",  "Direxion",     3),
    ("NAIL", "Direxion Daily Homebuilders Bull 3X",   "ITB",  "Direxion",     3),
    ("RETL", "Direxion Daily Retail Bull 3X",         "XRT",  "Direxion",     3),
    ("DPST", "Direxion Daily Regional Banks Bull 3X", "KRE",  "Direxion",     3),

    # Commodity / metal — reduced from 3X to 2X in 2020
    ("ERX",  "Direxion Daily Energy Bull 2X",         "XLE",  "Direxion",     2),
    ("NUGT", "Direxion Daily Gold Miners Bull 2X",    "GDX",  "Direxion",     2),
    ("GUSH", "Direxion Daily Oil & Gas E&P Bull 2X",  "XOP",  "Direxion",     2),
    ("JNUG", "Direxion Daily Junior Gold Miners 2X",  "GDXJ", "Direxion",     2),

    # Treasuries — 3X
    ("TMF",  "Direxion Daily 20+ Year Treasury 3X",   "TLT",  "Direxion",     3),
    ("TYD",  "Direxion Daily 7-10 Year Treasury 3X",  "IEF",  "Direxion",     3),
]


def fetch() -> list[ETF]:
    return [
        ETF(
            ticker_2x=ticker,
            name=name,
            underlying=underlying,
            issuer=issuer,
            url="",
            expense_ratio=None,
            leverage=leverage,
        )
        for ticker, name, underlying, issuer, leverage in LEVERAGED_ETFS
    ]
