"""Hand-curated list of major 3X leveraged sector/index ETFs.

Single-stock 3X is not permitted by the SEC in the US, so 3X ETFs are all
sector- or index-based. We keep these in the same dashboard universe but
flag them as leverage=3 and pair them with the matching 1X ETF as their
'underlying' so MA/RSI/sector signals work consistently.
"""
from __future__ import annotations

from .base import ETF

# (ticker, name, underlying 1X ETF, issuer)
THREE_X_ETFS: list[tuple[str, str, str, str]] = [
    # Broad indices
    ("TQQQ", "ProShares UltraPro QQQ",                "QQQ",  "ProShares"),
    ("UPRO", "ProShares UltraPro S&P 500",            "SPY",  "ProShares"),
    ("UDOW", "ProShares UltraPro Dow30",              "DIA",  "ProShares"),
    ("TNA",  "Direxion Daily Small Cap Bull 3X",      "IWM",  "Direxion"),
    ("MIDU", "Direxion Daily Mid Cap Bull 3X",        "MDY",  "Direxion"),

    # Tech / Semis
    ("SOXL", "Direxion Daily Semiconductor Bull 3X",  "SOXX", "Direxion"),
    ("TECL", "Direxion Daily Technology Bull 3X",     "XLK",  "Direxion"),
    ("FNGU", "MicroSectors FANG+ 3X",                 "FNGS", "MicroSectors"),
    ("WEBL", "MicroSectors Solactive FANG+ Internet 3X", "", "MicroSectors"),

    # Other sectors
    ("FAS",  "Direxion Daily Financial Bull 3X",      "XLF",  "Direxion"),
    ("ERX",  "Direxion Daily Energy Bull 3X",         "XLE",  "Direxion"),
    ("LABU", "Direxion Daily S&P Biotech Bull 3X",    "XBI",  "Direxion"),
    ("NAIL", "Direxion Daily Homebuilders Bull 3X",   "ITB",  "Direxion"),
    ("RETL", "Direxion Daily Retail Bull 3X",         "XRT",  "Direxion"),
    ("DPST", "Direxion Daily Regional Banks Bull 3X", "KRE",  "Direxion"),

    # Commodity / metal-linked
    ("NUGT", "Direxion Daily Gold Miners Bull 3X",    "GDX",  "Direxion"),
    ("GUSH", "Direxion Daily Oil & Gas E&P Bull 3X",  "XOP",  "Direxion"),
    ("JNUG", "Direxion Daily Junior Gold Miners 3X",  "GDXJ", "Direxion"),

    # Treasuries
    ("TMF",  "Direxion Daily 20+ Year Treasury 3X",   "TLT",  "Direxion"),
    ("TYD",  "Direxion Daily 7-10 Year Treasury 3X",  "IEF",  "Direxion"),
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
            leverage=3,
        )
        for ticker, name, underlying, issuer in THREE_X_ETFS
    ]
