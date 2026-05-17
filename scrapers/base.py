"""Shared HTTP fetch + ETF record helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional
import requests

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def http_get(url: str, *, timeout: int = 20, retries: int = 3, sleep: float = 1.5) -> str:
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(sleep * (i + 1))
    raise RuntimeError(f"GET {url} failed: {last_err}")


@dataclass
class ETF:
    ticker_2x: str
    name: str
    underlying: str
    issuer: str
    url: str = ""
    expense_ratio: Optional[float] = None  # percent, e.g. 1.31

    def as_dict(self) -> dict:
        return asdict(self)


def dedupe(records: list[ETF]) -> list[ETF]:
    seen: dict[str, ETF] = {}
    for r in records:
        if r.ticker_2x and r.ticker_2x not in seen:
            seen[r.ticker_2x] = r
    return list(seen.values())
