"""Empirical study: for tickers that had strong 5-day rallies in the past year,
what technical setups appeared in the days BEFORE the rally started?

Compares pre-rally hit rate vs. a random-baseline hit rate to estimate
predictive value of each setup.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"

RALLY_5D_THRESHOLD = 15.0   # 5-day cumulative >= +15% = "strong rally"
COOLDOWN_DAYS = 10           # don't double-count the same rally
PRE_LOOK = 1                 # check indicators 1 day before rally starts


def rsi_at(close: pd.Series, idx: int, period: int = 14) -> float | None:
    if idx < period: return None
    win = close.iloc[: idx + 1]
    delta = win.diff().dropna()
    if delta.size < period: return None
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    val = 100 - 100 / (1 + rs.iloc[-1])
    return None if pd.isna(val) else float(val)


def indicators_at(close: pd.Series, high: pd.Series, low: pd.Series, vol: pd.Series, idx: int) -> dict | None:
    """Compute pre-breakout setups using only data up to and including idx."""
    if idx < 60 or idx >= close.size: return None

    res: dict[str, bool] = {}
    win = close.iloc[: idx + 1]

    # Bollinger squeeze: current BB width < 50% of avg BB width over past 126 days
    last20 = win.iloc[-20:]
    bb_now = float((last20.std(ddof=0) * 2) / last20.mean()) if last20.mean() > 0 else float("nan")
    past = []
    for j in range(max(0, idx - 126), idx - 20):
        seg = close.iloc[j : j + 20]
        if seg.size == 20 and seg.mean() > 0:
            past.append(float((seg.std(ddof=0) * 2) / seg.mean()))
    if past and not np.isnan(bb_now):
        res["bb_squeeze"] = bb_now < np.mean(past) * 0.5

    # NR7 — today's range is the smallest of the last 7 sessions
    ranges = [float(high.iloc[idx - k] - low.iloc[idx - k]) for k in range(7) if idx - k >= 0]
    if len(ranges) == 7:
        res["nr7"] = ranges[0] == min(ranges)

    # MA convergence: spread of MA5/20/60 within 3% of mean
    ma5 = win.iloc[-5:].mean()
    ma20 = win.iloc[-20:].mean()
    ma60 = win.iloc[-60:].mean()
    if win.size >= 60:
        mean_ma = (ma5 + ma20 + ma60) / 3
        spread = (max(ma5, ma20, ma60) - min(ma5, ma20, ma60)) / mean_ma if mean_ma > 0 else 1
        res["ma_converge"] = spread < 0.03

    # Volume dry-up: 10-day avg < 70% of 60-day avg
    if vol.size > idx and idx >= 60:
        v10 = float(vol.iloc[idx - 9 : idx + 1].mean())
        v60 = float(vol.iloc[idx - 59 : idx + 1].mean())
        res["vol_dryup"] = v10 < v60 * 0.7 if v60 > 0 else False

    # Tight base: 20-day high/low range within 10% of mean
    if win.size >= 20:
        rng = (last20.max() - last20.min()) / last20.mean() if last20.mean() > 0 else 1
        res["tight_base"] = rng < 0.10

    # Bullish MA alignment
    if win.size >= 120:
        ma120 = win.iloc[-120:].mean()
        res["bull_align"] = ma5 > ma20 > ma60 > ma120

    # Bearish MA alignment
    if win.size >= 120:
        ma120 = win.iloc[-120:].mean()
        res["bear_align"] = ma5 < ma20 < ma60 < ma120

    # MA20 support: today's low touched MA20 from above
    if win.size >= 20:
        last_low = float(low.iloc[idx])
        last_close = float(close.iloc[idx])
        res["ma20_support"] = last_low <= ma20 * 1.02 and last_close > ma20

    # RSI 40-60 (neutral, room to run)
    rsi = rsi_at(close, idx)
    if rsi is not None:
        res["rsi_neutral"] = 40 <= rsi <= 60
        res["rsi_oversold"] = rsi < 35

    # 거래량 spike + 양봉 (institutional buying signal)
    if vol.size > idx and idx >= 20:
        v_today = float(vol.iloc[idx])
        v_avg20 = float(vol.iloc[idx - 19 : idx].mean())
        daily = float((close.iloc[idx] / close.iloc[idx - 1] - 1) * 100) if idx >= 1 else 0
        res["volup_candle"] = v_avg20 > 0 and v_today >= v_avg20 * 2 and daily >= 3

    return res


def find_rally_starts(close: pd.Series) -> list[int]:
    """Indices where a 5-day rally of >=+15% started. Cooldown to avoid duplicates."""
    if close.size < 10: return []
    starts: list[int] = []
    last_start = -10**9
    for i in range(5, close.size):
        ret5 = float((close.iloc[i] / close.iloc[i - 5] - 1) * 100)
        if ret5 >= RALLY_5D_THRESHOLD and (i - 5) - last_start > COOLDOWN_DAYS:
            starts.append(i - 5)  # rally start = 5 days before peak
            last_start = i - 5
    return starts


def main() -> int:
    prices = pd.read_pickle(DATA / "prices.pkl")
    etf_list = json.loads((DATA / "etf_list.json").read_text(encoding="utf-8"))
    underlyings = sorted({e["underlying"] for e in etf_list["etfs"] if e.get("underlying")})

    rallies: list[dict] = []
    skipped = 0

    for t in underlyings:
        try:
            df = prices[t]
        except KeyError:
            continue
        df = df.dropna(subset=["Close"])
        if df.shape[0] < 80:
            skipped += 1
            continue
        close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
        starts = find_rally_starts(close)
        for s in starts:
            pre_idx = s - PRE_LOOK
            ind = indicators_at(close, high, low, vol, pre_idx)
            if ind is None: continue
            peak_idx = min(s + 5, close.size - 1)
            rally_pct = float((close.iloc[peak_idx] / close.iloc[s] - 1) * 100)
            rallies.append({
                "ticker": t,
                "start_date": str(close.index[s].date()),
                "rally_5d_pct": rally_pct,
                "ind": ind,
            })

    # Baseline: same indicators on N random non-rally days (>=10 days from any rally)
    rally_idx_set: dict[str, set[int]] = {}
    for r in rallies:
        rally_idx_set.setdefault(r["ticker"], set())
    for t in underlyings:
        try:
            df = prices[t].dropna(subset=["Close"])
        except KeyError:
            continue
        starts = set(find_rally_starts(df["Close"]))
        rally_idx_set[t] = starts

    baseline: list[dict] = []
    target_baseline = max(len(rallies) * 5, 500)
    random.seed(42)
    underlyings_with_data = [t for t in underlyings if t in prices.columns.get_level_values(0)]
    while len(baseline) < target_baseline:
        t = random.choice(underlyings_with_data)
        try:
            df = prices[t].dropna(subset=["Close"])
        except KeyError:
            continue
        if df.shape[0] < 80: continue
        idx = random.randint(60, df.shape[0] - 6)
        if any(abs(idx - r) <= 10 for r in rally_idx_set.get(t, set())): continue
        ind = indicators_at(df["Close"], df["High"], df["Low"], df["Volume"], idx)
        if ind is None: continue
        baseline.append({"ticker": t, "ind": ind})

    # Aggregate
    def hit_rate(samples: list[dict]) -> dict[str, tuple[int, int]]:
        keys = set()
        for s in samples:
            keys.update(s["ind"].keys())
        out: dict[str, tuple[int, int]] = {}
        for k in keys:
            total = sum(1 for s in samples if k in s["ind"])
            hit = sum(1 for s in samples if s["ind"].get(k))
            out[k] = (hit, total)
        return out

    pre_rate = hit_rate(rallies)
    base_rate = hit_rate(baseline)

    print(f"\nAnalyzed {len(rallies)} rallies (5-day >= +{RALLY_5D_THRESHOLD}%) "
          f"vs {len(baseline)} random baseline days")
    print(f"Tickers with data: {len(underlyings_with_data)}  (skipped {skipped} short-history)\n")

    rows = []
    for key in sorted(set(pre_rate) | set(base_rate)):
        ph, pt = pre_rate.get(key, (0, 1))
        bh, bt = base_rate.get(key, (0, 1))
        pre_pct = ph / pt * 100 if pt else 0
        base_pct = bh / bt * 100 if bt else 0
        lift = pre_pct / base_pct if base_pct > 0 else float("inf")
        rows.append((key, pre_pct, base_pct, lift, ph, pt))
    rows.sort(key=lambda r: -r[3])

    print(f"{'INDICATOR':<18} {'PRE-RALLY':>10} {'BASELINE':>10} {'LIFT':>8} {'COUNT':>10}")
    print("-" * 65)
    for key, pre_pct, base_pct, lift, ph, pt in rows:
        lift_s = f"{lift:.2f}x" if lift != float("inf") else "  ∞"
        print(f"{key:<18} {pre_pct:>9.1f}% {base_pct:>9.1f}% {lift_s:>8} {ph:>4}/{pt:<5}")

    print("\nTop 15 rallies in our window:")
    rallies.sort(key=lambda r: -r["rally_5d_pct"])
    for r in rallies[:15]:
        active = ",".join(k for k, v in r["ind"].items() if v)
        print(f"  {r['ticker']:6s} {r['start_date']}  +{r['rally_5d_pct']:>5.1f}%  [{active}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
