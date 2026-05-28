"""Backtest: find common signal patterns before 2X ETF 30%+ surges.

For each 2X ticker in our universe, scan history for days where the next
5 trading days had cumulative return ≥ 30%. Capture the signal state at
that "pre-surge day-zero" and compare frequency vs baseline (all days).

Lift = signal_freq_in_pre_surge / signal_freq_in_baseline
- Lift > 1.5 = signal appears 1.5x more often before surges than random
- Lift > 2.0 = strong predictive value
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# ============================================================
# 1. Load data
# ============================================================
prices = pd.read_pickle(DATA / "prices.pkl")
etf_list = json.loads((DATA / "etf_list.json").read_text(encoding="utf-8"))

pairs = []
for e in etf_list["etfs"]:
    t2 = (e.get("ticker_2x") or "").strip().upper()
    und = (e.get("underlying") or "").strip().upper()
    if t2 and und and t2 in prices.columns.get_level_values(0):
        pairs.append((t2, und))

print(f"Loaded {len(pairs)} pairs with price data")

# ============================================================
# 2. Signal computation functions (lightweight, per-row state)
# ============================================================
def compute_signals_at(close_und: pd.Series, close_2x: pd.Series, vol_2x: pd.Series, low_und: pd.Series, i: int) -> dict:
    """Compute signal state on day i (using underlying for MA/MACD, 2X for vol)."""
    sigs = {}
    cu = close_und.iloc[:i+1]
    if cu.size < 130:
        return None
    cu_arr = cu.values

    # MA alignment (daily, on underlying)
    ma5  = cu_arr[-5:].mean()  if len(cu_arr) >= 5  else np.nan
    ma20 = cu_arr[-20:].mean() if len(cu_arr) >= 20 else np.nan
    ma60 = cu_arr[-60:].mean() if len(cu_arr) >= 60 else np.nan
    ma120= cu_arr[-120:].mean()if len(cu_arr) >= 120 else np.nan
    last = cu_arr[-1]

    sigs["ma_bull_align"] = ma5 > ma20 > ma60 > ma120
    sigs["ma_bear_align"] = ma5 < ma20 < ma60 < ma120
    sigs["above_ma20"] = last > ma20
    sigs["above_ma60"] = last > ma60
    sigs["above_ma120"] = last > ma120

    # Near MA support (within 2%)
    low_today = low_und.iloc[i] if i < low_und.size else last
    sigs["ma20_support"] = low_today <= ma20 * 1.02 and last > ma20 if not np.isnan(ma20) else False
    sigs["ma60_support"] = low_today <= ma60 * 1.02 and last > ma60 if not np.isnan(ma60) else False
    sigs["ma120_support"] = low_today <= ma120 * 1.02 and last > ma120 if not np.isnan(ma120) else False

    # Bull pullback composite
    sigs["bull_align_pullback"] = sigs["ma_bull_align"] and (
        sigs["ma20_support"] or sigs["ma60_support"] or sigs["ma120_support"]
    )

    # RSI 14
    delta = np.diff(cu_arr[-15:])
    gains = np.where(delta > 0, delta, 0).mean()
    losses = np.where(delta < 0, -delta, 0).mean()
    rsi = 100 - 100 / (1 + gains / (losses + 1e-9))
    sigs["rsi"] = float(rsi)
    sigs["rsi_oversold"] = rsi < 30
    sigs["rsi_overbought"] = rsi > 70
    sigs["rsi_30_50"] = 30 <= rsi <= 50

    # MACD (12/26/9)
    cu_series = pd.Series(cu_arr)
    ema12 = cu_series.ewm(span=12, adjust=False).mean()
    ema26 = cu_series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    sigs["macd_above_zero"] = macd.iloc[-1] > 0
    # Recent bull cross (last 3 days)
    bull_cross = False
    for k in range(-3, 0):
        if k-1 < -len(macd): continue
        if macd.iloc[k-1] <= signal.iloc[k-1] and macd.iloc[k] > signal.iloc[k]:
            bull_cross = True; break
    sigs["macd_bull_cross"] = bull_cross
    # Histogram momentum
    if len(hist) >= 3:
        sigs["macd_momentum_up"] = hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3]
        sigs["macd_momentum_down"] = hist.iloc[-1] < hist.iloc[-2] < hist.iloc[-3]
    else:
        sigs["macd_momentum_up"] = False
        sigs["macd_momentum_down"] = False

    # Volume ratio (on 2X)
    v2x = vol_2x.iloc[:i+1].values
    if len(v2x) >= 21:
        avg20 = v2x[-21:-1].mean()
        sigs["volume_ratio"] = float(v2x[-1] / avg20) if avg20 > 0 else 1.0
        sigs["volume_spike_2x"] = sigs["volume_ratio"] >= 2.0
    else:
        sigs["volume_ratio"] = 1.0
        sigs["volume_spike_2x"] = False

    # Recent return regime (on 2X)
    c2x = close_2x.iloc[:i+1].values
    if len(c2x) >= 6:
        ret_5d = (c2x[-1] / c2x[-6] - 1) * 100
    else:
        ret_5d = 0
    sigs["recent_5d_return_2x"] = float(ret_5d)
    sigs["recent_drop_5d"] = ret_5d <= -10  # 1주 -10% 이상 떨어진 상태
    sigs["recent_drop_20d"] = (c2x[-1] / c2x[-21] - 1) * 100 <= -20 if len(c2x) >= 21 else False

    # Bollinger band breakdown
    if cu.size >= 20:
        bb_mean = cu_arr[-20:].mean()
        bb_std = cu_arr[-20:].std(ddof=0)
        bb_low = bb_mean - 2 * bb_std
        bb_hi  = bb_mean + 2 * bb_std
        sigs["bb_breakdown"] = last < bb_low
        sigs["bb_breakout"] = last > bb_hi
        sigs["bb_position"] = (last - bb_low) / (bb_hi - bb_low + 1e-9)  # 0~1 within band
    else:
        sigs["bb_breakdown"] = sigs["bb_breakout"] = False
        sigs["bb_position"] = 0.5

    # HV30 percentile (on underlying)
    log_ret = np.diff(np.log(cu_arr))
    if len(log_ret) >= 60:
        hv = pd.Series(log_ret).rolling(30).std() * np.sqrt(252) * 100
        hv_current = hv.iloc[-1]
        hv_year = hv.iloc[-252:] if len(hv) > 252 else hv.dropna()
        hv_pct = (hv_year <= hv_current).sum() / len(hv_year) * 100
        sigs["hv_compressed"] = hv_pct <= 20
        sigs["hv_expanded"] = hv_pct >= 80
        sigs["hv_pct"] = float(hv_pct)
    else:
        sigs["hv_compressed"] = sigs["hv_expanded"] = False
        sigs["hv_pct"] = 50

    # 52w state
    window = cu_arr[-min(252, len(cu_arr)):]
    yr_high = window.max()
    yr_low = window.min()
    sigs["near_52w_high"] = last >= yr_high * 0.95
    sigs["near_52w_low"] = last <= yr_low * 1.05
    sigs["pct_from_52w_high"] = (last / yr_high - 1) * 100

    return sigs


# ============================================================
# 3. Scan history for surges + collect signal states
# ============================================================
FORWARD_WIN = 5      # 5 trading days
SURGE_THRESH = 30.0  # 30% surge

pre_surge_states = []
baseline_states = []

print(f"\nScanning for {SURGE_THRESH}% surges within {FORWARD_WIN} trading days...")

for ticker_2x, ticker_und in pairs:
    try:
        close_2x = prices[(ticker_2x, "Close")].dropna()
        if und := ticker_und:
            if (und, "Close") not in prices.columns:
                continue
            close_und = prices[(und, "Close")].dropna()
            low_und = prices[(und, "Low")].dropna()
        else:
            continue
        vol_2x = prices[(ticker_2x, "Volume")].dropna()
    except Exception:
        continue

    if close_2x.size < 130 or close_und.size < 130:
        continue

    # Align indices
    idx = close_2x.index.intersection(close_und.index)
    if len(idx) < 130:
        continue
    close_2x = close_2x.reindex(idx).ffill()
    close_und = close_und.reindex(idx).ffill()
    low_und = low_und.reindex(idx).ffill()
    vol_2x = vol_2x.reindex(idx).ffill()

    surge_count_this_ticker = 0
    for i in range(120, len(close_2x) - FORWARD_WIN):
        # Forward return
        fwd = (close_2x.iloc[i + FORWARD_WIN] / close_2x.iloc[i] - 1) * 100
        is_surge = fwd >= SURGE_THRESH

        sigs = compute_signals_at(close_und, close_2x, vol_2x, low_und, i)
        if sigs is None:
            continue
        sigs["ticker_2x"] = ticker_2x
        sigs["forward_5d_return"] = float(fwd)

        if is_surge:
            pre_surge_states.append(sigs)
            surge_count_this_ticker += 1
        else:
            # Sample baseline (every 5th day to keep memory reasonable)
            if i % 5 == 0:
                baseline_states.append(sigs)

print(f"  Found {len(pre_surge_states)} pre-surge instances")
print(f"  Sampled {len(baseline_states)} baseline days")

# ============================================================
# 4. Compute lift for each binary signal
# ============================================================
binary_signals = [
    "ma_bull_align", "ma_bear_align", "above_ma20", "above_ma60", "above_ma120",
    "ma20_support", "ma60_support", "ma120_support", "bull_align_pullback",
    "rsi_oversold", "rsi_overbought", "rsi_30_50",
    "macd_above_zero", "macd_bull_cross", "macd_momentum_up", "macd_momentum_down",
    "volume_spike_2x", "recent_drop_5d", "recent_drop_20d",
    "bb_breakdown", "bb_breakout",
    "hv_compressed", "hv_expanded",
    "near_52w_high", "near_52w_low",
]

print(f"\n{'Signal':<25s} {'Pre-Surge':>10s} {'Baseline':>10s} {'Lift':>8s} {'평가':>10s}")
print("=" * 70)

results = []
for sig in binary_signals:
    n_ps = sum(1 for s in pre_surge_states if s.get(sig))
    n_bl = sum(1 for s in baseline_states if s.get(sig))
    pct_ps = n_ps / len(pre_surge_states) * 100 if pre_surge_states else 0
    pct_bl = n_bl / len(baseline_states) * 100 if baseline_states else 0
    lift = pct_ps / pct_bl if pct_bl > 0 else float("inf") if pct_ps > 0 else 0
    results.append((sig, pct_ps, pct_bl, lift))

# Sort by lift descending
results.sort(key=lambda x: -x[3])

for sig, pct_ps, pct_bl, lift in results:
    if lift > 2.0:
        verdict = "*** STRONG"
    elif lift > 1.5:
        verdict = "** GOOD"
    elif lift > 1.2:
        verdict = "* WEAK"
    elif lift < 0.7:
        verdict = "x AVOID"
    else:
        verdict = ""
    print(f"{sig:<25s} {pct_ps:>9.1f}% {pct_bl:>9.1f}% {lift:>7.2f}x  {verdict}")

# ============================================================
# 5. Continuous metric averages
# ============================================================
print(f"\n{'Continuous metric':<25s} {'Pre-Surge':>15s} {'Baseline':>15s}")
print("=" * 65)
for metric in ("rsi", "recent_5d_return_2x", "bb_position", "hv_pct", "pct_from_52w_high", "volume_ratio"):
    avg_ps = np.mean([s[metric] for s in pre_surge_states])
    avg_bl = np.mean([s[metric] for s in baseline_states])
    print(f"{metric:<25s} {avg_ps:>15.2f} {avg_bl:>15.2f}")

# ============================================================
# 6. Top combos
# ============================================================
print(f"\n=== Most common 2-signal combos in pre-surge days ===")
combo_counts_ps = defaultdict(int)
combo_counts_bl = defaultdict(int)
top_sigs = [r[0] for r in results[:15]]  # top 15 by lift

for s in pre_surge_states:
    active = [k for k in top_sigs if s.get(k)]
    for i in range(len(active)):
        for j in range(i+1, len(active)):
            combo_counts_ps[(active[i], active[j])] += 1
for s in baseline_states:
    active = [k for k in top_sigs if s.get(k)]
    for i in range(len(active)):
        for j in range(i+1, len(active)):
            combo_counts_bl[(active[i], active[j])] += 1

combo_lifts = []
for combo, n_ps in combo_counts_ps.items():
    if n_ps < 3: continue
    n_bl = combo_counts_bl.get(combo, 0)
    pct_ps = n_ps / len(pre_surge_states) * 100
    pct_bl = n_bl / len(baseline_states) * 100 if len(baseline_states) > 0 else 0
    lift = pct_ps / pct_bl if pct_bl > 0 else float("inf")
    combo_lifts.append((combo, n_ps, pct_ps, pct_bl, lift))

combo_lifts.sort(key=lambda x: -x[4] if x[4] != float("inf") else -999)
for combo, n_ps, pct_ps, pct_bl, lift in combo_lifts[:10]:
    print(f"  {combo[0]:<22s} + {combo[1]:<22s}  pre={pct_ps:5.1f}%  base={pct_bl:5.1f}%  lift={lift:.2f}x  (n={n_ps})")
