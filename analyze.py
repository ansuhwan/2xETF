"""Stage 3+4: compute 5 alert conditions + volatility drag, write data/data.json.

Reads:
  - data/etf_list.json   : pair definitions (2X ↔ underlying)
  - data/prices.pkl      : 1y daily OHLCV, MultiIndex columns (ticker, field)

Writes:
  - data/data.json       : ready for the static dashboard
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ETF_LIST = DATA / "etf_list.json"
PRICES_PKL = DATA / "prices.pkl"
MONTHLY_PKL = DATA / "monthly.pkl"
WEEKLY_PKL = DATA / "weekly.pkl"
EARNINGS_JSON = DATA / "earnings.json"
OPTIONS_JSON = DATA / "options.json"
FUNDAMENTALS_JSON = DATA / "fundamentals.json"
OUT = DATA / "data.json"

KST = timezone(timedelta(hours=9))

# Sector → benchmark ETF (for relative strength). One per sector.
SECTOR_BENCHMARK: dict[str, str] = {
    "Mag7":    "QQQ",
    "AI":      "ARKK",
    "Semi":    "SOXX",
    "Nuclear": "URA",
    "Defense": "ITA",
    "Bio":     "XBI",
    "Fintech": "XLF",
    "Quantum": "ARKK",
    "EV":      "LIT",
    "Crypto":  "ARKK",
}

SECTORS: dict[str, set[str]] = {
    "Mag7":    {"AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA"},
    "AI":      {"NVDA", "AAOI", "ALAB", "ANET", "CRWV", "NBIS", "SOUN", "BBAI",
                "IONQ", "RGTI", "QBTS", "PATH", "AI", "PLTR", "DELL", "CRWD",
                "NET", "SNOW", "DDOG", "APP", "VRT", "MPWR", "ARM", "TSM"},
    "Semi":    {"NVDA", "AMD", "AVGO", "SMCI", "ARM", "AMAT", "LRCX", "KLAC",
                "MU", "AMKR", "INTC", "TSM", "ASML", "MRVL", "ON", "SWKS",
                "MCHP", "MPWR", "QCOM", "NXPI", "ALAB", "AAOI", "AXTI"},
    "Nuclear": {"OKLO", "SMR", "LEU", "CCJ", "UEC", "DNN", "NNE", "BWXT",
                "VST", "ASPI"},
    "Defense": {"LMT", "NOC", "RTX", "GD", "ASTS", "AVAV", "RKLB", "AUR",
                "BWXT", "LDOS"},
    "Bio":     {"MRNA", "BNTX", "NVAX", "VRTX", "REGN", "LLY", "NVO", "AMGN",
                "GILD", "BMRN"},
    "Fintech": {"PYPL", "SOFI", "AFRM", "HOOD", "UPST", "COIN", "AXP"},
    "Quantum": {"IONQ", "RGTI", "QBTS"},
    "EV":      {"TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "BYDDY", "ALB",
                "LIT", "BMNR"},
    "Crypto":  {"COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BTBT", "BITF",
                "WULF", "CLSH"},
}

THRESH = {
    "big_drop": -10.0,
    "vol_mult": 2.0,
    "vol_drop": -5.0,
    "rsi_low": 30.0,
    "five_day": -20.0,
}


def rsi(close: pd.Series, period: int = 14) -> float | None:
    if close.size < period + 1:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    val = 100 - 100 / (1 + rs.iloc[-1])
    if pd.isna(val):
        return None
    return float(val)


def bb_lower(close: pd.Series, period: int = 20, k: float = 2.0) -> float | None:
    if close.size < period:
        return None
    tail = close.iloc[-period:]
    return float(tail.mean() - k * tail.std(ddof=0))


def sectors_of(underlying: str) -> list[str]:
    if not underlying:
        return []
    return [name for name, members in SECTORS.items() if underlying in members]


def hv_percentile(close: pd.Series, window: int = 30, lookback: int = 252) -> float | None:
    """30-day realized volatility percentile rank over 252-day lookback.

    Returns 0.0~100.0 — where current HV30 falls in the 1-year distribution.
    Low percentile = volatility is compressed = squeeze candidate setup.
    """
    if close.size < window + 10:
        return None
    log_ret = np.log(close / close.shift(1)).dropna()
    if log_ret.size < window:
        return None
    # Annualized rolling std (252 trading days)
    hv = log_ret.rolling(window).std() * np.sqrt(252) * 100
    hv = hv.dropna()
    if hv.size < 10:
        return None
    current = hv.iloc[-1]
    if pd.isna(current):
        return None
    series = hv.iloc[-lookback:] if hv.size > lookback else hv
    if series.size < 10:
        return None
    rank = (series <= current).sum() / series.size * 100
    return float(rank)


def relative_strength(close_u: pd.Series, close_b: pd.Series, period: int = 20) -> float | None:
    """RS = underlying N-day return − benchmark N-day return (in pct)."""
    if close_u.size < period + 1 or close_b.size < period + 1:
        return None
    u_ret = (close_u.iloc[-1] / close_u.iloc[-period - 1] - 1) * 100
    b_ret = (close_b.iloc[-1] / close_b.iloc[-period - 1] - 1) * 100
    if pd.isna(u_ret) or pd.isna(b_ret):
        return None
    return float(u_ret - b_ret)


def categorize(underlying: str) -> str:
    # Legacy primary category — kept for back-compat in the JSON
    sects = sectors_of(underlying)
    for pref in ("Mag7", "AI", "Semi", "Nuclear", "Crypto"):
        if pref in sects:
            return pref
    return "Other"


def r(x, n=2):
    if x is None:
        return None
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return round(float(x), n)


def spy_regime(prices: pd.DataFrame) -> str:
    """SPY 기반 시장 국면 분류. 백테스트(backtest_v2.py)와 동일 정의."""
    if "SPY" not in prices.columns.get_level_values(0):
        return "unknown"
    try:
        spy = prices[("SPY", "Close")].dropna()
    except Exception:
        return "unknown"
    if spy.size < 200:
        return "unknown"
    ma200 = float(spy.iloc[-200:].mean())
    high_6m = float(spy.iloc[-126:].max())
    last = float(spy.iloc[-1])
    if high_6m <= 0:
        return "unknown"
    dd = (last / high_6m - 1) * 100
    if dd <= -15: return "bear"
    if dd <= -7:  return "correction"
    if last > ma200 and dd > -5: return "bull"
    return "transition"


def has_ticker(prices: pd.DataFrame, t: str) -> bool:
    """Check if ticker has the essential Close column.

    yfinance occasionally returns partial data (no Close field) for delisted
    or thinly-traded tickers. Just checking level-0 isn't enough.
    """
    if t not in prices.columns.get_level_values(0):
        return False
    return (t, "Close") in prices.columns


def close_of(prices: pd.DataFrame, t: str) -> pd.Series:
    if (t, "Close") not in prices.columns:
        return pd.Series(dtype=float)
    return prices[(t, "Close")].dropna()


def volume_of(prices: pd.DataFrame, t: str) -> pd.Series:
    if (t, "Volume") not in prices.columns:
        return pd.Series(dtype=float)
    return prices[(t, "Volume")].dropna()


def low_of(prices: pd.DataFrame, t: str) -> pd.Series:
    if (t, "Low") not in prices.columns:
        return pd.Series(dtype=float)
    return prices[(t, "Low")].dropna()


def high_of(prices: pd.DataFrame, t: str) -> pd.Series:
    if (t, "High") not in prices.columns:
        return pd.Series(dtype=float)
    return prices[(t, "High")].dropna()


def open_of(prices: pd.DataFrame, t: str) -> pd.Series:
    if (t, "Open") not in prices.columns:
        return pd.Series(dtype=float)
    return prices[(t, "Open")].dropna()


def _zigzag_pivots(p, pct):
    """% 임계 지그재그. 반환 [(idx, price, 'H'/'L')] (마지막은 진행 중 잠정 극값)."""
    thr = pct / 100.0
    n = len(p)
    piv = []
    if n < 2:
        return piv
    trend = 0
    ext_i, ext = 0, p[0]
    for i in range(1, n):
        if trend == 0:
            if p[i] >= ext * (1 + thr):
                piv.append((0, p[0], 'L')); trend = 1; ext_i, ext = i, p[i]
            elif p[i] <= ext * (1 - thr):
                piv.append((0, p[0], 'H')); trend = -1; ext_i, ext = i, p[i]
            else:
                if p[i] > ext: ext_i, ext = i, p[i]
        elif trend == 1:
            if p[i] > ext: ext_i, ext = i, p[i]
            elif p[i] <= ext * (1 - thr):
                piv.append((ext_i, ext, 'H')); trend = -1; ext_i, ext = i, p[i]
        else:
            if p[i] < ext: ext_i, ext = i, p[i]
            elif p[i] >= ext * (1 + thr):
                piv.append((ext_i, ext, 'L')); trend = 1; ext_i, ext = i, p[i]
    piv.append((ext_i, ext, 'H' if trend == 1 else 'L'))
    return piv


def rising_lows_support(close: pd.Series, pct=10.0, min_lows=3,
                        win=200, near_lo=-3.0, near_hi=10.0, min_below_high=7.0):
    """상승 추세선(저점 절상) 지지 테스트 여부. 현재가가 추세선 근접하면 True.
    고점 추격 배제: 52주 고점 대비 -min_below_high% 이상 눌린 것만.
    스크리닝용 — 백테스트상 엣지는 약함(자동 매수 신호 아님)."""
    if close.size < 60:
        return False
    # 고점 달리는 종목 제외 (52주 고점 대비 충분히 눌렸어야)
    high_52w = float(close.iloc[-252:].max()) if close.size >= 252 else float(close.max())
    if high_52w > 0 and (close.iloc[-1] / high_52w - 1) * 100 > -min_below_high:
        return False
    p = close.values[-win:]
    piv = _zigzag_pivots(p, pct)
    lows = [(i, pr) for (i, pr, t) in piv[:-1] if t == 'L']
    if len(lows) < min_lows:
        return False
    run = [lows[-1]]
    for j in range(len(lows) - 2, -1, -1):
        if lows[j][1] < run[-1][1]:
            run.append(lows[j])
        else:
            break
    if len(run) < min_lows:
        return False
    run = run[::-1]
    xs = np.array([x for x, _ in run], float)
    ys = np.array([y for _, y in run], float)
    slope, intercept = np.polyfit(xs, ys, 1)
    if slope <= 0:
        return False
    line = slope * (len(p) - 1) + intercept
    if line <= 0:
        return False
    dist = (p[-1] / line - 1) * 100
    return near_lo <= dist <= near_hi


def compute_macd_signals(close: pd.Series) -> list[str]:
    """MACD-based signals (12/26/9) on close. Computed on underlying."""
    sigs: list[str] = []
    if close.size < 35:
        return sigs
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    if pd.isna(macd.iloc[-1]) or pd.isna(signal.iloc[-1]):
        return sigs

    # Zero-line direction (trend regime)
    if macd.iloc[-1] > 0:
        sigs.append("macd_above_zero")
    else:
        sigs.append("macd_below_zero")

    # Recent cross — within last 3 days
    if macd.size >= 4:
        for i in range(-3, 0):
            pd_prev = macd.iloc[i-1] - signal.iloc[i-1]
            pd_curr = macd.iloc[i]   - signal.iloc[i]
            if pd.isna(pd_prev) or pd.isna(pd_curr):
                continue
            if pd_prev <= 0 < pd_curr:
                sigs.append("macd_bull_cross"); break
            if pd_prev >= 0 > pd_curr:
                sigs.append("macd_bear_cross"); break

    # Histogram momentum (last 3 bars monotonic)
    if hist.size >= 4:
        recent = hist.iloc[-3:].diff().dropna()
        if recent.size == 2:
            if (recent > 0).all() and hist.iloc[-1] > hist.iloc[-3]:
                sigs.append("macd_momentum_up")
            elif (recent < 0).all() and hist.iloc[-1] < hist.iloc[-3]:
                sigs.append("macd_momentum_down")

    return sigs


def compute_ma_signals(close: pd.Series, low: pd.Series) -> list[str]:
    """MA-based technical signals from a (close, low) series."""
    sigs: list[str] = []
    if close.size < 5:
        return sigs

    sma = {p: close.rolling(p).mean() for p in (5, 20, 60, 120) if close.size >= p}
    latest = {p: float(s.iloc[-1]) for p, s in sma.items() if not pd.isna(s.iloc[-1])}
    last_close = float(close.iloc[-1])
    last_low = float(low.iloc[-1]) if low.size >= 1 else last_close

    # Alignment (정배열/역배열) — needs all four MAs
    if all(p in latest for p in (5, 20, 60, 120)):
        m5, m20, m60, m120 = latest[5], latest[20], latest[60], latest[120]
        if m5 > m20 > m60 > m120:
            sigs.append("ma_bull_align")
        elif m5 < m20 < m60 < m120:
            sigs.append("ma_bear_align")

    # MA support — today's low touched the MA (within 2%) but close held above
    if 20 in latest and last_low <= latest[20] * 1.02 and last_close > latest[20]:
        sigs.append("ma20_support")
    if 60 in latest and last_low <= latest[60] * 1.02 and last_close > latest[60]:
        sigs.append("ma60_support")
    if 120 in latest and last_low <= latest[120] * 1.02 and last_close > latest[120]:
        sigs.append("ma120_support")

    # Composite: bullish alignment + key MA support (pullback-buy setup)
    if "ma_bull_align" in sigs and (
        "ma20_support" in sigs or "ma60_support" in sigs or "ma120_support" in sigs
    ):
        sigs.append("bull_align_pullback")

    # Cross MA20 / MA60 within last 5 trading days
    if 20 in sma and 60 in sma and close.size >= 65:
        s20, s60 = sma[20], sma[60]
        window = min(5, close.size - 1)
        for i in range(-window, 0):
            v20p, v60p = s20.iloc[i - 1], s60.iloc[i - 1]
            v20c, v60c = s20.iloc[i], s60.iloc[i]
            if any(pd.isna(v) for v in (v20p, v60p, v20c, v60c)):
                continue
            if v20p <= v60p and v20c > v60c:
                sigs.append("golden_cross")
                break
            if v20p >= v60p and v20c < v60c:
                sigs.append("dead_cross")
                break

    return sigs


def analyze_pair(etf: dict, prices: pd.DataFrame, earnings: dict[str, dict], today: date, monthly: pd.DataFrame | None = None, options: dict[str, dict] | None = None, fundamentals: dict[str, dict] | None = None, weekly: pd.DataFrame | None = None) -> dict | None:
    t2 = (etf.get("ticker_2x") or "").strip().upper()
    und = (etf.get("underlying") or "").strip().upper()
    leverage = int(etf.get("leverage") or 2)
    if not t2 or not has_ticker(prices, t2):
        return None

    close2 = close_of(prices, t2)
    vol2 = volume_of(prices, t2)
    cu = close_of(prices, und) if und and has_ticker(prices, und) else pd.Series(dtype=float)

    price = float(close2.iloc[-1]) if close2.size else None
    daily = float((close2.iloc[-1] / close2.iloc[-2] - 1) * 100) if close2.size >= 2 else None

    daily_und = float((cu.iloc[-1] / cu.iloc[-2] - 1) * 100) if cu.size >= 2 else None

    # 5-day cumulative — prefer 2X, fallback to ~2× underlying for new ETFs
    five_day = None
    five_day_proxy = False
    if close2.size >= 6:
        five_day = float((close2.iloc[-1] / close2.iloc[-6] - 1) * 100)
    elif cu.size >= 6:
        five_day = float((cu.iloc[-1] / cu.iloc[-6] - 1) * 100) * leverage
        five_day_proxy = True

    # Volume ratio — only meaningful on the 2X itself
    vol_ratio = None
    if vol2.size >= 21:
        avg20 = float(vol2.iloc[-21:-1].mean())
        if avg20 > 0:
            vol_ratio = float(vol2.iloc[-1]) / avg20

    # RSI — fallback to underlying when 2X is too new
    rsi_val = rsi(close2)
    rsi_proxy = False
    if rsi_val is None and cu.size >= 15:
        rsi_val = rsi(cu)
        rsi_proxy = rsi_val is not None

    # BB breakdown — fallback to underlying when 2X is too new
    bb_signal = False
    bb_proxy = False
    if close2.size >= 20:
        bb_low = bb_lower(close2)
        bb_signal = bb_low is not None and price is not None and price < bb_low
    elif cu.size >= 20:
        bb_low = bb_lower(cu)
        bb_signal = bb_low is not None and float(cu.iloc[-1]) < bb_low
        bb_proxy = bb_signal

    # 6m drag — needs both legs to have 126 days; otherwise N/A
    drag = None
    und_6m_pct = None
    if cu.size >= 126 and close2.size >= 126:
        und_6m = float((cu.iloc[-1] / cu.iloc[-126] - 1) * 100)
        two_6m = float((close2.iloc[-1] / close2.iloc[-126] - 1) * 100)
        drag = (und_6m * leverage) - two_6m
        und_6m_pct = und_6m

    alerts: list[str] = []
    if daily is not None and daily <= THRESH["big_drop"]:
        alerts.append("big_drop")
    if (vol_ratio is not None and vol_ratio >= THRESH["vol_mult"]
            and daily is not None and daily <= THRESH["vol_drop"]):
        alerts.append("volume_spike")
    if rsi_val is not None and rsi_val < THRESH["rsi_low"]:
        alerts.append("oversold")
    if five_day is not None and five_day <= THRESH["five_day"]:
        alerts.append("five_day_crash")
    if bb_signal:
        alerts.append("bb_breakdown")

    # Scanner flags on the 2X itself
    near_52w_high = False
    near_52w_low = False
    if close2.size >= 20 and price is not None:
        window = close2.iloc[-min(252, close2.size):]
        yr_high = float(window.max())
        yr_low = float(window.min())
        if yr_high > 0 and price >= yr_high * 0.95:
            near_52w_high = True
        if yr_low > 0 and price <= yr_low * 1.05:
            near_52w_low = True
    volume_up_candle = (
        vol_ratio is not None and vol_ratio >= 2.0
        and daily is not None and daily >= 3.0
    )
    is_new = close2.size < 60

    # 거래대금 (dollar volume) — 20-day average of close × volume on the 2X
    dollar_volume_20d = None
    if close2.size >= 20 and vol2.size >= 20:
        n = min(close2.size, vol2.size, 20)
        dv = (close2.iloc[-n:] * vol2.iloc[-n:]).mean()
        if not pd.isna(dv) and dv > 0:
            dollar_volume_20d = float(dv)

    # MA + MACD signals — always computed on the underlying. 2X has daily-compounding
    # distortion that makes its own indicators unreliable for trend identification.
    if und and has_ticker(prices, und):
        ma_sigs = compute_ma_signals(cu, low_of(prices, und))
        ma_sigs.extend(compute_macd_signals(cu))
    else:
        ma_sigs = []
    monthly_sig = monthly_alignment(monthly, und) if und else None
    if monthly_sig:
        ma_sigs.append(monthly_sig)

    # 급락 + 2X 60일선 터치: 기초 당일 -5%↓ + 2X 당일 Low가 2X MA60에 걸침(Low≤MA60≤High)
    high2 = high_of(prices, t2)
    low2 = low_of(prices, t2)
    if (daily_und is not None and daily_und <= -5
            and close2.size >= 60 and high2.size and low2.size):
        ma60_2x = float(close2.iloc[-60:].mean())
        lo_today = float(low2.iloc[-1])
        hi_today = float(high2.iloc[-1])
        if lo_today <= ma60_2x <= hi_today:
            ma_sigs.append("drop5_2x_ma60_touch")

    # 2X 종가가 자기 60일선과 120일선 사이에 위치 (정배/역배 구분)
    if close2.size >= 120:
        ma60_2x = float(close2.iloc[-60:].mean())
        ma120_2x = float(close2.iloc[-120:].mean())
        px2 = float(close2.iloc[-1])
        lo_b, hi_b = min(ma60_2x, ma120_2x), max(ma60_2x, ma120_2x)
        if lo_b <= px2 <= hi_b:
            if ma60_2x > ma120_2x:
                ma_sigs.append("px_between_ma60_120_bull")   # 정배 — 상승추세 눌림
            else:
                ma_sigs.append("px_between_ma60_120_bear")   # 역배 — 회복 초기

    # 상승 추세선(저점 절상) 지지 테스트 — 스크리닝 필터 (백테 엣지 약함, 점수 X)
    if cu.size >= 60 and rising_lows_support(cu):
        ma_sigs.append("rising_lows_support")

    # 딥 눌림 반등 — 저점 필터 (백테 non-bull 85%/+19%): 기초 3개월 -20~-40% + 양봉
    o_und = open_of(prices, und) if und else pd.Series(dtype=float)
    if cu.size >= 64 and o_und.size:
        o_und = o_und.reindex(cu.index).ffill()
        max3 = float(cu.iloc[-64:].max())
        dd3 = (cu.iloc[-1] / max3 - 1) * 100 if max3 > 0 else 0.0
        is_green_und = float(cu.iloc[-1]) > float(o_und.iloc[-1])
        if -40 <= dd3 <= -20 and is_green_und:
            ma_sigs.append("deep_pullback_bounce")
            ret = cu.pct_change()
            hv20 = float(ret.iloc[-20:].std() * 100)
            hv60 = float(ret.iloc[-60:].std() * 100)
            if hv60 > 0 and hv20 / hv60 <= 0.7:   # 변동성 압축 동반 = 추가 확신
                ma_sigs.append("deep_pullback_compressed")

    # HLBO (저점절상반등): 저점 2회 절상 + 양봉 + 3m DD≤-10% + 종가>MA20. non-bull 80%/+24%
    lo_und = low_of(prices, und) if und else pd.Series(dtype=float)
    if cu.size >= 64 and o_und.size and lo_und.size >= 9:
        o_und2 = o_und.reindex(cu.index).ffill()
        lov = lo_und.reindex(cu.index).ffill().values
        l1 = lov[-9:-5].min(); l2v = lov[-5:-1].min(); l3 = lov[-2]; lo_today = lov[-1]
        hl = (l1 < l2v < lo_today) and (l2v < l3)
        ma20_d = float(cu.iloc[-20:].mean())
        max3h = float(cu.iloc[-64:].max())
        dd3h = (cu.iloc[-1] / max3h - 1) * 100 if max3h > 0 else 0.0
        green_h = float(cu.iloc[-1]) > float(o_und2.iloc[-1])
        if hl and green_h and dd3h <= -10 and float(cu.iloc[-1]) > ma20_d:
            ma_sigs.append("hlbo")

    # 월봉 MA60/120 정배열(단기선>장기선) 기반 눌림목 셋업 — 사용자 정의 필터
    mbull_6012 = (
        bull_60_120(monthly[und]) if (und and monthly is not None and und in monthly.columns) else None
    )
    if mbull_6012:
        # A: 일봉 종가가 120일선보다 5%+ 아래 + 월봉 정배열 (장기추세 살아있는 깊은 눌림)
        if cu.size >= 120:
            ma120_d = float(cu.iloc[-120:].mean())
            if ma120_d > 0 and float(cu.iloc[-1]) <= ma120_d * 0.95:
                ma_sigs.append("daily_dip_mbull")
        # B: 월봉 정배열 + 주봉 역배열 (MA60<MA120) — 장기 강세 속 중기 조정
        wk_state = (
            bull_60_120(weekly[und]) if (weekly is not None and und in weekly.columns) else None
        )
        if wk_state is False:
            ma_sigs.append("mbull_wbear")

    # Sector relative strength: underlying vs primary sector ETF (20-day return spread)
    rs_20d = None
    rs_sector = None
    primary = next((s for s in sectors_of(und)), None)
    if primary and primary in SECTOR_BENCHMARK:
        bench = SECTOR_BENCHMARK[primary]
        if has_ticker(prices, bench) and und and has_ticker(prices, und):
            rs_20d = relative_strength(cu, close_of(prices, bench), 20)
            rs_sector = bench
            if rs_20d is not None:
                if rs_20d >= 5: ma_sigs.append("sector_leader")
                elif rs_20d <= -5: ma_sigs.append("sector_laggard")

    # Earnings surprise consistency (4 quarters)
    e_info = earnings.get(und) if und else None
    if e_info:
        bs = e_info.get("beat_streak", 0) or 0
        ms = e_info.get("miss_streak", 0) or 0
        if bs >= 4: ma_sigs.append("earnings_beat_streak")
        elif bs >= 2: ma_sigs.append("earnings_beats")
        if ms >= 3: ma_sigs.append("earnings_miss_streak")

    # Options flow signals (front-month underlying)
    o_info = (options or {}).get(und) if und else None
    pc_vol = None
    atm_iv = None
    call_oi_growth_pct = None
    if o_info:
        pc_vol = o_info.get("pc_ratio_vol")
        atm_iv = o_info.get("atm_iv")
        if pc_vol is not None:
            if pc_vol >= 1.5: ma_sigs.append("put_heavy")     # 풋 매수 우세 (방어/약세)
            elif pc_vol <= 0.4: ma_sigs.append("call_heavy")  # 콜 매수 우세 (강세)
        if atm_iv is not None:
            if atm_iv >= 60: ma_sigs.append("iv_elevated")    # 큰 변동 기대 (이벤트 임박)
            elif atm_iv <= 20: ma_sigs.append("iv_crushed")   # 변동성 짓눌림

        # Squeeze setup: 콜 우세 + 낮은 IV (≤30%) = 변동성 압축 + 누적 매수
        # 강한 상승 직전 패턴 (싼 가격에 콜이 쌓이고 있다)
        # iv_crushed(≤20%)는 너무 엄격해서 별도 임계값 사용
        if "call_heavy" in ma_sigs and atm_iv is not None and atm_iv <= 30:
            ma_sigs.append("squeeze_setup")

        # Call OI day-over-day growth — 신규 콜 포지션 진입 (≥20% 급증)
        call_oi_now  = o_info.get("call_oi")
        call_oi_prev = o_info.get("call_oi_prev")
        if call_oi_now and call_oi_prev and call_oi_prev > 0:
            call_oi_growth_pct = (call_oi_now - call_oi_prev) / call_oi_prev * 100
            if call_oi_growth_pct >= 20:
                ma_sigs.append("call_oi_growth")

        # Call IV premium — 콜 IV > 풋 IV * 1.05 (정상은 풋이 더 높음, 역전 시 강세 시그널)
        # 5% 마진을 둬서 ATM 행사가 선택 노이즈 제거
        call_iv = o_info.get("atm_call_iv")
        put_iv  = o_info.get("atm_put_iv")
        if call_iv is not None and put_iv is not None and put_iv > 0 and call_iv >= put_iv * 1.05:
            ma_sigs.append("call_iv_premium")

        # Unusual call activity — 콜 거래량이 기존 OI의 2배 이상 (헤지펀드 진입 가능)
        call_vol_now = o_info.get("call_volume")
        if call_vol_now and call_oi_now and call_oi_now > 0:
            v_oi_ratio = call_vol_now / call_oi_now
            if v_oi_ratio >= 2.0:
                ma_sigs.append("unusual_call_activity")

    # HV percentile rank — 변동성 압축 감지 (옵션 IV 히스토리 대체)
    hv_pct = hv_percentile(cu) if cu.size >= 40 else None
    if hv_pct is not None:
        if hv_pct <= 20: ma_sigs.append("hv_compressed")   # 1년 중 하위 20% — 스퀴즈 후보
        elif hv_pct >= 80: ma_sigs.append("hv_expanded")    # 상위 20% — 변동성 폭발 중

    # Fundamentals: short interest + insider + analyst
    f_info = (fundamentals or {}).get(und) if und else None
    short_pct = None
    insider_net = None
    upgrades_30d = None
    pt_raises_30d = None
    if f_info:
        short_pct = f_info.get("short_pct_of_float")
        insider_net = f_info.get("insider_net_value_90d")
        upgrades_30d = f_info.get("upgrades_30d")
        downgrades_30d = f_info.get("downgrades_30d") or 0
        pt_raises_30d = f_info.get("pt_raises_30d")
        pt_lowers_30d = f_info.get("pt_lowers_30d") or 0

        if short_pct is not None:
            if short_pct >= 15: ma_sigs.append("high_short_interest")
            # 숏 스퀴즈 후보: 숏 비중 높음 + 강세 셋업 동반
            if short_pct >= 20 and ("ma_bull_align" in ma_sigs or "call_heavy" in ma_sigs):
                ma_sigs.append("short_squeeze_setup")

        if insider_net is not None:
            if insider_net >= 1_000_000:
                ma_sigs.append("insider_buying_strong")  # $1M+ 순매수
            elif insider_net > 0:
                ma_sigs.append("insider_buying")          # 순매수 (작아도 의미)

        # 애널리스트: 업그레이드 2+ 또는 목표가 상향 5+ (다운 < 업)
        upgrade_pos = (upgrades_30d or 0) >= 2 and (upgrades_30d or 0) > downgrades_30d
        pt_pos = (pt_raises_30d or 0) >= 5 and (pt_raises_30d or 0) > pt_lowers_30d
        if upgrade_pos or pt_pos:
            ma_sigs.append("analyst_upgrades")

        # === 성장성 + 밸류에이션 시그널 ===
        rev_g  = f_info.get("revenue_growth_yoy")
        eps_g  = f_info.get("earnings_growth_yoy")
        opm    = f_info.get("operating_margins")
        pm     = f_info.get("profit_margins")
        fpe    = f_info.get("forward_pe")
        tpe    = f_info.get("trailing_pe")
        peg    = f_info.get("peg_ratio")
        roe    = f_info.get("roe")

        if rev_g is not None and rev_g >= 20:
            ma_sigs.append("strong_revenue_growth")  # 매출 +20%↑ (NVO 같은 안정형 포섭)
        elif rev_g is not None and rev_g <= -10:
            ma_sigs.append("revenue_decline")

        if eps_g is not None and eps_g >= 30:
            ma_sigs.append("strong_eps_growth")  # EPS +30%↑

        if opm is not None and opm >= 25:
            ma_sigs.append("high_margins")  # 영업이익률 25%↑

        if pm is not None and pm < 0:
            ma_sigs.append("loss_making")  # 적자

        # 저평가 성장주: PER 합리적 + 매출 두 자릿수 성장 + 흑자
        pe_eff = fpe if fpe is not None and fpe > 0 else tpe
        if (pe_eff is not None and 0 < pe_eff <= 18
            and rev_g is not None and rev_g >= 15
            and pm is not None and pm > 0):
            ma_sigs.append("value_growth")

        # PEG 1.5 이하 + EPS 성장 양수 = PEG 가치주
        if peg is not None and 0 < peg <= 1.5 and eps_g is not None and eps_g > 0:
            ma_sigs.append("peg_value")

        # ROE 20%↑ + 흑자 = 우량
        if roe is not None and roe >= 20 and pm is not None and pm > 0:
            ma_sigs.append("high_roe")

    # Monthly bull alignment + daily price near long-term MA (MA60 or MA120)
    # = long-term uptrend in pullback to key support
    if monthly_sig == "monthly_bull_align" and cu.size >= 60:
        last = float(cu.iloc[-1])
        ma60d  = float(cu.iloc[-60:].mean()) if cu.size >= 60 else None
        ma120d = float(cu.iloc[-120:].mean()) if cu.size >= 120 else None
        near60  = ma60d  is not None and ma60d  > 0 and abs(last - ma60d)  / ma60d  < 0.05
        near120 = ma120d is not None and ma120d > 0 and abs(last - ma120d) / ma120d < 0.05
        if near60 or near120:
            ma_sigs.append("monthly_bull_near_long")

    # Falling Knife Setup — 백테스트 결과 가장 강력한 시그널
    # 6개월 -40% 이상 drawdown + 5일 추가 -20% 하락 + MA20 위 유지
    # → 20일 내 30%+ 폭등 도달 확률 70.6%, 50%+ 64.7%, 60%+ 47.1% (n=17)
    # 1년에 약 43회 발생 (보통 시장 조정 시 집중)
    dd_6m_und = None
    if cu.size >= 126:
        und_6m_high = float(cu.iloc[-126:].max())
        if und_6m_high > 0:
            dd_6m_und = (float(cu.iloc[-1]) / und_6m_high - 1) * 100
    if dd_6m_und is not None and dd_6m_und <= -40 and close2.size >= 6:
        ret_5d_2x_now = float((close2.iloc[-1] / close2.iloc[-6] - 1) * 100)
        if ret_5d_2x_now <= -20 and cu.size >= 20:
            ma20_und = float(cu.iloc[-20:].mean())
            if float(cu.iloc[-1]) > ma20_und:
                ma_sigs.append("falling_knife_setup")

    # === 트리거 A/B/C — 3년 백테스트로 검증된 매수 트리거 ===
    # 트리거 A: Silent FK (조용한 폭락 코일링) — 30일 내 30%+ 확률 81.8% (n=11)
    #   6m 본주 -40% + 5일 2X -20% + 본주 MA20 위 + 본주 거래량 평소 수준 (<1.5x)
    # 트리거 B: 3일 +30% 거래량 증가 — 30일 30%+ 확률 60.0% (n=200)
    #   2X 3일 +30%↑ + 3일 평균 거래량 ≥1.5배
    # 트리거 C: Strong Runner Quiet — 30일 30%+ 확률 54.9% (n=583)
    #   2X 5일 +30%↑ + 5일 평균 거래량 정상 (<1.5x) + 본주 MA20 위
    if close2.size >= 6 and cu.size >= 20:
        # 본주 거래량 비율 (당일 + 5일 평균 vs 20일 평균)
        vol_und = None
        try:
            vol_und = prices[(und, "Volume")].dropna() if und and has_ticker(prices, und) else None
        except Exception:
            vol_und = None

        ma20_und_t = float(cu.iloc[-20:].mean())
        last_u = float(cu.iloc[-1])
        last_2x = float(close2.iloc[-1])
        ret_5d_2x_t = float((last_2x / close2.iloc[-6] - 1) * 100)
        ret_3d_2x_t = float((last_2x / close2.iloc[-4] - 1) * 100) if close2.size >= 4 else None

        # Underlying volume ratios
        vu_ratio_today = None
        vu_5d_ratio = None
        if vol_und is not None and vol_und.size >= 20:
            avg_vol20 = float(vol_und.iloc[-21:-1].mean())
            if avg_vol20 > 0:
                vu_ratio_today = float(vol_und.iloc[-1]) / avg_vol20
                vu_5d_ratio = float(vol_und.iloc[-5:].mean()) / avg_vol20

        # 2X volume ratios (for trigger B/C)
        v2_3d_ratio = None
        v2_5d_ratio = None
        if vol2.size >= 20:
            avg_v2_20 = float(vol2.iloc[-21:-1].mean())
            if avg_v2_20 > 0:
                if vol2.size >= 3:
                    v2_3d_ratio = float(vol2.iloc[-3:].mean()) / avg_v2_20
                v2_5d_ratio = float(vol2.iloc[-5:].mean()) / avg_v2_20

        # Trigger S: Silent FK Strict (가장 극단, 가장 높은 확률)
        # Silent FK + 6m -50% 깊은 폭락 → 30일 30%+ 확률 87.5% (n=8, CI [53%, 98%])
        if (dd_6m_und is not None and dd_6m_und <= -50
            and ret_5d_2x_t <= -20
            and last_u > ma20_und_t
            and vu_ratio_today is not None and vu_ratio_today < 1.5):
            ma_sigs.append("trigger_s_extreme_fk")

        # Trigger A: Silent FK (가장 강력 — 표본 큼)
        if (dd_6m_und is not None and dd_6m_und <= -40
            and ret_5d_2x_t <= -20
            and last_u > ma20_und_t
            and vu_ratio_today is not None and vu_ratio_today < 1.5):
            ma_sigs.append("trigger_a_silent_fk")

        # Trigger B: 3일 +30% + 거래량 증가 (자주, 높은 확률)
        if (ret_3d_2x_t is not None and ret_3d_2x_t >= 30
            and v2_3d_ratio is not None and v2_3d_ratio >= 1.5):
            ma_sigs.append("trigger_b_3d_momentum")

        # Trigger C: Strong Runner Quiet (안정 모멘텀, 큰 표본)
        if (ret_5d_2x_t >= 30
            and v2_5d_ratio is not None and v2_5d_ratio < 1.5
            and last_u > ma20_und_t):
            ma_sigs.append("trigger_c_runner_quiet")

        # === 저점 매수 트리거 D/E/F (3개월 drawdown 기반) ===
        # 본주 3개월 drawdown 계산
        dd_3m_und = None
        if cu.size >= 63:
            und_3m_high = float(cu.iloc[-63:].max())
            if und_3m_high > 0:
                dd_3m_und = (last_u / und_3m_high - 1) * 100

        # 본주 정배열
        bull_align_und = False
        if cu.size >= 120:
            ma5_und  = float(cu.iloc[-5:].mean())
            ma60_und = float(cu.iloc[-60:].mean())
            ma120_und= float(cu.iloc[-120:].mean())
            bull_align_und = ma5_und > ma20_und_t > ma60_und > ma120_und

        # Trigger D: Pullback Pro — 정배열 + 3m -15~25% + MA20 위 + 거래량 정상
        # 30일 30%+ 도달 확률 81.5% (n=27, CI [63%, 92%])
        if (bull_align_und
            and dd_3m_und is not None and -25 <= dd_3m_und <= -15
            and last_u > ma20_und_t
            and vu_ratio_today is not None and vu_ratio_today < 1.5):
            ma_sigs.append("trigger_d_pullback_pro")

        # Trigger E: Pullback Standard — 3m -20~25% + MA20 위 + 거래량 정상
        # 30일 30%+ 확률 55.5% (n=353, CI [50%, 61%]) — 큰 표본
        if (dd_3m_und is not None and -25 <= dd_3m_und <= -20
            and last_u > ma20_und_t
            and vu_ratio_today is not None and vu_ratio_today < 1.5):
            ma_sigs.append("trigger_e_pullback_std")

        # Trigger F: Pullback Recovery — 3m -20~30% + 어제·오늘 양봉 + MA20 위
        # 30일 30%+ 확률 50.7% (n=229, CI [44%, 57%])
        if (dd_3m_und is not None and -30 <= dd_3m_und <= -20
            and last_u > ma20_und_t
            and cu.size >= 2 and close_und.size if 'close_und' in dir() else False):
            pass
        # Recovery 패턴: 어제 종가 > 어제 시가 + 오늘 종가 > 오늘 시가
        open_u_series = None
        if dd_3m_und is not None and und and has_ticker(prices, und):
            try:
                open_u_series = prices[(und, "Open")].dropna()
                if open_u_series.size >= 2 and -30 <= dd_3m_und <= -20 and last_u > ma20_und_t:
                    yest_green = float(cu.iloc[-2]) > float(open_u_series.iloc[-2])
                    today_green = float(cu.iloc[-1]) > float(open_u_series.iloc[-1])
                    if yest_green and today_green:
                        ma_sigs.append("trigger_f_pullback_recovery")
            except Exception:
                open_u_series = None

        # === 신규 트리거 G~P (3년 백테스트 검증) ===
        # Common context: RSI, MACD, 6m/4m/2m drawdowns, MA60 reclaim
        rsi_und = None
        if cu.size >= 15:
            delta_und = cu.diff().iloc[-14:].values
            gains_u = np.where(delta_und > 0, delta_und, 0).mean()
            losses_u = np.where(delta_und < 0, -delta_und, 0).mean()
            rsi_und = 100 - 100 / (1 + gains_u / (losses_u + 1e-9))

        # 6m / 4m / 2m drawdowns on underlying
        dd_6m_real = None
        if cu.size >= 126:
            dd_6m_real = (float(cu.iloc[-1]) / float(cu.iloc[-126:].max()) - 1) * 100
        dd_4m = None
        if cu.size >= 84:
            dd_4m = (float(cu.iloc[-1]) / float(cu.iloc[-84:].max()) - 1) * 100
        dd_2m = None
        if cu.size >= 42:
            dd_2m = (float(cu.iloc[-1]) / float(cu.iloc[-42:].max()) - 1) * 100

        # MA60 reclaim (yesterday below, today above)
        ma60_reclaim = False
        if cu.size >= 60:
            ma60_und = float(cu.iloc[-60:].mean())
            if cu.size >= 6:
                recently_below = any(float(cu.iloc[-k]) < ma60_und for k in range(1, 6))
                ma60_reclaim = recently_below and last_u > ma60_und

        # MACD bull cross (last 3 days)
        macd_bull_cross_recent = False
        if cu.size >= 35:
            cu_series_m = pd.Series(cu.values)
            ema12_m = cu_series_m.ewm(span=12, adjust=False).mean()
            ema26_m = cu_series_m.ewm(span=26, adjust=False).mean()
            macd_m = ema12_m - ema26_m
            sig_m = macd_m.ewm(span=9, adjust=False).mean()
            for k in range(-3, 0):
                if k-1 >= -len(macd_m) and macd_m.iloc[k-1] <= sig_m.iloc[k-1] and macd_m.iloc[k] > sig_m.iloc[k]:
                    macd_bull_cross_recent = True; break

        # Days since 6m high
        days_since_high = 126
        if cu.size >= 126:
            high_6m = float(cu.iloc[-126:].max())
            for k in range(126):
                if k < cu.size and float(cu.iloc[-1-k]) >= high_6m * 0.99:
                    days_since_high = k; break

        # Bullish engulfing (today vs yesterday)
        bullish_engulfing = False
        if open_u_series is not None and open_u_series.size >= 2:
            try:
                yest_c = float(cu.iloc[-2]); yest_o = float(open_u_series.iloc[-2])
                today_c = float(cu.iloc[-1]); today_o = float(open_u_series.iloc[-1])
                bullish_engulfing = (
                    yest_c < yest_o and today_c > today_o and
                    today_o <= yest_c and today_c >= yest_o
                )
            except Exception:
                pass

        # 5d return on 2X (for K, L)
        ret_3d_2x_now = None
        if close2.size >= 4:
            ret_3d_2x_now = (float(close2.iloc[-1]) / float(close2.iloc[-4]) - 1) * 100

        # Trigger G — RSI Bottom (76%, n=25)
        if (rsi_und is not None and 30 <= rsi_und <= 45
            and dd_3m_und is not None and -30 <= dd_3m_und <= -20
            and last_u > ma20_und_t):
            ma_sigs.append("trigger_g_rsi_bottom")

        # Trigger H — Failed Breakdown (55%, n=146)
        if (ma60_reclaim and dd_3m_und is not None and dd_3m_und <= -15
            and open_u_series is not None and open_u_series.size >= 1):
            try:
                today_o = float(open_u_series.iloc[-1])
                if float(cu.iloc[-1]) > today_o:
                    ma_sigs.append("trigger_h_failed_breakdown")
            except Exception:
                pass

        # Trigger I — MACD Bottom (49%, n=238)
        if (macd_bull_cross_recent
            and dd_3m_und is not None and -30 <= dd_3m_und <= -20):
            ma_sigs.append("trigger_i_macd_bottom")

        # Trigger J — Deep RSI Pullback (45%, n=666) — 가장 큰 표본
        if (rsi_und is not None and 30 <= rsi_und <= 45
            and dd_6m_real is not None and -40 <= dd_6m_real <= -25
            and vu_ratio_today is not None and vu_ratio_today < 1.5):
            ma_sigs.append("trigger_j_deep_rsi")

        # Trigger K — Soft Recovery (45%, n=289)
        if (ret_3d_2x_now is not None and 3 <= ret_3d_2x_now <= 10
            and dd_3m_und is not None and -30 <= dd_3m_und <= -15
            and last_u > ma20_und_t):
            ma_sigs.append("trigger_k_soft_recovery")

        # Trigger L — Steady Recovery (44%, n=248)
        if (5 <= ret_5d_2x_t <= 15
            and dd_3m_und is not None and -30 <= dd_3m_und <= -20
            and v2_5d_ratio is not None and v2_5d_ratio < 1.5):
            ma_sigs.append("trigger_l_steady_recovery")

        # Trigger M — 2-Month Pullback (59%, n=34)
        if (dd_2m is not None and -25 <= dd_2m <= -15
            and bull_align_und
            and vu_ratio_today is not None and vu_ratio_today < 1.5):
            ma_sigs.append("trigger_m_2m_pullback")

        # Trigger N — Stale Drawdown (47%, n=294)
        if (40 <= days_since_high <= 80
            and dd_6m_real is not None and -35 <= dd_6m_real <= -20
            and last_u > ma20_und_t):
            ma_sigs.append("trigger_n_stale_drawdown")

        # Trigger O — 4-Month Bottom (42.5%, n=492)
        if (dd_4m is not None and -30 <= dd_4m <= -20
            and last_u > ma20_und_t
            and vu_ratio_today is not None and vu_ratio_today < 1.5):
            ma_sigs.append("trigger_o_4m_bottom")

        # Trigger P — Bullish Engulfing (59%, n=22)
        if (bullish_engulfing
            and dd_3m_und is not None and -30 <= dd_3m_und <= -20
            and last_u > ma20_und_t):
            ma_sigs.append("trigger_p_engulfing")

        # Trigger Q — Gap Hold Recovery (64.7%, n=34, 갭-8% 완화)
        # 본주 갭 ≤-8% → 5거래일 ±7% 횡보 → 오늘 종가 갭종가 위
        # 백테스트: 승률 64.7%, 평균 +10.41%, 손절 32.4% (baseline +9.90%p)
        # 인덱싱: 갭일=iloc[-6] (5거래일 전), 횡보=iloc[-5:](오늘 포함)
        if open_u_series is not None and open_u_series.size >= 7 and cu.size >= 7:
            try:
                gap_pos = -6  # 5거래일 전
                prev_close = float(cu.iloc[gap_pos - 1])
                gap_open = float(open_u_series.iloc[gap_pos])
                gap_close = float(cu.iloc[gap_pos])
                if prev_close > 0:
                    gap_pct = (gap_open / prev_close - 1) * 100
                    if gap_pct <= -8:
                        holds = cu.iloc[gap_pos + 1:]  # 갭 다음날 ~ 오늘 (5일)
                        if holds.size == 5:
                            rng_max = (float(holds.max()) / gap_close - 1) * 100
                            rng_min = (float(holds.min()) / gap_close - 1) * 100
                            if (rng_max <= 7 and rng_min >= -7
                                and last_u >= gap_close):
                                ma_sigs.append("trigger_q_gap_hold_recovery")

        # Trigger V — Gap Hold Recovery (Wide, 61.8%, n=55)
        # 갭-8% 완화 + 5일 ±10% 넓은 횡보 — 더 자주 발동
        # 백테스트: 승률 61.8%, 평균 +10.05% (baseline +9.54%p)
                    if gap_pct <= -8:
                        holds_wide = cu.iloc[gap_pos + 1:]
                        if holds_wide.size == 5:
                            rng_max_w = (float(holds_wide.max()) / gap_close - 1) * 100
                            rng_min_w = (float(holds_wide.min()) / gap_close - 1) * 100
                            if (rng_max_w <= 10 and rng_min_w >= -10
                                and last_u >= gap_close
                                and "trigger_q_gap_hold_recovery" not in ma_sigs):
                                ma_sigs.append("trigger_v_gap_hold_wide")
            except Exception:
                pass

        # Trigger R — Monthly MA60 Support + Daily Bear Align (48.6%, n=706)
        # 일봉 역배(MA5<MA20<MA60<MA120) + 월봉 MA60 ±10% 영역 + 종가 위
        # 백테스트: 승률 48.6%, 평균 +5.19% (baseline +0.51% 대비 +4.68%p)
        # 의미: 단기 조정 깊지만 5년 추세선이 지지하는 가치 매수 진입점
        if "ma_bear_align" in ma_sigs and monthly is not None and und in monthly.columns:
            try:
                mu = monthly[und].dropna()
                if mu.size >= 60:
                    ma60_monthly = float(mu.iloc[-60:].mean())
                    if ma60_monthly > 0:
                        proximity = abs(last_u / ma60_monthly - 1) * 100
                        if proximity <= 10 and last_u >= ma60_monthly:
                            ma_sigs.append("trigger_r_monthly_ma60_support")
            except Exception:
                pass

        # Trigger U — Institutional Accumulation (54.8%, n=42)
        # 본주 5일 거래량 평균이 20일 평균의 1.5배+ + 5일 가격 범위 ±3% + RSI 40~55
        # 백테스트: 승률 54.8%, 평균 +5.50%, 손절 14.3% (baseline +4.99%p alpha)
        # 의미: 큰 손이 가격 안 올리고 흡수 중 — 가격 폭발 전 조용한 매집
        if und and has_ticker(prices, und):
            try:
                vu_full = volume_of(prices, und)
                if vu_full.size >= 25 and cu.size >= 5:
                    avg5_vol = float(vu_full.iloc[-5:].mean())
                    avg20_vol = float(vu_full.iloc[-25:-5].mean())
                    if avg20_vol > 0 and (avg5_vol / avg20_vol) >= 1.5:
                        recent5 = cu.iloc[-5:]
                        if recent5.min() > 0:
                            pct_range = (float(recent5.max()) / float(recent5.min()) - 1) * 100
                            if (pct_range <= 3.0
                                and rsi_und is not None and 40 <= rsi_und <= 55):
                                ma_sigs.append("trigger_u_institutional_accum")
            except Exception:
                pass

        # Trigger T — Monthly Bull + (MA60 OR MA120) Support (47.4%, n=2732)
        # 월봉 MA20>MA60>MA120 정배 + 최근 20일 일봉 저가 MA60 OR MA120 ±10% 영역 + 종가 위
        # 백테스트: 승률 47.4%, 평균 +3.49% (baseline +0.51% 대비 +2.98%p)
        # 의미: 월봉 장기 추세 유지 + 일봉 일시 조정 후 지지선 반등
        if monthly is not None and und in monthly.columns:
            try:
                mu = monthly[und].dropna()
                if mu.size >= 120:
                    ma20m = float(mu.iloc[-20:].mean())
                    ma60m = float(mu.iloc[-60:].mean())
                    ma120m = float(mu.iloc[-120:].mean())
                    if ma20m > ma60m > ma120m:
                        lows_recent = low_of(prices, und).iloc[-20:]
                        if lows_recent.size > 0:
                            min_low = float(lows_recent.min())
                            in_ma60 = (ma60m * 0.9 <= min_low <= ma60m * 1.05) and last_u >= ma60m
                            in_ma120 = (ma120m * 0.9 <= min_low <= ma120m * 1.05) and last_u >= ma120m
                            if in_ma60 or in_ma120:
                                ma_sigs.append("trigger_t_monthly_bull_support")
            except Exception:
                pass

    proxies: list[str] = []
    if rsi_proxy:
        proxies.append("rsi")
    if bb_proxy:
        proxies.append("bb")
    if five_day_proxy:
        proxies.append("five_day")

    # 필터 보너스용 피처 (트리거 multiplier 계산에 사용)
    ma20_dist_und = None
    if und and has_ticker(prices, und) and cu.size >= 20:
        _ma20 = float(cu.iloc[-20:].mean())
        if _ma20 > 0:
            ma20_dist_und = (float(cu.iloc[-1]) / _ma20 - 1) * 100
    vol_2x_std = None
    if close2.size >= 21:
        _rets = close2.iloc[-21:].pct_change().dropna()
        if _rets.size >= 15:
            vol_2x_std = float(_rets.std() * 100)

    return {
        "ticker_2x": t2,
        "ticker_underlying": und,
        "issuer": etf.get("issuer"),
        "expense_ratio": etf.get("expense_ratio"),
        "leverage": leverage,
        "category": categorize(und),
        "price_2x": r(price),
        "daily_pct": r(daily),
        "daily_pct_underlying": r(daily_und),
        "five_day_pct": r(five_day),
        "volume_ratio": r(vol_ratio),
        "rsi": r(rsi_val, 1),
        "drag_6m": r(drag),
        "underlying_6m_pct": r(und_6m_pct),
        "history_days_2x": int(close2.size),
        "proxy_fields": proxies,
        "alerts": alerts,
        "signals": ma_sigs,
        "sectors": sectors_of(und),
        "near_52w_high": near_52w_high,
        "near_52w_low": near_52w_low,
        "volume_up_candle": volume_up_candle,
        "is_new": is_new,
        "dollar_volume_20d": r(dollar_volume_20d, 0) if dollar_volume_20d is not None else None,
        "rs_20d": r(rs_20d),
        "rs_benchmark": rs_sector,
        "pc_ratio_vol": pc_vol,
        "atm_iv": atm_iv,
        "call_oi_growth_pct": r(call_oi_growth_pct, 1) if call_oi_growth_pct is not None else None,
        "hv_pct_rank": r(hv_pct, 0) if hv_pct is not None else None,
        "dd_6m_und": r(dd_6m_und, 1) if dd_6m_und is not None else None,
        "ma20_dist_und": r(ma20_dist_und, 2) if ma20_dist_und is not None else None,
        "vol_2x_std": r(vol_2x_std, 2) if vol_2x_std is not None else None,
        "short_pct_of_float": short_pct,
        "insider_net_value_90d": insider_net,
        "revenue_growth_yoy":  (f_info or {}).get("revenue_growth_yoy") if f_info else None,
        "earnings_growth_yoy": (f_info or {}).get("earnings_growth_yoy") if f_info else None,
        "forward_pe":          (f_info or {}).get("forward_pe") if f_info else None,
        "trailing_pe":         (f_info or {}).get("trailing_pe") if f_info else None,
        "peg_ratio":           (f_info or {}).get("peg_ratio") if f_info else None,
        "operating_margins":   (f_info or {}).get("operating_margins") if f_info else None,
        "profit_margins":      (f_info or {}).get("profit_margins") if f_info else None,
        "roe":                 (f_info or {}).get("roe") if f_info else None,
        "upgrades_30d": upgrades_30d,
        "pt_raises_30d": pt_raises_30d,
        **_earnings_fields(earnings.get(und) if und else None, today),
    }


# === 트리거 메타 ===
# backtest_all_triggers.py 검증 결과 기반.
# base_score = 베이스 점수
# kind = "pullback" (regime≠bull 선호), "momentum" (bull+정배열 선호), "disabled" (100% SL → 거의 무효)
# label = 사용자 표시 이유
TRIGGER_META: dict[str, dict] = {
    "trigger_s_extreme_fk":       {"base": 7.0, "kind": "disabled", "label": "트리거S(극한폭락)"},
    "trigger_a_silent_fk":        {"base": 6.0, "kind": "disabled", "label": "트리거A(폭락코일링)"},
    "trigger_b_3d_momentum":      {"base": 4.5, "kind": "momentum", "label": "트리거B(3일모멘텀)"},
    "trigger_c_runner_quiet":     {"base": 4.0, "kind": "momentum", "label": "트리거C(조용한모멘텀)"},
    "trigger_d_pullback_pro":     {"base": 5.5, "kind": "disabled", "label": "트리거D(저점강세)"},
    "trigger_e_pullback_std":     {"base": 4.0, "kind": "pullback", "label": "트리거E(저점표준)"},
    "trigger_f_pullback_recovery":{"base": 3.5, "kind": "pullback", "label": "트리거F(저점회복)"},
    "trigger_g_rsi_bottom":       {"base": 5.0, "kind": "pullback", "label": "트리거G(RSI저점)"},
    "trigger_h_failed_breakdown": {"base": 4.0, "kind": "pullback", "label": "트리거H(지지사수)"},
    "trigger_i_macd_bottom":      {"base": 3.5, "kind": "pullback", "label": "트리거I(MACD저점)"},
    "trigger_j_deep_rsi":         {"base": 3.5, "kind": "pullback", "label": "트리거J(딥RSI)"},
    "trigger_k_soft_recovery":    {"base": 3.5, "kind": "pullback", "label": "트리거K(소프트회복)"},
    "trigger_l_steady_recovery":  {"base": 3.5, "kind": "pullback", "label": "트리거L(안정회복)"},
    "trigger_m_2m_pullback":      {"base": 4.5, "kind": "disabled", "label": "트리거M(2개월눌림)"},
    "trigger_n_stale_drawdown":   {"base": 3.5, "kind": "pullback", "label": "트리거N(스테일DD)"},
    "trigger_o_4m_bottom":        {"base": 3.5, "kind": "pullback", "label": "트리거O(4개월저점)"},
    "trigger_p_engulfing":        {"base": 4.0, "kind": "pullback", "label": "트리거P(불리시엔걸핑)"},
    "trigger_q_gap_hold_recovery":{"base": 5.5, "kind": "pullback", "label": "트리거Q(갭홀드회복)"},
    "trigger_v_gap_hold_wide":    {"base": 4.5, "kind": "pullback", "label": "트리거V(갭홀드넓은범위)"},
    "trigger_r_monthly_ma60_support":{"base": 4.0, "kind": "pullback", "label": "트리거R(월봉MA60지지)"},
    "trigger_t_monthly_bull_support":{"base": 4.5, "kind": "pullback", "label": "트리거T(월봉정배지지)"},
    "trigger_u_institutional_accum":{"base": 5.0, "kind": "pullback", "label": "트리거U(기관매집)"},
}


def trigger_multiplier(kind: str, regime: str | None) -> float:
    """필터 검증 결과: pullback은 regime≠bull, momentum은 bull+정배열."""
    if kind == "disabled":
        return 0.1  # 거의 무효 (전부 -15% 손절)
    if kind == "pullback":
        if regime == "bull":         return 0.3
        if regime in ("correction", "bear"): return 1.5
        return 1.0  # transition / unknown
    if kind == "momentum":
        if regime == "bull":         return 1.3
        if regime in ("correction", "bear"): return 0.3
        return 1.0
    return 1.0


def recommendation_score(p: dict) -> tuple[float, list[str]]:
    """Score a pair for daily recommendation. Returns (score, reasons[])."""
    score = 0.0
    reasons: list[str] = []
    sigs = p.get("signals") or []
    alerts = p.get("alerts") or []
    regime = p.get("spy_regime")
    ma20_dist = p.get("ma20_dist_und")
    vol_2x_std = p.get("vol_2x_std")

    # === 트리거 점수 (regime multiplier 적용) ===
    # S/A 중복 가산 방지
    skip = set()
    if "trigger_s_extreme_fk" in sigs:
        skip.add("trigger_a_silent_fk")
    if ("trigger_a_silent_fk" in sigs and "trigger_s_extreme_fk" not in sigs):
        skip.add("trigger_s_extreme_fk")  # noop

    triggered_pullback = False
    triggered_momentum = False
    for trig, meta in TRIGGER_META.items():
        if trig not in sigs or trig in skip: continue
        mult = trigger_multiplier(meta["kind"], regime)
        gained = meta["base"] * mult
        if gained < 0.5:  # 너무 미미하면 표시 생략
            continue
        score += gained
        reasons.append(f"{meta['label']}×{mult:.1f}")
        if meta["kind"] == "pullback": triggered_pullback = True
        if meta["kind"] == "momentum": triggered_momentum = True

    # === 보조 필터 보너스 ===
    # MA20 +2% 이상 위 (pullback 트리거의 최강 보조 필터)
    if triggered_pullback and ma20_dist is not None and ma20_dist >= 2:
        score += 1.0; reasons.append(f"MA20+{ma20_dist:.1f}%")
    # 2X 일간 변동성 ≥ 8% (반등 폭 ↑)
    if triggered_pullback and vol_2x_std is not None and vol_2x_std >= 8:
        score += 0.5; reasons.append(f"변동성{vol_2x_std:.1f}%")
    # regime 정보 reason에 표시
    if regime and (triggered_pullback or triggered_momentum):
        reasons.append(f"[regime={regime}]")

    # === 콤보 보너스 (다중 트리거 동시 발동) ===
    # E+G 콤보 (85.7%, n=14)
    if "trigger_e_pullback_std" in sigs and "trigger_g_rsi_bottom" in sigs:
        score += 2.5; reasons.append("콤보E+G(86%)")
    # I+O 콤보 (67.8%, n=59) — 가장 실용적
    if "trigger_i_macd_bottom" in sigs and "trigger_o_4m_bottom" in sigs:
        score += 2.0; reasons.append("콤보I+O(68%)")
    # E+I 콤보 (68.4%, n=38)
    if "trigger_e_pullback_std" in sigs and "trigger_i_macd_bottom" in sigs:
        score += 2.0; reasons.append("콤보E+I(68%)")
    # G+I 콤보 (69.2%, n=13)
    if "trigger_g_rsi_bottom" in sigs and "trigger_i_macd_bottom" in sigs:
        score += 1.5; reasons.append("콤보G+I(69%)")
    # E+N 콤보 (56.9%, n=116) — 큰 표본
    if "trigger_e_pullback_std" in sigs and "trigger_n_stale_drawdown" in sigs:
        score += 1.5; reasons.append("콤보E+N(57%)")
    # I+N 콤보 (67.6%, n=34)
    if "trigger_i_macd_bottom" in sigs and "trigger_n_stale_drawdown" in sigs:
        score += 1.5; reasons.append("콤보I+N(68%)")
    # J+N 콤보 (78.6%, n=14)
    if "trigger_j_deep_rsi" in sigs and "trigger_n_stale_drawdown" in sigs:
        score += 2.0; reasons.append("콤보J+N(79%)")
    # G+O, G+N, G+J 콤보 (75%+)
    if "trigger_g_rsi_bottom" in sigs:
        if "trigger_o_4m_bottom" in sigs:
            score += 1.5; reasons.append("콤보G+O(75%)")
        if "trigger_n_stale_drawdown" in sigs:
            score += 1.5; reasons.append("콤보G+N(76%)")
        if "trigger_j_deep_rsi" in sigs:
            score += 1.5; reasons.append("콤보G+J(75%)")

    # 3개 이상 트리거 동시 발동 시 보너스
    trigger_count = sum(1 for k in sigs if k.startswith("trigger_"))
    if trigger_count >= 4:
        score += 3.0; reasons.append(f"트리거{trigger_count}개동시")
    elif trigger_count >= 3:
        score += 2.0; reasons.append(f"트리거{trigger_count}개동시")
    if "falling_knife_setup" in sigs and "trigger_a_silent_fk" not in sigs and "trigger_s_extreme_fk" not in sigs:
        score += 5.0; reasons.append("폭락코일링")  # 트리거 S/A 미발동 시만
    if "monthly_bull_near_long" in sigs:
        score += 3.5; reasons.append("월정배+장기근접")
    if "bull_align_pullback" in sigs:
        score += 0.5; reasons.append("정배눌림")  # 실측 2.55% (base 5.35%의 절반) — 비효율

    # Trend confirmation
    if "monthly_bull_align" in sigs and "monthly_bull_near_long" not in sigs:
        score += 1.5; reasons.append("월정배")
    if "ma_bull_align" in sigs:
        score += 2.0; reasons.append("일정배")

    # Cross events (recent)
    if "golden_cross" in sigs:
        score += 2.0; reasons.append("골드크로스")
    if "dead_cross" in sigs:
        score -= 2.0

    # Support tests (only score once)
    if any(s in sigs for s in ("ma20_support", "ma60_support", "ma120_support")):
        score += 1.0; reasons.append("MA지지")

    # Buying momentum
    if p.get("volume_up_candle"):
        score += 2.0; reasons.append("거래량+양봉")
    if p.get("near_52w_high"):
        score += 1.0; reasons.append("52신고가")

    # MACD
    if "macd_bull_cross" in sigs: score += 2.0; reasons.append("MACD골크")
    if "macd_momentum_up" in sigs: score += 1.0; reasons.append("MACD모멘텀↑")
    if "macd_above_zero" in sigs: score += 0.5
    if "macd_bear_cross" in sigs: score -= 2.0
    if "macd_momentum_down" in sigs: score -= 1.0
    if "macd_below_zero" in sigs: score -= 0.5

    # Sector leadership
    if "sector_leader" in sigs: score += 1.5; reasons.append("섹터선도")
    if "sector_laggard" in sigs: score -= 1.0

    # Earnings track record
    if "earnings_beat_streak" in sigs: score += 1.5; reasons.append("4분기 연속 비트")
    elif "earnings_beats" in sigs: score += 0.5
    if "earnings_miss_streak" in sigs: score -= 1.0

    # Options flow
    if "call_heavy" in sigs: score += 2.0; reasons.append("콜 매수 우세")
    if "put_heavy" in sigs: score -= 1.5
    if "iv_elevated" in sigs: score -= 0.5  # 큰 이벤트 임박 — 양방향 위험
    if "squeeze_setup" in sigs: score += 1.5; reasons.append("스퀴즈 셋업")  # 콜+IV낮음 콤보 보너스
    if "call_oi_growth" in sigs: score += 1.5; reasons.append("콜OI급증")
    if "call_iv_premium" in sigs: score += 1.5; reasons.append("콜IV프리미엄")
    if "unusual_call_activity" in sigs: score += 1.0; reasons.append("이상콜거래")

    # Volatility compression
    if "hv_compressed" in sigs: score += 1.0; reasons.append("변동성압축")
    if "hv_expanded" in sigs: score -= 0.5  # 이미 폭발 중 — 추격 위험

    # Fundamentals
    if "short_squeeze_setup" in sigs: score += 2.5; reasons.append("숏스퀴즈셋업")
    elif "high_short_interest" in sigs: score += 0.5  # 단독으로는 약함
    if "insider_buying_strong" in sigs: score += 2.5; reasons.append("내부자대량매수")
    elif "insider_buying" in sigs: score += 1.0; reasons.append("내부자매수")
    if "analyst_upgrades" in sigs: score += 1.0; reasons.append("애널리스트상향")

    # === 펀더멘털 (성장 + 밸류에이션) ===
    # "주가 빠진 좋은 회사" 식별 — NVO 같은 디버전스 케이스를 잡음
    if "strong_revenue_growth" in sigs: score += 1.5; reasons.append("매출+20%↑")
    if "strong_eps_growth" in sigs:     score += 1.5; reasons.append("EPS+30%↑")
    if "high_margins" in sigs:          score += 1.0; reasons.append("영업이익률25%↑")
    if "high_roe" in sigs:              score += 1.0; reasons.append("ROE20%↑")
    if "value_growth" in sigs:          score += 2.0; reasons.append("저평가성장주")  # PER<18 + 매출15%+ 흑자
    if "peg_value" in sigs:             score += 1.5; reasons.append("PEG≤1.5")
    # 음수
    if "loss_making" in sigs:           score -= 1.5; reasons.append("적자")
    if "revenue_decline" in sigs:       score -= 1.5; reasons.append("매출역성장")

    # 펀더멘털 + 트리거 콤보: "주가 빠진 우량 성장주" 가산
    if "value_growth" in sigs and any(s.startswith("trigger_") for s in sigs):
        score += 1.5; reasons.append("우량+트리거")

    # RSI sweet spot
    rsi = p.get("rsi")
    if rsi is not None:
        if 30 <= rsi <= 50:
            score += 1.0; reasons.append(f"RSI {rsi:.0f}")
        elif 50 < rsi <= 65:
            score += 0.5
        elif rsi < 25:
            score += 0.5
        elif rsi > 75:
            score -= 1.0

    # Liquidity (avoid micro-cap traps)
    dv = p.get("dollar_volume_20d")
    if dv is not None:
        if dv >= 100e6:
            score += 1.5
        elif dv >= 10e6:
            score += 0.5
        elif dv < 1e6:
            score -= 2.0

    # Negatives
    if "ma_bear_align" in sigs: score -= 3.0
    if "monthly_bear_align" in sigs: score -= 2.5
    drag = p.get("drag_6m")
    if drag is not None and drag <= -15: score -= 1.5

    # Earnings risk (0~3d before) — binary event, neither buy nor avoid
    days = p.get("days_to_earnings")
    if days is not None and 0 <= days <= 3:
        score -= 0.5; reasons.append(f"실적 D-{days}")

    # Recent big drop = potentially bounce candidate, slight bonus if oversold
    if "big_drop" in alerts and rsi is not None and rsi < 35:
        score += 0.5; reasons.append("과매도 바닥")

    return round(score, 2), reasons


def hidden_gem_score(p: dict) -> tuple[float, list[str]]:
    """폭주 전 우량 종목 식별 — rec_score와 별개의 랭킹.

    철학: '시장이 아직 안 알아본 좋은 회사'를 찾기.
      - 펀더멘털 우량 (필수): 최소 2개 시그널 발동 못하면 후보 자격 박탈
      - 가격 빠진 상태 (눌림): 6m DD, 눌림 트리거
      - 폭주 페널티: 이미 5일 +15%↑, 모멘텀 트리거, 52신고가, 거래량 급증
    """
    score = 0.0
    reasons: list[str] = []
    sigs = p.get("signals") or []

    # === [1] 펀더멘털 우량 (필수 — 부족하면 즉시 자격 박탈) ===
    fund_n = 0
    if "value_growth" in sigs:
        score += 3.0; reasons.append("저평가성장주"); fund_n += 1
    if "peg_value" in sigs:
        score += 2.0; reasons.append("PEG≤1.5"); fund_n += 1
    if "strong_revenue_growth" in sigs:
        score += 1.5; reasons.append("매출+20%↑"); fund_n += 1
    if "strong_eps_growth" in sigs:
        score += 1.5; reasons.append("EPS+30%↑"); fund_n += 1
    if "high_margins" in sigs:
        score += 1.0; reasons.append("영업이익률25%↑"); fund_n += 1
    if "high_roe" in sigs:
        score += 1.0; reasons.append("ROE20%↑"); fund_n += 1

    if fund_n < 2:
        return 0.0, []  # 펀더멘털 부재 — Hidden Gem 후보 아님

    if "loss_making" in sigs:
        score -= 3.0; reasons.append("적자")
    if "revenue_decline" in sigs:
        score -= 2.0; reasons.append("매출역성장")

    # === [2] 가격 빠진 상태 (눌림) — 폭주 전이려면 빠져있어야 함 ===
    dd_6m = p.get("dd_6m_und")
    if dd_6m is not None:
        if -40 <= dd_6m <= -15:
            score += 2.0; reasons.append(f"6m {dd_6m:.0f}%")
        elif -15 < dd_6m <= -8:
            score += 1.0; reasons.append(f"6m {dd_6m:.0f}%")
        elif dd_6m < -40:
            score += 0.5; reasons.append(f"6m {dd_6m:.0f}%(심층)")
        elif dd_6m > -3:
            score -= 1.5; reasons.append("이미 고점권")  # 안 빠짐 = 폭주 시작했거나 이미 진행

    # 눌림 트리거 (모멘텀 트리거 제외) — 매수 타이밍 필수 조건
    PULLBACK = {
        "trigger_e_pullback_std", "trigger_f_pullback_recovery", "trigger_g_rsi_bottom",
        "trigger_h_failed_breakdown", "trigger_i_macd_bottom", "trigger_j_deep_rsi",
        "trigger_k_soft_recovery", "trigger_l_steady_recovery",
        "trigger_n_stale_drawdown", "trigger_o_4m_bottom", "trigger_p_engulfing",
        "trigger_q_gap_hold_recovery", "trigger_r_monthly_ma60_support",
        "trigger_t_monthly_bull_support", "trigger_u_institutional_accum",
        "trigger_v_gap_hold_wide",
    }
    n_pull = sum(1 for t in sigs if t in PULLBACK)
    if n_pull < 1:
        return 0.0, []  # 매수 타이밍 없음 — Hidden Gem 자격 박탈
    bonus = min(2.0, n_pull * 1.0)
    score += bonus; reasons.append(f"눌림트리거{n_pull}개")

    # === [3] 폭주 페널티 — 이미 터졌으면 후보 아님 ===
    five_day = p.get("five_day_pct")
    if five_day is not None:
        if five_day > 15:
            score -= 3.0; reasons.append(f"5일+{five_day:.0f}%(폭주중)")
        elif five_day > 10:
            score -= 1.5; reasons.append(f"5일+{five_day:.0f}%")
        elif five_day < -15:
            score -= 1.0; reasons.append(f"5일{five_day:+.0f}%(급락중)")

    if "trigger_b_3d_momentum" in sigs:
        score -= 3.0; reasons.append("3일+30%(폭주)")
    if "trigger_c_runner_quiet" in sigs:
        score -= 2.5; reasons.append("5일+30%(폭주)")
    if "trigger_s_extreme_fk" in sigs or "trigger_a_silent_fk" in sigs:
        score -= 2.0; reasons.append("폭락중(추가 하락 위험)")

    if p.get("near_52w_high"):
        score -= 2.0; reasons.append("52신고가 근접")
    if "golden_cross" in sigs:
        score -= 1.0; reasons.append("골크(이미 전환)")

    vr = p.get("volume_ratio")
    if vr is not None and vr >= 2.0:
        score -= 1.5; reasons.append(f"거래량{vr:.1f}x(인식됨)")

    # 메가캡 페널티 — Mag7은 절대 hidden 아님 (시장 모두가 보고 있음)
    MEGA_TICKERS = {"AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
                    "AVGO", "BRK-B", "BRK.B", "WMT", "JPM", "ORCL", "MA", "V"}
    und = p.get("ticker_underlying")
    if und in MEGA_TICKERS:
        score -= 3.0; reasons.append(f"메가캡({und})")

    # === [4] 보조 가산 ===
    if "ma20_support" in sigs or "ma60_support" in sigs or "ma120_support" in sigs:
        score += 0.5; reasons.append("MA지지")
    if "insider_buying_strong" in sigs:
        score += 1.5; reasons.append("내부자대량매수")
    elif "insider_buying" in sigs:
        score += 0.5; reasons.append("내부자매수")
    # 월정배는 가산 (장기 우상향 추세)
    if "monthly_bull_align" in sigs:
        score += 1.0; reasons.append("월정배")
    elif "monthly_bear_align" in sigs:
        score -= 1.5; reasons.append("월역배(추세 약함)")

    # 유동성 (마이크로캡 함정 회피)
    dv = p.get("dollar_volume_20d")
    if dv is not None and dv < 1e6:
        score -= 2.0; reasons.append("저유동성")

    return round(score, 2), reasons


def _earnings_fields(info: dict | None, today: date) -> dict:
    if not info or not info.get("next_earnings"):
        return {
            "next_earnings": None,
            "days_to_earnings": None,
            "earnings_window": None,
            "eps_estimate": None,
        }
    try:
        ed = date.fromisoformat(info["next_earnings"])
    except Exception:
        return {"next_earnings": None, "days_to_earnings": None, "earnings_window": None, "eps_estimate": None}
    days = (ed - today).days
    window = None
    if 0 <= days <= 7:
        window = "soon"
    elif -7 <= days < 0:
        window = "post"
    return {
        "next_earnings": info["next_earnings"],
        "days_to_earnings": days,
        "earnings_window": window,
        "eps_estimate": info.get("eps_estimate"),
        "beat_streak": info.get("beat_streak", 0),
        "miss_streak": info.get("miss_streak", 0),
        "avg_surprise_pct": info.get("avg_surprise_pct"),
    }


def load_earnings() -> dict[str, dict]:
    if not EARNINGS_JSON.exists():
        return {}
    try:
        return json.loads(EARNINGS_JSON.read_text(encoding="utf-8")).get("tickers", {})
    except Exception:
        return {}


def load_options() -> dict[str, dict]:
    if not OPTIONS_JSON.exists():
        return {}
    try:
        return json.loads(OPTIONS_JSON.read_text(encoding="utf-8")).get("tickers", {})
    except Exception:
        return {}


def load_fundamentals() -> dict[str, dict]:
    if not FUNDAMENTALS_JSON.exists():
        return {}
    try:
        return json.loads(FUNDAMENTALS_JSON.read_text(encoding="utf-8")).get("tickers", {})
    except Exception:
        return {}


def load_monthly() -> pd.DataFrame | None:
    if not MONTHLY_PKL.exists():
        return None
    try:
        return pd.read_pickle(MONTHLY_PKL)
    except Exception:
        return None


def load_weekly() -> pd.DataFrame | None:
    if not WEEKLY_PKL.exists():
        return None
    try:
        return pd.read_pickle(WEEKLY_PKL)
    except Exception:
        return None


def bull_60_120(series: pd.Series | None) -> bool | None:
    """MA60 > MA120 정배열 여부 (말단 60/120 평균 기준). 데이터 부족 시 None."""
    if series is None:
        return None
    s = series.dropna()
    if s.size < 120:
        return None
    ma60 = s.iloc[-60:].mean()
    ma120 = s.iloc[-120:].mean()
    if pd.isna(ma60) or pd.isna(ma120):
        return None
    return bool(ma60 > ma120)


def monthly_alignment(monthly: pd.DataFrame | None, ticker: str) -> str | None:
    """Returns 'monthly_bull_align' / 'monthly_bear_align' / None based on
    monthly MA5/MA20/MA60 alignment on the underlying."""
    if monthly is None or ticker not in monthly.columns:
        return None
    s = monthly[ticker].dropna()
    if s.size < 60:
        return None
    ma5 = s.iloc[-5:].mean()
    ma20 = s.iloc[-20:].mean()
    ma60 = s.iloc[-60:].mean()
    if any(pd.isna(v) for v in (ma5, ma20, ma60)):
        return None
    if ma5 > ma20 > ma60:
        return "monthly_bull_align"
    if ma5 < ma20 < ma60:
        return "monthly_bear_align"
    return None


def main() -> int:
    payload = json.loads(ETF_LIST.read_text(encoding="utf-8"))
    prices = pd.read_pickle(PRICES_PKL)
    earnings = load_earnings()
    monthly = load_monthly()
    weekly = load_weekly()
    options = load_options()
    fundamentals = load_fundamentals()
    today = date.today()

    # Reference last trading date = max across all tickers; ETFs that haven't
    # printed a tape in >10 calendar days are treated as delisted/halted.
    close_all = prices.xs("Close", axis=1, level="field")
    last_dates = close_all.apply(lambda s: s.dropna().index.max())
    ref_date = last_dates.max()
    stale_cutoff = ref_date - pd.Timedelta(days=10)
    stale_set = {t for t, d in last_dates.items() if pd.notna(d) and d < stale_cutoff}

    pairs: list[dict] = []
    skipped: list[str] = []
    delisted: list[dict] = []
    seen_tickers: set[str] = set()
    for e in payload["etfs"]:
        t2 = (e.get("ticker_2x") or "").strip().upper()
        if t2 in seen_tickers:
            continue  # dedupe — scrape_all가 중복 추가하는 경우 방어
        seen_tickers.add(t2)
        if t2 in stale_set:
            ld = last_dates.get(t2)
            delisted.append({
                "ticker_2x": t2,
                "underlying": e.get("underlying"),
                "issuer": e.get("issuer"),
                "last_trade": str(ld.date()) if pd.notna(ld) else None,
            })
            continue
        rec = analyze_pair(e, prices, earnings, today, monthly, options, fundamentals, weekly)
        if rec is None:
            skipped.append(t2 or "?")
            continue
        pairs.append(rec)

    daily_rets = [p["daily_pct"] for p in pairs if p.get("daily_pct") is not None]
    alerts_count = sum(1 for p in pairs if p["alerts"])
    proxy_count = sum(1 for p in pairs if p.get("proxy_fields"))

    # SPY regime — 모든 pair에 동일하게 적용 (시장 국면 필터)
    regime = spy_regime(prices)
    for p in pairs:
        p["spy_regime"] = regime

    # Recommendation scoring — top picks for today
    for p in pairs:
        score, reasons = recommendation_score(p)
        p["rec_score"] = score
        p["rec_reasons"] = reasons
        gem_score, gem_reasons = hidden_gem_score(p)
        p["gem_score"] = gem_score
        p["gem_reasons"] = gem_reasons
    REC_MIN_SCORE = 5.0
    REC_TOP_N = 10
    rec_candidates = sorted(
        [p for p in pairs if p["rec_score"] >= REC_MIN_SCORE],
        key=lambda p: -p["rec_score"],
    )[:REC_TOP_N]
    rec_tickers = {p["ticker_2x"] for p in rec_candidates}
    for p in pairs:
        p["is_recommended"] = p["ticker_2x"] in rec_tickers

    # Hidden Gem 랭킹 — '폭주 전 우량주' 별도 후보
    GEM_MIN_SCORE = 5.0
    GEM_TOP_N = 10
    gem_pool = sorted(
        [p for p in pairs if p.get("gem_score", 0) >= GEM_MIN_SCORE],
        key=lambda p: -p["gem_score"],
    )
    # underlying 별 dedupe — 같은 기초자산의 2X ETF 여러개는 가장 점수 높은 것만
    seen_und: set[str] = set()
    gem_candidates: list[dict] = []
    for p in gem_pool:
        und = p.get("ticker_underlying") or p["ticker_2x"]
        if und in seen_und: continue
        seen_und.add(und)
        gem_candidates.append(p)
        if len(gem_candidates) >= GEM_TOP_N: break
    gem_tickers = {p["ticker_2x"] for p in gem_candidates}
    for p in pairs:
        p["is_hidden_gem"] = p["ticker_2x"] in gem_tickers

    summary = {
        "total_pairs": len(pairs),
        "alerts_count": alerts_count,
        "using_underlying_proxy": proxy_count,
        "delisted_count": len(delisted),
        "avg_daily_return": round(float(np.mean(daily_rets)), 2) if daily_rets else None,
        "as_of": str(ref_date.date()),
        "recommended_count": len(rec_candidates),
        "recommended": [
            {"ticker_2x": p["ticker_2x"], "underlying": p["ticker_underlying"],
             "score": p["rec_score"], "reasons": p["rec_reasons"]}
            for p in rec_candidates
        ],
        "hidden_gems_count": len(gem_candidates),
        "hidden_gems": [
            {"ticker_2x": p["ticker_2x"], "underlying": p["ticker_underlying"],
             "score": p["gem_score"], "reasons": p["gem_reasons"]}
            for p in gem_candidates
        ],
        # 갱신 건강 상태 — 스크래퍼 폴백(stale) 발행사 + 유니버스 갱신 시각
        "stale_issuers": payload.get("stale_issuers") or [],
        "universe_updated_at": payload.get("updated_at"),
    }

    pairs_sorted = sorted(
        pairs,
        key=lambda p: (-len(p["alerts"]), p.get("daily_pct") if p.get("daily_pct") is not None else 0),
    )

    out = {
        "updated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "summary": summary,
        "skipped_no_price": skipped,
        "delisted": delisted,
        "pairs": pairs_sorted,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {OUT}")
    print(f"  as of        : {summary['as_of']}")
    print(f"  pairs        : {len(pairs)}")
    print(f"  alerts       : {alerts_count}")
    print(f"  proxy(under) : {proxy_count}")
    print(f"  delisted     : {len(delisted)}  {[d['ticker_2x'] for d in delisted][:10]}")
    print(f"  skipped(no $): {len(skipped)}  {skipped[:10]}")
    print(f"  avg daily    : {summary['avg_daily_return']}%")
    print(f"  recommended  : {len(rec_candidates)}")
    for p in rec_candidates:
        rs = ", ".join(p["rec_reasons"][:4])
        print(f"    {p['ticker_2x']:6s} {p['ticker_underlying']:6s}  score={p['rec_score']:>4.1f}  [{rs}]")
    print(f"  hidden gems  : {len(gem_candidates)}")
    for p in gem_candidates:
        rs = ", ".join(p["gem_reasons"][:5])
        print(f"    {p['ticker_2x']:6s} {p['ticker_underlying']:6s}  gem={p['gem_score']:>4.1f}  [{rs}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
