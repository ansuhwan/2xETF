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
EARNINGS_JSON = DATA / "earnings.json"
OPTIONS_JSON = DATA / "options.json"
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


def has_ticker(prices: pd.DataFrame, t: str) -> bool:
    return t in prices.columns.get_level_values(0)


def close_of(prices: pd.DataFrame, t: str) -> pd.Series:
    return prices[(t, "Close")].dropna()


def volume_of(prices: pd.DataFrame, t: str) -> pd.Series:
    return prices[(t, "Volume")].dropna()


def low_of(prices: pd.DataFrame, t: str) -> pd.Series:
    return prices[(t, "Low")].dropna()


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


def analyze_pair(etf: dict, prices: pd.DataFrame, earnings: dict[str, dict], today: date, monthly: pd.DataFrame | None = None, options: dict[str, dict] | None = None) -> dict | None:
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

    proxies: list[str] = []
    if rsi_proxy:
        proxies.append("rsi")
    if bb_proxy:
        proxies.append("bb")
    if five_day_proxy:
        proxies.append("five_day")

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
        **_earnings_fields(earnings.get(und) if und else None, today),
    }


def recommendation_score(p: dict) -> tuple[float, list[str]]:
    """Score a pair for daily recommendation. Returns (score, reasons[])."""
    score = 0.0
    reasons: list[str] = []
    sigs = p.get("signals") or []
    alerts = p.get("alerts") or []

    # Premium setups (long-term + tactical entry)
    if "monthly_bull_near_long" in sigs:
        score += 3.5; reasons.append("월정배+장기근접")
    if "bull_align_pullback" in sigs:
        score += 3.0; reasons.append("정배눌림")

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


def load_monthly() -> pd.DataFrame | None:
    if not MONTHLY_PKL.exists():
        return None
    try:
        return pd.read_pickle(MONTHLY_PKL)
    except Exception:
        return None


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
    options = load_options()
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
    for e in payload["etfs"]:
        t2 = (e.get("ticker_2x") or "").strip().upper()
        if t2 in stale_set:
            ld = last_dates.get(t2)
            delisted.append({
                "ticker_2x": t2,
                "underlying": e.get("underlying"),
                "issuer": e.get("issuer"),
                "last_trade": str(ld.date()) if pd.notna(ld) else None,
            })
            continue
        rec = analyze_pair(e, prices, earnings, today, monthly, options)
        if rec is None:
            skipped.append(t2 or "?")
            continue
        pairs.append(rec)

    daily_rets = [p["daily_pct"] for p in pairs if p.get("daily_pct") is not None]
    alerts_count = sum(1 for p in pairs if p["alerts"])
    proxy_count = sum(1 for p in pairs if p.get("proxy_fields"))

    # Recommendation scoring — top picks for today
    for p in pairs:
        score, reasons = recommendation_score(p)
        p["rec_score"] = score
        p["rec_reasons"] = reasons
    REC_MIN_SCORE = 5.0
    REC_TOP_N = 10
    rec_candidates = sorted(
        [p for p in pairs if p["rec_score"] >= REC_MIN_SCORE],
        key=lambda p: -p["rec_score"],
    )[:REC_TOP_N]
    rec_tickers = {p["ticker_2x"] for p in rec_candidates}
    for p in pairs:
        p["is_recommended"] = p["ticker_2x"] in rec_tickers

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
