"""量价指标计算:位置分位、均量线、量比、地量、单日象限、5日趋势、背离、突破站稳、筹码峰。"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from chip_distribution import analyze as chip_analyze


def to_df(daily: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(daily)
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def position_percentile(close: pd.Series, window: int) -> float:
    """当前价在过去 window 日的百分位 (0-100)。"""
    if len(close) < window:
        window = len(close)
    seg = close.tail(window)
    hi, lo = seg.max(), seg.min()
    if hi == lo:
        return 50.0
    return round((close.iloc[-1] - lo) / (hi - lo) * 100, 1)


def position_label(pct: float) -> str:
    if pct >= 80:
        return "高位"
    if pct <= 20:
        return "低位"
    return "中位"


def vol_ma(volume: pd.Series, n: int) -> float:
    if len(volume) < n:
        n = len(volume)
    return round(float(volume.tail(n).mean()), 0)


def vol_ma_cross(volume: pd.Series) -> Dict[str, Any]:
    """5日 / 20日成交量均线金叉/死叉判定。

    金叉:过去 3 日内 ma5 从下方上穿 ma20。
    死叉:过去 3 日内 ma5 从上方下穿 ma20。
    """
    if len(volume) < 20:
        return {"state": "数据不足", "ma5": None, "ma20": None}
    ma5 = volume.rolling(5).mean()
    ma20 = volume.rolling(20).mean()
    diff = ma5 - ma20
    cross = "无交叉"
    if len(diff) >= 6:
        recent = diff.iloc[-6:].tolist()
        for i in range(1, len(recent)):
            if recent[i-1] <= 0 and recent[i] > 0 and recent[-1] > 0:
                cross = "金叉"
                break
            if recent[i-1] >= 0 and recent[i] < 0 and recent[-1] < 0:
                cross = "死叉"
                break
    return {
        "state": cross,
        "ma5": round(float(ma5.iloc[-1]), 0),
        "ma20": round(float(ma20.iloc[-1]), 0),
    }


def vol_ratio(volume: pd.Series) -> float:
    """相对量比 ROVL:今日量 / 过去 20 日均量。

    用于五步核对的"量比 1.5-3 倍"判断 — 突破日量能是否充分放大。
    20 日基准反映真实筹码活跃度,5 日太短易被单日异常带偏。
    """
    if len(volume) < 21:
        return 1.0
    prev20_avg = volume.iloc[-21:-1].mean()
    if prev20_avg == 0:
        return 0.0
    return round(float(volume.iloc[-1] / prev20_avg), 2)


def is_ground_vol(volume: pd.Series) -> bool:
    """地量:今日量 < 20 日均量 × 0.5。"""
    if len(volume) < 21:
        return False
    ma20 = volume.tail(20).mean()
    return bool(volume.iloc[-1] < ma20 * 0.5)


def quadrant(df: pd.DataFrame) -> Dict[str, Any]:
    """今日 vs 昨日的单日量价四象限(辅助信号)。

    返回:量增/缩 + 价涨/跌 + 象限名。
    主信号应使用 trend_5day(),单日象限仅作拐点验证。
    """
    if len(df) < 2:
        return {"vol_change": "数据不足", "price_change": "数据不足", "quadrant": "数据不足"}
    today = df.iloc[-1]
    yest = df.iloc[-2]
    vol_up = today["volume"] > yest["volume"]
    price_up = today["close"] > yest["close"]
    vol_chg = "量增" if vol_up else "量缩"
    price_chg = "价涨" if price_up else "价跌"
    q = f"{vol_chg}{price_chg}"
    return {
        "vol_change": vol_chg,
        "price_change": price_chg,
        "quadrant": q,
        "today_close": float(today["close"]),
        "yest_close": float(yest["close"]),
        "today_vol": float(today["volume"]),
        "yest_vol": float(yest["volume"]),
    }


def trend_5day(df: pd.DataFrame) -> Dict[str, Any]:
    """近 5 日量价趋势(线性回归斜率法)。

    单日象限噪音大,5 日趋势反映真实主力意图。
    对近 5 个交易日的收盘价和成交量分别做最小二乘线性回归,
    斜率归一化为"日均变化率%",避免不同价位/量级的股票斜率不可比。

    阈值:
      - 价格:日均变化 > 0.3% = 价涨,< -0.3% = 价跌,中间为价平
      - 成交量:日均变化 > 3% = 量增,< -3% = 量缩,中间为量平
      (量阈值大于价阈值,因为成交量自然波动比价格大)
    """
    if len(df) < 5:
        return {"price_trend": "数据不足", "vol_trend": "数据不足", "quadrant": "数据不足"}

    last5 = df.tail(5).reset_index(drop=True)
    price = last5["close"].astype(float).values
    vol = last5["volume"].astype(float).values
    x = np.arange(5)

    price_slope = float(np.polyfit(x, price, 1)[0])
    vol_slope = float(np.polyfit(x, vol, 1)[0])

    price_mean = float(price.mean())
    vol_mean = float(vol.mean())

    price_slope_pct = (price_slope / price_mean * 100) if price_mean > 0 else 0.0
    vol_slope_pct = (vol_slope / vol_mean * 100) if vol_mean > 0 else 0.0

    PRICE_THRESHOLD = 0.3
    VOL_THRESHOLD = 3.0

    if price_slope_pct > PRICE_THRESHOLD:
        price_trend = "价涨"
    elif price_slope_pct < -PRICE_THRESHOLD:
        price_trend = "价跌"
    else:
        price_trend = "价平"

    if vol_slope_pct > VOL_THRESHOLD:
        vol_trend = "量增"
    elif vol_slope_pct < -VOL_THRESHOLD:
        vol_trend = "量缩"
    else:
        vol_trend = "量平"

    quadrant_5d = f"{vol_trend}{price_trend}"

    max_slope_ratio = max(
        abs(price_slope_pct / PRICE_THRESHOLD) if price_trend != "价平" else 0,
        abs(vol_slope_pct / VOL_THRESHOLD) if vol_trend != "量平" else 0,
    )
    if max_slope_ratio >= 3:
        strength = "强"
    elif max_slope_ratio >= 1.5:
        strength = "中"
    else:
        strength = "弱"

    daily_quads = []
    for i in range(1, 5):
        v_up = vol[i] > vol[i - 1]
        p_up = price[i] > price[i - 1]
        daily_quads.append(f"{'量增' if v_up else '量缩'}{'价涨' if p_up else '价跌'}")

    if vol_trend != "量平" and price_trend != "价平":
        trend_quad = f"{vol_trend}{price_trend}"
        consistent = sum(1 for q in daily_quads if q == trend_quad)
        consistency = f"{consistent}/4"
    else:
        consistency = "-"

    return {
        "price_trend": price_trend,
        "vol_trend": vol_trend,
        "quadrant": quadrant_5d,
        "price_slope_pct": round(price_slope_pct, 2),
        "vol_slope_pct": round(vol_slope_pct, 2),
        "strength": strength,
        "daily_quadrants": daily_quads,
        "consistency": consistency,
        "start_close": float(price[0]),
        "end_close": float(price[-1]),
        "start_vol": float(vol[0]),
        "end_vol": float(vol[-1]),
    }


def top_divergence(df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    """顶背离:近 window 日内价格创区间新高,但成交量未创新高。"""
    if len(df) < window:
        window = len(df)
    seg = df.tail(window).reset_index(drop=True)
    price_max_idx = seg["close"].idxmax()
    vol_at_price_high = seg.loc[price_max_idx, "volume"]
    vol_max = seg["volume"].max()
    is_div = price_max_idx == len(seg) - 1 and vol_at_price_high < vol_max * 0.8
    return {
        "detected": bool(is_div),
        "price_new_high": bool(price_max_idx == len(seg) - 1),
        "vol_at_price_high": float(vol_at_price_high),
        "vol_max": float(vol_max),
        "window": window,
    }


def bottom_divergence(df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    """底背离:近 window 日内价格创区间新低,但成交量萎缩或放量滞跌。"""
    if len(df) < window:
        window = len(df)
    seg = df.tail(window).reset_index(drop=True)
    price_min_idx = seg["close"].idxmin()
    vol_at_price_low = seg.loc[price_min_idx, "volume"]
    vol_mean = seg["volume"].mean()
    # 价新低 + 当日量 < 均量(无量新低)或当日量 > 均量但涨幅收窄(放量滞跌)
    price_new_low = price_min_idx == len(seg) - 1
    no_vol_new_low = price_new_low and vol_at_price_low < vol_mean * 0.8
    # 放量滞跌:今日量 > 均量,但价格变化幅度 < 1%
    today = seg.iloc[-1]
    yest = seg.iloc[-2] if len(seg) > 1 else today
    price_chg_pct = abs((today["close"] - yest["close"]) / yest["close"]) if yest["close"] > 0 else 0
    vol_surge_stall = today["volume"] > vol_mean * 1.2 and price_chg_pct < 0.01
    return {
        "detected": bool(no_vol_new_low or vol_surge_stall),
        "price_new_low": bool(price_new_low),
        "no_vol_new_low": bool(no_vol_new_low),
        "vol_surge_stall": bool(vol_surge_stall),
        "vol_at_price_low": float(vol_at_price_low),
        "vol_mean": float(vol_mean),
        "window": window,
    }


def breakout_3day(df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    """是否在近端阻力位(近 window 日最高价)上方站稳 3 天。

    站稳:连续 3 天收盘价 > 阻力位。
    突破位:取近 window 日最高价(不含最后 3 天)。
    """
    if len(df) < window + 3:
        return {"detected": False, "resistance": None, "days_above": 0, "note": "数据不足"}
    lookback = df.iloc[-(window + 3):-3]
    resistance = float(lookback["high"].max())
    last3 = df.tail(3)
    days_above = int((last3["close"] > resistance).sum())
    return {
        "detected": days_above == 3,
        "resistance": resistance,
        "days_above": days_above,
        "last3_closes": last3["close"].round(2).tolist(),
    }


def detect_traps(df: pd.DataFrame, quad: Dict[str, Any]) -> Dict[str, Any]:
    """主力骗局风险标记。

    - 假突破:近 5 日内曾有 1 日收盘价 > 近 20 日最高价,但最新 3 日内跌回阻力下方
    - 对倒造量:量比 > 2.5 但价格涨幅 < 1%(放量不涨)
    - 算法无量空涨:量比 < 0.7 但价格涨幅 > 2%(无量拉升)
    """
    if len(df) < 25:
        return {"false_breakout": False, "wash_trade": False, "algo_no_vol_rise": False}
    last5 = df.tail(5)
    lookback = df.iloc[-25:-5]
    resistance = float(lookback["high"].max())
    false_breakout = bool(
        (last5["high"] > resistance).any() and df.iloc[-1]["close"] < resistance
    )
    today = df.iloc[-1]
    yest = df.iloc[-2]
    price_chg_pct = (today["close"] - yest["close"]) / yest["close"] if yest["close"] > 0 else 0
    vr = vol_ratio(df["volume"])
    wash_trade = bool(vr > 2.5 and abs(price_chg_pct) < 0.01)
    algo_no_vol = bool(vr < 0.7 and price_chg_pct > 0.02)
    return {
        "false_breakout": false_breakout,
        "wash_trade": wash_trade,
        "algo_no_vol_rise": algo_no_vol,
        "vol_ratio": vr,
        "price_chg_pct": round(float(price_chg_pct) * 100, 2),
    }


def compute(daily: List[Dict[str, Any]]) -> Dict[str, Any]:
    """主入口:输入 daily 列表,输出全部指标。"""
    df = to_df(daily)
    if len(df) < 20:
        raise ValueError(f"数据不足:仅 {len(df)} 行,至少需要 20 行")

    pos60 = position_percentile(df["close"], 60)
    pos120 = position_percentile(df["close"], 120)
    vol_cross = vol_ma_cross(df["volume"])
    vr = vol_ratio(df["volume"])
    ground = is_ground_vol(df["volume"])
    quad = quadrant(df)
    trend5 = trend_5day(df)
    top_div = top_divergence(df)
    bot_div = bottom_divergence(df)
    breakout = breakout_3day(df)
    traps = detect_traps(df, quad)
    chip = chip_analyze(daily)

    five_step = {
        "1_position": {
            "pass": pos120 <= 50,
            "detail": f"60日分位 {pos60}% / 120日分位 {pos120}% → {position_label(pos120)}",
        },
        "2_ground_vol": {
            "pass": ground,
            "detail": f"今日量 {df['volume'].iloc[-1]:.0f} vs 20日均量 {vol_ma(df['volume'], 20):.0f}",
        },
        "3_vol_ratio": {
            "pass": 1.5 <= vr <= 3.0,
            "detail": f"量比 {vr}",
        },
        "4_vol_ma_cross": {
            "pass": vol_cross["state"] == "金叉",
            "detail": f"5/20日均量 {vol_cross['state']}",
        },
        "5_breakout_3day": {
            "pass": breakout["detected"],
            "detail": f"阻力位 {breakout.get('resistance')}, 站稳 {breakout.get('days_above', 0)}/3 天",
        },
    }

    return {
        "position": {
            "pct_60d": pos60,
            "pct_120d": pos120,
            "label": position_label(pos120),
            "current_close": float(df["close"].iloc[-1]),
            "current_vol": float(df["volume"].iloc[-1]),
            "pct_chg_today": round(
                float((df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100),
                2,
            ) if len(df) >= 2 else 0.0,
        },
        "volume": {
            "ma5": vol_ma(df["volume"], 5),
            "ma20": vol_ma(df["volume"], 20),
            "ma_cross": vol_cross["state"],
            "vol_ratio": vr,
            "is_ground_vol": ground,
        },
        "quadrant": quad,
        "trend_5d": trend5,
        "divergence": {
            "top": top_div,
            "bottom": bot_div,
        },
        "traps": traps,
        "breakout": breakout,
        "five_step": five_step,
        "chip": chip,
    }
