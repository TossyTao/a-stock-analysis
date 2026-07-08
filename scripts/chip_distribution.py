"""筹码峰分析:三角形分布 + 指数衰减 + 峰态识别 + 支撑压力。

核心原理:股票交易是筹码换手,汇总各价位持仓占比形成筹码峰。
- 峰越高 = 该价位持仓越多
- 下方筹码 = 支撑,上方筹码 = 压力
- 越集中 = 市场分歧越小 = 波动合力越强
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_chip_distribution(
    daily: List[Dict[str, Any]],
    decay: float = 0.05,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算筹码分布。

    算法:
    - 价格区间分 n_bins 个桶
    - 每个交易日:当日成交量按三角形分布在 [low, high](峰在 close)
    - 每天现有筹码 × (1-decay) 衰减,模拟换手
    - 归一化为百分比

    decay=0.05:半衰期约 13 天,60 天前的筹码权重 < 5%。
    """
    if len(daily) < 20:
        raise ValueError(f"数据不足: {len(daily)} < 20")

    df = pd.DataFrame(daily)
    for c in ["low", "high", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    all_low = float(df["low"].min())
    all_high = float(df["high"].max())
    if all_high <= all_low:
        raise ValueError("价格区间无效")

    bins = np.linspace(all_low, all_high, n_bins)
    chip = np.zeros(n_bins)

    for _, row in df.iterrows():
        chip *= (1 - decay)
        low, high, close, vol = row["low"], row["high"], row["close"], row["volume"]
        if vol <= 0 or high <= low:
            continue
        in_range = (bins >= low) & (bins <= high)
        if not in_range.any():
            continue
        if close <= low or close >= high:
            weights = in_range.astype(float)
        else:
            weights = np.zeros(n_bins)
            for j, p in enumerate(bins):
                if p < low or p > high:
                    continue
                if p <= close:
                    weights[j] = (p - low) / (close - low)
                else:
                    weights[j] = (high - p) / (high - close)
        w_sum = weights.sum()
        if w_sum > 0:
            chip += vol * weights / w_sum

    total = chip.sum()
    if total > 0:
        chip = chip / total * 100

    return bins, chip


def find_peaks(
    bins: np.ndarray,
    chip: np.ndarray,
    smoothing_window: int = 3,
) -> List[Dict[str, Any]]:
    """识别筹码峰。返回 [{price, pct, index}, ...] 按价格升序。"""
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        chip_s = np.convolve(chip, kernel, mode="same")
    else:
        chip_s = chip.copy()

    max_pct = chip_s.max()
    if max_pct < 1.0:
        return []

    threshold = max(1.0, max_pct * 0.3)

    peaks = []
    for i in range(1, len(chip_s) - 1):
        if (
            chip_s[i] >= threshold
            and chip_s[i] >= chip_s[i - 1]
            and chip_s[i] > chip_s[i + 1]
        ):
            peaks.append({"price": float(bins[i]), "pct": round(float(chip_s[i]), 2), "index": i})

    merged = []
    for p in peaks:
        if merged and abs(p["price"] - merged[-1]["price"]) / merged[-1]["price"] < 0.02:
            if p["pct"] > merged[-1]["pct"]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged


def classify_pattern(
    peaks: List[Dict[str, Any]],
    current_close: float,
    all_low: float,
    all_high: float,
) -> Dict[str, Any]:
    """识别筹码峰形态(单峰/双峰/多峰/分散)+ 位置 + 解读。"""
    if not peaks:
        return {
            "pattern": "分散",
            "dominant_peak": None,
            "all_peaks": [],
            "position_label": None,
            "relative_to_price": None,
            "peak_pos_pct": None,
            "interpretation": "筹码分散无主导成本区,市场分歧大,股价缺乏方向",
        }

    dominant = max(peaks, key=lambda p: p["pct"])
    price_range = all_high - all_low
    peak_pos_pct = (dominant["price"] - all_low) / price_range * 100 if price_range > 0 else 50.0
    if peak_pos_pct <= 33:
        pos_label = "低位"
    elif peak_pos_pct >= 67:
        pos_label = "高位"
    else:
        pos_label = "中位"

    if dominant["price"] < current_close * 0.95:
        relative = "下方(支撑)"
    elif dominant["price"] > current_close * 1.05:
        relative = "上方(压力)"
    else:
        relative = "贴近"

    n_peaks = len(peaks)
    if n_peaks == 1:
        pattern = "单峰"
    elif n_peaks == 2:
        pattern = "双峰"
    else:
        pattern = "多峰"

    interp = ""
    if pattern == "单峰":
        if pos_label == "低位" and relative == "贴近":
            interp = "低位单峰:主力吸筹信号,长期下跌横盘后等放量易拉升"
        elif pos_label == "高位" and relative == "贴近":
            interp = "高位单峰:出货风险,散户盼回本抛压大,股价易回落"
        elif relative == "下方(支撑)":
            interp = "单峰支撑:峰位在下方构成强支撑,回调到峰位附近可低吸"
        elif relative == "上方(压力)":
            interp = "单峰压力:峰位在上方构成强压力,反弹到峰位附近需谨慎"
        else:
            interp = "中位单峰:筹码集中,方向待选择"
    elif pattern == "双峰":
        peaks_sorted = sorted(peaks, key=lambda p: p["price"])
        interp = (
            f"双峰形态:适合波段做T,股价大概率在两峰之间"
            f"({peaks_sorted[0]['price']:.2f} - {peaks_sorted[-1]['price']:.2f})震荡"
        )
    else:
        interp = "多峰分散:筹码不集中,市场分歧大,趋势不明朗"

    return {
        "pattern": pattern,
        "dominant_peak": dominant,
        "all_peaks": peaks,
        "position_label": pos_label,
        "relative_to_price": relative,
        "peak_pos_pct": round(peak_pos_pct, 1),
        "interpretation": interp,
    }


def support_resistance(
    bins: np.ndarray,
    chip: np.ndarray,
    current_close: float,
) -> Dict[str, Any]:
    """找当前价下方最近的筹码密集(支撑)和上方最近密集(压力)。"""
    kernel = np.ones(3) / 3
    chip_s = np.convolve(chip, kernel, mode="same")

    support = None
    resistance = None
    for i in range(1, len(chip_s) - 1):
        if chip_s[i] >= chip_s[i - 1] and chip_s[i] > chip_s[i + 1] and chip_s[i] >= 1.0:
            if bins[i] < current_close * 0.99:
                if support is None or chip_s[i] > support["pct"]:
                    support = {"price": round(float(bins[i]), 2), "pct": round(float(chip_s[i]), 2)}
            elif bins[i] > current_close * 1.01:
                if resistance is None or chip_s[i] > resistance["pct"]:
                    resistance = {"price": round(float(bins[i]), 2), "pct": round(float(chip_s[i]), 2)}

    if support is None:
        below_idx = [i for i in range(len(bins)) if bins[i] < current_close and chip_s[i] >= 0.5]
        if below_idx:
            best = max(below_idx, key=lambda i: chip_s[i])
            support = {"price": round(float(bins[best]), 2), "pct": round(float(chip_s[best]), 2)}
    if resistance is None:
        above_idx = [i for i in range(len(bins)) if bins[i] > current_close and chip_s[i] >= 0.5]
        if above_idx:
            best = max(above_idx, key=lambda i: chip_s[i])
            resistance = {"price": round(float(bins[best]), 2), "pct": round(float(chip_s[best]), 2)}

    return {"support": support, "resistance": resistance}


def concentration_ratio(
    bins: np.ndarray,
    chip: np.ndarray,
    current_close: float,
    band_pct: float = 0.05,
) -> float:
    """当前价 ±band_pct 范围内的筹码占比(集中度)。"""
    low_bound = current_close * (1 - band_pct)
    high_bound = current_close * (1 + band_pct)
    mask = (bins >= low_bound) & (bins <= high_bound)
    return round(float(chip[mask].sum()), 2)


def analyze(daily: List[Dict[str, Any]]) -> Dict[str, Any]:
    """筹码峰分析主入口。"""
    df = pd.DataFrame(daily)
    for c in ["low", "high", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bins, chip = compute_chip_distribution(daily)
    current_close = float(df["close"].iloc[-1])
    all_low = float(df["low"].min())
    all_high = float(df["high"].max())

    peaks = find_peaks(bins, chip)
    pattern = classify_pattern(peaks, current_close, all_low, all_high)
    sr = support_resistance(bins, chip, current_close)
    conc = concentration_ratio(bins, chip, current_close)
    conc_label = "高度集中" if conc >= 40 else ("较集中" if conc >= 25 else "分散")

    return {
        "current_close": current_close,
        "price_range": {"low": round(all_low, 2), "high": round(all_high, 2)},
        "peaks": peaks,
        "pattern": pattern,
        "support_resistance": sr,
        "concentration_5pct": conc,
        "concentration_label": conc_label,
    }
