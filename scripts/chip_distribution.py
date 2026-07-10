"""筹码峰分析:换手率衰减模型(筹码守恒)+ 底仓追踪 + ASR/CYQK + 形态信号。

核心原理:
- **筹码守恒**:股票交易是筹码换手。每日换手率 turnover_t = vol_t / 流通股本。
  当日 turnover_t 比例的旧筹码被转移,新筹码按 [low, high] 三角分布(峰在 close)沉淀。
  主力再怎么对倒、藏仓,真实换手数据藏不住 - 这是底仓追踪的物理基础。
- 峰越高 = 该价位持仓越多;下方筹码 = 支撑,上方筹码 = 压力;越集中 = 合力越强。
- **底仓追踪**:涨 30-50% 后底部筹码不萎缩 = 主力还在(主升浪续涨信号);
  高位底仓消失 = 主力出货(见顶信号)。散户早止盈了,底仓不动说明主力未跑。

无流通股本数据时降级到固定 decay=0.05(半衰期约 13 天)。
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def _daily_turnover(vol: float, free_float_shares: Optional[float]) -> float:
    """计算当日换手率。无流通股本时返回 None(调用方降级到固定 decay)。"""
    if free_float_shares is None or free_float_shares <= 0:
        return None
    return min(max(vol / free_float_shares, 0.0), 1.0)


def compute_chip_distribution(
    daily: List[Dict[str, Any]],
    free_float_shares: Optional[float] = None,
    n_bins: int = 100,
    fallback_decay: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """计算筹码分布(换手率衰减模型)。

    算法:
    - 价格区间分 n_bins 个桶
    - 每个交易日:当日成交量按三角形分布在 [low, high](峰在 close)
    - 每天现有筹码 × (1 - turnover_t) 保留,turnover_t 比例被新筹码替换
    - 无流通股本时降级到固定 fallback_decay
    - 归一化为百分比

    返回 (bins, chip, meta),meta 包含 decay_mode / avg_turnover 等信息。
    """
    if len(daily) < 5:
        raise ValueError(f"数据不足: {len(daily)} < 5")

    df = pd.DataFrame(daily)
    for c in ["low", "high", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    all_low = float(df["low"].min())
    all_high = float(df["high"].max())
    if all_high <= all_low:
        raise ValueError("价格区间无效")

    bins = np.linspace(all_low, all_high, n_bins)
    chip = np.zeros(n_bins)

    turnovers = []
    for _, row in df.iterrows():
        vol = float(row["volume"]) if pd.notna(row["volume"]) else 0.0
        low, high, close = row["low"], row["high"], row["close"]
        if pd.isna(low) or pd.isna(high) or pd.isna(close):
            continue

        # 换手率衰减:turnover_t 比例的旧筹码被替换
        turnover_t = _daily_turnover(vol, free_float_shares) if vol > 0 else None
        if turnover_t is not None:
            chip *= (1 - turnover_t)
            turnovers.append(turnover_t)
            decay_mode = "turnover"
        else:
            chip *= (1 - fallback_decay)
            decay_mode = "fixed"

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

    avg_turnover = float(np.mean(turnovers)) if turnovers else None
    meta = {
        "decay_mode": decay_mode,
        "avg_turnover": avg_turnover,
        "free_float_shares": free_float_shares,
    }
    return bins, chip, meta


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


def compute_asr(
    bins: np.ndarray,
    chip: np.ndarray,
    current_close: float,
    band_pct: float = 0.10,
) -> Dict[str, Any]:
    """ASR(活动筹码指标)= 当前价 ±band_pct 带内筹码占比。

    下跌时 ASR 高 = 筹码锁死(没人割肉,主力控盘);
    下跌时 ASR 低 = 筹码松动(恐慌抛售)。
    """
    low_bound = current_close * (1 - band_pct)
    high_bound = current_close * (1 + band_pct)
    mask = (bins >= low_bound) & (bins <= high_bound)
    asr = float(chip[mask].sum())
    label = "高" if asr >= 50 else ("中" if asr >= 30 else "低")
    return {
        "value": round(asr, 2),
        "band_pct": band_pct,
        "label": label,
        "interpretation": (
            "活动筹码占比高,下跌中筹码锁死 = 主力控盘" if asr >= 50
            else "活动筹码占比中,需结合趋势判断" if asr >= 30
            else "活动筹码占比低,筹码松动"
        ),
    }


def compute_cyqk(
    bins: np.ndarray,
    chip: np.ndarray,
    current_close: float,
) -> Dict[str, Any]:
    """CYQK(筹码K线)。CYQK_WIN = 获利比例 = 当前价下方筹码占比。

    获利比例高(>80%)= 大部分人盈利,抛压大;
    获利比例低(<20%)= 大部分人套牢,反弹抛压小。
    低位长阳 + 低换手 + 获利比例上升 = 主力控盘。
    """
    below_mask = bins < current_close
    win_ratio = float(chip[below_mask].sum())
    label = (
        "套牢多" if win_ratio < 20
        else "套牢偏多" if win_ratio < 40
        else "均衡" if win_ratio < 60
        else "获利偏多" if win_ratio < 80
        else "获利多"
    )
    return {
        "win_ratio": round(win_ratio, 2),
        "label": label,
        "interpretation": (
            f"获利比例 {win_ratio:.1f}% - "
            + ("大部分套牢,反弹抛压小" if win_ratio < 40
               else "均衡" if win_ratio < 60
               else "大部分盈利,抛压渐增")
        ),
    }


def compute_bottom_chip_retention(
    daily: List[Dict[str, Any]],
    free_float_shares: Optional[float] = None,
    lookback_days: int = 30,
    min_price_rise_pct: float = 0.30,
    bottom_threshold_pct: float = 1.0,
    retention_threshold: float = 0.5,
    disappearance_threshold: float = 0.2,
) -> Dict[str, Any]:
    """底仓追踪:比较当前与 lookback_days 日前的筹码分布,判断底仓是否保留。

    逻辑:
    - 取 lookback_days 日前的筹码分布(用前 N-lookback 日数据),找最低峰(底仓)
    - 看当前分布在同一价位区间的筹码是否保留
    - 若价格涨 ≥30% 且底仓保留率 ≥50% -> "底仓不动"(主升浪续涨信号)
    - 若底仓保留率 <20% -> "底仓消失"(主力出货,见顶信号)

    返回:
    - bottom_peak_price: 底仓价位
    - bottom_pct_then / bottom_pct_now: 当时/现在的底仓占比
    - retention_ratio: 保留率 = now / then
    - price_rise_pct: 价格涨幅
    - signal: "底仓不动" / "底仓消失" / "无信号"
    """
    df = pd.DataFrame(daily)
    for c in ["low", "high", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n = len(df)
    if n < lookback_days + 30:
        return {
            "available": False,
            "reason": f"数据不足 {n} < {lookback_days + 30}",
            "signal": "无信号",
        }

    # 切片:lookback_days 日前用前 n-lookback 数据
    daily_then = df.iloc[: n - lookback_days].to_dict("records")
    price_then = float(daily_then[-1]["close"])
    price_now = float(df["close"].iloc[-1])
    price_rise_pct = (price_now - price_then) / price_then if price_then > 0 else 0.0

    bins_then, chip_then, _ = compute_chip_distribution(daily_then, free_float_shares)
    peaks_then = find_peaks(bins_then, chip_then)

    if not peaks_then:
        return {
            "available": True,
            "reason": "lookback 日前无显著筹码峰",
            "signal": "无信号",
            "price_rise_pct": round(price_rise_pct, 4),
            "price_then": round(price_then, 2),
            "price_now": round(price_now, 2),
        }

    # 底仓 = 最低价的峰
    bottom_peak = min(peaks_then, key=lambda p: p["price"])
    bottom_price = bottom_peak["price"]

    # 用同一 ±2% 带宽比较两个时刻的底仓筹码占比(口径一致)
    band_low = bottom_price * 0.98
    band_high = bottom_price * 1.02

    mask_then = (bins_then >= band_low) & (bins_then <= band_high)
    bottom_pct_then = float(chip_then[mask_then].sum())

    bins_now, chip_now, _ = compute_chip_distribution(daily, free_float_shares)
    mask_now = (bins_now >= band_low) & (bins_now <= band_high)
    bottom_pct_now = float(chip_now[mask_now].sum())

    retention_ratio = bottom_pct_now / bottom_pct_then if bottom_pct_then > 0 else 0.0

    # 判断信号
    signal = "无信号"
    interpretation = ""
    if price_rise_pct >= min_price_rise_pct:
        if retention_ratio >= retention_threshold:
            signal = "底仓不动"
            interpretation = (
                f"价格已涨 {price_rise_pct*100:.1f}%,但底仓(价位 {bottom_price:.2f})"
                f"保留率 {retention_ratio*100:.0f}%(>={retention_threshold*100:.0f}%),"
                "散户早止盈了,底仓不动说明主力还在 - 主升浪续涨信号"
            )
        elif retention_ratio < disappearance_threshold:
            signal = "底仓消失"
            interpretation = (
                f"底仓保留率仅 {retention_ratio*100:.0f}%(<{disappearance_threshold*100:.0f}%),"
                "主力已出货,见顶信号"
            )
        else:
            signal = "底仓部分转移"
            interpretation = (
                f"底仓保留率 {retention_ratio*100:.0f}%,部分转移,信号不明确"
            )
    else:
        interpretation = (
            f"价格涨幅 {price_rise_pct*100:.1f}% < {min_price_rise_pct*100:.0f}%,"
            "底仓追踪信号不适用(需涨 30%+ 才有意义)"
        )

    return {
        "available": True,
        "bottom_peak_price": round(bottom_price, 2),
        "bottom_pct_then": round(bottom_pct_then, 2),
        "bottom_pct_now": round(bottom_pct_now, 2),
        "retention_ratio": round(retention_ratio, 3),
        "price_rise_pct": round(price_rise_pct, 4),
        "price_then": round(price_then, 2),
        "price_now": round(price_now, 2),
        "lookback_days": lookback_days,
        "signal": signal,
        "interpretation": interpretation,
    }


def classify_pattern(
    peaks: List[Dict[str, Any]],
    current_close: float,
    all_low: float,
    all_high: float,
    today_turnover: Optional[float] = None,
    bottom_retention: Optional[Dict[str, Any]] = None,
    breakout_resistance: Optional[float] = None,
) -> Dict[str, Any]:
    """识别筹码峰形态 + 升级信号(主升浪 / 洗盘 / 见顶)。

    升级信号:
    - 低位单峰 + 今日换手 <1.5%(无量突破) + 价格接近/突破阻力 = 主升浪信号
    - 双峰 + 底仓保留率 ≥0.5 = 洗盘信号(底仓不萎缩)
    - 高位单峰 + 底仓消失(保留率 <0.2) = 见顶信号
    """
    if not peaks:
        return {
            "pattern": "分散",
            "dominant_peak": None,
            "all_peaks": [],
            "position_label": None,
            "relative_to_price": None,
            "peak_pos_pct": None,
            "enhanced_signal": None,
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

    # 升级信号判断
    enhanced_signal = None
    br = bottom_retention or {}
    br_signal = br.get("signal")
    br_ratio = br.get("retention_ratio")

    if pattern == "单峰":
        if pos_label == "低位" and relative == "贴近":
            # 低位单峰 + 无量突破(换手 <1.5%) = 主升浪信号
            if today_turnover is not None and today_turnover < 0.015:
                enhanced_signal = "主升浪信号"
                interp = (
                    f"低位单峰密集 + 今日换手 {today_turnover*100:.2f}%(<1.5%,无量突破)"
                    " = 主力吸筹完毕,主升浪启动信号"
                )
            else:
                enhanced_signal = "低位单峰待突破"
                interp = (
                    "低位单峰:主力吸筹信号,长期下跌横盘后等放量(换手 <1.5%)易拉升"
                )
                if today_turnover is not None:
                    interp += f"(今日换手 {today_turnover*100:.2f}%)"
        elif pos_label == "高位" and relative == "贴近":
            # 高位单峰 + 底仓消失 = 见顶
            if br_signal == "底仓消失":
                enhanced_signal = "见顶信号"
                interp = (
                    f"高位单峰密集 + 底仓消失(保留率 {br_ratio*100:.0f}%) = "
                    "主力出货完成,见顶信号,警惕减仓"
                )
            else:
                enhanced_signal = "高位预警"
                interp = "高位单峰:出货风险,散户盼回本抛压大,股价易回落"
        elif relative == "下方(支撑)":
            interp = "单峰支撑:峰位在下方构成强支撑,回调到峰位附近可低吸"
        elif relative == "上方(压力)":
            interp = "单峰压力:峰位在上方构成强压力,反弹到峰位附近需谨慎"
        else:
            interp = "中位单峰:筹码集中,方向待选择"
    elif pattern == "双峰":
        peaks_sorted = sorted(peaks, key=lambda p: p["price"])
        # 双峰峡谷 + 底仓不萎缩 = 洗盘
        if br_signal == "底仓不动" or (br_ratio is not None and br_ratio >= 0.5):
            enhanced_signal = "洗盘信号"
            interp = (
                f"双峰峡谷 + 底仓保留率 {br_ratio*100:.0f}%(不萎缩) = "
                f"主力洗盘,股价大概率在两峰之间"
                f"({peaks_sorted[0]['price']:.2f} - {peaks_sorted[-1]['price']:.2f})震荡,底仓未跑"
            )
        else:
            interp = (
                f"双峰形态:适合波段做T,股价大概率在两峰之间"
                f"({peaks_sorted[0]['price']:.2f} - {peaks_sorted[-1]['price']:.2f})震荡"
            )
            if br_ratio is not None:
                interp += f"(底仓保留率 {br_ratio*100:.0f}%)"
    else:
        interp = "多峰分散:筹码不集中,市场分歧大,趋势不明朗"
        if br_signal == "底仓不动":
            interp += ";但底仓不动,关注突破方向"

    return {
        "pattern": pattern,
        "dominant_peak": dominant,
        "all_peaks": peaks,
        "position_label": pos_label,
        "relative_to_price": relative,
        "peak_pos_pct": round(peak_pos_pct, 1),
        "enhanced_signal": enhanced_signal,
        "today_turnover": round(today_turnover, 4) if today_turnover is not None else None,
        "bottom_retention_signal": br_signal,
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


def analyze(
    daily: List[Dict[str, Any]],
    free_float_shares: Optional[float] = None,
    breakout_resistance: Optional[float] = None,
) -> Dict[str, Any]:
    """筹码峰分析主入口(换手率衰减 + 底仓追踪 + ASR + CYQK)。"""
    df = pd.DataFrame(daily)
    for c in ["low", "high", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bins, chip, decay_meta = compute_chip_distribution(daily, free_float_shares)
    current_close = float(df["close"].iloc[-1])
    all_low = float(df["low"].min())
    all_high = float(df["high"].max())

    peaks = find_peaks(bins, chip)
    asr = compute_asr(bins, chip, current_close)
    cyqk = compute_cyqk(bins, chip, current_close)
    bottom_retention = compute_bottom_chip_retention(daily, free_float_shares)

    # 今日换手率(用于形态升级判断)
    today_vol = float(df["volume"].iloc[-1])
    today_turnover = _daily_turnover(today_vol, free_float_shares)

    pattern = classify_pattern(
        peaks, current_close, all_low, all_high,
        today_turnover=today_turnover,
        bottom_retention=bottom_retention,
        breakout_resistance=breakout_resistance,
    )
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
        "asr": asr,
        "cyqk": cyqk,
        "bottom_retention": bottom_retention,
        "decay_meta": decay_meta,
    }


def compute_short_term_chip_trend(
    daily: List[Dict[str, Any]],
    free_float_shares: Optional[float] = None,
    windows: List[int] = None,
) -> Dict[str, Any]:
    """计算 5/10/20 日筹码分布对比,识别筹码迁移趋势。

    对每个窗口用换手率衰减模型单独计算筹码分布,输出:
    - 主峰位置 + 占比
    - 集中度(±5%)
    - CYQK(获利比例)
    - ASR(活动筹码)

    趋势判断:
    - 主峰位置随窗口缩短而上移 = 筹码向上迁移(高位承接增强)
    - 主峰位置随窗口缩短而下移 = 筹码向下迁移(低位承接增强)
    - 主峰位置稳定 = 筹码锁定
    - 集中度随窗口缩短而上升 = 短期筹码集中(主力吸筹或锁定)
    - CYQK 随窗口缩短而上升 = 短期获利盘增加(抛压渐增)

    Args:
        daily: 日线数据(至少 max(windows)+30 天)
        free_float_shares: 流通股本
        windows: 窗口列表,默认 [5, 10, 20]

    Returns:
        {
            "available": bool,
            "windows": {5: {...}, 10: {...}, 20: {...}},
            "trend": {
                "peak_migration": "向上/向下/稳定",
                "concentration_trend": "上升/下降/稳定",
                "cyqk_trend": "上升/下降/稳定",
            },
            "interpretation": str,
        }
    """
    if windows is None:
        windows = [5, 10, 20]

    df = pd.DataFrame(daily)
    for c in ["low", "high", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if len(df) < max(windows) + 5:
        return {"available": False, "reason": f"数据不足 {len(df)} < {max(windows)+5}"}

    current_close = float(df["close"].iloc[-1])
    window_results: Dict[int, Dict[str, Any]] = {}

    for w in windows:
        # 取最近 w 天
        seg = df.tail(w).to_dict("records")
        try:
            bins, chip, meta = compute_chip_distribution(seg, free_float_shares)
            peaks = find_peaks(bins, chip)
            asr = compute_asr(bins, chip, current_close)
            cyqk = compute_cyqk(bins, chip, current_close)
            conc = concentration_ratio(bins, chip, current_close)

            # 主峰(占比最高的)
            if peaks:
                dominant = max(peaks, key=lambda p: p.get("pct", 0))
                dom_price = dominant.get("price")
                dom_pct = dominant.get("pct")
            else:
                idx = int(np.argmax(chip))
                dom_price = round(float(bins[idx]), 2)
                dom_pct = round(float(chip[idx]), 2)

            # 主峰相对当前价的位置
            if dom_price is not None and current_close > 0:
                rel = (dom_price - current_close) / current_close
                if rel < -0.02:
                    pos_label = "下方"
                elif rel > 0.02:
                    pos_label = "上方"
                else:
                    pos_label = "贴近"
            else:
                pos_label = "未知"

            window_results[w] = {
                "window": w,
                "dominant_peak": {"price": dom_price, "pct": dom_pct, "position": pos_label},
                "concentration_5pct": conc,
                "cyqk_win_ratio": cyqk.get("win_ratio"),
                "cyqk_label": cyqk.get("label"),
                "asr_value": asr.get("value"),
                "asr_label": asr.get("label"),
                "decay_mode": meta.get("decay_mode"),
                "peak_count": len(peaks),
            }
        except Exception as e:
            window_results[w] = {"window": w, "error": str(e)}

    # 趋势判断(需要 5/10/20 都有数据)
    valid_windows = {w: r for w, r in window_results.items() if "error" not in r}
    if len(valid_windows) < 2:
        return {
            "available": False,
            "windows": window_results,
            "reason": "有效窗口不足",
        }

    # 主峰迁移:对比短窗口(5)和长窗口(20)的主峰位置
    short_w = min(valid_windows.keys())
    long_w = max(valid_windows.keys())
    short_peak = valid_windows[short_w].get("dominant_peak", {}).get("price")
    long_peak = valid_windows[long_w].get("dominant_peak", {}).get("price")

    peak_migration = "稳定"
    if short_peak is not None and long_peak is not None and current_close > 0:
        # 短窗口主峰相对当前价的偏移 vs 长窗口主峰相对当前价的偏移
        short_off = (short_peak - current_close) / current_close
        long_off = (long_peak - current_close) / current_close
        diff = short_off - long_off
        if diff > 0.02:
            peak_migration = "向上迁移"
        elif diff < -0.02:
            peak_migration = "向下迁移"
        else:
            peak_migration = "稳定"

    # 集中度趋势:短窗口 vs 长窗口
    short_conc = valid_windows[short_w].get("concentration_5pct", 0) or 0
    long_conc = valid_windows[long_w].get("concentration_5pct", 0) or 0
    conc_diff = short_conc - long_conc
    if conc_diff > 3:
        conc_trend = "上升"
    elif conc_diff < -3:
        conc_trend = "下降"
    else:
        conc_trend = "稳定"

    # CYQK 趋势:短窗口 vs 长窗口
    short_cyqk = valid_windows[short_w].get("cyqk_win_ratio", 0) or 0
    long_cyqk = valid_windows[long_w].get("cyqk_win_ratio", 0) or 0
    cyqk_diff = short_cyqk - long_cyqk
    if cyqk_diff > 5:
        cyqk_trend = "上升"
    elif cyqk_diff < -5:
        cyqk_trend = "下降"
    else:
        cyqk_trend = "稳定"

    # 综合解读
    interpretations = []
    if peak_migration == "向上迁移":
        interpretations.append("短期主峰上移,高位承接增强(可能是吸筹或派发,需结合量价)")
    elif peak_migration == "向下迁移":
        interpretations.append("短期主峰下移,低位承接增强(可能是主力吸筹)")
    else:
        interpretations.append("主峰位置稳定,筹码锁定")

    if conc_trend == "上升":
        interpretations.append("短期集中度上升,筹码集中(主力吸筹或锁定)")
    elif conc_trend == "下降":
        interpretations.append("短期集中度下降,筹码分散(主力可能派发)")

    if cyqk_trend == "上升":
        interpretations.append("短期获利盘增加,抛压渐增")
    elif cyqk_trend == "下降":
        interpretations.append("短期获利盘减少,抛压减轻")

    return {
        "available": True,
        "windows": window_results,
        "current_close": current_close,
        "trend": {
            "peak_migration": peak_migration,
            "concentration_trend": conc_trend,
            "cyqk_trend": cyqk_trend,
            "short_window": short_w,
            "long_window": long_w,
            "concentration_diff": round(conc_diff, 1),
            "cyqk_diff": round(cyqk_diff, 1),
        },
        "interpretation": "; ".join(interpretations) if interpretations else "无明显趋势",
    }


# ---------- 筹码 × 主力资金流 交叉验证 ----------

def cross_validate_chip_capital(
    chip: Dict[str, Any],
    short_term_chip: Dict[str, Any],
    capital_flow: Dict[str, Any],
) -> Dict[str, Any]:
    """筹码 × 主力资金流交叉验证,确认主力意图。

    矩阵(短期筹码趋势 × 主力资金):
    | 短期筹码      | 主力净流入        | 主力净流出        |
    |--------------|-------------------|-------------------|
    | 主峰上移     | ✅ 强吸筹(资金+筹码双确认) | ⚠️ 派发(高位换手出货) |
    | 主峰稳定     | 低位吸筹(悄然收集) | 暗中派发(主力撤离) |
    | 主峰下移     | 抄底进场(恐慌盘接货) | 弱势阴跌(无承接) |
    | 集中度上升   | ✅ 强吸筹(筹码集中) | 派发尾声(散户接盘) |
    | 集中度下降   | 派发(筹码分散) | 强派发(主力+散户双撤) |

    Args:
        chip: compute_chip_distribution 的输出(indicators.chip)
        short_term_chip: compute_short_term_chip_trend 的输出(indicators.short_term_chip)
        capital_flow: fetch_capital_flow 的输出(indicators.capital_flow)

    Returns:
        {
            "available": bool,
            "main_force_intent": "强吸筹" | "吸筹" | "派发" | "强派发" | "中性" | "矛盾",
            "confidence": "高" | "中" | "低",
            "evidence": [str, ...],
            "interpretation": str,
            "action_hint": str,
        }
    """
    # 任一输入 unavailable -> 整体不可用
    if not capital_flow or not capital_flow.get("available"):
        return {"available": False, "reason": "资金流数据不可用"}
    if not short_term_chip or not short_term_chip.get("available"):
        return {"available": False, "reason": "短期筹码数据不可用"}

    # 提取筹码信号
    st_trend = short_term_chip.get("trend", {})
    peak_migration = st_trend.get("peak_migration", "稳定")  # 向上迁移/向下迁移/稳定
    conc_trend = st_trend.get("concentration_trend", "稳定")  # 上升/下降/稳定
    cyqk_trend = st_trend.get("cyqk_trend", "稳定")

    # 提取资金流信号
    cf_signals = capital_flow.get("signals", {})
    cf_action = cf_signals.get("main_force_action", "中性")  # 吸筹/派发/中性
    cf_strength = cf_signals.get("strength", "弱")  # 强/中/弱
    cf_trend = capital_flow.get("trend", {})
    cf_consecutive = cf_trend.get("consecutive_days", 0)
    cf_source = capital_flow.get("source", "eastmoney")  # eastmoney / ths
    cf_trend_available = cf_trend.get("available", False)

    # 今日主力净额
    today = capital_flow.get("today", {})
    today_main_net = today.get("main_net_amount") or 0
    today_main_pct = today.get("main_net_pct") or 0

    # 5日累计(东财数据有,THS fallback 没有)
    cum_5d = capital_flow.get("cumulative", {}).get("5d", {})
    cum_5d_net = cum_5d.get("main_net_amount") or 0

    evidence: List[str] = []
    evidence.append(f"短期主峰{peak_migration}")
    evidence.append(f"集中度{conc_trend}")
    evidence.append(f"主力资金{cf_action}({cf_strength})")
    evidence.append(f"今日主力净额 {today_main_net:,.0f}({today_main_pct}%)")
    if cf_trend_available:
        evidence.append(f"连续 {abs(cf_consecutive)} 日{'流入' if cf_consecutive > 0 else '流出' if cf_consecutive < 0 else '无持续'}")
        if cum_5d_net:
            evidence.append(f"5日累计主力净额 {cum_5d_net:,.0f}")
    else:
        evidence.append("资金流仅今日数据(THS 降级源),无趋势/累计")

    # ---- 交叉验证逻辑 ----
    # 资金方向:inflow(净流入)/ outflow(净流出)/ neutral
    capital_inflow = cf_action == "吸筹" or (cf_consecutive > 0 and today_main_net > 0)
    capital_outflow = cf_action == "派发" or (cf_consecutive < 0 and today_main_net < 0)

    # 筹码方向
    peak_up = peak_migration == "向上迁移"
    peak_down = peak_migration == "向下迁移"
    conc_up = conc_trend == "上升"
    conc_down = conc_trend == "下降"

    intent = "中性"
    confidence = "中"
    interpretation = ""
    action_hint = ""

    # 强吸筹:筹码集中 + 资金流入(双确认)
    if capital_inflow and (conc_up or (peak_up and conc_up)):
        intent = "强吸筹"
        confidence = "高"
        interpretation = "筹码集中 + 主力资金流入,双确认吸筹,主力在收集筹码"
        action_hint = "可逢低跟随,止损设在主力成本区下方"
    # 强吸筹:主峰上移 + 资金流入(高位承接)
    elif capital_inflow and peak_up and not conc_down:
        intent = "强吸筹"
        confidence = "高"
        interpretation = "主峰上移 + 资金流入,高位承接增强且资金进场,主力向上拉升"
        action_hint = "可持有/轻仓跟进,注意上方压力位"
    # 派发:主峰上移 + 资金流出(高位换手出货)
    elif capital_outflow and peak_up:
        intent = "派发"
        confidence = "高"
        interpretation = "主峰上移 + 主力资金流出,高位换手出货,散户接盘主力撤离"
        action_hint = "警惕减仓,不追高,跌破主峰支撑离场"
    # 强派发:集中度下降 + 资金流出(主力+散户双撤)
    elif capital_outflow and conc_down:
        intent = "强派发"
        confidence = "高"
        interpretation = "筹码分散 + 主力资金流出,主力散户双撤,趋势走弱"
        action_hint = "立即减仓,反弹不参与,等筹码重新集中"
    # 派发:集中度上升 + 资金流出(派发尾声,散户接盘)
    elif capital_outflow and conc_up:
        intent = "派发"
        confidence = "中"
        interpretation = "筹码集中但主力流出,可能是派发尾声散户接盘,或换庄"
        action_hint = "观望,等资金信号转正再介入"
    # 低位吸筹:主峰稳定 + 资金流入
    elif capital_inflow and (not peak_up and not peak_down):
        intent = "吸筹"
        confidence = "中"
        interpretation = "主峰稳定 + 资金流入,主力在低位悄然收集筹码"
        action_hint = "可分批低吸,等放量突破再加仓"
    # 暗中派发:主峰稳定 + 资金流出
    elif capital_outflow and (not peak_up and not peak_down):
        intent = "派发"
        confidence = "中"
        interpretation = "主峰稳定但资金流出,主力在震荡中悄然撤离"
        action_hint = "减仓防范,破位果断止损"
    # 抄底进场:主峰下移 + 资金流入(恐慌盘接货)
    elif capital_inflow and peak_down:
        intent = "吸筹"
        confidence = "中"
        interpretation = "主峰下移 + 资金流入,主力在恐慌中接货抄底"
        action_hint = "等企稳信号再进场,不抢反弹"
    # 弱势阴跌:主峰下移 + 资金流出
    elif capital_outflow and peak_down:
        intent = "强派发"
        confidence = "高"
        interpretation = "主峰下移 + 资金流出,无承接阴跌,远离"
        action_hint = "不参与,等资金流入 + 主峰企稳"
    # 矛盾场景
    elif (peak_up and conc_down) or (peak_down and conc_up):
        intent = "矛盾"
        confidence = "低"
        interpretation = "筹码信号矛盾(主峰与集中度方向不一致),需更多数据确认"
        action_hint = "观望,等信号一致再操作"
    # 中性
    else:
        intent = "中性"
        confidence = "低"
        interpretation = "筹码与资金信号均不明确,无方向"
        action_hint = "观望为主"

    # 资金强度调整置信度
    if cf_strength == "强" and confidence == "中":
        confidence = "高"
    if cf_strength == "弱" and confidence == "高" and intent not in ("强吸筹", "强派发"):
        confidence = "中"

    # THS 降级源只有今日数据,置信度最高为"中"(无趋势/连续性确认)
    if cf_source == "ths" and confidence == "高":
        confidence = "中"
        interpretation += "(注:资金流仅今日快照,置信度受限于数据源)"

    return {
        "available": True,
        "main_force_intent": intent,
        "confidence": confidence,
        "evidence": evidence,
        "interpretation": interpretation,
        "action_hint": action_hint,
        "inputs": {
            "peak_migration": peak_migration,
            "concentration_trend": conc_trend,
            "cyqk_trend": cyqk_trend,
            "capital_action": cf_action,
            "capital_strength": cf_strength,
            "today_main_net": round(today_main_net, 0),
            "consecutive_days": cf_consecutive,
        },
    }
