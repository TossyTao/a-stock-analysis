"""矩阵 14:日内走势诊断。

数据源:ak.stock_zh_a_minute(period="1") 1 分钟 K 线。

四大维度:
  1. 路径效率 = |收盘-开盘| / (最高-最低)
     - 高(>0.7):价格路径平滑,资金一致推动
     - 低(<0.3):反复反转,戏精型
  2. 收盘位置 = (收盘-最低) / (最高-最低)
     - 高(>0.8):买方控盘,收盘近高点
     - 低(<0.5):卖方主导,收盘近低点
  3. 量价转化 = 涨幅 / 换手率(无换手则用 涨幅/百万股)
     - 衡量单位换手推动的涨幅,高=资金效率高
  4. 尾盘涨幅占比 = (14:30 后涨幅) / 全天涨幅
     - <30%:持续型(老黄牛)
     - >50%:尾盘突击型(警惕)

四类走势分类:
  - 🐂 老黄牛型:路径效率>0.7 + 收盘位置>0.8 + 尾盘占比<30%
  - 🎭 戏精型:振幅>3% + 路径效率<0.3
  - 📉 装忙型:成交量>20日均量×1.5 + 涨幅<1%
  - ⚡ 尾盘突击型:尾盘占比>50%
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.dirname(__file__))
from fetch_data import (
    _cache_get, _cache_set, _retry_with_backoff, normalize_code, code_with_market,
)


def _fetch_minute_df(code: str):
    """拉取 1 分钟 K 线,带缓存 + 重试。

    缓存原始 DataFrame(网络昂贵),不缓存计算结果(参数可能变)。
    """
    import akshare as ak
    sym = code_with_market(code)

    cache_key = f"raw_{code}"
    cached = _cache_get("intraday", cache_key)
    if cached is not None:
        import pandas as pd
        try:
            return pd.read_pickle(cached) if isinstance(cached, str) else cached
        except Exception:
            pass

    def _fetch():
        return ak.stock_zh_a_minute(symbol=sym, period="1", adjust="")

    df = _retry_with_backoff(_fetch, max_retries=3, base_delay=1.0)
    if df is None or len(df) == 0:
        return None
    # 缓存原始 DataFrame(直接 pickle,避免路径问题)
    _cache_set("intraday", cache_key, df)
    return df


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _parse_minute_df(df) -> List[Dict[str, Any]]:
    """将 akshare DataFrame 转为统一格式的分钟列表(按时间升序)。

    原始列:day(时间字符串 '2026-07-11 09:31')/ open / high / low / close / volume / amount
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return []
    # 列名标准化
    cols = list(df.columns)
    # akshare 返回的列通常为 ['day', 'open', 'high', 'low', 'close', 'volume', 'amount']
    df = df.copy()
    df["day"] = df["day"].astype(str)
    # 按时间升序
    df = df.sort_values("day").reset_index(drop=True)
    out = []
    for _, row in df.iterrows():
        out.append({
            "time": str(row["day"]),
            "open": _to_float(row["open"]),
            "high": _to_float(row["high"]),
            "low": _to_float(row["low"]),
            "close": _to_float(row["close"]),
            "volume": _to_float(row["volume"]),
            "amount": _to_float(row.get("amount", 0)),
        })
    return out


def _filter_by_date(minutes: List[Dict], date: Optional[str]) -> tuple:
    """筛选指定日期的分钟数据。

    返回 (filtered_minutes, date_str)。若 date 为 None,取最后一个交易日。
    """
    if not minutes:
        return [], None
    if date is None:
        # 取最后一根 K 线的日期
        last_time = minutes[-1]["time"]
        # 时间格式 '2026-07-11 09:31' 或 '2026-07-11 09:31:00'
        target_date = last_time[:10]
    else:
        target_date = date[:10]
    filtered = [m for m in minutes if m["time"][:10] == target_date]
    return filtered, target_date


def _compute_metrics(minutes: List[Dict], daily_volume_20avg: Optional[float] = None,
                     free_float_shares: Optional[float] = None) -> Dict[str, Any]:
    """计算四大维度指标。"""
    if not minutes or len(minutes) < 2:
        return {"available": False, "error": "分钟数据不足"}

    open_price = minutes[0]["open"]
    close_price = minutes[-1]["close"]
    high = max(m["high"] for m in minutes)
    low = min(m["low"] for m in minutes)
    total_vol = sum(m["volume"] for m in minutes)
    total_amount = sum(m["amount"] for m in minutes)

    # 涨幅
    pct_chg = (close_price - open_price) / open_price * 100 if open_price > 0 else 0.0
    # 振幅
    amplitude = (high - low) / open_price * 100 if open_price > 0 else 0.0

    # 1. 路径效率 = |收盘-开盘| / (最高-最低)
    price_range = high - low
    path_efficiency = abs(close_price - open_price) / price_range if price_range > 0 else 0.0

    # 2. 收盘位置 = (收盘-最低) / (最高-最低)
    close_position = (close_price - low) / price_range if price_range > 0 else 0.0

    # 3. 量价转化 = 涨幅 / 换手率(无换手用 涨幅/百万股)
    vol_price_ratio = 0.0
    if free_float_shares and free_float_shares > 0:
        # 换手率 = 总成交量(股) / 流通股本 × 100
        turnover_pct = total_vol / free_float_shares * 100
        vol_price_ratio = pct_chg / turnover_pct if turnover_pct > 0 else 0.0
    else:
        # 无流通股本:用 涨幅/百万股
        vol_millions = total_vol / 1_000_000 if total_vol > 0 else 0.001
        vol_price_ratio = pct_chg / vol_millions if vol_millions > 0 else 0.0

    # 4. 尾盘涨幅占比 = (14:30 后涨幅) / 全天涨幅
    # 14:30 之前的最后一根 K 线作为基准
    tail_baseline_price = None
    for m in minutes:
        # 时间格式 '2026-07-11 14:30' 或 '2026-07-11 14:30:00'
        time_str = m["time"]
        hhmm = time_str[11:16] if len(time_str) >= 16 else ""
        if hhmm <= "14:30":
            tail_baseline_price = m["close"]
    if tail_baseline_price is None or tail_baseline_price <= 0:
        tail_ratio = 0.0
    elif abs(close_price - open_price) < 1e-9:
        # 全天涨幅为 0,无法算占比
        tail_ratio = 0.0
    else:
        tail_gain = close_price - tail_baseline_price
        full_gain = close_price - open_price
        # 尾盘占比 = 尾盘涨幅 / 全天涨幅(同方向为正,反方向为负)
        tail_ratio = tail_gain / full_gain if abs(full_gain) > 1e-9 else 0.0

    return {
        "available": True,
        "metrics": {
            "path_efficiency": round(path_efficiency, 3),
            "close_position": round(close_position, 3),
            "vol_price_ratio": round(vol_price_ratio, 4),
            "tail_ratio": round(tail_ratio, 3),
            "amplitude": round(amplitude, 2),
            "pct_chg": round(pct_chg, 2),
            "open": round(open_price, 2),
            "close": round(close_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "total_volume": total_vol,
            "total_amount": total_amount,
            "daily_volume_20avg": daily_volume_20avg,
            "minute_count": len(minutes),
        },
    }


def _classify(metrics: Dict[str, Any]) -> tuple:
    """四类走势分类。返回 (classification, interpretation, action_hint)。"""
    m = metrics["metrics"]
    pe = m["path_efficiency"]
    cp = m["close_position"]
    tr = m["tail_ratio"]
    amp = m["amplitude"]
    pct = m["pct_chg"]
    total_vol = m["total_volume"]
    avg_20 = m.get("daily_volume_20avg")

    # 优先级:尾盘突击 > 老黄牛 > 戏精 > 装忙
    # 尾盘突击型:尾盘占比>50%(且全天有涨幅)
    if tr > 0.5 and abs(pct) > 0.1:
        return ("尾盘突击型",
                "尾盘拉升占比过高(>50%),全天涨幅主要靠尾盘,资金急于做收盘价,⚠️ 警惕",
                "⚠️ 警惕,不追")

    # 老黄牛型:路径效率>0.7 + 收盘位置>0.8 + 尾盘占比<30%
    if pe > 0.7 and cp > 0.8 and tr < 0.3:
        return ("老黄牛型",
                "路径平滑 + 收盘高位 + 涨幅均匀分布,资金一致买盘持续,✅ 跟随",
                "✅ 跟随")

    # 戏精型:振幅>3% + 路径效率<0.3(反复反转)
    if amp > 3.0 and pe < 0.3:
        return ("戏精型",
                "振幅大但路径反复,多空拉锯,收盘接近开盘,⚠️ 观望",
                "⚠️ 观望")

    # 装忙型:成交量>20日均量×1.5 + 涨幅<1%
    if avg_20 and avg_20 > 0 and total_vol > avg_20 * 1.5 and abs(pct) < 1.0:
        return ("装忙型",
                "成交量放大但涨幅小,放量不涨,主力对倒或出货,❌ 回避",
                "❌ 回避")

    # 混合型:不满足任一典型分类
    return ("混合型",
            f"路径效率{pe:.2f} + 收盘位置{cp:.2f} + 尾盘占比{tr:.2f},无典型走势特征",
            "中性观察")


def analyze_intraday(code: str, date: Optional[str] = None,
                     daily_volume_20avg: Optional[float] = None,
                     free_float_shares: Optional[float] = None) -> Dict[str, Any]:
    """日内走势诊断(矩阵 14)。

    参数:
      code: 股票代码
      date: 指定日期(YYYY-MM-DD),None 取最近交易日
      daily_volume_20avg: 20 日均成交量(股),用于装忙型识别。None 则跳过该维度
      free_float_shares: 流通股本,用于换手率计算。None 则用 涨幅/百万股 替代

    返回:dict,见模块 docstring
    """
    code = normalize_code(code)
    # 不缓存计算结果(参数 daily_volume_20avg/free_float_shares 可能在不同调用中不同)
    # _fetch_minute_df 内部已缓存原始 DataFrame
    try:
        df = _fetch_minute_df(code)
    except Exception as e:
        return {"available": False, "code": code, "error": f"分钟数据拉取失败: {e}"}

    if df is None or len(df) == 0:
        return {"available": False, "code": code, "error": "无分钟数据"}

    minutes = _parse_minute_df(df)
    filtered, target_date = _filter_by_date(minutes, date)
    if not filtered or len(filtered) < 10:
        return {"available": False, "code": code, "error": f"{target_date} 分钟数据不足"}

    result = _compute_metrics(filtered, daily_volume_20avg, free_float_shares)
    if not result.get("available"):
        return {"available": False, "code": code, "error": result.get("error", "计算失败")}

    classification, interp, action = _classify(result)
    out = {
        "available": True,
        "code": code,
        "date": target_date,
        "metrics": result["metrics"],
        "classification": classification,
        "interpretation": interp,
        "action_hint": action,
    }
    return out


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser(description="矩阵 14:日内走势诊断")
    p.add_argument("code", help="股票代码")
    p.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD,默认最近交易日")
    args = p.parse_args()
    r = analyze_intraday(args.code, args.date)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
