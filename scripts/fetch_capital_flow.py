"""东方财富个股主力资金流拉取 + 吸筹/派发信号判定。

数据源优先级:
  1. 东方财富 push2his.eastmoney.com/api/qt/stock/fflow/daykline/get
     返回 ~121 天日线资金流,klines 数组每行 15 字段:
     date, main_net, small_net, mid_net, large_net, super_large_net,
     main_pct, small_pct, mid_pct, large_pct, super_large_pct,
     close, pct_chg, "", ""
     主力 = 超大单(单笔>100万) + 大单(20-100万),东财标准
  2. 同花顺 data.10jqka.com.cn/funds/ggzjl/ (fallback)
     当东财 push2his 对带 secid 的个股 API 做反爬拦截时启用。
     只返回今日全市场快照(流入/流出/净额/成交额),无日线序列。
     通过 akshare.stock_fund_flow_individual(symbol="即时") 拉取,需 hexin-v token。

判定规则(东财数据):
  - 吸筹:连续≥3日主力净流入 + 5日均主力净占比≥5%
  - 派发:连续≥3日主力净流出 + 5日均主力净占比≤-5%
  - 强吸筹/强派发:连续≥5日 + 5日均占比≥10%(吸筹)/≤-10%(派发)
  - 中性:其他

判定规则(THS fallback):
  - 吸筹:今日净额 > 0 且 占比≥5%
  - 派发:今日净额 < 0 且 占比≤-5%
  - 强度:|占比|≥10%=强, ≥5%=中, <5%=弱
  - 中性:|占比|<5%

复用 fetch_data 的网络层(Session + 限流 + 重试 + 缓存)。
"""
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd

from fetch_data import (
    _session, _throttle, _retry_with_backoff,
    _cache_get, _cache_set,
    normalize_code,
)


# ---------- 数据拉取 ----------

_FLOW_API = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"


def _detect_secid(code: str) -> str:
    """按代码首位推断 eastmoney secid 格式(0.XXXXXX for SZ/BJ, 1.XXXXXX for SH)。"""
    code = normalize_code(code)
    if code.startswith("6"):
        return f"1.{code}"  # SH
    if code.startswith(("0", "3")):
        return f"0.{code}"  # SZ
    if code.startswith(("8", "4")):
        return f"0.{code}"  # BJ(东财 BJ 也用 0 前缀)
    raise ValueError(f"无法识别市场: {code}")


def _fetch_raw(code: str) -> List[Dict[str, Any]]:
    """直接调东方财富资金流 API,返回原始 klines 列表。"""
    secid = _detect_secid(code)
    params = {
        "lmt": 0,
        "klt": 101,  # 日线
        "secid": secid,
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
    }
    headers = {"Referer": "https://data.eastmoney.com/zjlx/"}

    def _do():
        _throttle()
        r = _session.get(_FLOW_API, params=params, timeout=(10, 30), headers=headers)
        r.raise_for_status()
        data = r.json()
        if not data or "data" not in data or not data["data"]:
            raise RuntimeError(f"资金流响应为空: {code}")
        klines = data["data"].get("klines") or []
        if not klines:
            raise RuntimeError(f"资金流 klines 为空: {code}")
        return klines

    return _retry_with_backoff(_do, max_retries=6, base_delay=3.0)


def _parse_klines(klines: List[str]) -> pd.DataFrame:
    """解析 klines 字符串列表为 DataFrame。

    klines 格式:date,main_net,small_net,mid_net,large_net,super_large_net,
                main_pct,small_pct,mid_pct,large_pct,super_large_pct,close,pct_chg,"",""
    """
    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 13:
            continue
        try:
            rows.append({
                "date": parts[0],
                "main_net": float(parts[1]),
                "small_net": float(parts[2]),
                "mid_net": float(parts[3]),
                "large_net": float(parts[4]),
                "super_large_net": float(parts[5]),
                "main_pct": float(parts[6]),
                "small_pct": float(parts[7]),
                "mid_pct": float(parts[8]),
                "large_pct": float(parts[9]),
                "super_large_pct": float(parts[10]),
                "close": float(parts[11]),
                "pct_chg": float(parts[12]),
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        raise RuntimeError("klines 解析失败:无有效行")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------- 统计计算 ----------

def _compute_cumulative(df: pd.DataFrame, windows: List[int] = None) -> Dict[str, Any]:
    """滚动累计:每个窗口的主力净额累计 + 平均占比 + 流入/流出天数。"""
    if windows is None:
        windows = [5, 10, 20]
    out: Dict[str, Any] = {}
    for w in windows:
        if len(df) < w:
            out[f"{w}d"] = {"available": False, "reason": f"数据不足 {len(df)} < {w}"}
            continue
        seg = df.tail(w)
        main_net_sum = float(seg["main_net"].sum())
        main_pct_mean = float(seg["main_pct"].mean())
        days_inflow = int((seg["main_net"] > 0).sum())
        days_outflow = int((seg["main_net"] < 0).sum())
        super_large_sum = float(seg["super_large_net"].sum())
        large_sum = float(seg["large_net"].sum())
        out[f"{w}d"] = {
            "available": True,
            "window": w,
            "main_net_amount": round(main_net_sum, 0),
            "main_pct_mean": round(main_pct_mean, 2),
            "days_inflow": days_inflow,
            "days_outflow": days_outflow,
            "super_large_net_sum": round(super_large_sum, 0),
            "large_net_sum": round(large_sum, 0),
        }
    return out


def _compute_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """主力净流入 MA5/MA20 + 金叉/死叉 + 连续流入/流出天数。"""
    if len(df) < 20:
        return {"available": False, "reason": f"数据不足 {len(df)} < 20"}

    main_net = df["main_net"]
    ma5 = float(main_net.tail(5).mean())
    ma20 = float(main_net.tail(20).mean())

    # 金叉/死叉:对比昨日 MA5 vs MA20 和今日 MA5 vs MA20
    if len(df) >= 21:
        prev_ma5 = float(main_net.iloc[-6:-1].mean())
        prev_ma20 = float(main_net.iloc[-21:-1].mean())
        if prev_ma5 <= prev_ma20 and ma5 > ma20:
            cross = "金叉"
        elif prev_ma5 >= prev_ma20 and ma5 < ma20:
            cross = "死叉"
        else:
            cross = "无交叉"
    else:
        cross = "无交叉"

    # 连续流入/流出天数(从今日倒推)
    consecutive = 0
    sign = 0
    for v in main_net.iloc[::-1]:
        v = float(v)
        if sign == 0:
            if v > 0:
                sign = 1
            elif v < 0:
                sign = -1
            else:
                break
        if v * sign > 0:
            consecutive += 1
        else:
            break
    consecutive_days = sign * consecutive
    if consecutive >= 5:
        label = "持续流入" if sign > 0 else "持续流出"
    elif consecutive >= 3:
        label = "连续流入" if sign > 0 else "连续流出"
    else:
        label = "无持续"

    return {
        "available": True,
        "main_net_ma5": round(ma5, 0),
        "main_net_ma20": round(ma20, 0),
        "ma_cross": cross,
        "consecutive_days": consecutive_days,
        "consecutive_label": label,
    }


def _compute_signals(df: pd.DataFrame, trend: Dict[str, Any],
                     cumulative: Dict[str, Any]) -> Dict[str, Any]:
    """主力意图判定:吸筹/派发/中性 + 强度。"""
    evidence: List[str] = []
    if not trend.get("available"):
        return {
            "main_force_action": "中性",
            "strength": "弱",
            "evidence": ["趋势数据不足"],
            "interpretation": "资金流数据不足,无法判定主力意图",
        }

    consecutive = trend.get("consecutive_days", 0)
    ma5_pct = cumulative.get("5d", {}).get("main_pct_mean", 0) or 0
    ma5_net = trend.get("main_net_ma5", 0) or 0
    cross = trend.get("ma_cross", "无交叉")
    abs_consec = abs(consecutive)

    evidence.append(f"连续 {abs_consec} 日{'净流入' if consecutive > 0 else '净流出' if consecutive < 0 else '无持续'}")
    evidence.append(f"5日均主力净占比 {ma5_pct}%")
    evidence.append(f"5日均主力净额 {ma5_net:,.0f}")
    evidence.append(f"MA5/MA20 {cross}")

    # 判定(用 abs_consec 比较,因为 consecutive 流出为负)
    action = "中性"
    strength = "弱"

    if abs_consec >= 5 and ma5_pct >= 10:
        action = "吸筹"
        strength = "强"
    elif abs_consec >= 5 and ma5_pct <= -10:
        action = "派发"
        strength = "强"
    elif abs_consec >= 3 and ma5_pct >= 5:
        action = "吸筹"
        strength = "中"
    elif abs_consec >= 3 and ma5_pct <= -5:
        action = "派发"
        strength = "中"
    elif ma5_pct >= 5 and ma5_net > 0:
        action = "吸筹"
        strength = "弱"
    elif ma5_pct <= -5 and ma5_net < 0:
        action = "派发"
        strength = "弱"

    # MA 金叉/死叉增强信号
    if cross == "金叉" and action == "吸筹":
        strength = "强" if strength == "中" else strength
        evidence.append("MA5 上穿 MA20,资金加速进场")
    if cross == "死叉" and action == "派发":
        strength = "强" if strength == "中" else strength
        evidence.append("MA5 下穿 MA20,资金加速离场")

    if action == "中性":
        interpretation = "主力资金无明显方向,观望"
    elif action == "吸筹":
        interpretation = f"主力资金{'强' if strength == '强' else ''}吸筹:连续净流入 + 占比达标,资金进场收集筹码"
    else:  # 派发
        interpretation = f"主力资金{'强' if strength == '强' else ''}派发:连续净流出 + 占比达标,资金撤离出货"

    return {
        "main_force_action": action,
        "strength": strength,
        "evidence": evidence,
        "interpretation": interpretation,
    }


# ---------- THS fallback(东财反爬时的降级路径) ----------

_THS_CACHE_KEY = "ths_realtime_all_stocks"


def _fetch_ths_realtime_all() -> Optional[pd.DataFrame]:
    """拉取同花顺"即时"全市场个股资金流(5194 只,~15s)。

    通过 akshare.stock_fund_flow_individual(symbol="即时") 调用,
    内部使用 hexin-v token(JS 挑战)绕过反爬。

    Returns:
        DataFrame[code, name, price, pct_chg, turnover_rate,
                 inflow, outflow, net, turnover] 或 None(失败)
    """
    cached = _cache_get("capital_flow", _THS_CACHE_KEY)
    if cached is not None:
        try:
            return pd.DataFrame(cached)
        except Exception:
            pass

    try:
        import akshare as ak
    except ImportError:
        return None

    try:
        df = ak.stock_fund_flow_individual(symbol="即时")
    except Exception as e:
        return None

    if df is None or len(df) == 0:
        return None

    # 归一化列名(akshare 即时返回中文列名,已去单位后缀)
    col_map = {
        "股票代码": "code_raw",
        "股票简称": "name",
        "最新价": "price",
        "涨跌幅": "pct_chg",
        "换手率": "turnover_rate",
        "流入资金": "inflow",
        "流出资金": "outflow",
        "净额": "net",
        "成交额": "turnover",
    }
    df = df.rename(columns=col_map)

    # 解析中文金额("1.05亿" / "3596.95万")为元
    def _parse_amount(s):
        if pd.isna(s) or s == "" or s == "0.00":
            return 0.0
        s = str(s).strip()
        try:
            if s.endswith("亿"):
                return float(s[:-1]) * 1e8
            if s.endswith("万"):
                return float(s[:-1]) * 1e4
            return float(s)
        except ValueError:
            return 0.0

    for col in ["inflow", "outflow", "net", "turnover"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_amount)

    # 解析百分比
    for col in ["pct_chg", "turnover_rate"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 归一化代码为 6 位(THS 返回的 code 是 int 去前导零)
    def _norm_code(c):
        try:
            return f"{int(c):06d}"
        except (ValueError, TypeError):
            return str(c).zfill(6)
    df["code"] = df["code_raw"].apply(_norm_code)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    keep = ["code", "name", "price", "pct_chg", "turnover_rate",
            "inflow", "outflow", "net", "turnover"]
    df = df[[c for c in keep if c in df.columns]]

    # 缓存全市场表(4h TTL,与 capital_flow 同)
    _cache_set("capital_flow", _THS_CACHE_KEY, df.to_dict(orient="records"))
    return df


def _parse_ths_realtime(code: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """从 THS 全市场表中过滤目标股票,构建简化 result。"""
    code = normalize_code(code)
    row = df[df["code"] == code]
    if row.empty:
        return None

    r = row.iloc[0]
    net = float(r.get("net", 0) or 0)
    turnover = float(r.get("turnover", 0) or 0)
    inflow = float(r.get("inflow", 0) or 0)
    outflow = float(r.get("outflow", 0) or 0)
    price = float(r["price"]) if pd.notna(r.get("price")) else None
    pct_chg = float(r["pct_chg"]) if pd.notna(r.get("pct_chg")) else None

    main_pct = (net / turnover * 100) if turnover > 0 else 0.0

    today = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "close": price,
        "pct_chg": pct_chg,
        "main_net_amount": round(net, 0),
        "main_net_pct": round(main_pct, 2),
        "inflow_amount": round(inflow, 0),
        "outflow_amount": round(outflow, 0),
        "turnover": round(turnover, 0),
    }

    # 简化信号:只用今日数据
    if net > 0 and main_pct >= 10:
        action, strength = "吸筹", "强"
    elif net < 0 and main_pct <= -10:
        action, strength = "派发", "强"
    elif net > 0 and main_pct >= 5:
        action, strength = "吸筹", "中"
    elif net < 0 and main_pct <= -5:
        action, strength = "派发", "中"
    elif net > 0:
        action, strength = "吸筹", "弱"
    elif net < 0:
        action, strength = "派发", "弱"
    else:
        action, strength = "中性", "弱"

    evidence = [
        f"今日主力净额 {net:,.0f} 元({main_pct:.2f}% of 成交额)",
        f"流入 {inflow:,.0f} / 流出 {outflow:,.0f}",
        f"成交额 {turnover:,.0f}",
        "数据源:同花顺(东财反爬降级)",
    ]
    if action == "中性":
        interpretation = "主力资金今日无明显方向(THS 即时数据)"
    elif action == "吸筹":
        interpretation = f"主力资金今日净流入 {net:,.0f} 元,占比 {main_pct:.2f}%"
    else:
        interpretation = f"主力资金今日净流出 {abs(net):,.0f} 元,占比 {abs(main_pct):.2f}%"

    return {
        "code": code,
        "available": True,
        "source": "ths",
        "days_returned": 1,
        "today": today,
        "cumulative": {
            "today": {
                "available": True,
                "main_net_amount": round(net, 0),
                "main_pct": round(main_pct, 2),
                "inflow_amount": round(inflow, 0),
                "outflow_amount": round(outflow, 0),
                "turnover": round(turnover, 0),
            }
        },
        "trend": {
            "available": False,
            "reason": "THS 即时数据无日线序列,无法计算 MA/连续天数",
        },
        "signals": {
            "main_force_action": action,
            "strength": strength,
            "evidence": evidence,
            "interpretation": interpretation,
        },
    }


# ---------- 主入口 ----------

def fetch_capital_flow(code: str, days: int = 60) -> Dict[str, Any]:
    """拉取个股主力资金流数据,计算吸筹/派发信号。

    Args:
        code: 股票代码(6 位,如 002472)
        days: 返回的最近天数(原始数据 ~121 天,默认截取 60 天分析)

    Returns:
        {
            "code": "002472",
            "available": True,
            "days_returned": 60,
            "today": {...},
            "cumulative": {"5d": {...}, "10d": {...}, "20d": {...}},
            "trend": {...},
            "signals": {...},
        }
    """
    code = normalize_code(code)
    cache_key = hashlib.md5(f"{code}_{days}".encode()).hexdigest()
    cached = _cache_get("capital_flow", cache_key)
    if cached is not None:
        return cached

    try:
        klines = _fetch_raw(code)
        df = _parse_klines(klines)
    except Exception as e:
        # 东财失败 -> THS fallback
        ths_df = _fetch_ths_realtime_all()
        if ths_df is not None and not ths_df.empty:
            ths_result = _parse_ths_realtime(code, ths_df)
            if ths_result is not None:
                _cache_set("capital_flow", cache_key, ths_result)
                return ths_result
        return {"code": code, "available": False, "error": f"资金流拉取失败: {e}"}

    if len(df) < 5:
        return {"code": code, "available": False, "error": f"数据不足 {len(df)} 行"}

    # 截取最近 days 天用于分析(但保留全部原始数据用于 MA20)
    analyze_df = df.tail(days) if len(df) > days else df

    cumulative = _compute_cumulative(analyze_df)
    trend = _compute_trend(analyze_df)
    signals = _compute_signals(analyze_df, trend, cumulative)

    # 今日数据
    today_row = df.iloc[-1]
    today = {
        "date": today_row["date"].strftime("%Y-%m-%d") if pd.notna(today_row["date"]) else None,
        "close": float(today_row["close"]) if pd.notna(today_row["close"]) else None,
        "pct_chg": float(today_row["pct_chg"]) if pd.notna(today_row["pct_chg"]) else None,
        "main_net_amount": float(today_row["main_net"]) if pd.notna(today_row["main_net"]) else None,
        "main_net_pct": float(today_row["main_pct"]) if pd.notna(today_row["main_pct"]) else None,
        "super_large_net": float(today_row["super_large_net"]) if pd.notna(today_row["super_large_net"]) else None,
        "large_net": float(today_row["large_net"]) if pd.notna(today_row["large_net"]) else None,
        "mid_net": float(today_row["mid_net"]) if pd.notna(today_row["mid_net"]) else None,
        "small_net": float(today_row["small_net"]) if pd.notna(today_row["small_net"]) else None,
    }

    result = {
        "code": code,
        "available": True,
        "days_returned": len(analyze_df),
        "total_days_available": len(df),
        "today": today,
        "cumulative": cumulative,
        "trend": trend,
        "signals": signals,
    }

    _cache_set("capital_flow", cache_key, result)
    return result


if __name__ == "__main__":
    import sys
    import json
    code = sys.argv[1] if len(sys.argv) > 1 else "002472"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    result = fetch_capital_flow(code, days=days)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
