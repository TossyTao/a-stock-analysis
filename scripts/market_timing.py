"""矩阵 13:市场择时五维度投票。

五维度:
  1. 估值(ERP)= 1/PE - 十年国债,高位看多(股票便宜)
  2. 资金(融资布林带)= 融资买入额 20 日布林带,突破上轨看多
  3. 技术(指数布林带 + 市场广度)= 上证 20 日布林带 + 上涨家数占比
  4. 情绪(PCR + 基差)= 认沽/认购比 + 期货基差(IV/期货持仓降级)
  5. 基本面(反向底部)= CPI+PMI 双弱 -> 高赔率底部信号

投票:≥4 看多 -> 100% 仓位 / 3 看多 -> 50% / ≤2 看多 -> 0%

数据源:
  - stock_index_pe_lg('沪深300') -> PE
  - bond_china_yield -> 十年国债
  - stock_margin_detail_sse/szse -> 融资买入额
  - stock_zh_index_daily('sh000001') -> 上证指数
  - stock_zh_a_spot -> 全市场行情(市场广度)
  - option_cffex_hs300_spot_sina('IOYYMM') -> 期权 PCR
  - futures_main_sina('IF0') -> 股指期货主力(基差)
  - macro_china_cpi_yearly / macro_china_pmi_yearly -> 宏观
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.dirname(__file__))
from fetch_data import _cache_get, _cache_set, _retry_with_backoff


# ---------- 工具 ----------

def _to_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _verdict_label(v: str) -> int:
    """verdict 转 +1/0/-1"""
    if "看多" in v:
        return 1
    if "看空" in v:
        return -1
    return 0


# ---------- 维度 1:估值 ERP ----------

def _dimension_valuation() -> Dict[str, Any]:
    """ERP = 1/PE - 十年国债。高位(股票便宜)看多。"""
    import akshare as ak
    try:
        pe_df = _retry_with_backoff(lambda: ak.stock_index_pe_lg(symbol="沪深300"), max_retries=3, base_delay=1.0)
        # bond_china_yield 仅返回近 6 个月数据,start_date 用最近半年
        bond_df = _retry_with_backoff(
            lambda: ak.bond_china_yield(start_date="20260101", end_date=datetime.now().strftime("%Y%m%d")),
            max_retries=3, base_delay=1.0,
        )
    except Exception as e:
        return {"available": False, "error": str(e)}

    if pe_df is None or len(pe_df) == 0 or bond_df is None or len(bond_df) == 0:
        return {"available": False, "error": "PE 或国债数据为空"}

    # PE:取最近 250 日的滚动市盈率
    pe_series = pe_df["滚动市盈率"].dropna().tail(250)
    if len(pe_series) < 30:
        return {"available": False, "error": "PE 数据不足"}
    pe_current = _to_float(pe_series.iloc[-1])
    pe_median = float(pe_series.median())
    pe_std = float(pe_series.std())

    # 十年国债:过滤"中债国债收益率曲线",取 10 年列
    bond_gov = bond_df[bond_df["曲线名称"] == "中债国债收益率曲线"]
    if len(bond_gov) == 0:
        return {"available": False, "error": "无国债收益率数据"}
    bond_10y = _to_float(bond_gov.iloc[-1]["10年"])
    # 百分比转小数:1.74% -> 0.0174
    bond_yield = bond_10y / 100 if bond_10y > 1 else bond_10y

    # ERP = 1/PE - 国债利率
    erp = 1.0 / pe_current - bond_yield if pe_current > 0 else 0.0

    # ERP 历史分布:用 PE 倒推各日 ERP(假设国债不变,粗略)
    erp_history = 1.0 / pe_series - bond_yield
    erp_median = float(erp_history.median())
    erp_std = float(erp_history.std())

    # 判定:ERP > 中位数 + 1σ -> 看多;< 中位数 - 1σ -> 看空
    if erp > erp_median + erp_std:
        verdict = "看多"
    elif erp < erp_median - erp_std:
        verdict = "看空"
    else:
        verdict = "中性"

    return {
        "available": True,
        "verdict": verdict,
        "detail": {
            "pe": round(pe_current, 2),
            "pe_median": round(pe_median, 2),
            "bond_10y": round(bond_yield * 100, 3),
            "erp": round(erp * 100, 3),
            "erp_median": round(erp_median * 100, 3),
            "erp_std": round(erp_std * 100, 3),
        },
        "interpretation": f"PE {pe_current:.1f}(中位 {pe_median:.1f}) / 十年国债 {bond_yield*100:.2f}% / ERP {erp*100:.2f}%(中位 {erp_median*100:.2f}% ± {erp_std*100:.2f}%) -> {verdict}",
    }


# ---------- 维度 2:融资布林带 ----------

def _dimension_capital() -> Dict[str, Any]:
    """融资买入额 20 日布林带。突破上轨看多,跌破下轨看空。"""
    import akshare as ak
    from datetime import timedelta
    # 拉近 30 个交易日的沪市融资数据(深市接口名可能变化,优先沪市)
    today = datetime.now()
    dates_to_try = []
    for d in range(50):
        dt = today - timedelta(days=d)
        # 跳过周末
        if dt.weekday() < 5:
            dates_to_try.append(dt.strftime("%Y%m%d"))
        if len(dates_to_try) >= 30:
            break

    daily_buys: List[float] = []
    last_date = None
    for date_str in dates_to_try:
        try:
            df = _retry_with_backoff(
                lambda d=date_str: ak.stock_margin_detail_sse(date=d),
                max_retries=2, base_delay=1.0,
            )
            if df is not None and len(df) > 0:
                total = df["融资买入额"].apply(_to_float).sum()
                daily_buys.append(total)
                last_date = date_str
        except Exception:
            continue
        if len(daily_buys) >= 25:
            break

    if len(daily_buys) < 20:
        return {"available": False, "error": f"融资数据不足({len(daily_buys)} 日)"}

    import statistics
    series = daily_buys[-20:]
    mu = statistics.mean(series)
    sigma = statistics.stdev(series) if len(series) > 1 else 0
    upper = mu + 2 * sigma
    lower = mu - 2 * sigma
    current = series[-1]

    if sigma > 0 and current > upper:
        verdict = "看多"
    elif sigma > 0 and current < lower:
        verdict = "看空"
    else:
        verdict = "中性"

    return {
        "available": True,
        "verdict": verdict,
        "detail": {
            "current": round(current / 1e8, 2),  # 亿元
            "boll_upper": round(upper / 1e8, 2),
            "boll_lower": round(lower / 1e8, 2),
            "boll_mid": round(mu / 1e8, 2),
            "date": last_date,
        },
        "interpretation": f"融资买入额 {current/1e8:.0f}亿(20日布林带 {lower/1e8:.0f}亿 ~ {upper/1e8:.0f}亿,中位 {mu/1e8:.0f}亿) -> {verdict}",
    }


def timedelta_days(days: int):
    """已废弃,保留兼容。请直接用 datetime.now() - timedelta(days=...)"""
    from datetime import timedelta
    return datetime.now() - timedelta(days=days)


# ---------- 维度 3:技术(指数布林带 + 市场广度)----------

def _dimension_technique() -> Dict[str, Any]:
    """上证 20 日布林带 + 全市场上涨家数占比。"""
    import akshare as ak
    try:
        idx_df = _retry_with_backoff(lambda: ak.stock_zh_index_daily(symbol="sh000001"), max_retries=3, base_delay=1.0)
    except Exception as e:
        return {"available": False, "error": str(e)}

    if idx_df is None or len(idx_df) < 20:
        return {"available": False, "error": "指数数据不足"}

    import statistics
    closes = idx_df["close"].apply(_to_float).tail(20).tolist()
    mu = statistics.mean(closes)
    sigma = statistics.stdev(closes) if len(closes) > 1 else 0
    upper = mu + 2 * sigma
    lower = mu - 2 * sigma
    current = closes[-1]

    if sigma > 0 and current > upper:
        idx_verdict = "看多"
    elif sigma > 0 and current < lower:
        idx_verdict = "看空"
    else:
        idx_verdict = "中性"

    # 市场广度:全市场上涨家数占比
    breadth = None
    breadth_verdict = "中性"
    try:
        spot_df = _retry_with_backoff(lambda: ak.stock_zh_a_spot(), max_retries=2, base_delay=1.0)
        if spot_df is not None and len(spot_df) > 0:
            pct = spot_df["涨跌幅"].apply(_to_float)
            up = int((pct > 0).sum())
            down = int((pct < 0).sum())
            flat = int((pct == 0).sum())
            total = up + down + flat
            breadth = up / total if total > 0 else 0.0
            if breadth > 0.7:
                breadth_verdict = "看多"
            elif breadth < 0.3:
                breadth_verdict = "看空"
            else:
                breadth_verdict = "中性"
    except Exception:
        pass

    # 综合:两个子指标都看多 -> 看多;都看空 -> 看空;否则中性
    if idx_verdict == breadth_verdict:
        verdict = idx_verdict
    elif idx_verdict == "中性" or breadth_verdict == "中性":
        verdict = idx_verdict if idx_verdict != "中性" else breadth_verdict
    else:
        verdict = "中性"

    return {
        "available": True,
        "verdict": verdict,
        "detail": {
            "index_close": round(current, 2),
            "index_boll_upper": round(upper, 2),
            "index_boll_lower": round(lower, 2),
            "index_boll_mid": round(mu, 2),
            "index_verdict": idx_verdict,
            "breadth": round(breadth, 3) if breadth is not None else None,
            "breadth_verdict": breadth_verdict,
        },
        "interpretation": f"上证 {current:.0f}(布林带 {lower:.0f}~{upper:.0f}) {idx_verdict};上涨家数占比 {breadth*100:.0f}% {breadth_verdict}" if breadth is not None else f"上证 {current:.0f} {idx_verdict}(广度数据缺失)",
    }


# ---------- 维度 4:情绪(PCR + 基差)----------

def _dimension_sentiment() -> Dict[str, Any]:
    """PCR(认沽/认购比)+ 期货基差。至少 2 因子同向才算有效。"""
    import akshare as ak
    # PCR:用沪深300股指期权(IO 主力)
    pcr = None
    try:
        # 当月合约 symbol:IO + YYMM
        now = datetime.now()
        sym = f"IO{now.strftime('%y%m')}"
        df = _retry_with_backoff(lambda: ak.option_cffex_hs300_spot_sina(symbol=sym), max_retries=2, base_delay=1.0)
        if df is not None and len(df) > 0:
            call_oi = df["看涨合约-持仓量"].apply(_to_float).sum()
            put_oi = df["看跌合约-持仓量"].apply(_to_float).sum()
            if call_oi > 0:
                pcr = put_oi / call_oi
    except Exception:
        pass

    # 基差:IF 主力(沪深300期货)- 沪深300 指数(高升水=乐观=看空,深贴水=悲观=看多)
    basis = None
    try:
        if_df = _retry_with_backoff(lambda: ak.futures_main_sina(symbol="IF0"), max_retries=2, base_delay=1.0)
        idx_df = _retry_with_backoff(lambda: ak.stock_zh_index_daily(symbol="sh000300"), max_retries=2, base_delay=1.0)
        if if_df is not None and idx_df is not None and len(if_df) > 0 and len(idx_df) > 0:
            if_close = _to_float(if_df.iloc[-1]["收盘价"])
            idx_close = _to_float(idx_df.iloc[-1]["close"])
            if idx_close > 0:
                basis = (if_close - idx_close) / idx_close
    except Exception:
        pass

    # 判定:PCR > 1.2 -> 看多(恐慌);PCR < 0.8 -> 看空(贪婪)
    pcr_verdict = "中性"
    if pcr is not None:
        if pcr > 1.2:
            pcr_verdict = "看多"
        elif pcr < 0.8:
            pcr_verdict = "看空"

    # 基差:高升水(>1%) -> 看空;深贴水(<-1%) -> 看多
    basis_verdict = "中性"
    if basis is not None:
        if basis > 0.01:
            basis_verdict = "看空"
        elif basis < -0.01:
            basis_verdict = "看多"

    # 至少 2 因子同向才算有效
    if pcr_verdict == basis_verdict and pcr_verdict != "中性":
        verdict = pcr_verdict
    else:
        verdict = "中性"

    detail = {
        "pcr": round(pcr, 3) if pcr is not None else None,
        "pcr_verdict": pcr_verdict,
        "basis": round(basis * 100, 3) if basis is not None else None,
        "basis_verdict": basis_verdict,
    }
    interp_parts = []
    if pcr is not None:
        interp_parts.append(f"PCR {pcr:.2f}({pcr_verdict})")
    if basis is not None:
        interp_parts.append(f"基差 {basis*100:.2f}%({basis_verdict})")
    interp = " + ".join(interp_parts) if interp_parts else "情绪数据缺失"
    interp += f" -> {verdict}"

    return {
        "available": pcr is not None or basis is not None,
        "verdict": verdict,
        "detail": detail,
        "interpretation": interp,
    }


# ---------- 维度 5:基本面(反向底部)----------

def _dimension_fundamental() -> Dict[str, Any]:
    """CPI + PMI 双弱 -> 高赔率底部信号(反向)。"""
    import akshare as ak
    try:
        cpi_df = _retry_with_backoff(lambda: ak.macro_china_cpi_yearly(), max_retries=2, base_delay=1.0)
        pmi_df = _retry_with_backoff(lambda: ak.macro_china_pmi_yearly(), max_retries=2, base_delay=1.0)
    except Exception as e:
        return {"available": False, "error": str(e)}

    if cpi_df is None or len(cpi_df) == 0 or pmi_df is None or len(pmi_df) == 0:
        return {"available": False, "error": "宏观数据为空"}

    # CPI:今值 < 前值 -> 走弱(看多,反向底部)
    cpi_now = _to_float(cpi_df.iloc[-1]["今值"])
    cpi_prev = _to_float(cpi_df.iloc[-1]["前值"])
    cpi_trend = "走弱" if cpi_now < cpi_prev else "走强"

    # PMI:今值 < 前值 -> 走弱(看多,反向底部)
    pmi_now = _to_float(pmi_df.iloc[-1]["今值"])
    pmi_prev = _to_float(pmi_df.iloc[-1]["前值"])
    pmi_trend = "走弱" if pmi_now < pmi_prev else "走强"

    # 双弱 -> 高赔率底部(看多);双强 -> 看空(过热)
    if cpi_trend == "走弱" and pmi_trend == "走弱":
        verdict = "看多"
        bottom_signal = True
    elif cpi_trend == "走强" and pmi_trend == "走强":
        verdict = "看空"
        bottom_signal = False
    else:
        verdict = "中性"
        bottom_signal = False

    return {
        "available": True,
        "verdict": verdict,
        "detail": {
            "cpi_now": cpi_now,
            "cpi_prev": cpi_prev,
            "cpi_trend": cpi_trend,
            "pmi_now": pmi_now,
            "pmi_prev": pmi_prev,
            "pmi_trend": pmi_trend,
            "bottom_signal": bottom_signal,
        },
        "interpretation": f"CPI {cpi_now}({cpi_trend}) / PMI {pmi_now}({pmi_trend}) -> {verdict}{'(反向底部)' if bottom_signal else ''}",
    }


# ---------- 主函数 ----------

def _vote_to_position(bullish: int, neutral: int, bearish: int) -> Dict[str, str]:
    """投票结果转仓位建议。可测试纯逻辑函数。

    规则(5 维度):
      - bullish >= 4 -> 100% 看多(强多)
      - bullish == 3 -> 50% 中性偏多
      - bearish >= 4 -> 0% 看空(强空)
      - bearish == 3 -> 20% 中性偏空
      - bullish == 2 -> 50% 中性偏多(偏多)
      - bullish == 1 -> 30% 中性
      - bullish == 0 and bearish >= 2 -> 0% 看空
      - bullish == 0 and bearish <= 1 -> 30% 中性(无明显方向)
    """
    if bullish >= 4:
        return {"position": "100%", "verdict": "看多"}
    if bullish == 3:
        return {"position": "50%", "verdict": "中性偏多"}
    if bearish >= 4:
        return {"position": "0%", "verdict": "看空"}
    if bearish == 3:
        return {"position": "20%", "verdict": "中性偏空"}
    if bullish == 2:
        return {"position": "50%", "verdict": "中性偏多"}
    if bullish == 1:
        return {"position": "30%", "verdict": "中性"}
    # bullish == 0
    if bearish >= 2:
        return {"position": "0%", "verdict": "看空"}
    return {"position": "30%", "verdict": "中性"}


def analyze_market_timing() -> Dict[str, Any]:
    """市场择时五维度投票(矩阵 13)。

    返回:看多/中性/看空 + 推荐仓位(100%/50%/0%)。
    """
    # 缓存 1 天
    cached = _cache_get("market_timing", "latest")
    if cached:
        return cached

    dims = {}
    for name, fn in [
        ("valuation", _dimension_valuation),
        ("capital", _dimension_capital),
        ("technique", _dimension_technique),
        ("sentiment", _dimension_sentiment),
        ("fundamental", _dimension_fundamental),
    ]:
        try:
            dims[name] = fn()
        except Exception as e:
            dims[name] = {"available": False, "error": str(e)}

    # 投票
    vote = {"看多": 0, "中性": 0, "看空": 0, "unavailable": 0}
    for name, d in dims.items():
        if not d.get("available"):
            vote["unavailable"] += 1
            continue
        v = d.get("verdict", "中性")
        if v in vote:
            vote[v] += 1
        else:
            vote["中性"] += 1

    bullish = vote["看多"]
    bearish = vote["看空"]
    neutral = vote["中性"]
    available_count = sum(1 for d in dims.values() if d.get("available"))

    pos = _vote_to_position(bullish, neutral, bearish)

    # 综合解读
    parts = []
    for name, d in dims.items():
        if d.get("available"):
            parts.append(f"{name}: {d.get('verdict')}({d.get('interpretation', '')[:60]})")
        else:
            parts.append(f"{name}: 不可用({d.get('error', '')[:40]})")
    interp = " | ".join(parts) + f" -> 投票:看多 {bullish} / 中性 {neutral} / 看空 {bearish} / 不可用 {vote['unavailable']}"

    out = {
        "available": available_count >= 3,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "vote": {"bullish": bullish, "neutral": neutral, "bearish": bearish, "unavailable": vote["unavailable"]},
        "position_recommendation": pos["position"],
        "verdict": pos["verdict"],
        "dimensions": dims,
        "interpretation": interp,
    }
    _cache_set("market_timing", "latest", out)
    return out


def _print_report(r: Dict[str, Any]) -> None:
    """格式化打印市场择时报告。"""
    if not r.get("available"):
        print(f"❌ 市场择时不可用:{r.get('dimensions', {})}")
        return
    print(f"=== 矩阵 13 市场择时({r['date']})===")
    print(f"综合判定:{r['verdict']} / 推荐仓位:{r['position_recommendation']}")
    print(f"投票:看多 {r['vote']['bullish']} / 中性 {r['vote']['neutral']} / 看空 {r['vote']['bearish']} / 不可用 {r['vote']['unavailable']}")
    print()
    print("--- 五维度详情 ---")
    for name, d in r["dimensions"].items():
        if d.get("available"):
            print(f"[{name}] {d['verdict']}: {d.get('interpretation', '')}")
        else:
            print(f"[{name}] ❌ 不可用: {d.get('error', '未知')}")
    print()
    print(f"解读:{r['interpretation']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="矩阵 13:市场择时五维度投票")
    p.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = p.parse_args()
    r = analyze_market_timing()
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    else:
        _print_report(r)
