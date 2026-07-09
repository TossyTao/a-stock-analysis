"""东方财富研报中心数据拉取 + 外资/港资/台资合资券商识别 + 评级共识/目标价聚合。

数据源:reportapi.eastmoney.com/report/list
  - 支持 code(股票代码)和 orgCode(机构代码)双重筛选
  - 字段:title/orgSName/publishDate/emRatingName(评级)/predictThisYearEps/Pe
          /predictNextYearEps/Pe/indvAimPriceT(目标价)/researcher(分析师)/encodeUrl(PDF)

外资/港资/台资券商识别:用关键词匹配 orgSName,不硬编码 orgCode(避免数据源变更失效)
  - 国际投行在华合资:高盛高华、瑞银证券、摩根士丹利华鑫、瑞信方正、野村东方国际
  - 中资国际子公司:中银国际、招银国际、建银国际、交银国际、海通国际、华泰国际 等
  - 港台资:群益、元大、凯基、第一上海、汇丰前海

评级标准化:东财评级 -> {买入, 增持, 中性, 减持, 卖出}(未评级单独标记)
"""
import os
import hashlib
import pickle
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import requests

# 复用 fetch_data 的网络层(Session + 限流 + 重试 + 浏览器 headers)
from fetch_data import _session, _throttle, _retry_with_backoff, _cache_get, _cache_set, _CACHE_TTL


# ---------- 外资/港资/台资券商识别 ----------

# 关键词:匹配 orgSName 即视为外资/港资/台资背景
FOREIGN_BROKER_KEYWORDS = [
    # 国际投行在华合资券商
    "高盛", "摩根", "瑞银", "瑞信", "野村", "巴克莱", "花旗", "美林",
    "杰富瑞", "汇丰", "星展", "麦格理", "富瑞", "德意志", "瑞万",
    "大和", "日兴", "三菱",
    # 中资券商的国际子公司(港资背景)
    "中银国际", "招银国际", "建银国际", "交银国际", "国泰君安国际",
    "申万国际", "海通国际", "华泰国际", "中信国际", "兴证国际",
    # 港台资券商
    "群益", "元大", "凯基", "第一上海", "复星", "申港",
]

# 评级标准化:东财评级 -> 标准化标签
RATING_NORMALIZE = {
    "买入": "buy",
    "增持": "overweight",
    "推荐": "overweight",
    "中性": "neutral",
    "持有": "neutral",
    "减持": "reduce",
    "卖出": "sell",
    "回避": "sell",
}

# 评级分数(buy=2, overweight=1, neutral=0, reduce=-1, sell=-2)
RATING_SCORE = {
    "buy": 2,
    "overweight": 1,
    "neutral": 0,
    "reduce": -1,
    "sell": -2,
}


def is_foreign_broker(org_name: str) -> bool:
    """判断机构是否为外资/港资/台资背景。"""
    if not org_name:
        return False
    return any(kw in org_name for kw in FOREIGN_BROKER_KEYWORDS)


def normalize_rating(rating: str) -> str:
    """东财评级 -> 标准化标签。未识别返回 unknown。"""
    if not rating:
        return "unknown"
    rating = rating.strip()
    return RATING_NORMALIZE.get(rating, "unknown")


# ---------- 东方财富研报 API ----------

_REPORT_API = "https://reportapi.eastmoney.com/report/list"


def _fetch_report_page(code: str, page_no: int = 1, page_size: int = 100,
                       begin_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
    """拉取一页研报数据。返回原始 item 列表。"""
    if begin_date is None:
        # 默认拉最近 1 年
        end_dt = datetime.now()
        begin_dt = end_dt - timedelta(days=365)
        begin_date = begin_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

    params = {
        "industryCode": "*",
        "pageSize": page_size,
        "industry": "*",
        "rating": "*",
        "ratingChange": "*",
        "beginTime": begin_date,
        "endTime": end_date,
        "pageNo": page_no,
        "qType": 0,
        "code": code,
    }
    _throttle()
    r = _session.get(_REPORT_API, params=params, timeout=(10, 30),
                     headers={"Referer": "https://data.eastmoney.com/report/"})
    r.raise_for_status()
    data = r.json()
    return data.get("data") or []


def _normalize_report_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """把东财原始 item 标准化为简洁结构。"""
    org_name = item.get("orgSName") or item.get("orgName") or ""
    rating_raw = item.get("emRatingName") or ""
    last_rating_raw = item.get("lastEmRatingName") or ""
    rating_change = item.get("ratingChange")

    # ratingChange: 1=首次, 2=上调, 3=维持, 4=下调
    change_map = {1: "首次", 2: "上调", 3: "维持", 4: "下调"}
    change_label = change_map.get(rating_change, "未变") if rating_change else "未变"

    # 目标价:优先 indvAimPriceT,其次 indvAimPriceL
    aim_price = _safe_float(item.get("indvAimPriceT"))
    if aim_price is None:
        aim_price = _safe_float(item.get("indvAimPriceL"))

    return {
        "title": item.get("title", ""),
        "org": org_name,
        "is_foreign": is_foreign_broker(org_name),
        "publish_date": (item.get("publishDate") or "")[:10],
        "rating": rating_raw,
        "rating_norm": normalize_rating(rating_raw),
        "last_rating": last_rating_raw,
        "rating_change": change_label,
        "aim_price": aim_price,
        "eps_this_year": _safe_float(item.get("predictThisYearEps")),
        "pe_this_year": _safe_float(item.get("predictThisYearPe")),
        "eps_next_year": _safe_float(item.get("predictNextYearEps")),
        "pe_next_year": _safe_float(item.get("predictNextYearPe")),
        "eps_year_after_next": _safe_float(item.get("predictNextTwoYearEps")),
        "pe_year_after_next": _safe_float(item.get("predictNextTwoYearPe")),
        "researcher": item.get("researcher", ""),
        "encode_url": item.get("encodeUrl", ""),
        "pdf_url": f"https://pdf.dfcfw.com/pdf/H3_{item.get('infoCode', '')}_1.pdf" if item.get("infoCode") else "",
    }


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
            return None
        return f
    except (TypeError, ValueError):
        return None


# ---------- 主入口 ----------

def fetch_research_reports(code: str, lookback_days: int = 365,
                           current_price: Optional[float] = None,
                           max_pages: int = 5) -> Dict[str, Any]:
    """拉取某股票的研报数据,聚合评级共识/目标价/外资观点。

    Args:
        code: 股票代码(6 位,如 600519)
        lookback_days: 回看天数,默认 365 天
        current_price: 当前价(用于计算目标价上涨空间),可选
        max_pages: 最多拉取页数,默认 5 页(500 条)

    Returns:
        {
            "code": "600519",
            "total_reports": 212,
            "reports": [...],            # 最近 N 条标准化研报
            "rating_consensus": {...},   # 评级共识
            "target_price": {...},       # 目标价统计
            "eps_forecast": {...},       # 盈利预测
            "foreign_reports": [...],    # 外资/港资/台资合资券商研报
            "foreign_summary": {...},    # 外资观点汇总
            "divergence": {...},         # 外资 vs 内资分歧度
        }
    """
    # 磁盘缓存:TTL 1 天(同 fundamental)
    cache_key = hashlib.md5(f"{code}_{lookback_days}".encode()).hexdigest()
    cached = _cache_get("research_report", cache_key)
    if cached is not None:
        # 缓存命中也要重新算目标价空间(因为 current_price 可能变化)
        return _attach_upside(cached, current_price)

    end_dt = datetime.now()
    begin_dt = end_dt - timedelta(days=lookback_days)
    begin_date = begin_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    all_items: List[Dict[str, Any]] = []

    def _do_fetch():
        nonlocal all_items
        all_items = []
        for page in range(1, max_pages + 1):
            items = _fetch_report_page(code, page_no=page, page_size=100,
                                       begin_date=begin_date, end_date=end_date)
            if not items:
                break
            all_items.extend(items)
            if len(items) < 100:
                break
        return all_items

    try:
        _retry_with_backoff(_do_fetch, max_retries=3, base_delay=1.0)
    except Exception as e:
        return {
            "code": code,
            "error": f"研报数据拉取失败: {e}",
            "total_reports": 0,
            "reports": [],
        }

    reports = [_normalize_report_item(item) for item in all_items]
    # 按发布日期降序(新 -> 旧)
    reports.sort(key=lambda r: r.get("publish_date", ""), reverse=True)

    result = {
        "code": code,
        "total_reports": len(reports),
        "reports": reports,
        "rating_consensus": _compute_rating_consensus(reports),
        "target_price": _compute_target_price(reports, current_price),
        "eps_forecast": _compute_eps_forecast(reports),
        "foreign_reports": [r for r in reports if r.get("is_foreign")],
        "foreign_summary": {},
        "divergence": {},
    }

    foreign_reports = result["foreign_reports"]
    result["foreign_summary"] = _compute_foreign_summary(foreign_reports, current_price)
    result["divergence"] = _compute_divergence(reports, foreign_reports)

    # 缓存(不带 current_price 的 upside,后面再补)
    _cache_set("research_report", cache_key, result)

    return _attach_upside(result, current_price)


# ---------- 聚合统计 ----------

def _compute_rating_consensus(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """评级共识:统计各评级数量 + 主导评级 + 共识强度。"""
    if not reports:
        return {"available": False}

    rating_count: Dict[str, int] = {}
    for r in reports:
        rn = r.get("rating_norm", "unknown")
        rating_count[rn] = rating_count.get(rn, 0) + 1

    total = len(reports)
    # 主导评级(忽略 unknown)
    valid_ratings = {k: v for k, v in rating_count.items() if k != "unknown"}
    if not valid_ratings:
        return {
            "available": True,
            "total": total,
            "rating_count": rating_count,
            "dominant": None,
            "dominant_pct": 0.0,
            "consensus_strength": 0.0,
            "score_mean": None,
            "label": "无评级",
        }

    dominant = max(valid_ratings, key=valid_ratings.get)
    dominant_pct = valid_ratings[dominant] / sum(valid_ratings.values())
    # 共识强度:主导评级占比(0-1)
    consensus_strength = dominant_pct

    # 评级分数均值
    scores = [RATING_SCORE.get(k, 0) * v for k, v in valid_ratings.items()]
    score_mean = sum(scores) / sum(valid_ratings.values())

    # 标签:共识强(>60%) / 中(40-60%) / 弱(<40%)
    if consensus_strength >= 0.6:
        label = "共识强"
    elif consensus_strength >= 0.4:
        label = "共识中"
    else:
        label = "分歧大"

    label_map = {
        "buy": "买入", "overweight": "增持", "neutral": "中性",
        "reduce": "减持", "sell": "卖出",
    }
    return {
        "available": True,
        "total": total,
        "rating_count": rating_count,
        "dominant": dominant,
        "dominant_label": label_map.get(dominant, dominant),
        "dominant_pct": round(dominant_pct * 100, 1),
        "consensus_strength": round(consensus_strength, 3),
        "score_mean": round(score_mean, 2),
        "label": label,
    }


def _compute_target_price(reports: List[Dict[str, Any]],
                          current_price: Optional[float] = None) -> Dict[str, Any]:
    """目标价统计:均值/中位/最大/最小/上涨空间。"""
    prices = [r["aim_price"] for r in reports if r.get("aim_price") is not None and r["aim_price"] > 0]
    if not prices:
        return {"available": False}

    prices.sort()
    n = len(prices)
    mean = sum(prices) / n
    median = prices[n // 2] if n % 2 == 1 else (prices[n // 2 - 1] + prices[n // 2]) / 2
    max_p = max(prices)
    min_p = min(prices)
    spread = (max_p - min_p) / mean if mean > 0 else 0.0

    result = {
        "available": True,
        "count": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "max": round(max_p, 2),
        "min": round(min_p, 2),
        "spread_pct": round(spread * 100, 1),  # (max-min)/mean
        "current_price": current_price,
        "upside_pct": None,
        "label": None,
    }

    if current_price and current_price > 0:
        upside = (mean - current_price) / current_price
        result["upside_pct"] = round(upside * 100, 1)
        if upside >= 0.2:
            result["label"] = "空间大"
        elif upside >= 0.05:
            result["label"] = "空间中"
        elif upside >= -0.05:
            result["label"] = "空间小"
        else:
            result["label"] = "已超目标"

    return result


def _compute_eps_forecast(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """盈利预测:取最近有预测的研报,聚合当年/明年/后年 EPS/PE。"""
    if not reports:
        return {"available": False}

    # 按发布日期降序取最近的(已排序)
    def _latest_with(field_eps: str, field_pe: str) -> Dict[str, Any]:
        for r in reports:
            eps = r.get(field_eps)
            pe = r.get(field_pe)
            if eps is not None or pe is not None:
                return {"eps": eps, "pe": pe, "source_date": r.get("publish_date"), "org": r.get("org")}
        return {}

    return {
        "available": True,
        "current_year": _latest_with("eps_this_year", "pe_this_year"),
        "next_year": _latest_with("eps_next_year", "pe_next_year"),
        "year_after_next": _latest_with("eps_year_after_next", "pe_year_after_next"),
    }


def _compute_foreign_summary(foreign_reports: List[Dict[str, Any]],
                             current_price: Optional[float] = None) -> Dict[str, Any]:
    """外资/港资/台资合资券商研报汇总。"""
    if not foreign_reports:
        return {"available": False, "count": 0}

    consensus = _compute_rating_consensus(foreign_reports)
    target = _compute_target_price(foreign_reports, current_price)

    # 最近一条
    latest = foreign_reports[0] if foreign_reports else None

    return {
        "available": True,
        "count": len(foreign_reports),
        "rating_consensus": consensus,
        "target_price": target,
        "latest": {
            "title": latest.get("title"),
            "org": latest.get("org"),
            "date": latest.get("publish_date"),
            "rating": latest.get("rating"),
            "rating_norm": latest.get("rating_norm"),
            "aim_price": latest.get("aim_price"),
        } if latest else None,
    }


def _compute_divergence(reports: List[Dict[str, Any]],
                        foreign_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """外资 vs 内资评级分歧度。"""
    if not foreign_reports or not reports:
        return {"available": False}

    domestic_reports = [r for r in reports if not r.get("is_foreign")]
    if not domestic_reports:
        return {"available": False}

    foreign_consensus = _compute_rating_consensus(foreign_reports)
    domestic_consensus = _compute_rating_consensus(domestic_reports)

    if not foreign_consensus.get("available") or not domestic_consensus.get("available"):
        return {"available": False}

    f_score = foreign_consensus.get("score_mean")
    d_score = domestic_consensus.get("score_mean")
    if f_score is None or d_score is None:
        return {"available": False}

    diff = f_score - d_score  # 正:外资更乐观,负:外资更悲观

    if abs(diff) >= 1.0:
        label = "外资明显更乐观" if diff > 0 else "外资明显更悲观"
    elif abs(diff) >= 0.5:
        label = "外资略乐观" if diff > 0 else "外资略悲观"
    else:
        label = "内外资一致"

    # 目标价分歧
    f_tp = _compute_target_price(foreign_reports).get("mean")
    d_tp = _compute_target_price(domestic_reports).get("mean")
    tp_div = None
    if f_tp and d_tp and d_tp > 0:
        tp_div = (f_tp - d_tp) / d_tp

    return {
        "available": True,
        "foreign_score": f_score,
        "domestic_score": d_score,
        "diff": round(diff, 2),
        "label": label,
        "foreign_target_price": f_tp,
        "domestic_target_price": d_tp,
        "target_price_divergence_pct": round(tp_div * 100, 1) if tp_div is not None else None,
    }


def _attach_upside(result: Dict[str, Any], current_price: Optional[float]) -> Dict[str, Any]:
    """缓存命中后,根据 current_price 重新计算上涨空间。"""
    if not current_price or current_price <= 0:
        return result

    # 重算 target_price.upside_pct
    tp = result.get("target_price")
    if tp and tp.get("available") and tp.get("mean"):
        mean = tp["mean"]
        upside = (mean - current_price) / current_price
        tp["current_price"] = current_price
        tp["upside_pct"] = round(upside * 100, 1)
        if upside >= 0.2:
            tp["label"] = "空间大"
        elif upside >= 0.05:
            tp["label"] = "空间中"
        elif upside >= -0.05:
            tp["label"] = "空间小"
        else:
            tp["label"] = "已超目标"

    # 重算 foreign_summary.target_price.upside_pct
    fs = result.get("foreign_summary")
    if fs and fs.get("available"):
        ftp = fs.get("target_price")
        if ftp and ftp.get("available") and ftp.get("mean"):
            mean = ftp["mean"]
            upside = (mean - current_price) / current_price
            ftp["current_price"] = current_price
            ftp["upside_pct"] = round(upside * 100, 1)

    # 重算 divergence 的目标价
    div = result.get("divergence")
    if div and div.get("available"):
        # 不需要重算,因为 div 用的是相对值
        pass

    return result


if __name__ == "__main__":
    import sys
    import json
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    price = float(sys.argv[2]) if len(sys.argv) > 2 else None
    r = fetch_research_reports(code, current_price=price)
    # 简化输出(不含 reports 列表)
    out = {k: v for k, v in r.items() if k != "reports"}
    out["reports_count"] = len(r.get("reports", []))
    out["latest_3_reports"] = r.get("reports", [])[:3]
    out["latest_3_foreign"] = [r for r in r.get("foreign_reports", [])][:3]
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
