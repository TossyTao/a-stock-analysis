"""个股最近新闻拉取 + 舆情分类 + 关键事件提取。

数据源:akshare.stock_news_em(symbol=code)
  - 东方财富搜索聚合(证券时报/上海证券报/华夏时报等主流财经媒体)
  - 返回 10 条最近新闻:关键词, 新闻标题, 新闻内容, 发布时间, 文章来源, 新闻链接
  - 响应 ~2 秒,反爬宽松

舆情分类(关键词词库 + 标题权重 ×2):
  - 标题命中权重 ×2,内容命中权重 ×1
  - positive_score vs negative_score -> 利好/利空/中性
  - 关键事件从标题提取(关键词 + 前后 10 字),去重

舆情汇总:
  - 偏正面/偏负面/中性偏正/中性偏负/分歧
  - 基于 positive/negative/neutral 条数占比

复用 fetch_data 的网络层(限流 + 重试 + 缓存)。缓存 TTL 4h(新闻盘中可能更新)。
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fetch_data import (
    _throttle, _retry_with_backoff,
    _cache_get, _cache_set,
    normalize_code,
)


# ---------- 关键词词库 ----------

POSITIVE_KEYWORDS = [
    # 业绩
    "净利预增", "净利大增", "业绩大增", "业绩预增", "业绩扭亏", "盈利大增",
    "营收增长", "营收创历史", "毛利率提升",
    # 订单/合同
    "订单", "中标", "签约", "合同", "框架协议",
    # 技术/产能
    "突破", "攻克", "量产", "扩产", "投产", "达产", "首发", "首试",
    # 资本动作(正面)
    "回购", "增持", "员工持股", "股权激励",
    # 政策/叙事
    "政策支持", "补贴", "税收优惠", "国产替代", "自主可控",
    "AI", "算力", "HBM", "DDR5", "光模块", "高速互联",
    "入选", "纳入", "指数",
    # 市场表现
    "涨停", "强势", "创新高",
    # 分红
    "分红", "派息", "特别分红",
]

NEGATIVE_KEYWORDS = [
    # 业绩
    "净利预减", "净利下滑", "亏损", "减亏", "业绩爆雷", "商誉减值", "资产减值",
    # 资本动作(负面)
    "减持", "质押", "平仓", "强制平仓",
    # 合规/风险
    "爆雷", "违规", "立案", "处罚", "警示", "问询", "关注函",
    "诉讼", "仲裁", "被执行人",
    # 经营
    "停产", "召回", "事故", "安全",
    # 地缘
    "制裁", "实体清单", "出口管制", "关税",
    # 退市
    "退市", "ST", "*ST",
]


# ---------- 舆情分类 ----------

def _classify_sentiment(title: str, content: str) -> tuple:
    """对单条新闻做舆情分类。

    标题命中权重 ×2,内容命中权重 ×1。
    返回 (sentiment, keywords_matched)。
    """
    title = title or ""
    content = content or ""

    positive_score = 0
    negative_score = 0
    keywords_matched = []

    for kw in POSITIVE_KEYWORDS:
        title_hits = title.count(kw)
        content_hits = content.count(kw)
        if title_hits + content_hits > 0:
            positive_score += title_hits * 2 + content_hits * 1
            if kw not in keywords_matched:
                keywords_matched.append(kw)

    for kw in NEGATIVE_KEYWORDS:
        title_hits = title.count(kw)
        content_hits = content.count(kw)
        if title_hits + content_hits > 0:
            negative_score += title_hits * 2 + content_hits * 1
            if kw not in keywords_matched:
                keywords_matched.append(kw)

    if positive_score > negative_score:
        sentiment = "利好"
    elif negative_score > positive_score:
        sentiment = "利空"
    else:
        sentiment = "中性"

    return sentiment, keywords_matched


# ---------- 关键事件提取 ----------

def _extract_key_events(news_list: List[Dict[str, Any]], max_events: int = 5) -> List[str]:
    """从标题中提取包含关键词的关键事件片段(关键词 + 前后 10 字),去重。"""
    events = []
    seen = set()

    for news in news_list:
        title = news.get("title", "")
        if not title:
            continue
        for kw in POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS:
            idx = title.find(kw)
            if idx >= 0:
                start = max(0, idx - 10)
                end = min(len(title), idx + len(kw) + 10)
                snippet = title[start:end].strip()
                # 简单去重:完全相同或包含关系
                dup = False
                for existing in events:
                    if snippet in existing or existing in snippet:
                        dup = True
                        break
                if not dup and snippet not in seen:
                    events.append(snippet)
                    seen.add(snippet)
                    break  # 每条新闻只取第一个关键词事件
        if len(events) >= max_events:
            break

    return events


# ---------- 舆情汇总 ----------

def _compute_sentiment_summary(sentiments: List[str]) -> Dict[str, Any]:
    """统计 positive/negative/neutral 条数,生成汇总标签。"""
    positive = sentiments.count("利好")
    negative = sentiments.count("利空")
    neutral = sentiments.count("中性")
    total = len(sentiments)

    if total == 0:
        return {
            "positive": 0, "negative": 0, "neutral": 0,
            "dominant": "无", "dominant_pct": 0.0, "label": "无数据",
        }

    dominant_count = max(positive, negative, neutral)
    if positive == negative == neutral:
        dominant = "分歧"
    elif positive == dominant_count and negative == dominant_count:
        dominant = "分歧"
    elif positive == dominant_count:
        dominant = "利好"
    elif negative == dominant_count:
        dominant = "利空"
    else:
        dominant = "中性"

    dominant_pct = round(dominant_count / total * 100, 1)

    # 标签
    if dominant == "分歧":
        label = "分歧"
    elif dominant == "利好":
        if dominant_pct >= 60:
            label = "偏正面"
        else:
            label = "中性偏正"
    elif dominant == "利空":
        if dominant_pct >= 60:
            label = "偏负面"
        else:
            label = "中性偏负"
    else:  # 中性
        if dominant_pct >= 60:
            label = "中性"
        else:
            label = "分歧"

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "dominant": dominant,
        "dominant_pct": dominant_pct,
        "label": label,
    }


# ---------- 新闻拉取 ----------

def _fetch_raw_news(code: str) -> List[Dict[str, Any]]:
    """调 akshare.stock_news_em 拉取原始新闻列表。"""
    try:
        import akshare as ak
    except ImportError as e:
        raise RuntimeError(f"akshare 未安装: {e}")

    _throttle()
    df = _retry_with_backoff(lambda: ak.stock_news_em(symbol=code))
    if df is None or len(df) == 0:
        return []

    # 列名映射:中文 -> 英文
    col_map = {
        "关键词": "keyword",
        "新闻标题": "title",
        "新闻内容": "content",
        "发布时间": "datetime",
        "文章来源": "source",
        "新闻链接": "url",
    }
    df = df.rename(columns=col_map)

    records = []
    for _, row in df.iterrows():
        dt_str = str(row.get("datetime", ""))
        date_str = ""
        try:
            # 尝试解析时间
            if dt_str:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d")
                date_str = dt_str
            except (ValueError, TypeError):
                date_str = dt_str[:10] if len(dt_str) >= 10 else dt_str

        records.append({
            "title": str(row.get("title", "")),
            "content": str(row.get("content", ""))[:200],  # 前 200 字摘要
            "datetime": dt_str,
            "date": date_str,
            "source": str(row.get("source", "")),
            "url": str(row.get("url", "")),
        })

    return records


def fetch_news(code: str, limit: int = 10, lookback_days: int = 30) -> Dict[str, Any]:
    """拉取个股最近新闻 + 舆情分类。

    数据源:ak.stock_news_em(symbol=code) -> 东方财富搜索聚合
    缓存:TTL 4h(新闻盘中可能更新,但不必实时)

    Args:
        code: 股票代码(6 位)
        limit: 返回条数(默认 10,API 上限 10)
        lookback_days: 回看天数(默认 30),过滤超出范围的旧新闻

    Returns:
        Dict:新闻列表 + 舆情汇总 + 关键事件 + 解读
    """
    code = normalize_code(code)
    cache_key = f"news_{code}_{lookback_days}"

    cached = _cache_get("news", cache_key)
    if cached is not None:
        return cached

    try:
        raw_news = _fetch_raw_news(code)
    except Exception as e:
        result = {"code": code, "available": False, "error": str(e)}
        _cache_set("news", cache_key, result)
        return result

    if not raw_news:
        result = {
            "code": code, "available": True, "source": "eastmoney_search",
            "count": 0, "lookback_days": lookback_days,
            "news": [], "sentiment_summary": _compute_sentiment_summary([]),
            "key_events": [], "interpretation": "无新闻数据",
        }
        _cache_set("news", cache_key, result)
        return result

    # lookback_days 过滤
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    filtered = []
    for n in raw_news:
        try:
            dt = datetime.strptime(n["date"], "%Y-%m-%d")
            if dt >= cutoff_date:
                filtered.append(n)
        except (ValueError, TypeError):
            # 时间解析失败的新闻保留(可能格式异常)
            filtered.append(n)

    # 舆情分类
    for n in filtered:
        sentiment, keywords = _classify_sentiment(n["title"], n["content"])
        n["sentiment"] = sentiment
        n["keywords_matched"] = keywords

    # 限制条数
    filtered = filtered[:limit]

    # 舆情汇总
    sentiments = [n["sentiment"] for n in filtered]
    summary = _compute_sentiment_summary(sentiments)

    # 关键事件
    key_events = _extract_key_events(filtered)

    # 时间范围
    dates = [n["date"] for n in filtered if n["date"]]
    date_range = {
        "earliest": min(dates) if dates else None,
        "latest": max(dates) if dates else None,
    }

    # 解读
    pos = summary["positive"]
    neg = summary["negative"]
    label = summary["label"]
    events_str = "、".join(key_events[:3]) if key_events else "无明显事件"
    interpretation = f"最近 {lookback_days} 天舆情{label}({pos} 利好 vs {neg} 利空),核心事件:{events_str}"

    result = {
        "code": code,
        "available": True,
        "source": "eastmoney_search",
        "count": len(filtered),
        "lookback_days": lookback_days,
        "date_range": date_range,
        "news": filtered,
        "sentiment_summary": summary,
        "key_events": key_events,
        "interpretation": interpretation,
    }

    _cache_set("news", cache_key, result)
    return result
