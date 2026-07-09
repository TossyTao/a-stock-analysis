"""akshare 数据拉取:股票代码归一化、名称查询、前复权日线。

网络层优化(综合 CSDN 多篇方案):
  - Session + HTTPAdapter + Retry:连接池 + 智能重试(连接异常 + 5xx/429)+ 指数退避
  - 强化浏览器 headers(UA + Referer + Accept-Encoding + sec-ch-ua 等),模拟真实浏览器
  - 请求间隔限流(_throttle):同源请求最少 500ms 间隔 + 随机抖动,避免触发限流
  - 重试随机抖动:避免请求模式规律被识别
  - 数据源 fallback:eastmoney 不可达时切 sina K线/sina hq/硬编码映射
  - 轻量磁盘缓存(日线 4h,财务摘要 1d,名称表 7d),避免短时重复拉取
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 强化浏览器 headers:模拟真实 Chrome 请求,避免被 TLS/UA 指纹检测拦截
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,application/json,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "Upgrade-Insecure-Requests": "1",
}

# 全局 Session:连接池 + 重试策略
# - total/connect/read=5:连接异常(含 RemoteDisconnected)自动重试 5 次
# - backoff_factor=0.5:重试间隔 0.5,1,2,4,8 秒(指数退避)
# - status_forcelist:5xx + 429 状态码重试
_RETRY_STRATEGY = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST", "HEAD"],
    raise_on_status=False,
)

_session = requests.Session()
_adapter = HTTPAdapter(max_retries=_RETRY_STRATEGY, pool_connections=10, pool_maxsize=10)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
_session.headers.update(_BROWSER_HEADERS)

_original_get = requests.get
_original_post = requests.post

# 请求限流:同源请求最小间隔 500ms + 随机抖动,避免触发限流(参考 CSDN 方案 time.sleep)
_LAST_REQUEST_TIME = 0.0
_MIN_REQUEST_INTERVAL = 0.5


def _throttle():
    """请求间隔限流:确保两次请求间至少 500ms + 随机抖动 0-300ms。"""
    global _LAST_REQUEST_TIME
    now = time.time()
    elapsed = now - _LAST_REQUEST_TIME
    if elapsed < _MIN_REQUEST_INTERVAL:
        wait = _MIN_REQUEST_INTERVAL - elapsed + random.uniform(0, 0.3)
        time.sleep(wait)
    _LAST_REQUEST_TIME = time.time()


def _patched_get(url, headers=None, **kwargs):
    """让 akshare 内部的 requests.get 走我们的 Session(限流 + 连接池 + 重试 + headers)。"""
    _throttle()
    merged = {**_BROWSER_HEADERS, **(headers or {})}
    kwargs.setdefault("timeout", (10, 30))
    return _session.get(url, headers=merged, **kwargs)


def _patched_post(url, headers=None, **kwargs):
    _throttle()
    merged = {**_BROWSER_HEADERS, **(headers or {})}
    kwargs.setdefault("timeout", (10, 30))
    return _session.post(url, headers=merged, **kwargs)


requests.get = _patched_get
requests.post = _patched_post


def check_akshare_version() -> None:
    """检查 akshare 版本,过旧则提示升级(参考 CSDN 方案:pip install --upgrade akshare)。"""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            v = version("akshare")
            # 简单判断:版本号 < 1.10 建议升级
            parts = v.split(".")
            major = int(parts[0]) if parts else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            if major < 1 or (major == 1 and minor < 10):
                print(
                    f"[warn] akshare 当前版本 {v} 偏旧,建议升级:pip install --upgrade akshare",
                    file=sys.stderr,
                )
        except PackageNotFoundError:
            pass
    except Exception:
        pass


# ---------- 轻量磁盘缓存 ----------
_CACHE_DIR = os.path.expanduser("~/.claude/skills/stock-analysis/.cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

_CACHE_TTL = {
    "daily": timedelta(hours=4),       # 日线数据:4 小时(覆盖盘中+盘后)
    "fundamental": timedelta(days=1),  # 财务摘要:1 天
    "name": timedelta(days=7),         # 股票名称表:7 天
}


def _cache_key(kind: str, *parts) -> str:
    key_str = "_".join(str(p) for p in parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _cache_get(kind: str, key: str):
    path = os.path.join(_CACHE_DIR, f"{kind}_{key}.pkl")
    if not os.path.exists(path):
        return None
    ttl = _CACHE_TTL.get(kind)
    if ttl and datetime.now() - datetime.fromtimestamp(os.path.getmtime(path)) > ttl:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_set(kind: str, key: str, data) -> None:
    if data is None:
        return
    path = os.path.join(_CACHE_DIR, f"{kind}_{key}.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """外层指数退避重试 + 随机抖动:即使 Session 层重试也失败时,再兜底一次。"""
    last_err = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                # 指数退避 + 随机抖动(0-1s),避免请求模式规律
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
    raise last_err


def normalize_code(code: str) -> str:
    """归一化 A 股代码为 akshare 接受的 6 位纯数字形式。

    支持:
      - "000001" / "sz000001" / "SZ000001" / "000001.SZ"
      - "600000" / "sh600000"
      - "830879" / "bj830879" (北交所)
    """
    code = str(code).strip().lower().replace(".", "")
    if code.startswith(("sh", "sz", "bj")):
        code = code[2:]
    if not (len(code) == 6 and code.isdigit()):
        raise ValueError(f"无法识别的股票代码: {code}")
    return code


def code_with_market(code: str) -> str:
    """按代码首位推断市场前缀(用于 akshare 部分接口)。"""
    code = normalize_code(code)
    if code.startswith("6"):
        return f"sh{code}"
    if code.startswith(("0", "3")):
        return f"sz{code}"
    if code.startswith(("8", "4")):
        return f"bj{code}"
    raise ValueError(f"无法识别市场: {code}")


def resolve_name(query: str) -> str:
    """如果输入是名称(如"平安银行"),返回对应代码;否则原样返回。"""
    query = str(query).strip()
    if not query:
        raise ValueError("空查询")
    if query.isdigit() and len(query) == 6:
        return query
    try:
        query_lower = query.lower()
        if query_lower.startswith(("sh", "sz", "bj")) and len(query_lower) >= 8:
            return normalize_code(query)
    except ValueError:
        pass
    try:
        info = ak.stock_info_a_code_name()
    except Exception as e:
        raise RuntimeError(f"查询股票名称失败(网络?): {e}")
    matches = info[info["name"].str.contains(query, na=False)]
    if matches.empty:
        raise ValueError(f"未找到名称含 '{query}' 的股票")
    if len(matches) > 1:
        names = matches["name"].tolist()[:5]
        raise ValueError(f"名称 '{query}' 匹配多只: {names},请更精确")
    return matches.iloc[0]["code"]


def fetch_name(code: str) -> str:
    """取股票中文名称(带缓存)。

    主路径:akshare stock_info_a_code_name(eastmoney)
    兜底:sina hq 实时报价接口(返回第一个字段就是中文名)
    """
    cache_key = _cache_key("name", code)
    cached = _cache_get("name", cache_key)
    if cached is not None:
        return cached

    # 主路径:akshare 名称表
    try:
        info = ak.stock_info_a_code_name()
        row = info[info["code"] == code]
        if not row.empty:
            name = row.iloc[0]["name"]
            _cache_set("name", cache_key, name)
            return name
    except Exception:
        pass

    # 兜底:sina hq
    try:
        name = _fetch_name_sina(code)
        if name:
            _cache_set("name", cache_key, name)
            return name
    except Exception:
        pass
    return code


def _fetch_name_sina(code: str) -> str:
    """sina hq 实时报价拿股票中文名称。"""
    symbol = code_with_market(code)
    url = f"https://hq.sinajs.cn/list={symbol}"
    r = _session.get(url, timeout=(10, 15), headers={"Referer": "https://finance.sina.com.cn/"})
    r.encoding = "gbk"
    m = re.search(r'hq_str_\w+="([^"]+)"', r.text)
    if not m:
        return ""
    return m.group(1).split(",")[0]


# 常见公司 -> 行业映射(push2 不可达时兜底,覆盖用户常问的股票)
_INDUSTRY_FALLBACK = {
    "600938": "石油石化(海上油气开采)",
    "601857": "石油石化",
    "601938": "银行",
    "000001": "银行",
    "603986": "半导体(存储芯片+MCU)",
    "002472": "汽车零部件(齿轮+减速器)",
    "002028": "电力设备(输配电一次设备)",  # 思源电气
    "600519": "食品饮料(白酒)",
    "603288": "食品饮料(调味品)",
    "601899": "有色金属(黄金+铜)",
    "600547": "有色金属(黄金)",
    "002460": "有色金属(锂)",
    "002466": "有色金属(锂)",
    "600585": "建材(水泥)",
    "000858": "食品饮料(白酒)",
    "600036": "银行",
    "601318": "保险",
    "601398": "银行",
    "600276": "医药生物(创新药)",
    "300750": "电力设备(电池)",
    "300059": "非银金融",
    "600406": "电力设备(电网自动化)",  # 国电南瑞
    "601882": "电力设备(电网自动化)",  # 海兴电力
    "601179": "电力设备(电网自动化)",  # 中国西电
    "300341": "电力设备(输配电)",  # 麦克奥迪
}


def _fetch_industry_fallback(code: str) -> str:
    """行业兜底:常见公司硬编码映射。"""
    return _INDUSTRY_FALLBACK.get(code, "")


def fetch_daily(code: str, days: int = 120) -> pd.DataFrame:
    """拉前复权日线,返回最近 N 个交易日。

    带磁盘缓存(4h TTL)+ 外层指数退避重试 + sina K线 fallback。
    eastmoney push2his 持续不可达时,自动切到 sina K线端点。
    """
    code = normalize_code(code)

    cache_key = _cache_key("daily", code, days)
    cached = _cache_get("daily", cache_key)
    if cached is not None:
        return cached

    # 主路径:akshare(eastmoney)
    end = pd.Timestamp.now().strftime("%Y%m%d")
    start = (pd.Timestamp.now() - pd.Timedelta(days=int(days * 1.8) + 30)).strftime("%Y%m%d")

    def _fetch_em():
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        if df is None or df.empty:
            raise RuntimeError(f"akshare 返回空数据(code={code})")
        rename = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
        df = df.rename(columns=rename)[list(rename.values())]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume"]).reset_index(drop=True)
        return df.tail(days).reset_index(drop=True)

    try:
        df = _retry_with_backoff(_fetch_em, max_retries=3, base_delay=1.0)
        _cache_set("daily", cache_key, df)
        return df
    except Exception as em_err:
        # fallback:sina K线(不复权,近 120 日内除权罕见)
        try:
            df = _fetch_daily_sina(code, days)
            _cache_set("daily", cache_key, df)
            return df
        except Exception as sina_err:
            raise RuntimeError(
                f"akshare 拉取失败(重试 3 次): {em_err}; sina fallback 也失败: {sina_err}"
            )


def _fetch_daily_sina(code: str, days: int) -> pd.DataFrame:
    """sina K线 fallback:不复权,返回近 N 个交易日。

    endpoint: https://quotes.sina.cn/cn/api/jsonp_v2.php/.../getKLineData
    返回字段: day, open, high, low, close, volume(无 amount)
    """
    symbol = code_with_market(code)
    datalen = min(max(days + 10, 30), 250)  # sina 限制 datalen
    url = (
        f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_k=/CN_MarketDataService.getKLineData"
        f"?symbol={symbol}&scale=240&ma=no&datalen={datalen}"
    )
    r = _session.get(url, timeout=(10, 30), headers={"Referer": "https://finance.sina.com.cn/"})
    r.raise_for_status()
    text = r.text
    m = re.search(r"var _k=\((.+)\);?\s*$", text.strip(), re.S)
    if not m:
        raise RuntimeError(f"sina K线返回异常: {text[:200]}")
    data = json.loads(m.group(1))
    if not data:
        raise RuntimeError("sina K线返回空数据")
    df = pd.DataFrame(data)
    df = df.rename(columns={"day": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["amount"] = 0.0
    df = df.dropna(subset=["close", "volume"]).reset_index(drop=True)
    return df.tail(days).reset_index(drop=True)


def fetch(code_or_name: str, days: int = 120) -> dict:
    """主入口:输入代码或名称,返回 JSON 可序列化的 dict。"""
    code = resolve_name(code_or_name)
    name = fetch_name(code)
    df = fetch_daily(code, days)
    if len(df) < 20:
        raise RuntimeError(f"数据不足:仅 {len(df)} 个交易日,至少需要 20 天")
    daily = df.to_dict(orient="records")
    return {
        "code": code,
        "name": name,
        "days_returned": len(daily),
        "data_insufficient": len(daily) < days,
        "daily": daily,
    }


def fetch_fundamental(code: str) -> dict:
    """拉取基本面数据:行业、PE、PB、多年财务摘要(净利润、ROIC、FCF)。

    stock_individual_info_em 提供行业/PE/PB(可能失败);
    stock_financial_abstract 是 sina 版,提供多年财务数据。
    行业/PE/PB 失败时用 eastmoney push2 直接 API 兜底。
    财务摘要带 1 天磁盘缓存。
    """
    code = normalize_code(code)
    info = {"industry": None, "pe": None, "pb": None, "free_float_shares": None}
    financials = []

    # 行业/PE/PB/流通股本(em 接口,失败则用 push2 直接 API 兜底)
    def _fetch_individual():
        return ak.stock_individual_info_em(symbol=code)

    try:
        individual = _retry_with_backoff(_fetch_individual, max_retries=2, base_delay=1.0)
        for _, row in individual.iterrows():
            k = str(row["item"])
            v = row["value"]
            if "行业" in k:
                info["industry"] = str(v)
            elif k == "市盈率(动态)" or "市盈率" in k:
                info["pe"] = _safe_float(v)
            elif k == "市净率" or "市净率" in k:
                info["pb"] = _safe_float(v)
            elif "流通股本" in k or k == "流通股" or "流通股" in k:
                # 单位通常为股,需解析 "X.XX亿股" / "X万股" 等中文格式
                # eastmoney 用 "流通股",可能值为 "X.XX亿股" 或纯数字(股)
                info["free_float_shares"] = _parse_shares(v)
    except Exception:
        # 兜底:直接调 eastmoney push2 单股实时接口
        pe_pb = _fetch_pe_pb_push2(code)
        info["pe"] = pe_pb.get("pe")
        info["pb"] = pe_pb.get("pb")
        if pe_pb.get("free_float_shares"):
            info["free_float_shares"] = pe_pb["free_float_shares"]

    # 行业兜底:常见公司硬编码映射
    if not info["industry"]:
        info["industry"] = _fetch_industry_fallback(code) or None

    # 财务摘要(sina 版,带 1 天缓存)
    sina_symbol = code_with_market(code)
    cache_key = _cache_key("fundamental", sina_symbol)
    cached = _cache_get("fundamental", cache_key)
    if cached is not None:
        financials = cached
    else:
        def _fetch_abstract():
            return ak.stock_financial_abstract(symbol=sina_symbol)
        try:
            abstract = _retry_with_backoff(_fetch_abstract, max_retries=2, base_delay=1.0)
            if abstract is not None and not abstract.empty:
                financials = _parse_sina_financials(abstract)
                _cache_set("fundamental", cache_key, financials)
        except Exception:
            pass

    return {
        "code": code,
        "industry": info["industry"],
        "pe": info["pe"],
        "pb": info["pb"],
        "free_float_shares": info["free_float_shares"],
        "financials": financials,
    }


def _fetch_pe_pb_push2(code: str) -> dict:
    """eastmoney push2 单股实时接口兜底拉 PE/PB/流通股本。

    fields:
    - f9: PE(动态)
    - f23: PB
    - f2: 最新价
    - f84: 流通股本(股)
    - f117: 流通市值(元)
    无 f84 时用 f117/f2 反推流通股本。
    """
    secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f9,f23,f2,f84,f117,f116,f162"
    try:
        r = _session.get(url, timeout=(10, 15))
        r.raise_for_status()
        data = r.json().get("data", {})
        pe = _safe_float(data.get("f9"))
        pb = _safe_float(data.get("f23"))
        price = _safe_float(data.get("f2"))
        float_shares = _safe_float(data.get("f84"))
        float_mv = _safe_float(data.get("f117"))

        # f84 缺失时用 流通市值 / 最新价 反推
        if not float_shares and float_mv and price and price > 0:
            float_shares = float_mv / price

        return {
            "pe": pe,
            "pb": pb,
            "free_float_shares": float_shares,
        }
    except Exception:
        return {}


def _parse_em_financials(abstract) -> list:
    """解析 em 财务摘要为标准 financials 列表。"""
    result = []
    for _, row in abstract.iterrows():
        fin = {
            "period": str(row.get("选项", row.get("日期", ""))),
            "net_profit": _safe_float(row.get("净利润")),
            "revenue": _safe_float(row.get("营业总收入")),
            "operating_cf": _safe_float(row.get("经营活动产生的现金流量净额")),
            "roe": _safe_float(row.get("净资产收益率(加权)", row.get("ROE"))),
            "roic": _safe_float(row.get("ROIC")),
        }
        capex = _safe_float(row.get("固定资产折旧、油气资产折耗、生产性生物资产折旧"))
        if fin["operating_cf"] is not None and capex is not None:
            fin["fcf"] = fin["operating_cf"] - capex
        else:
            fin["fcf"] = fin["operating_cf"]
        result.append(fin)
    return result


def _parse_sina_financials(abstract) -> list:
    """解析 sina 财务摘要(宽表:行=指标,列=日期)。

    sina 返回 80 行 × 60 列,需要转置并提取关键指标。
    FCF 用经营现金流代理(sina 摘要无 capex)。
    扩展提取:扣非净利润、营业成本(算毛利率)、每股收益。
    """
    # 指标 -> 标准字段映射(顺序敏感:更具体的指标放前面避免被通用名匹配)
    metric_map = {
        "扣非净利润": "operating_profit",  # 扣除非经常性损益的净利润
        "归母净利润": "net_profit",
        "营业总收入": "revenue",
        "营业成本": "operating_cost",  # 用于算毛利率
        "经营现金流量净额": "operating_cf",
        "投入资本回报率": "roic",
        "净资产收益率(ROE)": "roe",
        "基本每股收益": "eps",
        "销售毛利率": "gross_margin_pct",  # 部分公司直接提供
        "资产负债率": "debt_ratio_pct",  # 直接提供百分比
        "资产总计": "total_assets",
        "负债合计": "total_liabilities",
        "归属于母公司股东权益合计": "net_assets",
        "股东权益合计": "net_assets",
        "所有者权益合计": "net_assets",
        "权益乘数": "equity_multiplier",  # sina 直接提供,用于杜邦分析
        "总资产周转率": "asset_turnover",  # sina 直接提供
        "总资产净利率_平均": "roa",
    }

    # 收集各指标的时间序列
    series = {}
    for _, row in abstract.iterrows():
        metric_name = str(row["指标"])
        for sina_key, std_key in metric_map.items():
            if sina_key in metric_name and std_key not in series:
                # 取该行的所有日期列(前两列是 选项、指标)
                date_cols = list(row.index[2:])
                values = []
                for col in date_cols:
                    v = _safe_float(row[col])
                    if v is not None:
                        values.append((col, v))
                if values:
                    series[std_key] = values
                break

    # 对齐时间轴(以净利润的日期为准)
    if "net_profit" not in series:
        return []

    periods = [p for p, _ in series["net_profit"]]
    result = []
    for i, period in enumerate(periods):
        fin = {"period": period}
        for key, vals in series.items():
            if i < len(vals):
                fin[key] = vals[i][1]
            else:
                fin[key] = None
        # FCF 用经营现金流代理(OCF - capex,sina 摘要无 capex,用 OCF 近似)
        if fin.get("operating_cf") is not None and fin.get("net_profit") is not None:
            fin["fcf"] = fin["operating_cf"]
        else:
            fin["fcf"] = None
        # 毛利率:若 sina 未直接提供,用 (revenue - operating_cost) / revenue 计算
        if fin.get("gross_margin_pct") is None and fin.get("revenue") and fin.get("operating_cost") and fin["revenue"] != 0:
            fin["gross_margin_pct"] = round((fin["revenue"] - fin["operating_cost"]) / fin["revenue"] * 100, 2)
        # 资产负债率:若无直接提供,用 (负债合计 / 资产总计) * 100 兜底
        if fin.get("debt_ratio_pct") is None and fin.get("total_liabilities") and fin.get("total_assets") and fin["total_assets"] != 0:
            fin["debt_ratio_pct"] = round(fin["total_liabilities"] / fin["total_assets"] * 100, 2)
        # 净资产兜底:若无直接提供,用 总资产 - 负债合计
        if fin.get("net_assets") is None and fin.get("total_assets") and fin.get("total_liabilities"):
            fin["net_assets"] = fin["total_assets"] - fin["total_liabilities"]
        # 总资产兜底:若无直接提供但有净资产+资产负债率,用 net_assets / (1 - debt_ratio/100)
        if fin.get("total_assets") is None and fin.get("net_assets") and fin.get("debt_ratio_pct") is not None and fin["debt_ratio_pct"] < 100:
            fin["total_assets"] = fin["net_assets"] / (1 - fin["debt_ratio_pct"] / 100)
        result.append(fin)
    return result


def _safe_float(v):
    try:
        if v is None or v == "" or v == "--":
            return None
        import re
        s = str(v).replace(",", "").replace("%", "").strip()
        if s in ("", "nan", "None", "-"):
            return None
        f = float(s)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _parse_shares(v) -> float:
    """解析流通股本,返回股数。支持 "X.XX亿股" / "X万股" / "X股" / 纯数字。"""
    if v is None or v == "" or v == "--":
        return None
    s = str(v).replace(",", "").strip()
    if s in ("", "nan", "None", "-"):
        return None
    try:
        if "亿" in s:
            return float(s.replace("亿", "").replace("股", "").strip()) * 1e8
        if "万" in s:
            return float(s.replace("万", "").replace("股", "").strip()) * 1e4
        return float(s.replace("股", "").strip())
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description="A股日线数据拉取(akshare,前复权)")
    p.add_argument("code", help="股票代码(000001)或名称(平安银行)")
    p.add_argument("--days", type=int, default=120, help="回看天数,默认 120")
    args = p.parse_args()
    check_akshare_version()
    try:
        result = fetch(args.code, args.days)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
