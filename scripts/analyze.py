"""主入口:串联 fetch_data + compute_indicators + fundamental,输出结构化 JSON。

用法:
    python analyze.py 000001
    python analyze.py 平安银行 --days 120
    python analyze.py 000001 --days 60 --no-fundamental
"""
import argparse
import json
import sys
import numpy as np

from fetch_data import fetch, fetch_fundamental, check_akshare_version
from compute_indicators import compute
from fundamental import analyze_fundamental
from fetch_research_reports import fetch_research_reports
from fetch_capital_flow import fetch_capital_flow


def _json_default(o):
    """处理 numpy 类型无法 JSON 序列化的问题。"""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def analyze(code_or_name: str, days: int = 120, include_fundamental: bool = True) -> dict:
    raw = fetch(code_or_name, days)
    indicators = compute(raw["daily"])

    fundamental = None
    free_float_shares = None
    research_reports = None
    if include_fundamental:
        try:
            fund_data = fetch_fundamental(raw["code"])
            free_float_shares = fund_data.get("free_float_shares")
            # 当前价(用于目标价上涨空间计算):从 indicators 取
            current_price = indicators.get("position", {}).get("current_close")
            # 拉取机构研报(失败不阻塞主流程)
            try:
                research_reports = fetch_research_reports(
                    raw["code"], lookback_days=365, current_price=current_price
                )
            except Exception as e:
                research_reports = {"error": str(e), "total_reports": 0, "reports": []}
            fundamental = analyze_fundamental(
                code=fund_data["code"],
                name=raw["name"],
                industry=fund_data["industry"],
                pe=fund_data["pe"],
                pb=fund_data["pb"],
                financials=fund_data["financials"],
                research_reports=research_reports,
            )
        except Exception as e:
            fundamental = {"error": str(e)}

    # 用流通股本重算筹码(换手率衰减模型);失败降级到固定 decay
    if free_float_shares is not None:
        try:
            from compute_indicators import recompute_chip_with_float
            indicators = recompute_chip_with_float(indicators, raw["daily"], free_float_shares)
        except Exception:
            pass

    # 换手率指标 + 量价细化(需要流通股本)
    if free_float_shares is not None:
        try:
            from compute_indicators import compute_turnover, compute_volume_price_detail
            indicators["turnover"] = compute_turnover(raw["daily"], free_float_shares)
            position_label = indicators.get("position", {}).get("label", "")
            indicators["volume_price_detail"] = compute_volume_price_detail(
                raw["daily"],
                indicators["turnover"],
                position_label,
            )
        except Exception:
            pass

    # 拉取主力资金流(失败不阻塞主流程)
    try:
        capital_flow = fetch_capital_flow(raw["code"], days=60)
    except Exception as e:
        capital_flow = {"available": False, "error": str(e)}
    indicators["capital_flow"] = capital_flow

    # 筹码 × 主力资金流交叉验证(在 recompute_chip_with_float 之后,确保 short_term_chip 已重算)
    try:
        from chip_distribution import cross_validate_chip_capital
        indicators["chip_cross_validation"] = cross_validate_chip_capital(
            chip=indicators.get("chip", {}),
            short_term_chip=indicators.get("short_term_chip", {}),
            capital_flow=capital_flow,
        )
    except Exception:
        pass

    # 拉取最近新闻(失败不阻塞主流程)
    try:
        from fetch_news import fetch_news
        indicators["news"] = fetch_news(raw["code"], limit=10, lookback_days=30)
    except Exception as e:
        indicators["news"] = {"available": False, "error": str(e)}

    return {
        "basic": {
            "code": raw["code"],
            "name": raw["name"],
            "days_returned": raw["days_returned"],
            "data_insufficient": raw["data_insufficient"],
            "free_float_shares": free_float_shares,
        },
        "indicators": indicators,
        "fundamental": fundamental,
    }


def main():
    p = argparse.ArgumentParser(description="A股量价 + 筹码 + 公司质地分析")
    p.add_argument("code", help="股票代码(000001)或名称(平安银行)")
    p.add_argument("--days", type=int, default=120, help="回看天数,默认 120")
    p.add_argument("--no-fundamental", action="store_true", help="跳过基本面分析")
    args = p.parse_args()
    check_akshare_version()
    try:
        result = analyze(args.code, args.days, include_fundamental=not args.no_fundamental)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, default=_json_default), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
