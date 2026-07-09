"""公司质地判断:成长 vs 周期分类、PE 陷阱检测、ROIC/FCF/毛利率/扣非/营收 鉴别。

底层逻辑:
- 成长股:靠需求扩张 + 技术/品牌护城河,利润复利增长
- 周期股:需求固定,利润由供给松紧决定,重资产扩产滞后易过剩暴跌
- 周期转成长潜力:财务呈现周期性,但行业有 AI/国产替代/机器人等叙事,
  主业可能因结构性需求转成长(需业绩验证)

PE 陷阱:
- 周期股利润暴增 PE 极低 = 见顶信号(行业亏损 PE 极高反而是买点)
- 成长股高 PE 靠持续业绩增长消化

鉴别指标:
- ROIC + 再投资率:真成长长期高 ROIC 且利润可再投;伪成长 ROIC 下滑、扩产消耗价值
- FCF / 净利润:优质成长现金流匹配利润;周期股景气期盈利但现金流长期为负
- 毛利率趋势:上升 = 定价权增强 / 成本下行;下降 = 竞争加剧 / 成本上行
- 扣非净利润 vs 净利润:差异大 = 一次性损益多,盈利质量差
- 营收增长:利润增长前提;营收不增利润增 = 利润操纵嫌疑
"""
from typing import List, Dict, Any, Optional
import statistics


CYCLICAL_INDUSTRIES = {
    "钢铁", "有色金属", "煤炭", "化工", "建筑材料", "机械", "房地产",
    "证券", "保险", "航运", "航空", "造纸", "纺织服装",
    "石油石化", "基础化学", "农药兽药",
    # 注意:银行从 CYCLICAL 移除,改走 bank_quality 专用路径
}

# 银行股识别:A 股主要银行股代码(国有大行 + 股份制 + 城商行)
BANK_CODES = {
    "600036",  # 招商银行
    "601398",  # 工商银行
    "601939",  # 建设银行
    "601288",  # 农业银行
    "601988",  # 中国银行
    "601328",  # 交通银行
    "600000",  # 浦发银行
    "600016",  # 民生银行
    "600015",  # 华夏银行
    "600011",  # 照片银行(原 600011 为华能国际,此处应为 600015 华夏)
    "601166",  # 兴业银行
    "601818",  # 光大银行
    "601998",  # 中信银行
    "601009",  # 南京银行
    "601169",  # 北京银行
    "601229",  # 上海银行
    "601577",  # 长沙银行
    "601838",  # 成都银行
    "601916",  # 浙商银行
    "002142",  # 宁波银行
    "002839",  # 张家港行
    "002936",  # 郑州银行
    "002948",  # 青岛银行
    "002958",  # 青农商行
    "6066",    # 重庆银行(H)
}


def is_bank(industry: str, code: str = "", name: str = "") -> bool:
    """识别银行股:行业含'银行'或代码在 BANK_CODES。"""
    if industry and "银行" in industry:
        return True
    if code and code in BANK_CODES:
        return True
    if name and "银行" in name:
        return True
    return False

GROWTH_INDUSTRIES = {
    "医药生物", "医疗器械", "食品饮料", "电子", "半导体", "计算机",
    "软件开发", "互联网服务", "通信", "消费电子", "生物制品",
}

GROWTH_TO_CYCLICAL_RISK = {
    "光伏设备", "电池", "汽车整车", "新能源车", "风电设备",
    "消费电子", "半导体", "显示器件",
}

# 成长叙事(AI 算力 / 国产替代 / 机器人 / 创新药等)
# 用于识别"财务周期但主业有转成长潜力"的情况
GROWTH_NARRATIVES = {
    "ai_compute": {
        "keywords": ["存储芯片", "DRAM", "NAND", "HBM", "DDR5", "光模块",
                     "AI算力", "算力", "高速连接器", "PCB"],
        "narrative": "AI 算力链(AI 需求拉动高端存储/光模块/高速互联涨价周期)",
        "growth_potential": "周期转成长潜力(AI 拉动高端产品结构性需求,DDR5/HBM 涨价持续)",
    },
    "domestic_substitution": {
        "keywords": ["半导体设备", "半导体材料", "EDA", "MCU", "国产替代",
                     "自主可控", "光刻", "刻蚀", "薄膜沉积"],
        "narrative": "国产替代(政策驱动 + 美国制裁加速国产化)",
        "growth_potential": "成长性强(国产化率低,提升空间大)",
    },
    "robotics": {
        "keywords": ["减速器", "伺服", "谐波", "机器人", "RV减速器", "丝杠"],
        "narrative": "人形机器人产业链(Optimus / 国产机器人量产预期)",
        "growth_potential": "成长叙事(需量产验证,题材估值溢价高)",
    },
    "innovative_drug": {
        "keywords": ["ADC", "GLP-1", "创新药", "CXO", "生物医药"],
        "narrative": "创新药出海 + ADC/GLP-1 大单品",
        "growth_potential": "成长性强(若管线兑现)",
    },
    "high_end_manufacturing": {
        "keywords": ["数控机床", "工业软件", "高端装备", "航空航天",
                     "电力设备", "输配电", "电网", "特高压", "智能电网",
                     "新型电力系统", "海外电网", "工程机械", "叉车"],
        "narrative": "高端制造升级(进口替代 + 出海 + 电网投资)",
        "growth_potential": "成长性中等(传统制造向高端升级 + 海外业务扩张)",
    },
}

# 政治/地缘/政策风险类型
# 用于识别公司业务的外部环境风险,需结合最新新闻和企业公告动态评估
GEOPOLITICAL_RISK_TYPES = {
    "overseas_resource": {
        "keywords": ["矿业", "有色金属", "黄金", "铜矿", "锂矿", "钴矿", "铁矿",
                     "石油", "天然气", "油气开采", "钾肥", "镍", "铝", "煤炭"],
        "risk": "海外资产地缘风险",
        "description": "海外矿山/油气田所在国政局动荡、国有化、税收政策变化、内战冲突",
        "examples": "紫金矿业(塞尔维亚/刚果金/苏里南)、赣锋锂业(墨西哥/阿根廷)、中海油(海外油气)、洛阳钼业(刚果金)",
        "verification": "查最新新闻:资源国政局/政策/冲突;查公告:海外资产减值/税收变化",
    },
    "overseas_business": {
        "keywords": ["电力设备", "输配电", "电网", "工程机械", "家电", "白色家电",
                     "黑电", "海外工程", "国际工程", "出海", "海外业务", "港口机械",
                     "叉车", "客车", "重卡", "纺织服装", "鞋服", "消费电子出海"],
        "risk": "海外业务地缘/汇率风险",
        "description": "海外业务占比高,面临汇率波动、贸易摩擦、海外项目政治风险、客户所在国政策变化",
        "examples": "思源电气(欧洲电网改造订单)、三一重工(海外工程机械)、海尔智家(全球白电)、宇通客车(海外客车)",
        "verification": "查最新新闻:汇率波动/贸易摩擦/海外项目履约;查公告:海外业务占比/汇率影响/海外子公司",
    },
    "sanction": {
        "keywords": ["半导体", "芯片", "EDA", "光刻", "刻蚀", "AI算力", "GPU",
                     "先进制程", "军工", "航天", "超算", "量子"],
        "risk": "美国制裁/实体清单风险",
        "description": "美国出口管制、实体清单、技术封锁,影响设备/材料/EDA/先进制程供应",
        "examples": "中芯国际(设备受限)、寒武纪(实体清单)、北方华创(国产替代反向受益)",
        "verification": "查最新新闻:美国 BIS 实体清单更新、出口管制新规;查公告:供应链影响/豁免到期",
    },
    "policy_dependency": {
        "keywords": ["光伏", "新能源车", "风电", "储能", "创新药", "CXO",
                     "医疗器械", "教育培训", "游戏", "平台经济"],
        "risk": "政策依赖/监管变化风险",
        "description": "补贴退坡、集采降价、监管政策变化,业绩依赖政策延续",
        "examples": "光伏(补贴退坡+产能过剩)、创新药(集采)、CXO(海外制裁+国内监管)、教育(双减)、游戏(版号)",
        "verification": "查最新政策:补贴/集采/监管文件;查公告:政府补助占比/集采中标价",
    },
    "strategic_resource": {
        "keywords": ["稀土", "钨", "锑", "锗", "镓", "铟"],
        "risk": "战略资源出口管制风险(反向受益)",
        "description": "中国对战略资源实施出口管制,可能限制海外收入但提升国内定价权",
        "examples": "北方稀土(出口管制受益)、厦门钨业、湖南黄金",
        "verification": "查最新政策:出口管制清单;查公告:出口业务占比变化",
    },
    "consumer_policy": {
        "keywords": ["白酒", "高端消费", "医美", "食品饮料"],
        "risk": "消费政策/反腐风险",
        "description": "消费税改革、反腐影响高端消费、医美监管收紧",
        "examples": "茅台(消费税预期)、爱美客(医美监管)",
        "verification": "查最新政策:消费税/医美监管;查公告:渠道库存/批价变化",
    },
}


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


def classify_by_industry(industry: str) -> Dict[str, Any]:
    """按行业做初始分类(可被财务指标覆盖)。"""
    ind = industry or ""
    if any(c in ind for c in CYCLICAL_INDUSTRIES):
        initial = "周期"
    elif any(g in ind for g in GROWTH_INDUSTRIES):
        initial = "成长"
    else:
        initial = "待定"

    risk_flag = any(r in ind for r in GROWTH_TO_CYCLICAL_RISK)
    return {
        "industry": ind,
        "initial_guess": initial,
        "growth_to_cyclical_risk": risk_flag,
        "note": "成长赛道可能转周期(渗透率触顶 + 产能集中释放)" if risk_flag else "",
    }


def classify_by_narrative(industry: str, name: str = "") -> Dict[str, Any]:
    """识别行业成长叙事(AI / 国产替代 / 机器人等)。

    财务指标是滞后的,只看历史财务会把"周期转成长中"的公司错判为周期股。
    行业叙事识别用于提示"虽财务周期,但主业有转成长潜力"。
    叙事需结合最新季报业绩验证,不能仅凭叙事买入。
    """
    text = f"{industry} {name}"
    matched = []
    for narrative_id, info in GROWTH_NARRATIVES.items():
        if any(kw in text for kw in info["keywords"]):
            matched.append({
                "id": narrative_id,
                "narrative": info["narrative"],
                "growth_potential": info["growth_potential"],
            })
    return {
        "has_narrative": len(matched) > 0,
        "narratives": matched,
        "note": "行业有成长叙事,财务周期判定可能低估其成长潜力,需业绩验证" if matched else "",
    }


def classify_geopolitical_risk(industry: str, name: str = "") -> Dict[str, Any]:
    """识别公司业务的政治/地缘/政策风险类型。

    脚本只做"风险敞口识别"——判断公司业务是否暴露在某类外部风险下。
    风险是否兑现、兑现程度,需结合最新新闻和企业公告动态评估。
    脚本输出的 risk_types 是"需要核查的风险清单",不是"已发生的风险"。
    """
    text = f"{industry} {name}"
    matched = []
    for risk_id, info in GEOPOLITICAL_RISK_TYPES.items():
        if any(kw in text for kw in info["keywords"]):
            matched.append({
                "id": risk_id,
                "risk": info["risk"],
                "description": info["description"],
                "examples": info["examples"],
                "verification": info["verification"],
            })
    return {
        "has_risk": len(matched) > 0,
        "risk_types": matched,
        "note": "识别到外部风险敞口,需结合最新新闻和企业公告核查兑现情况" if matched else "无明显政治/地缘/政策风险敞口",
    }


def compute_roic_stability(roics: List[float], periods: List[str] = None) -> Dict[str, Any]:
    """ROIC 稳定性:均值、标准差、变异系数。

    若提供 periods,优先用年度数据(period 末尾为 '1231')算 cv,避免季度季节性失真。
    电力设备、电网、工程机械等季节性行业 Q4 集中回款,季度 ROIC 波动天然大,
    用季度数据算 cv 会高估不稳定性,误判为周期股。
    """
    if periods and len(periods) == len(roics):
        annual_pairs = [(str(p), r) for p, r in zip(periods, roics)
                        if r is not None and str(p).endswith("1231")]
        if len(annual_pairs) >= 2:
            valid = [r for _, r in annual_pairs]
            used_periods = [p for p, _ in annual_pairs]
            seasonal_adjusted = True
        else:
            valid = [r for r in roics if r is not None]
            used_periods = None
            seasonal_adjusted = False
    else:
        valid = [r for r in roics if r is not None]
        used_periods = None
        seasonal_adjusted = False

    if len(valid) < 2:
        return {"available": False, "mean": None, "std": None, "cv": None, "trend": None,
                "seasonal_adjusted": seasonal_adjusted}
    mean = statistics.mean(valid)
    std = statistics.stdev(valid)
    cv = abs(std / mean) if mean != 0 else float("inf")
    if len(valid) >= 2:
        if valid[-1] > valid[0] * 1.1:
            trend = "上升"
        elif valid[-1] < valid[0] * 0.9:
            trend = "下降"
        else:
            trend = "平稳"
    else:
        trend = "数据不足"
    return {
        "available": True,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "cv": round(cv, 4),
        "trend": trend,
        "values": valid,
        "seasonal_adjusted": seasonal_adjusted,
        "used_periods": used_periods,
    }


def compute_roe_stability(roes: List[float], periods: List[str] = None) -> Dict[str, Any]:
    """ROE 稳定性(巴菲特选股核心指标)。

    巴菲特标准:
    - 5-10 年均 ROE > 15%
    - 单年 ROE 不低于 12%

    返回:mean / std / cv / min / trend / buffett_pass(均>15% + 单年≥12%)
    """
    valid_idx = [(i, r) for i, r in enumerate(roes) if r is not None]
    if len(valid_idx) < 2:
        return {"available": False, "reason": "ROE 数据不足"}

    # 季节性调整:优先用年报(period endswith "1231")
    if periods and len(periods) == len(roes):
        annual_pairs = [(str(p), r) for p, r in zip(periods, roes)
                        if r is not None and str(p).endswith("1231")]
        if len(annual_pairs) >= 2:
            valid = [r for _, r in annual_pairs]
            used_periods = [p for p, _ in annual_pairs]
            seasonal_adjusted = True
        else:
            valid = [r for _, r in valid_idx]
            used_periods = None
            seasonal_adjusted = False
    else:
        valid = [r for _, r in valid_idx]
        used_periods = None
        seasonal_adjusted = False

    mean = sum(valid) / len(valid)
    std = (sum((r - mean) ** 2 for r in valid) / len(valid)) ** 0.5
    cv = std / mean if mean != 0 else None
    min_roe = min(valid)
    max_roe = max(valid)

    # 趋势:比较前 1/3 和后 1/3
    n = len(valid)
    if n >= 4:
        first_third = valid[: max(1, n // 3)]
        last_third = valid[-(max(1, n // 3)):]
        first_mean = sum(first_third) / len(first_third)
        last_mean = sum(last_third) / len(last_third)
        if last_mean > first_mean * 1.10:
            trend = "上升"
        elif last_mean < first_mean * 0.90:
            trend = "下降"
        else:
            trend = "平稳"
    else:
        trend = "数据不足"

    # 巴菲特标准
    buffett_mean_pass = mean > 15
    buffett_min_pass = min_roe >= 12
    buffett_pass = buffett_mean_pass and buffett_min_pass

    return {
        "available": True,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "cv": round(cv, 2) if cv is not None else None,
        "min": round(min_roe, 2),
        "max": round(max_roe, 2),
        "trend": trend,
        "values": valid,
        "seasonal_adjusted": seasonal_adjusted,
        "used_periods": used_periods,
        "buffett_filter": {
            "mean_pass": buffett_mean_pass,
            "min_pass": buffett_min_pass,
            "pass": buffett_pass,
            "criteria": "5-10 年均 ROE > 15% + 单年 ≥ 12%",
        },
    }


def compute_dupont_analysis(financials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """杜邦分析:ROE = 净利率 × 总资产周转率 × 权益乘数。

    三种模式:
    - 高净利率(茅台/爱马仕)- 安全,定价权强
    - 高周转率(Costco/沃尔玛)- 稳健,运营效率高
    - 高杠杆(房企/银行)- 风险大,杠杆依赖

    数据来源优先级:
    - 优先用 sina 直接提供的 equity_multiplier / asset_turnover(更准)
    - 否则用 total_assets / net_assets 计算

    返回最新一期的三因素拆解 + 主导模式判断。
    """
    # 优先用 sina 直接提供的字段
    valid_direct = [
        f for f in financials
        if f.get("equity_multiplier") is not None
        and f.get("asset_turnover") is not None
        and f.get("net_profit") is not None
        and f.get("revenue") is not None and f.get("revenue") != 0
    ]
    # 兜底:用 total_assets / net_assets 计算
    valid_compute = [
        f for f in financials
        if f.get("total_assets") is not None and f.get("net_assets") is not None
        and f.get("net_profit") is not None
        and f.get("revenue") is not None and f.get("revenue") != 0
        and f.get("total_assets", 0) != 0 and f.get("net_assets", 0) != 0
    ]

    if valid_direct:
        latest = valid_direct[-1]
        net_margin = latest["net_profit"] / latest["revenue"]
        asset_turnover = latest["asset_turnover"]
        equity_multiplier = latest["equity_multiplier"]
        source = "sina_direct"
    elif valid_compute:
        latest = valid_compute[-1]
        net_margin = latest["net_profit"] / latest["revenue"]
        asset_turnover = latest["revenue"] / latest["total_assets"]
        equity_multiplier = latest["total_assets"] / latest["net_assets"]
        source = "computed"
    else:
        return {"available": False, "reason": "缺杜邦分析所需字段(净利/营收/总资产/净资产 或 权益乘数/周转率)"}

    derived_roe = net_margin * asset_turnover * equity_multiplier

    # 模式判断(对比三因素的水平)
    # 高净利率:net_margin > 15% (茅台 ~50%,爱马仕 ~30%)
    # 高周转:asset_turnover > 1.0 (Costco ~3.5,沃尔玛 ~2.5)
    # 高杠杆:equity_multiplier > 5 (房企 ~10,银行 ~12)
    mode_signals = {
        "high_margin": net_margin > 0.15,
        "high_turnover": asset_turnover > 1.0,
        "high_leverage": equity_multiplier > 5.0,
    }
    # 主导模式:取最显著的那个(标准化对比)
    norm_margin = min(net_margin / 0.15, 1.0) if net_margin > 0 else 0
    norm_turnover = min(asset_turnover / 1.0, 1.0) if asset_turnover > 0 else 0
    norm_leverage = min(equity_multiplier / 5.0, 1.0) if equity_multiplier > 0 else 0
    norms = {"高净利率": norm_margin, "高周转": norm_turnover, "高杠杆": norm_leverage}
    dominant_mode = max(norms, key=norms.get) if any(norms.values()) else "无明显主导"

    mode_label = {
        "高净利率": "高净利率驱动(茅台/爱马仕式,定价权强,安全)",
        "高周转": "高周转驱动(Costco/沃尔玛式,运营效率,稳健)",
        "高杠杆": "高杠杆驱动(房企/银行式,风险大,杠杆依赖)",
        "无明显主导": "无明显主导模式",
    }[dominant_mode]

    return {
        "available": True,
        "period": latest.get("period"),
        "net_margin": round(net_margin * 100, 2),  # %
        "asset_turnover": round(asset_turnover, 3),
        "equity_multiplier": round(equity_multiplier, 2),
        "derived_roe": round(derived_roe * 100, 2),  # %
        "reported_roe": latest.get("roe"),
        "dominant_mode": dominant_mode,
        "mode_label": mode_label,
        "mode_signals": mode_signals,
        "source": source,
    }


def compute_buffett_filter(financials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """巴菲特三步筛选:
    1. 5-10 年均 ROE > 15% + 单年 ≥ 12%
    2. 资产负债率 < 50%
    3. 经营现金流 ≥ 净利润

    返回每步通过情况 + 综合判断。
    """
    roes = [_safe_float(f.get("roe")) for f in financials]
    periods = [f.get("period") for f in financials]
    roe_stab = compute_roe_stability(roes, periods)

    # 第 1 步:ROE 标准
    step1_pass = roe_stab.get("buffett_filter", {}).get("pass", False) if roe_stab.get("available") else False
    step1_detail = (
        f"5-10 年均 ROE {roe_stab.get('mean')}% (>{15}%? {roe_stab.get('buffett_filter',{}).get('mean_pass')}) / "
        f"单年最低 {roe_stab.get('min')}% (≥12%? {roe_stab.get('buffett_filter',{}).get('min_pass')})"
        if roe_stab.get("available") else "ROE 数据不足"
    )

    # 第 2 步:资产负债率 < 50%(最新一期)
    latest_with_debt = [f for f in financials if f.get("debt_ratio_pct") is not None]
    if latest_with_debt:
        latest_debt = latest_with_debt[-1]["debt_ratio_pct"]
        step2_pass = latest_debt < 50
        step2_detail = f"最新资产负债率 {latest_debt}% (<50%? {step2_pass})"
    else:
        step2_pass = False
        step2_detail = "资产负债率数据缺失"

    # 第 3 步:经营现金流 ≥ 净利润(最新一期)
    latest_with_cf = [f for f in financials if f.get("operating_cf") is not None and f.get("net_profit") is not None]
    if latest_with_cf:
        cf = latest_with_cf[-1]["operating_cf"]
        np_ = latest_with_cf[-1]["net_profit"]
        step3_pass = cf >= np_
        step3_detail = f"最新经营现金流 {cf:,.0f} vs 净利润 {np_:,.0f} (≥? {step3_pass})"
    else:
        step3_pass = False
        step3_detail = "现金流/净利润数据缺失"

    all_pass = step1_pass and step2_pass and step3_pass

    return {
        "available": True,
        "step1_roe": {"pass": step1_pass, "detail": step1_detail},
        "step2_debt": {"pass": step2_pass, "detail": step2_detail},
        "step3_cashflow": {"pass": step3_pass, "detail": step3_detail},
        "all_pass": all_pass,
        "interpretation": (
            "巴菲特三步全过 - 长期稳、低负债、现金流匹配,优质标的"
            if all_pass else
            "未全过 - " + "; ".join(
                f"步骤{ i+1 } 未通过"
                for i, p in enumerate([step1_pass, step2_pass, step3_pass]) if not p
            )
        ),
    }


def detect_fake_roe(financials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """假高 ROE 识别:
    1. 高杠杆驱动(权益乘数 > 5 且净利率 < 10%) - 银行/房企式,风险大
    2. 一次性收益(扣非/NI < 0.7) - 卖资产凑利润
    3. 回购缩分母(净资产同比下降但净利润持平/上升) - 波音案例

    返回各项警告 + 综合判断。
    """
    warnings = []

    # 1. 高杠杆驱动
    dupont = compute_dupont_analysis(financials)
    if dupont.get("available"):
        em = dupont["equity_multiplier"]
        nm = dupont["net_margin"]
        if em > 5 and nm < 10:
            warnings.append({
                "type": "high_leverage",
                "severity": "高" if em > 8 else "中",
                "detail": f"权益乘数 {em} (>5) + 净利率 {nm}% (<10%) - ROE 由杠杆驱动,非经营效率,风险大",
                "case": "房企/银行式高杠杆,波音举债回购案例",
            })

    # 2. 一次性收益(用最新一期的扣非/NI)
    latest_with_op = [f for f in financials if f.get("operating_profit") is not None and f.get("net_profit") is not None and f.get("net_profit") != 0]
    if latest_with_op:
        latest = latest_with_op[-1]
        ratio = latest["operating_profit"] / latest["net_profit"]
        if ratio < 0.7:
            warnings.append({
                "type": "one_shot_gain",
                "severity": "高" if ratio < 0.4 else "中",
                "detail": f"扣非/NI = {ratio:.2f} (<0.7) - 主业贡献低,ROE 靠一次性损益(卖资产/政府补贴)撑高",
                "case": "卖资产、政府补贴、投资收益一次性抬高净利润",
            })

    # 3. 回购缩分母(净资产同比下降,净利润未同比下降)
    valid_assets = [(i, f) for i, f in enumerate(financials) if f.get("net_assets") is not None and f.get("net_profit") is not None]
    if len(valid_assets) >= 2:
        prev = valid_assets[-2][1]
        curr = valid_assets[-1][1]
        assets_change = (curr["net_assets"] - prev["net_assets"]) / prev["net_assets"] if prev["net_assets"] != 0 else 0
        profit_change = (curr["net_profit"] - prev["net_profit"]) / abs(prev["net_profit"]) if prev["net_profit"] != 0 else 0
        # 净资产下降 > 5% 且净利润未下降(持平或上升)= 回购缩分母嫌疑
        if assets_change < -0.05 and profit_change >= -0.05:
            warnings.append({
                "type": "buyback_shrink",
                "severity": "中",
                "detail": f"净资产同比下降 {abs(assets_change)*100:.1f}% 但净利润未下降(变化 {profit_change*100:.1f}%) - 疑似回购缩分母抬高 ROE",
                "case": "波音式举债回购,净资产缩小推高 ROE,但债务风险累积",
            })

    return {
        "available": True,
        "warnings": warnings,
        "is_fake": len(warnings) > 0,
        "warning_count": len(warnings),
        "interpretation": (
            f"⚠️ 检测到 {len(warnings)} 个假高 ROE 信号 - 需警惕 ROE 来源"
            if warnings else "✅ 未检测到假高 ROE 信号,ROE 来源健康"
        ),
    }


def compute_profit_growth(profits: List[float]) -> Dict[str, Any]:
    """利润增长一致性:逐年增长率 + 是否全部为正。"""
    valid = [p for p in profits if p is not None and p != 0]
    if len(valid) < 3:
        return {"available": False, "growth_rates": [], "all_positive": None, "volatile": None}
    rates = []
    for i in range(1, len(valid)):
        if valid[i - 1] > 0:
            rates.append((valid[i] - valid[i - 1]) / abs(valid[i - 1]))
        else:
            rates.append(None)
    rates_clean = [r for r in rates if r is not None]
    if not rates_clean:
        return {"available": False, "growth_rates": [], "all_positive": None, "volatile": None}
    all_pos = all(r > 0 for r in rates_clean)
    volatile = any(r < -0.3 for r in rates_clean) or (statistics.stdev(rates_clean) > 0.5 if len(rates_clean) > 1 else False)
    return {
        "available": True,
        "growth_rates": [round(r, 4) for r in rates_clean],
        "all_positive": all_pos,
        "volatile": volatile,
        "min_rate": round(min(rates_clean), 4),
        "max_rate": round(max(rates_clean), 4),
    }


def compute_fcf_quality(fcfs: List[float], profits: List[float]) -> Dict[str, Any]:
    """FCF / 净利润比率:现金流是否匹配利润。"""
    pairs = [
        (f, p) for f, p in zip(fcfs, profits)
        if f is not None and p is not None and p > 0
    ]
    if len(pairs) < 2:
        return {"available": False, "ratios": [], "mean_ratio": None, "match": None}
    ratios = [f / p for f, p in pairs]
    mean_ratio = statistics.mean(ratios)
    if mean_ratio >= 0.7:
        match = "良好(现金流匹配利润)"
    elif mean_ratio >= 0.3:
        match = "一般(现金流部分匹配)"
    else:
        match = "差(盈利但现金流不足,警惕)"
    return {
        "available": True,
        "ratios": [round(r, 4) for r in ratios],
        "mean_ratio": round(mean_ratio, 4),
        "match": match,
    }


def compute_gross_margin_trend(margins: List[float]) -> Dict[str, Any]:
    """毛利率趋势:均值/最近值/趋势(上升=定价权增强,下降=竞争加剧)。"""
    valid = [m for m in margins if m is not None]
    if len(valid) < 2:
        return {"available": False, "mean": None, "latest": None, "trend": None}
    mean = statistics.mean(valid)
    latest = valid[-1]
    # 趋势:最近值 vs 均值,差异 > 2 个百分点算显著
    if latest > mean + 2:
        trend = "上升"
    elif latest < mean - 2:
        trend = "下降"
    else:
        trend = "平稳"
    # 计算最近 3 期斜率(若数据足够)
    recent = valid[-3:] if len(valid) >= 3 else valid
    if len(recent) >= 2:
        delta = recent[-1] - recent[0]
    else:
        delta = 0
    return {
        "available": True,
        "mean": round(mean, 2),
        "latest": round(latest, 2),
        "trend": trend,
        "recent_delta": round(delta, 2),
        "values": [round(v, 2) for v in valid],
    }


def compute_revenue_growth(revenues: List[float]) -> Dict[str, Any]:
    """营收增长趋势:增长率 / 是否持续正增长 / 波动性。

    营收是利润增长的前提,营收不增利润增 = 利润操纵嫌疑。
    """
    valid = [r for r in revenues if r is not None and r != 0]
    if len(valid) < 3:
        return {"available": False, "growth_rates": [], "all_positive": None, "volatile": None}
    rates = []
    for i in range(1, len(valid)):
        if valid[i - 1] > 0:
            rates.append((valid[i] - valid[i - 1]) / abs(valid[i - 1]))
        else:
            rates.append(None)
    rates_clean = [r for r in rates if r is not None]
    if not rates_clean:
        return {"available": False, "growth_rates": [], "all_positive": None, "volatile": None}
    all_pos = all(r > 0 for r in rates_clean)
    volatile = statistics.stdev(rates_clean) > 0.3 if len(rates_clean) > 1 else False
    return {
        "available": True,
        "growth_rates": [round(r, 4) for r in rates_clean],
        "all_positive": all_pos,
        "volatile": volatile,
        "mean_rate": round(statistics.mean(rates_clean), 4),
        "latest_rate": round(rates_clean[-1], 4),
    }


def compute_operating_profit_quality(
    operating_profits: List[float],
    net_profits: List[float],
) -> Dict[str, Any]:
    """扣非净利润质量:与净利润的差异,识别一次性损益。

    扣非/净利 < 0.7 = 一次性损益多,盈利质量差
    扣非/净利 >= 0.9 = 主业贡献利润,盈利质量好
    """
    pairs = [
        (o, n) for o, n in zip(operating_profits, net_profits)
        if o is not None and n is not None and n > 0
    ]
    if len(pairs) < 2:
        return {"available": False, "ratios": [], "mean_ratio": None, "quality": None}
    ratios = [o / n for o, n in pairs]
    mean_ratio = statistics.mean(ratios)
    if mean_ratio >= 0.9:
        quality = "良好(主业贡献利润,一次性损益少)"
    elif mean_ratio >= 0.7:
        quality = "一般(有一定一次性损益)"
    else:
        quality = "差(一次性损益多,警惕盈利质量)"
    return {
        "available": True,
        "ratios": [round(r, 4) for r in ratios],
        "mean_ratio": round(mean_ratio, 4),
        "quality": quality,
    }


def classify_stock_type(
    industry_info: Dict[str, Any],
    roic_stability: Dict[str, Any],
    profit_growth: Dict[str, Any],
    fcf_quality: Dict[str, Any],
    narrative_info: Optional[Dict[str, Any]] = None,
    gross_margin: Optional[Dict[str, Any]] = None,
    revenue_growth: Optional[Dict[str, Any]] = None,
    operating_profit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """综合分类:成长 / 周期 / 周期(有成长潜力) / 伪成长 / 待定。

    优先用财务指标,行业 + 叙事作为补充。
    若财务显示周期但行业有成长叙事,标记为"周期(有成长潜力)"。
    """
    initial = industry_info["initial_guess"]
    evidence = []

    # 周期信号:利润波动大(某年跌幅 > 30%)或 ROIC 变异系数大
    is_cyclical_by_fin = (
        (profit_growth.get("available") and profit_growth.get("volatile"))
        or (roic_stability.get("available") and roic_stability.get("cv", 1) > 0.4)
    )
    # 成长信号:利润持续正增长 + ROIC 高且稳定 + FCF 良好
    is_growth_by_fin = (
        profit_growth.get("available")
        and profit_growth.get("all_positive")
        and roic_stability.get("available")
        and roic_stability.get("mean", 0) > 0.12
        and roic_stability.get("cv", 1) < 0.3
    )
    # 伪成长:表面增长但 ROIC 下滑或 FCF 差
    is_fake_growth = (
        profit_growth.get("available")
        and profit_growth.get("all_positive")
        and (
            (roic_stability.get("available") and roic_stability.get("trend") == "下降")
            or (fcf_quality.get("available") and fcf_quality.get("mean_ratio", 1) < 0.3)
        )
    )

    # 加毛利率 / 营收 / 扣非 辅助证据
    if gross_margin and gross_margin.get("available"):
        gm_trend = gross_margin.get("trend")
        if gm_trend == "上升":
            evidence.append(f"毛利率上升(最新 {gross_margin['latest']}% vs 均值 {gross_margin['mean']}%),定价权增强或成本下行")
        elif gm_trend == "下降":
            evidence.append(f"毛利率下降(最新 {gross_margin['latest']}% vs 均值 {gross_margin['mean']}%),竞争加剧或成本上行")
        else:
            evidence.append(f"毛利率平稳(均值 {gross_margin['mean']}%,最新 {gross_margin['latest']}%)")

    if revenue_growth and revenue_growth.get("available"):
        if revenue_growth.get("all_positive"):
            evidence.append(f"营收持续正增长(均值 {revenue_growth['mean_rate']*100:.1f}%),增长有收入支撑")
        else:
            evidence.append(f"营收增长波动(均值 {revenue_growth['mean_rate']*100:.1f}%,最新 {revenue_growth['latest_rate']*100:.1f}%)")
        # 营收不增但利润增 = 利润操纵嫌疑
        if (not revenue_growth.get("all_positive")
            and profit_growth.get("available")
            and profit_growth.get("all_positive")):
            evidence.append("⚠️ 营收不增但利润增,警惕利润操纵(降成本/一次性损益)")

    if operating_profit and operating_profit.get("available"):
        op_ratio = operating_profit.get("mean_ratio", 0)
        if op_ratio < 0.7:
            evidence.append(f"⚠️ 扣非/净利仅 {op_ratio:.2f},一次性损益多,盈利质量差")
        elif op_ratio >= 0.9:
            evidence.append(f"扣非/净利 {op_ratio:.2f},主业贡献利润,盈利质量好")
        else:
            evidence.append(f"扣非/净利 {op_ratio:.2f},有一定一次性损益")

    # 叙事标记
    has_narrative = narrative_info and narrative_info.get("has_narrative")
    if has_narrative:
        for n in narrative_info["narratives"]:
            evidence.append(f"📈 行业叙事:{n['narrative']} -> {n['growth_potential']}")

    # 最终分类
    if is_fake_growth:
        stock_type = "伪成长"
        evidence.append("利润在增长但 ROIC 下滑或 FCF 不匹配,扩产可能消耗价值")
    elif is_growth_by_fin and fcf_quality.get("mean_ratio", 0) >= 0.5:
        stock_type = "成长"
        evidence.append("ROIC 高且稳定 + 利润持续增长 + FCF 匹配利润")
        if has_narrative:
            stock_type = "成长(叙事强化)"
            evidence.append("财务成长 + 行业叙事支撑,长期逻辑更强")
    elif is_cyclical_by_fin:
        if has_narrative:
            stock_type = "周期(有成长潜力)"
            evidence.append("财务呈现周期性,但行业有成长叙事,主业可能转成长(需业绩验证)")
        else:
            stock_type = "周期"
            evidence.append("利润大幅波动 / ROIC 不稳定,符合周期股特征")
    elif initial == "周期":
        stock_type = "周期(按行业)"
        evidence.append("财务数据不足,按行业归类为周期")
    elif initial == "成长":
        stock_type = "成长(按行业)"
        evidence.append("财务数据不足,按行业归类为成长")
    else:
        stock_type = "待定"
        evidence.append("数据不足,无法判断")

    if industry_info.get("growth_to_cyclical_risk"):
        evidence.append("⚠️ " + industry_info["note"])

    return {
        "type": stock_type,
        "evidence": evidence,
        "initial_guess": initial,
        "roic": roic_stability,
        "profit_growth": profit_growth,
        "fcf_quality": fcf_quality,
        "gross_margin": gross_margin or {"available": False},
        "revenue_growth": revenue_growth or {"available": False},
        "operating_profit": operating_profit or {"available": False},
        "narrative": narrative_info or {"has_narrative": False, "narratives": []},
    }


def detect_pe_trap(
    stock_type: str,
    pe: Optional[float],
    profit_growth: Dict[str, Any],
) -> Dict[str, Any]:
    """PE 陷阱检测。

    周期股:
    - PE 极低(<10)+ 利润近期暴增 -> 见顶信号(卖)
    - PE 极高或为负(行业亏损)-> 买点信号
    成长股:
    - 高 PE + ROIC 稳定增长 -> 可接受(业绩消化)
    - 高 PE + ROIC 下滑 -> 危险(伪成长)
    周期(有成长潜力):
    - 高 PE 可接受(叙事溢价),但需业绩兑现,否则估值杀
    """
    if pe is None:
        return {"available": False, "warning": None, "interpretation": "PE 数据缺失"}

    warning = None
    interp = ""

    if "周期(有成长潜力)" in stock_type:
        if pe > 60:
            interp = f"周期股(有成长潜力)PE {pe}(高),叙事溢价,需 Q2/Q3 业绩验证转成长逻辑,否则估值杀风险"
            if not (profit_growth.get("available") and profit_growth.get("all_positive")):
                warning = "高 PE + 利润未持续增长,叙事未兑现"
        elif pe < 15:
            interp = f"周期股(有成长潜力)PE {pe}(低),可能市场未识别叙事或叙事已破"
        else:
            interp = f"周期股(有成长潜力)PE {pe},中性,等业绩验证"
    elif "周期" in stock_type:
        if pe < 10:
            if profit_growth.get("available") and profit_growth.get("max_rate", 0) > 0.5:
                warning = "见顶信号"
                interp = f"周期股 PE 仅 {pe}(利润暴增后),行业景气顶部,警惕供给过剩暴跌"
            else:
                interp = f"周期股低 PE {pe},需结合利润趋势判断是否见顶"
        elif pe > 50 or pe < 0:
            warning = "潜在买点"
            interp = f"周期股 PE {pe}(行业亏损或微利),低谷期可能正是布局时机"
        else:
            interp = f"周期股 PE {pe},处于中性区间"
    elif "成长" in stock_type:
        if pe > 50:
            if profit_growth.get("available") and profit_growth.get("all_positive"):
                interp = f"成长股高 PE {pe},若 ROIC 稳定可由业绩增长消化"
            else:
                warning = "高 PE + 增长不确定"
                interp = f"成长股高 PE {pe} 但利润趋势不稳,警惕估值杀"
        else:
            interp = f"成长股 PE {pe},估值合理"
    elif stock_type == "伪成长":
        warning = "伪成长陷阱"
        interp = f"表面成长但 ROIC/FCF 暴露问题,PE {pe} 不可靠"
    else:
        interp = f"PE {pe},类型待定无法判断陷阱"

    return {
        "available": True,
        "pe": pe,
        "warning": warning,
        "interpretation": interp,
    }


def investment_approach(stock_type: str) -> Dict[str, Any]:
    """根据类型给投资思路。"""
    approaches = {
        "成长": {
            "approach": "长期持有",
            "rationale": "时间是优势,利润复利增长 + 护城河 + 持续高 ROIC",
            "action": "逢低加仓,长期持有,不轻易波段",
        },
        "成长(叙事强化)": {
            "approach": "长期持有(叙事强化)",
            "rationale": "财务成长 + 行业叙事双重支撑,长期逻辑更确定",
            "action": "逢低加仓,长期持有,关注叙事兑现进度",
        },
        "成长(按行业)": {
            "approach": "长期持有(待财务验证)",
            "rationale": "按行业归为成长,需用 ROIC/FCF 进一步验证",
            "action": "验证财务后决定,默认按长期持有思路",
        },
        "周期(有成长潜力)": {
            "approach": "波段为主 + 跟踪业绩验证成长性",
            "rationale": "财务仍呈周期性,但行业叙事可能让主业转成长。不能完全死扛,也不能纯波段",
            "action": "波段操作为主,关注 Q2/Q3 业绩:若营收+扣非持续增长,可逐步转为长期持有;若业绩未兑现,按周期股波段",
        },
        "周期": {
            "approach": "波段操作",
            "rationale": "利润由供给松紧决定,重资产扩产滞后易过剩暴跌",
            "action": "低谷布局(行业亏损 PE 极高时),高峰离场(利润暴增 PE 极低时),别死扛",
        },
        "周期(按行业)": {
            "approach": "波段操作(待财务验证)",
            "rationale": "按行业归为周期,需用利润波动验证",
            "action": "默认按波段思路,低谷买高峰卖",
        },
        "伪成长": {
            "approach": "回避",
            "rationale": "ROIC 下滑 + 扩产消耗价值 + FCF 不匹配,表面成长实为价值毁灭",
            "action": "不参与,或做空",
        },
        "待定": {
            "approach": "观望",
            "rationale": "数据不足,无法判断类型",
            "action": "补充财务数据后再决策",
        },
    }
    return approaches.get(stock_type, approaches["待定"])


def analyze_bank_quality(
    code: str,
    name: str,
    industry: str,
    pe: Optional[float],
    pb: Optional[float],
    financials: List[Dict[str, Any]],
    research_reports: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """银行股专用分析:用 ROE 替代 ROIC,PB/股息率替代 PE。

    银行股特殊性:
    - 不适用 ROIC(资本结构不同,银行无"投入资本"概念)
    - 不适用 FCF/NI(经营现金流本身就是业务)
    - 不适用 PE 陷阱(银行低 PE 是常态,非顶部信号)
    - 估值锚是 PB + 股息率,而非 PE

    银行类型判定:
    - 成长(优质银行):ROE 稳定 > 12% + 利润持续正增长 + 不良率稳定
    - 周期(普通银行):ROE 波动大 + 利润负增长

    核心指标(从 sina 摘要可获取):
    - ROE(净资产收益率)- 替代 ROIC
    - ROA(总资产报酬率)
    - 资产负债率(银行普遍 > 90%,差异小)
    - 净息差/不良率/拨备率 - sina 摘要无,需查年报附注
    """
    if financials and len(financials) >= 2:
        first_p = str(financials[0].get("period", ""))
        last_p = str(financials[-1].get("period", ""))
        if first_p > last_p:
            financials = list(reversed(financials))

    roes = [_safe_float(f.get("roe")) for f in financials]
    profits = [_safe_float(f.get("net_profit")) for f in financials]
    revenues = [_safe_float(f.get("revenue")) for f in financials]
    periods = [str(f.get("period", "")) for f in financials]

    roe_stability = compute_roic_stability(roes, periods)  # 复用,语义是 ROE
    profit_growth = compute_profit_growth(profits)
    revenue_growth = compute_revenue_growth(revenues)

    # 银行类型判定:ROE 稳定性 + 利润增长
    roe_mean = roe_stability.get("mean")
    roe_cv = roe_stability.get("cv")
    roe_available = roe_stability.get("available")

    bank_type = "待定"
    evidence = ["银行股,走专用分析路径(ROE 替代 ROIC,PB 替代 PE)"]

    if roe_available and roe_mean is not None:
        evidence.append(f"ROE 均值 {roe_mean:.2f}%")
        if roe_cv is not None:
            evidence.append(f"ROE cv {roe_cv:.2f}")
        # ROE > 12% 且 cv < 0.3 = 优质银行
        if roe_mean > 12 and (roe_cv is None or roe_cv < 0.3):
            bank_type = "成长(优质银行)"
            evidence.append("ROE 高且稳定,符合优质银行特征(如招行/宁波)")
        elif roe_mean > 10:
            bank_type = "周期(普通银行)"
            evidence.append("ROE 中等,普通银行")
        else:
            bank_type = "周期(弱质银行)"
            evidence.append("ROE 偏低,经营压力大")

    if profit_growth.get("available"):
        if profit_growth.get("all_positive"):
            evidence.append("利润持续正增长")
        else:
            evidence.append(f"利润波动大(min {profit_growth.get('min_rate')}, max {profit_growth.get('max_rate')})")

    # 银行估值锚:PB + 股息率(非 PE)
    pe_trap = {
        "available": pb is not None,
        "warning": None,
        "interpretation": (
            f"银行股估值锚为 PB = {pb}(当前)"
            if pb is not None
            else "银行股估值锚为 PB + 股息率(PE 不适用),PB 数据缺失"
        ),
        "pb": pb,
        "valuation_anchor": "PB + 股息率",
        "note": "银行低 PE 是常态非顶部信号;PB<1 + ROE>12% = 低估;股息率>4% = 高息配置价值",
    }

    # 投资思路
    if "成长" in bank_type:
        approach = {
            "approach": "长期持有(优质银行)",
            "rationale": "ROE 高且稳定,零售/对公护城河,长期复利",
            "action": "逢低加仓,PB<1 时重点配置;长期持有吃股息,不轻易波段",
        }
    elif "普通" in bank_type:
        approach = {
            "approach": "波段操作(普通银行)",
            "rationale": "ROE 中等,跟随经济周期和息差周期",
            "action": "PB<0.7 低估布局,PB>1 减仓;关注息差/不良率拐点",
        }
    else:
        approach = {
            "approach": "观望(弱质银行)",
            "rationale": "ROE 偏低,经营压力或风险暴露",
            "action": "不建议配置,等基本面改善",
        }

    geopolitical_risk = {
        "has_risk": True,
        "risk_types": [
            {
                "id": "bank_policy",
                "risk": "政策依赖(银行专用)",
                "description": "LPR 下行压息差、准备金率调整、房地产敞口风险、地方债风险暴露、利率市场化",
                "examples": "招行(房地产+零售信用)、工行(对公+地方债)",
                "verification": "查最新:LPR 变化、季度净息差、不良贷款率、拨备覆盖率、房地产敞口",
            }
        ],
        "note": "银行股天然受政策强烈影响,需跟踪息差/不良/房地产",
    }

    # 机构研报评估
    research_report = summarize_research_report(research_reports)
    if research_report.get("available"):
        evidence = research_report["evidence"] + evidence

    return {
        "code": code,
        "name": name,
        "industry": industry or "银行",
        "pe": pe,
        "pb": pb,
        "is_bank": True,
        "classification": {
            "type": bank_type,
            "evidence": evidence,
            "roe_stability": roe_stability,
            "profit_growth": profit_growth,
            "revenue_growth": revenue_growth,
            "research_quality": {
                "quality_signal": research_report.get("quality_signal"),
                "total_reports": research_report.get("total_reports"),
                "rating_consensus": research_report.get("rating_consensus"),
                "target_price": research_report.get("target_price"),
                "foreign_summary": research_report.get("foreign_summary"),
                "divergence": research_report.get("divergence"),
            } if research_report.get("available") else {},
            "note": "银行股专用分析,不适用 ROIC/FCF/PE 陷阱框架",
        },
        "pe_trap": pe_trap,
        "investment_approach": approach,
        "geopolitical_risk": geopolitical_risk,
        "research_report": research_report,
    }


def summarize_research_report(reports_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把 fetch_research_reports 的输出翻译成公司质地维度的评估摘要。

    返回:
    {
        "available": bool,
        "rating_consensus": {...},    # 评级共识(主导评级 + 强度)
        "target_price": {...},        # 目标价(均值 + 上涨空间 + 标签)
        "foreign_summary": {...},     # 外资观点
        "divergence": {...},          # 外资 vs 内资分歧
        "evidence": [str, ...],       # 给 classification.evidence 用的证据列表
        "quality_signal": str,        # 机构认可度信号: 强 / 中 / 弱 / 无覆盖
    }
    """
    if not reports_data or reports_data.get("error") or not reports_data.get("available", True):
        return {
            "available": False,
            "evidence": ["无机构研报覆盖或拉取失败"],
            "quality_signal": "无覆盖",
        }
    if reports_data.get("total_reports", 0) == 0:
        return {
            "available": False,
            "evidence": ["无机构研报覆盖"],
            "quality_signal": "无覆盖",
        }

    total = reports_data.get("total_reports", 0)
    rc = reports_data.get("rating_consensus", {})
    tp = reports_data.get("target_price", {})
    fs = reports_data.get("foreign_summary", {})
    div = reports_data.get("divergence", {})
    eps_f = reports_data.get("eps_forecast", {})

    evidence: List[str] = []

    # 评级共识
    if rc.get("available"):
        dominant_label = rc.get("dominant_label", "未明")
        dominant_pct = rc.get("dominant_pct", 0)
        strength_label = rc.get("label", "")
        evidence.append(
            f"机构评级共识:{dominant_label}({dominant_pct}%)/{strength_label},共 {rc.get('total', 0)} 篇研报"
        )

    # 目标价
    if tp.get("available"):
        mean = tp.get("mean")
        upside = tp.get("upside_pct")
        label = tp.get("label")
        if upside is not None:
            evidence.append(f"目标价均值 {mean}(相对当前价 {upside:+.1f}%/{label})")
        else:
            evidence.append(f"目标价均值 {mean}(无当前价对比)")

    # 外资观点
    if fs.get("available"):
        fcount = fs.get("count", 0)
        f_rc = fs.get("rating_consensus", {})
        f_latest = fs.get("latest", {}) or {}
        if f_rc.get("available"):
            evidence.append(
                f"外资/港资/台资合资券商 {fcount} 篇,共识 {f_rc.get('dominant_label', '未明')}"
                f"({f_rc.get('dominant_pct', 0)}%)"
            )
        if f_latest.get("org"):
            evidence.append(
                f"最近外资研报:{f_latest.get('org')} {f_latest.get('date')} "
                f"评级 {f_latest.get('rating') or '未明'}"
            )

    # 分歧度
    if div.get("available"):
        evidence.append(f"内外资分歧:{div.get('label', '未明')}")

    # 盈利预测
    if eps_f.get("available"):
        cur = eps_f.get("current_year", {}) or {}
        nxt = eps_f.get("next_year", {}) or {}
        if cur.get("eps") is not None and nxt.get("eps") is not None:
            growth = (nxt["eps"] - cur["eps"]) / cur["eps"] if cur["eps"] else 0
            evidence.append(
                f"盈利预测:今年 EPS {cur.get('eps')} / 明年 {nxt.get('eps')}"
                f"(同比 {growth*100:+.1f}%)"
            )

    # 机构认可度信号:强(≥10 篇 + 共识强) / 中(≥5 篇 + 共识中以上) / 弱(<5 篇) / 无覆盖(0)
    if total >= 10 and rc.get("label") in ("共识强",):
        quality_signal = "强"
    elif total >= 5 and rc.get("consensus_strength", 0) >= 0.5:
        quality_signal = "中"
    elif total >= 1:
        quality_signal = "弱"
    else:
        quality_signal = "无覆盖"

    return {
        "available": True,
        "total_reports": total,
        "rating_consensus": rc,
        "target_price": tp,
        "foreign_summary": fs,
        "divergence": div,
        "eps_forecast": eps_f,
        "evidence": evidence,
        "quality_signal": quality_signal,
    }


def analyze_fundamental(
    code: str,
    name: str,
    industry: str,
    pe: Optional[float],
    pb: Optional[float],
    financials: List[Dict[str, Any]],
    research_reports: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """主入口:综合公司质地分析。

    financials: 多年财务数据,每项含 net_profit, revenue, operating_cf, roic, roe, fcf,
                 operating_profit(扣非), operating_cost(营业成本), gross_margin_pct, eps

    时间顺序:financials 按时间正序(旧 -> 新)传入,最新一期在 financials[-1]。
    若上游(sina)返回的是倒序(新 -> 旧),会自动检测并 reverse。

    银行股走专用路径(analyze_bank_quality),不适用 ROIC/FCF/PE 陷阱框架。
    """
    # 银行股分流:用 ROE 替代 ROIC,PB/股息率替代 PE
    if is_bank(industry, code, name):
        return analyze_bank_quality(code, name, industry, pe, pb, financials)

    industry_info = classify_by_industry(industry)
    narrative_info = classify_by_narrative(industry, name)
    geopolitical_risk = classify_geopolitical_risk(industry, name)

    # 检测时间顺序:sina 返回 20260331 -> 20111231(新->旧),需 reverse 为旧->新
    # 检测方法:period 字符串如果递减则 reverse
    if financials and len(financials) >= 2:
        first_p = str(financials[0].get("period", ""))
        last_p = str(financials[-1].get("period", ""))
        if first_p > last_p:
            financials = list(reversed(financials))

    roics = [_safe_float(f.get("roic")) for f in financials]
    profits = [_safe_float(f.get("net_profit")) for f in financials]
    fcfs = [_safe_float(f.get("fcf")) for f in financials]
    margins = [_safe_float(f.get("gross_margin_pct")) for f in financials]
    revenues = [_safe_float(f.get("revenue")) for f in financials]
    operating_profits = [_safe_float(f.get("operating_profit")) for f in financials]
    periods = [str(f.get("period", "")) for f in financials]

    roic_stability = compute_roic_stability(roics, periods)
    profit_growth = compute_profit_growth(profits)
    fcf_quality = compute_fcf_quality(fcfs, profits)
    gross_margin = compute_gross_margin_trend(margins)
    revenue_growth = compute_revenue_growth(revenues)
    operating_profit = compute_operating_profit_quality(operating_profits, profits)

    # ROE 深度分析:稳定性 + 杜邦 + 巴菲特筛选 + 假高 ROE 识别
    roes = [_safe_float(f.get("roe")) for f in financials]
    roe_stability = compute_roe_stability(roes, periods)
    dupont = compute_dupont_analysis(financials)
    buffett = compute_buffett_filter(financials)
    fake_roe = detect_fake_roe(financials)

    classification = classify_stock_type(
        industry_info, roic_stability, profit_growth, fcf_quality,
        narrative_info=narrative_info,
        gross_margin=gross_margin,
        revenue_growth=revenue_growth,
        operating_profit=operating_profit,
    )

    # 把 ROE 证据加入 classification.evidence
    roe_evidence = []
    if roe_stability.get("available"):
        roe_evidence.append(
            f"ROE 均值 {roe_stability['mean']}%(min {roe_stability['min']}%,{roe_stability['trend']})"
        )
        bf = roe_stability.get("buffett_filter", {})
        if bf.get("pass"):
            roe_evidence.append("✅ ROE 达巴菲特标准(均>15% + 单年≥12%)")
        else:
            roe_evidence.append(
                f"⚠️ ROE 未达巴菲特标准(均{'>' if bf.get('mean_pass') else '<'}15% "
                f"/ 单年{'≥' if bf.get('min_pass') else '<'}12%)"
            )
    if dupont.get("available"):
        roe_evidence.append(
            f"杜邦:{dupont['mode_label']} "
            f"(净利率 {dupont['net_margin']}% / 周转 {dupont['asset_turnover']} / 权益乘数 {dupont['equity_multiplier']})"
        )
    if fake_roe.get("is_fake"):
        for w in fake_roe["warnings"]:
            roe_evidence.append(f"⚠️ 假高 ROE - {w['detail']}")
    if roe_evidence:
        classification["evidence"] = roe_evidence + classification.get("evidence", [])
        classification["roe_quality"] = {
            "roe_stability": roe_stability,
            "dupont": dupont,
            "buffett_filter": buffett,
            "fake_roe": fake_roe,
        }

    pe_trap = detect_pe_trap(classification["type"], pe, profit_growth)
    approach = investment_approach(classification["type"])

    # 机构研报评估:评级共识 + 目标价 + 外资观点 + 分歧度
    research_report = summarize_research_report(research_reports)
    if research_report.get("available"):
        classification["evidence"] = (
            research_report["evidence"] + classification.get("evidence", [])
        )
        classification["research_quality"] = {
            "quality_signal": research_report.get("quality_signal"),
            "total_reports": research_report.get("total_reports"),
            "rating_consensus": research_report.get("rating_consensus"),
            "target_price": research_report.get("target_price"),
            "foreign_summary": research_report.get("foreign_summary"),
            "divergence": research_report.get("divergence"),
        }

    return {
        "code": code,
        "name": name,
        "industry": industry,
        "pe": pe,
        "pb": pb,
        "classification": classification,
        "pe_trap": pe_trap,
        "investment_approach": approach,
        "geopolitical_risk": geopolitical_risk,
        "roe_quality": {
            "roe_stability": roe_stability,
            "dupont": dupont,
            "buffett_filter": buffett,
            "fake_roe": fake_roe,
        },
        "research_report": research_report,
    }
