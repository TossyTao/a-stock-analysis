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
    "银行", "证券", "保险", "航运", "航空", "造纸", "纺织服装",
    "石油石化", "基础化学", "农药兽药",
}

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


def analyze_fundamental(
    code: str,
    name: str,
    industry: str,
    pe: Optional[float],
    pb: Optional[float],
    financials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """主入口:综合公司质地分析。

    financials: 多年财务数据,每项含 net_profit, revenue, operating_cf, roic, roe, fcf,
                 operating_profit(扣非), operating_cost(营业成本), gross_margin_pct, eps

    时间顺序:financials 按时间正序(旧 -> 新)传入,最新一期在 financials[-1]。
    若上游(sina)返回的是倒序(新 -> 旧),会自动检测并 reverse。
    """
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

    classification = classify_stock_type(
        industry_info, roic_stability, profit_growth, fcf_quality,
        narrative_info=narrative_info,
        gross_margin=gross_margin,
        revenue_growth=revenue_growth,
        operating_profit=operating_profit,
    )
    pe_trap = detect_pe_trap(classification["type"], pe, profit_growth)
    approach = investment_approach(classification["type"])

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
    }
