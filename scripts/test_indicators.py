"""compute_indicators 单元测试 — 用造的数据验证指标计算逻辑。

运行:python test_indicators.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from compute_indicators import (
    position_percentile, position_label, vol_ma_cross, vol_ratio,
    is_ground_vol, quadrant, trend_5day, top_divergence, bottom_divergence,
    breakout_3day, detect_traps, compute,
)
from chip_distribution import (
    compute_chip_distribution, find_peaks, classify_pattern,
    support_resistance, concentration_ratio, analyze as chip_analyze,
)
from fundamental import (
    classify_by_industry, compute_roic_stability, compute_profit_growth,
    compute_fcf_quality, classify_stock_type, detect_pe_trap,
    investment_approach, analyze_fundamental,
    classify_by_narrative, compute_gross_margin_trend,
    compute_revenue_growth, compute_operating_profit_quality,
    classify_geopolitical_risk,
)
import pandas as pd


def make_daily(closes, vols=None, base=10.0):
    """造 daily 数据:closes 是价格列表,vols 默认递增。"""
    n = len(closes)
    if vols is None:
        vols = [1000 + i * 10 for i in range(n)]
    return [
        {"date": f"2026-01-{i+1:02d}", "open": c - 0.1, "high": c + 0.2,
         "low": c - 0.2, "close": c, "volume": v, "amount": v * c}
        for i, (c, v) in enumerate(zip(closes, vols))
    ]


def test_position_percentile():
    closes = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert position_percentile(closes, 10) == 100.0
    closes = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    assert position_percentile(closes, 10) == 0.0
    closes = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 5])
    assert position_percentile(closes, 10) == 50.0
    print("✅ test_position_percentile passed")


def test_position_label():
    assert position_label(85) == "高位"
    assert position_label(15) == "低位"
    assert position_label(50) == "中位"
    assert position_label(80) == "高位"
    assert position_label(20) == "低位"
    print("✅ test_position_label passed")


def test_vol_ma_cross_golden():
    # 造金叉:先 ma5 < ma20,然后放量让 ma5 上穿
    vols = [100] * 25 + [500, 600, 700]
    s = pd.Series(vols)
    r = vol_ma_cross(s)
    assert r["state"] == "金叉", f"expected 金叉, got {r['state']}"
    print("✅ test_vol_ma_cross_golden passed")


def test_vol_ma_cross_death():
    # 造死叉:先高量后缩量,交叉发生在最近 5 日内
    vols = [200] * 25 + [100, 100, 100]
    s = pd.Series(vols)
    r = vol_ma_cross(s)
    assert r["state"] == "死叉", f"expected 死叉, got {r['state']}"
    print("✅ test_vol_ma_cross_death passed")


def test_vol_ratio():
    # ROVL: 今日量 / 过去 20 日均量
    vols = [100] * 20 + [200]
    s = pd.Series(vols)
    assert vol_ratio(s) == 2.0
    vols = [100] * 20 + [50]
    s = pd.Series(vols)
    assert vol_ratio(s) == 0.5
    print("✅ test_vol_ratio passed")


def test_ground_vol():
    vols = [1000] * 20 + [400]
    s = pd.Series(vols)
    assert is_ground_vol(s) is True
    vols = [1000] * 20 + [600]
    s = pd.Series(vols)
    assert is_ground_vol(s) is False
    print("✅ test_ground_vol passed")


def test_quadrant_vol_up_price_up():
    daily = make_daily([10, 11], vols=[1000, 1500])
    q = quadrant(pd.DataFrame(daily))
    assert q["quadrant"] == "量增价涨"
    print("✅ test_quadrant_vol_up_price_up passed")


def test_quadrant_vol_down_price_down():
    daily = make_daily([10, 9], vols=[1500, 1000])
    q = quadrant(pd.DataFrame(daily))
    assert q["quadrant"] == "量缩价跌"
    print("✅ test_quadrant_vol_down_price_down passed")


def test_trend_5d_vol_up_price_up():
    # 5 日连续放量上涨:价从 10 -> 12,量从 1000 -> 1800
    closes = [10, 10.5, 11, 11.5, 12]
    vols = [1000, 1200, 1400, 1600, 1800]
    daily = make_daily(closes, vols)
    r = trend_5day(pd.DataFrame(daily))
    assert r["price_trend"] == "价涨", f"expected 价涨, got {r['price_trend']}"
    assert r["vol_trend"] == "量增", f"expected 量增, got {r['vol_trend']}"
    assert r["quadrant"] == "量增价涨"
    assert r["price_slope_pct"] > 0
    assert r["vol_slope_pct"] > 0
    print(f"✅ test_trend_5d_vol_up_price_up passed (slope_p={r['price_slope_pct']}%, slope_v={r['vol_slope_pct']}%)")


def test_trend_5d_vol_down_price_down():
    # 5 日连续缩量下跌:价从 12 -> 10,量从 1800 -> 1000
    closes = [12, 11.5, 11, 10.5, 10]
    vols = [1800, 1600, 1400, 1200, 1000]
    daily = make_daily(closes, vols)
    r = trend_5day(pd.DataFrame(daily))
    assert r["price_trend"] == "价跌", f"expected 价跌, got {r['price_trend']}"
    assert r["vol_trend"] == "量缩", f"expected 量缩, got {r['vol_trend']}"
    assert r["quadrant"] == "量缩价跌"
    assert r["price_slope_pct"] < 0
    assert r["vol_slope_pct"] < 0
    print(f"✅ test_trend_5d_vol_down_price_down passed (slope_p={r['price_slope_pct']}%, slope_v={r['vol_slope_pct']}%)")


def test_trend_5d_vol_up_price_down():
    # 放量下跌:价跌 + 量增
    closes = [12, 11, 10, 9, 8]
    vols = [1000, 1300, 1600, 2000, 2500]
    daily = make_daily(closes, vols)
    r = trend_5day(pd.DataFrame(daily))
    assert r["price_trend"] == "价跌"
    assert r["vol_trend"] == "量增"
    assert r["quadrant"] == "量增价跌"
    print(f"✅ test_trend_5d_vol_up_price_down passed (放量下跌)")


def test_trend_5d_vol_down_price_up():
    # 缩量上涨:价涨 + 量缩
    closes = [10, 10.5, 11, 11.5, 12]
    vols = [1800, 1600, 1400, 1200, 1000]
    daily = make_daily(closes, vols)
    r = trend_5day(pd.DataFrame(daily))
    assert r["price_trend"] == "价涨"
    assert r["vol_trend"] == "量缩"
    assert r["quadrant"] == "量缩价涨"
    print(f"✅ test_trend_5d_vol_down_price_up passed (缩量上涨)")


def test_trend_5d_flat():
    # 5 日窄幅震荡:价平 + 量平
    closes = [10.0, 10.05, 9.98, 10.02, 10.0]
    vols = [1000, 1010, 990, 1005, 1000]
    daily = make_daily(closes, vols)
    r = trend_5day(pd.DataFrame(daily))
    assert r["price_trend"] == "价平", f"expected 价平, got {r['price_trend']}"
    assert r["vol_trend"] == "量平", f"expected 量平, got {r['vol_trend']}"
    assert r["quadrant"] == "量平价平"
    assert r["strength"] == "弱"
    print(f"✅ test_trend_5d_flat passed (slope_p={r['price_slope_pct']}%, slope_v={r['vol_slope_pct']}%)")


def test_trend_5d_strength_strong():
    # 强趋势:价从 10 -> 13(30%),量从 1000 -> 3000(200%)
    closes = [10, 11, 12, 12.5, 13]
    vols = [1000, 1500, 2000, 2500, 3000]
    daily = make_daily(closes, vols)
    r = trend_5day(pd.DataFrame(daily))
    assert r["strength"] == "强", f"expected 强, got {r['strength']}"
    print(f"✅ test_trend_5d_strength_strong passed (strength={r['strength']})")


def test_trend_5d_consistency():
    # 4/4 天与趋势一致
    closes = [10, 10.5, 11, 11.5, 12]
    vols = [1000, 1200, 1400, 1600, 1800]
    daily = make_daily(closes, vols)
    r = trend_5day(pd.DataFrame(daily))
    assert r["consistency"] == "4/4", f"expected 4/4, got {r['consistency']}"
    print(f"✅ test_trend_5d_consistency passed (consistency={r['consistency']})")


def test_trend_5d_in_compute():
    # compute() 输出应包含 trend_5d
    closes = [10] * 25 + [11, 12, 13, 14, 15]
    vols = [1000] * 25 + [1200, 1400, 1600, 1800, 2000]
    daily = make_daily(closes, vols)
    r = compute(daily)
    assert "trend_5d" in r, "compute output missing trend_5d"
    assert r["trend_5d"]["quadrant"] == "量增价涨"
    assert "quadrant" in r, "compute output missing daily quadrant"
    print(f"✅ test_trend_5d_in_compute passed (trend={r['trend_5d']['quadrant']}, daily={r['quadrant']['quadrant']})")


def test_trend_5d_vs_daily_divergence():
    # 单日与 5 日趋势不一致:5 日上涨但今日下跌(可能拐点)
    closes = [10, 11, 12, 13, 12.5]
    vols = [1000, 1200, 1400, 1600, 1300]
    daily = make_daily(closes, vols)
    df = pd.DataFrame(daily)
    t = trend_5day(df)
    q = quadrant(df)
    assert t["quadrant"] == "量增价涨", f"5 日应仍是量增价涨, got {t['quadrant']}"
    assert q["quadrant"] == "量缩价跌", f"单日应是量缩价跌, got {q['quadrant']}"
    print(f"✅ test_trend_5d_vs_daily_divergence passed (5日={t['quadrant']}, 单日={q['quadrant']} -> 拐点警示)")


def test_top_divergence_detected():
    # 价格新高但量未新高:价新高在最后一天,量 < 区间最大量 × 0.8
    closes = [10] * 19 + [12]
    vols = [2000] * 19 + [1500]
    daily = make_daily(closes, vols)
    r = top_divergence(pd.DataFrame(daily))
    assert r["detected"] is True, f"expected divergence, got {r}"
    print("✅ test_top_divergence_detected passed")


def test_top_divergence_not_detected():
    # 价新高量也新高
    closes = [10] * 19 + [12]
    vols = [1000] * 19 + [3000]
    daily = make_daily(closes, vols)
    r = top_divergence(pd.DataFrame(daily))
    assert r["detected"] is False
    print("✅ test_top_divergence_not_detected passed")


def test_breakout_3day():
    # 前 20 日最高 10 元,最后 3 天收盘 11, 11, 11
    closes = [8, 9, 10, 9, 8] * 4 + [11, 11, 11]
    daily = make_daily(closes)
    r = breakout_3day(pd.DataFrame(daily))
    assert r["detected"] is True, f"expected breakout, got {r}"
    print("✅ test_breakout_3day passed")


def test_breakout_failed():
    # 站稳 2 天第 3 天跌回
    closes = [8, 9, 10, 9, 8] * 4 + [11, 11, 9]
    daily = make_daily(closes)
    r = breakout_3day(pd.DataFrame(daily))
    assert r["detected"] is False
    assert r["days_above"] == 2
    print("✅ test_breakout_failed passed")


def test_compute_full():
    # 端到端:造 120 天数据
    closes = [10 + i * 0.05 for i in range(120)]
    vols = [1000 + i * 5 for i in range(120)]
    daily = make_daily(closes, vols)
    r = compute(daily)
    assert "position" in r
    assert "volume" in r
    assert "quadrant" in r
    assert "divergence" in r
    assert "traps" in r
    assert "breakout" in r
    assert "five_step" in r
    assert len(r["five_step"]) == 5
    print("✅ test_compute_full passed")


def test_wash_trade_detection():
    # 量比 > 2.5 但价格涨幅 < 1%(对倒造量)
    closes = [10.00, 10.05]  # 涨幅 0.5%
    vols = [1000, 3000]  # 量比 3.0
    daily = make_daily(closes, vols)
    df = pd.DataFrame(daily)
    # 需要更多数据让 detect_traps 不返回全 False
    closes = [10] * 23 + [10.00, 10.05]
    vols = [1000] * 23 + [1000, 3000]
    daily = make_daily(closes, vols)
    r = detect_traps(pd.DataFrame(daily), {})
    assert r["wash_trade"] is True, f"expected wash_trade, got {r}"
    print("✅ test_wash_trade_detection passed")


def test_algo_no_vol_rise():
    # 量比 < 0.7 但价格涨幅 > 2%(算法无量空涨)
    closes = [10] * 23 + [10.00, 10.25]  # 涨 2.5%
    vols = [1000] * 23 + [1000, 600]  # 量比 0.6
    daily = make_daily(closes, vols)
    r = detect_traps(pd.DataFrame(daily), {})
    assert r["algo_no_vol_rise"] is True, f"expected algo_no_vol_rise, got {r}"
    print("✅ test_algo_no_vol_rise passed")


# ===== 筹码峰测试 =====

def make_chip_daily(prices_vols):
    """造 daily 数据:prices_vols = [(close, volume), ...],open=high=low=close 简化。"""
    return [
        {"date": f"2026-01-{i+1:02d}", "open": c, "high": c + 0.1,
         "low": c - 0.1, "close": c, "volume": v, "amount": v * c}
        for i, (c, v) in enumerate(prices_vols)
    ]


def test_chip_low_single_peak():
    """低位单峰:长期在低价横盘,主力吸筹,末尾小幅放量。"""
    # 50 天在 10 元横盘,最后 5 天价升到 11
    pv = [(10.0, 1000)] * 50 + [(10.2, 1500), (10.5, 1800), (10.8, 2000), (11.0, 2200), (11.0, 2000)]
    daily = make_chip_daily(pv)
    r = chip_analyze(daily)
    assert r["pattern"]["pattern"] == "单峰", f"expected 单峰, got {r['pattern']['pattern']}"
    assert r["pattern"]["position_label"] == "低位", f"expected 低位, got {r['pattern']['position_label']}"
    print(f"✅ test_chip_low_single_peak passed (峰位 {r['pattern']['dominant_peak']['price']:.2f}, {r['pattern']['dominant_peak']['pct']}%)")


def test_chip_high_single_peak():
    """高位单峰:价格从低升到高后在高价横盘,出货风险。"""
    # 10 天 8 元,10 天 12 元,30 天 15 元横盘
    pv = [(8.0, 1000)] * 10 + [(10.0, 1500)] * 5 + [(13.0, 2000)] * 5 + [(15.0, 2000)] * 30
    daily = make_chip_daily(pv)
    r = chip_analyze(daily)
    assert r["pattern"]["pattern"] == "单峰", f"expected 单峰, got {r['pattern']['pattern']}"
    assert r["pattern"]["position_label"] == "高位", f"expected 高位, got {r['pattern']['position_label']}"
    print(f"✅ test_chip_high_single_peak passed (峰位 {r['pattern']['dominant_peak']['price']:.2f}, {r['pattern']['dominant_peak']['pct']}%)")


def test_chip_double_peak():
    """双峰:两段横盘形成两个筹码峰。旧峰需更高量能补偿衰减。"""
    # 15 天 10 元高量吸筹,3 天过渡,15 天 14 元,5 天回到 12 元
    pv = [(10.0, 5000)] * 15 + [(11.0, 500), (12.0, 500), (13.0, 500)] + [(14.0, 2000)] * 15 + [(13.0, 500), (12.5, 500), (12.2, 500), (12.0, 500), (12.0, 500)]
    daily = make_chip_daily(pv)
    r = chip_analyze(daily)
    assert r["pattern"]["pattern"] == "双峰", f"expected 双峰, got {r['pattern']['pattern']} (peaks: {len(r['pattern']['all_peaks'])})"
    print(f"✅ test_chip_double_peak passed (peaks: {[(p['price'], p['pct']) for p in r['pattern']['all_peaks']]})")


def test_chip_support_resistance():
    """支撑压力:低位筹码形成支撑,高位筹码形成压力。"""
    # 30 天在 10 元吸筹,然后价升到 14,10 元处应被识别为支撑
    pv = [(10.0, 2000)] * 30 + [(11.0, 1000), (12.0, 1000), (13.0, 1000), (14.0, 1000)]
    daily = make_chip_daily(pv)
    r = chip_analyze(daily)
    assert r["support_resistance"]["support"] is not None, "expected support"
    assert r["support_resistance"]["support"]["price"] < 12, f"support should be below 12, got {r['support_resistance']['support']['price']}"
    print(f"✅ test_chip_support_resistance passed (support: {r['support_resistance']['support']})")


def test_chip_concentration():
    """集中度:长期横盘后筹码高度集中。"""
    pv = [(10.0, 1000)] * 60
    daily = make_chip_daily(pv)
    r = chip_analyze(daily)
    assert r["concentration_5pct"] >= 40, f"expected high concentration, got {r['concentration_5pct']}%"
    assert r["concentration_label"] == "高度集中"
    print(f"✅ test_chip_concentration passed (集中度 {r['concentration_5pct']}%)")


def test_chip_in_compute():
    """端到端:compute() 输出包含 chip 字段。"""
    closes = [10 + i * 0.02 for i in range(120)]
    vols = [1000 + i * 5 for i in range(120)]
    daily = make_daily(closes, vols)
    r = compute(daily)
    assert "chip" in r, "compute() should include chip analysis"
    assert "pattern" in r["chip"]
    assert "support_resistance" in r["chip"]
    print(f"✅ test_chip_in_compute passed (chip.pattern: {r['chip']['pattern']['pattern']})")


# ===== 公司质地判断测试 =====

def test_classify_growth_stock():
    """成长股:ROIC 高且稳定 + 利润持续增长 + FCF 良好。"""
    industry_info = classify_by_industry("医药生物")
    assert industry_info["initial_guess"] == "成长"
    roic_stab = compute_roic_stability([0.15, 0.16, 0.17, 0.18])
    profits = compute_profit_growth([10, 12, 15, 18, 22])
    fcf = compute_fcf_quality([8, 10, 13, 16, 20], [10, 12, 15, 18, 22])
    r = classify_stock_type(industry_info, roic_stab, profits, fcf)
    assert r["type"] == "成长", f"expected 成长, got {r['type']}"
    print(f"✅ test_classify_growth_stock passed (type: {r['type']})")


def test_classify_cyclical_stock():
    """周期股:利润大幅波动 + ROIC 不稳定。"""
    industry_info = classify_by_industry("钢铁")
    assert industry_info["initial_guess"] == "周期"
    roic_stab = compute_roic_stability([0.25, 0.05, -0.10, 0.20])
    profits = compute_profit_growth([100, 20, -30, 80, 120])
    fcf = compute_fcf_quality([50, -20, -40, 30, 60], [100, 20, -30, 80, 120])
    r = classify_stock_type(industry_info, roic_stab, profits, fcf)
    assert "周期" in r["type"], f"expected 周期, got {r['type']}"
    print(f"✅ test_classify_cyclical_stock passed (type: {r['type']})")


def test_classify_fake_growth():
    """伪成长:利润增长但 ROIC 下滑或 FCF 差。"""
    industry_info = classify_by_industry("电子")
    roic_stab = compute_roic_stability([0.20, 0.18, 0.15, 0.12])
    profits = compute_profit_growth([10, 12, 14, 16, 18])
    fcf = compute_fcf_quality([2, 1, 0, -1, -2], [10, 12, 14, 16, 18])
    r = classify_stock_type(industry_info, roic_stab, profits, fcf)
    assert r["type"] == "伪成长", f"expected 伪成长, got {r['type']}"
    print(f"✅ test_classify_fake_growth passed (type: {r['type']})")


def test_pe_trap_cyclical_low():
    """周期股低 PE + 利润暴增 = 见顶信号。"""
    pe_trap = detect_pe_trap(
        "周期",
        pe=5.0,
        profit_growth={"available": True, "all_positive": False, "max_rate": 1.5, "growth_rates": [1.5]},
    )
    assert pe_trap["warning"] == "见顶信号", f"expected 见顶信号, got {pe_trap['warning']}"
    print(f"✅ test_pe_trap_cyclical_low passed (warning: {pe_trap['warning']})")


def test_pe_trap_cyclical_high():
    """周期股高 PE(行业亏损)= 买点信号。"""
    pe_trap = detect_pe_trap(
        "周期",
        pe=80.0,
        profit_growth={"available": True, "all_positive": False, "max_rate": -0.5, "growth_rates": [-0.5]},
    )
    assert pe_trap["warning"] == "潜在买点", f"expected 潜在买点, got {pe_trap['warning']}"
    print(f"✅ test_pe_trap_cyclical_high passed (warning: {pe_trap['warning']})")


def test_pe_trap_growth_high():
    """成长股高 PE + 持续增长 = 可接受。"""
    pe_trap = detect_pe_trap(
        "成长",
        pe=60.0,
        profit_growth={"available": True, "all_positive": True, "max_rate": 0.3, "growth_rates": [0.2, 0.25, 0.3]},
    )
    assert pe_trap["warning"] is None, f"expected no warning, got {pe_trap['warning']}"
    print(f"✅ test_pe_trap_growth_high passed (no warning)")


def test_investment_approach():
    """投资思路匹配类型。"""
    assert investment_approach("成长")["approach"] == "长期持有"
    assert investment_approach("周期")["approach"] == "波段操作"
    assert investment_approach("伪成长")["approach"] == "回避"
    print("✅ test_investment_approach passed")


def test_growth_to_cyclical_risk():
    """成长赛道转周期风险标记(光伏/新能源车)。"""
    industry_info = classify_by_industry("光伏设备")
    assert industry_info["growth_to_cyclical_risk"] is True
    industry_info = classify_by_industry("医药生物")
    assert industry_info["growth_to_cyclical_risk"] is False
    print("✅ test_growth_to_cyclical_risk passed")


def test_fundamental_end_to_end():
    """端到端:analyze_fundamental 完整流程。"""
    financials = [
        {"net_profit": 10, "roic": 0.15, "fcf": 8, "revenue": 100},
        {"net_profit": 12, "roic": 0.16, "fcf": 10, "revenue": 120},
        {"net_profit": 15, "roic": 0.17, "fcf": 13, "revenue": 150},
        {"net_profit": 18, "roic": 0.18, "fcf": 16, "revenue": 180},
    ]
    r = analyze_fundamental(
        code="600276",
        name="测试医药",
        industry="医药生物",
        pe=35.0,
        pb=5.0,
        financials=financials,
    )
    assert r["classification"]["type"] == "成长"
    assert r["investment_approach"]["approach"] == "长期持有"
    assert r["pe_trap"]["pe"] == 35.0
    print(f"✅ test_fundamental_end_to_end passed (type: {r['classification']['type']}, approach: {r['investment_approach']['approach']})")


# ===== 行业叙事 + 扩展财务指标测试 =====

def test_classify_by_narrative_ai_compute():
    """AI 算力叙事:存储芯片行业识别为 ai_compute。"""
    r = classify_by_narrative("半导体(存储芯片)", "兆易创新")
    assert r["has_narrative"] is True
    ids = [n["id"] for n in r["narratives"]]
    assert "ai_compute" in ids
    print(f"✅ test_classify_by_narrative_ai_compute passed (narratives: {ids})")


def test_classify_by_narrative_robotics():
    """机器人叙事:减速器行业识别为 robotics。"""
    r = classify_by_narrative("汽车零部件", "双环传动")  # 名称含"传动"但不命中,需测试减速器
    r2 = classify_by_narrative("减速器", "XX传动")
    assert r2["has_narrative"] is True
    assert any(n["id"] == "robotics" for n in r2["narratives"])
    print(f"✅ test_classify_by_narrative_robotics passed")


def test_classify_by_narrative_domestic_substitution():
    """国产替代叙事:半导体设备识别为 domestic_substitution。"""
    r = classify_by_narrative("半导体设备", "北方华创")
    assert r["has_narrative"] is True
    assert any(n["id"] == "domestic_substitution" for n in r["narratives"])
    print(f"✅ test_classify_by_narrative_domestic_substitution passed")


def test_classify_by_narrative_none():
    """无叙事:传统行业不命中。"""
    r = classify_by_narrative("钢铁", "宝钢股份")
    assert r["has_narrative"] is False
    assert r["narratives"] == []
    print(f"✅ test_classify_by_narrative_none passed")


def test_gross_margin_trend_up():
    """毛利率上升趋势:定价权增强。"""
    margins = [20.0, 22.0, 25.0, 28.0, 30.0]
    r = compute_gross_margin_trend(margins)
    assert r["available"] is True
    assert r["trend"] == "上升"
    assert r["latest"] == 30.0
    print(f"✅ test_gross_margin_trend_up passed (mean={r['mean']}%, latest={r['latest']}%, trend={r['trend']})")


def test_gross_margin_trend_down():
    """毛利率下降趋势:竞争加剧。"""
    margins = [35.0, 32.0, 28.0, 24.0, 20.0]
    r = compute_gross_margin_trend(margins)
    assert r["available"] is True
    assert r["trend"] == "下降"
    print(f"✅ test_gross_margin_trend_down passed (mean={r['mean']}%, latest={r['latest']}%, trend={r['trend']})")


def test_gross_margin_trend_flat():
    """毛利率平稳。"""
    margins = [25.0, 25.5, 24.8, 25.2, 25.0]
    r = compute_gross_margin_trend(margins)
    assert r["available"] is True
    assert r["trend"] == "平稳"
    print(f"✅ test_gross_margin_trend_flat passed (mean={r['mean']}%, latest={r['latest']}%, trend={r['trend']})")


def test_revenue_growth_steady():
    """营收持续正增长。"""
    revenues = [100, 120, 145, 175, 210]
    r = compute_revenue_growth(revenues)
    assert r["available"] is True
    assert r["all_positive"] is True
    assert r["mean_rate"] > 0.2
    print(f"✅ test_revenue_growth_steady passed (mean_rate={r['mean_rate']*100:.1f}%)")


def test_revenue_growth_volatile():
    """营收波动:某年下滑。"""
    revenues = [100, 120, 80, 150, 130]
    r = compute_revenue_growth(revenues)
    assert r["available"] is True
    assert r["all_positive"] is False
    print(f"✅ test_revenue_growth_volatile passed (mean_rate={r['mean_rate']*100:.1f}%, all_pos={r['all_positive']})")


def test_operating_profit_quality_good():
    """扣非/净利 >= 0.9,主业贡献利润。"""
    op = [9, 12, 15, 18]
    np = [10, 13, 16, 19]
    r = compute_operating_profit_quality(op, np)
    assert r["available"] is True
    assert r["mean_ratio"] >= 0.9
    assert "良好" in r["quality"]
    print(f"✅ test_operating_profit_quality_good passed (ratio={r['mean_ratio']:.2f}, quality={r['quality']})")


def test_operating_profit_quality_poor():
    """扣非/净利 < 0.7,一次性损益多。"""
    op = [5, 6, 7, 8]
    np = [10, 12, 14, 16]  # 净利大幅高于扣非 -> 一次性损益多
    r = compute_operating_profit_quality(op, np)
    assert r["available"] is True
    assert r["mean_ratio"] < 0.7
    assert "差" in r["quality"]
    print(f"✅ test_operating_profit_quality_poor passed (ratio={r['mean_ratio']:.2f}, quality={r['quality']})")


def test_cyclical_with_growth_narrative_zhaoyi():
    """仿真:兆易创新场景 - 财务周期 + AI 叙事 = 周期(有成长潜力)。"""
    industry_info = classify_by_industry("半导体(存储芯片)")
    narrative_info = classify_by_narrative("半导体(存储芯片)", "兆易创新")
    # 财务呈现周期性:2021 高 -> 2023 低谷 -> 2025 反弹
    roic_stab = compute_roic_stability([0.175, 0.136, 0.0105, 0.0668, 0.0872])
    profits = compute_profit_growth([23.4, 20.5, 1.6, 11.0, 16.5])
    fcf = compute_fcf_quality([20, 18, 5, 10, 15], [23.4, 20.5, 1.6, 11.0, 16.5])

    margins = compute_gross_margin_trend([35.0, 30.0, 25.0, 28.0, 32.0])
    revs = compute_revenue_growth([85, 81, 58, 74, 92])
    op = compute_operating_profit_quality([22, 19, 1.5, 10, 15], [23.4, 20.5, 1.6, 11.0, 16.5])

    r = classify_stock_type(
        industry_info, roic_stab, profits, fcf,
        narrative_info=narrative_info,
        gross_margin=margins,
        revenue_growth=revs,
        operating_profit=op,
    )
    assert r["type"] == "周期(有成长潜力)", f"expected 周期(有成长潜力), got {r['type']}"
    assert r["narrative"]["has_narrative"] is True
    print(f"✅ test_cyclical_with_growth_narrative_zhaoyi passed (type: {r['type']})")
    print(f"   evidence: {r['evidence']}")


def test_pe_trap_cyclical_with_narrative_high_pe():
    """周期(有成长潜力)+ 高 PE = 叙事溢价,需业绩验证。"""
    pe_trap = detect_pe_trap(
        "周期(有成长潜力)",
        pe=140.0,
        profit_growth={"available": True, "all_positive": False, "max_rate": 9.0, "growth_rates": [9.0]},
    )
    assert pe_trap["available"] is True
    assert "叙事" in pe_trap["interpretation"] or "验证" in pe_trap["interpretation"]
    print(f"✅ test_pe_trap_cyclical_with_narrative_high_pe passed (interp: {pe_trap['interpretation']})")


def test_investment_approach_cyclical_with_narrative():
    """周期(有成长潜力)投资思路:波段为主 + 跟踪业绩。"""
    r = investment_approach("周期(有成长潜力)")
    assert "波段" in r["approach"]
    assert "业绩" in r["action"]
    print(f"✅ test_investment_approach_cyclical_with_narrative passed (approach: {r['approach']})")


def test_investment_approach_growth_with_narrative():
    """成长(叙事强化)投资思路:长期持有 + 叙事支撑。"""
    r = investment_approach("成长(叙事强化)")
    assert "长期持有" in r["approach"]
    assert "叙事" in r["rationale"]
    print(f"✅ test_investment_approach_growth_with_narrative passed (approach: {r['approach']})")


def test_revenue_down_profit_up_warning():
    """营收降但利润增 = 利润操纵嫌疑,在 evidence 中标记。"""
    industry_info = classify_by_industry("电子")
    roic_stab = compute_roic_stability([0.15, 0.16, 0.17, 0.18])
    profits = compute_profit_growth([10, 12, 14, 16, 18])  # 利润增
    fcf = compute_fcf_quality([8, 10, 13, 16, 20], [10, 12, 14, 16, 18])
    revs = compute_revenue_growth([100, 95, 90, 92, 88])  # 营收降
    r = classify_stock_type(
        industry_info, roic_stab, profits, fcf,
        revenue_growth=revs,
    )
    has_warning = any("利润操纵" in e for e in r["evidence"])
    assert has_warning, "expected 利润操纵 warning in evidence"
    print(f"✅ test_revenue_down_profit_up_warning passed (warning detected)")


# ===== 政治/地缘/政策风险测试 =====

def test_geopolitical_risk_overseas_mining():
    """海外矿业风险:有色金属行业识别为 overseas_resource。"""
    r = classify_geopolitical_risk("有色金属/铜矿", "紫金矿业")
    assert r["has_risk"] is True
    ids = [rt["id"] for rt in r["risk_types"]]
    assert "overseas_resource" in ids
    print(f"✅ test_geopolitical_risk_overseas_mining passed (risks: {ids})")


def test_geopolitical_risk_oil_gas():
    """海外油气风险:石油开采识别为 overseas_resource。"""
    r = classify_geopolitical_risk("石油石化/油气开采", "中国海油")
    assert r["has_risk"] is True
    assert any(rt["id"] == "overseas_resource" for rt in r["risk_types"])
    print(f"✅ test_geopolitical_risk_oil_gas passed")


def test_geopolitical_risk_semiconductor_sanction():
    """半导体制裁风险:半导体识别为 sanction。"""
    r = classify_geopolitical_risk("半导体(存储芯片+MCU)", "兆易创新")
    assert r["has_risk"] is True
    ids = [rt["id"] for rt in r["risk_types"]]
    assert "sanction" in ids
    print(f"✅ test_geopolitical_risk_semiconductor_sanction passed (risks: {ids})")


def test_geopolitical_risk_policy_dependency():
    """政策依赖风险:光伏识别为 policy_dependency。"""
    r = classify_geopolitical_risk("光伏设备", "隆基绿能")
    assert r["has_risk"] is True
    assert any(rt["id"] == "policy_dependency" for rt in r["risk_types"])
    print(f"✅ test_geopolitical_risk_policy_dependency passed")


def test_geopolitical_risk_cxo_double():
    """CXO 双重风险:创新药 + CXO 命中 sanction(海外)和 policy_dependency(国内集采)。"""
    r = classify_geopolitical_risk("医药生物/CXO", "药明康德")
    assert r["has_risk"] is True
    ids = [rt["id"] for rt in r["risk_types"]]
    # CXO 同时受海外制裁(美国生物安全法案)和国内集采影响
    assert "policy_dependency" in ids
    print(f"✅ test_geopolitical_risk_cxo_double passed (risks: {ids})")


def test_geopolitical_risk_strategic_resource():
    """战略资源:稀土识别为 strategic_resource(出口管制反向受益)。"""
    r = classify_geopolitical_risk("有色金属/稀土", "北方稀土")
    assert r["has_risk"] is True
    assert any(rt["id"] == "strategic_resource" for rt in r["risk_types"])
    print(f"✅ test_geopolitical_risk_strategic_resource passed")


def test_geopolitical_risk_none():
    """无风险敞口:普通公用事业行业不命中任何风险。"""
    r = classify_geopolitical_risk("公用事业/水务", "XX水务")
    assert r["has_risk"] is False
    assert r["risk_types"] == []
    print(f"✅ test_geopolitical_risk_none passed")


def test_geopolitical_risk_in_analyze_fundamental():
    """端到端:analyze_fundamental 输出包含 geopolitical_risk。"""
    financials = [
        {"net_profit": 10, "roic": 0.15, "fcf": 8, "revenue": 100, "gross_margin_pct": 30},
        {"net_profit": 12, "roic": 0.16, "fcf": 10, "revenue": 120, "gross_margin_pct": 32},
        {"net_profit": 15, "roic": 0.17, "fcf": 13, "revenue": 150, "gross_margin_pct": 33},
    ]
    r = analyze_fundamental(
        code="603986", name="兆易创新", industry="半导体",
        pe=50.0, pb=5.0, financials=financials,
    )
    assert "geopolitical_risk" in r
    assert r["geopolitical_risk"]["has_risk"] is True
    assert any(rt["id"] == "sanction" for rt in r["geopolitical_risk"]["risk_types"])
    print(f"✅ test_geopolitical_risk_in_analyze_fundamental passed (risks: {[rt['id'] for rt in r['geopolitical_risk']['risk_types']]})")


def test_geopolitical_risk_overseas_business_power_equipment():
    """海外业务风险:电力设备(输配电)识别为 overseas_business。"""
    r = classify_geopolitical_risk("电力设备(输配电一次设备)", "思源电气")
    assert r["has_risk"] is True
    ids = [rt["id"] for rt in r["risk_types"]]
    assert "overseas_business" in ids
    print(f"✅ test_geopolitical_risk_overseas_business_power_equipment passed (risks: {ids})")


def test_geopolitical_risk_overseas_business_construction_machinery():
    """海外业务风险:工程机械识别为 overseas_business。"""
    r = classify_geopolitical_risk("工程机械", "三一重工")
    assert r["has_risk"] is True
    assert any(rt["id"] == "overseas_business" for rt in r["risk_types"])
    print(f"✅ test_geopolitical_risk_overseas_business_construction_machinery passed")


def test_geopolitical_risk_overseas_business_home_appliance():
    """海外业务风险:家电(白电)识别为 overseas_business。"""
    r = classify_geopolitical_risk("白色家电", "海尔智家")
    assert r["has_risk"] is True
    assert any(rt["id"] == "overseas_business" for rt in r["risk_types"])
    print(f"✅ test_geopolitical_risk_overseas_business_home_appliance passed")


def test_roic_stability_seasonal_adjustment():
    """季节性调整:有年度数据(period 末尾 1231)时优先用年度数据算 cv。"""
    # 8 期季度数据,Q4 回款季节性导致 ROIC 大幅波动
    roics = [0.05, 0.02, 0.03, 0.15, 0.06, 0.02, 0.04, 0.16]
    periods = ["20210331", "20210630", "20210930", "20211231",
               "20220331", "20220630", "20220930", "20221231"]
    r = compute_roic_stability(roics, periods)
    assert r["available"] is True
    assert r["seasonal_adjusted"] is True
    # 只用了 2 期年度数据(20211231, 20221231)
    assert len(r["values"]) == 2
    assert r["used_periods"] == ["20211231", "20221231"]
    # 年度数据 cv 应远小于季度数据 cv
    print(f"✅ test_roic_stability_seasonal_adjustment passed (cv={r['cv']}, periods={r['used_periods']})")


def test_roic_stability_no_periods_fallback():
    """无 periods 时退化为原逻辑(全部数据),seasonal_adjusted=False。"""
    r = compute_roic_stability([0.15, 0.16, 0.17])
    assert r["available"] is True
    assert r["seasonal_adjusted"] is False
    assert r["used_periods"] is None
    assert len(r["values"]) == 3
    print(f"✅ test_roic_stability_no_periods_fallback passed")


def test_roic_stability_only_quarterly_no_annual():
    """只有季度数据(无 1231 期)时退化为原逻辑。"""
    roics = [0.05, 0.02, 0.03]
    periods = ["20210331", "20210630", "20210930"]
    r = compute_roic_stability(roics, periods)
    assert r["available"] is True
    assert r["seasonal_adjusted"] is False
    assert len(r["values"]) == 3
    print(f"✅ test_roic_stability_only_quarterly_no_annual passed")


def test_siyuan_industry_fallback():
    """思源电气 002028 行业映射兜底:电力设备(输配电一次设备)。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fetch_data import _fetch_industry_fallback
    ind = _fetch_industry_fallback("002028")
    assert "电力设备" in ind
    assert "输配电" in ind
    print(f"✅ test_siyuan_industry_fallback passed (industry: {ind})")


def test_is_bank_detection():
    """银行识别:行业含'银行'/code 在 BANK_CODES/名称含'银行'。"""
    from fundamental import is_bank
    assert is_bank("银行", "600036") is True
    assert is_bank("", "600036") is True  # code 匹配
    assert is_bank("股份制银行", "600036") is True
    assert is_bank("半导体", "603986") is False
    assert is_bank("", "", name="招商银行") is True
    assert is_bank("", "", name="兆易创新") is False
    print("✅ test_is_bank_detection passed")


def test_bank_quality_growth_bank():
    """优质银行(招行类):ROE > 12% 且稳定 -> 成长(优质银行)。"""
    from fundamental import analyze_fundamental
    financials = [
        {"period": "20221231", "net_profit": 100, "revenue": 1000, "roe": 16.5},
        {"period": "20231231", "net_profit": 110, "revenue": 1100, "roe": 16.2},
        {"period": "20241231", "net_profit": 120, "revenue": 1200, "roe": 15.8},
    ]
    r = analyze_fundamental("600036", "招商银行", "银行", pe=8, pb=1.0, financials=financials)
    assert r["is_bank"] is True
    assert r["classification"]["type"] == "成长(优质银行)"
    assert "长期持有" in r["investment_approach"]["approach"]
    assert r["pe_trap"]["valuation_anchor"] == "PB + 股息率"
    assert r["geopolitical_risk"]["risk_types"][0]["id"] == "bank_policy"
    print(f"✅ test_bank_quality_growth_bank passed (type={r['classification']['type']})")


def test_bank_quality_ordinary_bank():
    """普通银行:ROE 中等(10-12%) -> 周期(普通银行)。"""
    from fundamental import analyze_fundamental
    financials = [
        {"period": "20221231", "net_profit": 100, "revenue": 1000, "roe": 11.0},
        {"period": "20231231", "net_profit": 95, "revenue": 980, "roe": 10.5},
        {"period": "20241231", "net_profit": 105, "revenue": 1020, "roe": 10.8},
    ]
    r = analyze_fundamental("601398", "工商银行", "银行", pe=6, pb=0.8, financials=financials)
    assert r["is_bank"] is True
    assert "普通银行" in r["classification"]["type"]
    assert "波段操作" in r["investment_approach"]["approach"]
    print(f"✅ test_bank_quality_ordinary_bank passed (type={r['classification']['type']})")


def test_bank_quality_weak_bank():
    """弱质银行:ROE < 10% -> 周期(弱质银行)。"""
    from fundamental import analyze_fundamental
    financials = [
        {"period": "20221231", "net_profit": 50, "revenue": 1000, "roe": 8.0},
        {"period": "20231231", "net_profit": 40, "revenue": 950, "roe": 6.5},
        {"period": "20241231", "net_profit": 35, "revenue": 900, "roe": 5.8},
    ]
    r = analyze_fundamental("600015", "华夏银行", "银行", pe=5, pb=0.5, financials=financials)
    assert r["is_bank"] is True
    assert "弱质银行" in r["classification"]["type"]
    assert "观望" in r["investment_approach"]["approach"]
    print(f"✅ test_bank_quality_weak_bank passed (type={r['classification']['type']})")


def test_bank_quality_pb_valuation_anchor():
    """银行股估值锚是 PB,不是 PE;pe_trap 输出 PB + 股息率。"""
    from fundamental import analyze_fundamental
    financials = [
        {"period": "20231231", "net_profit": 100, "revenue": 1000, "roe": 15.0},
        {"period": "20241231", "net_profit": 110, "revenue": 1100, "roe": 15.2},
    ]
    r = analyze_fundamental("600036", "招商银行", "银行", pe=8, pb=1.2, financials=financials)
    assert r["pe_trap"]["available"] is True
    assert r["pe_trap"]["pb"] == 1.2
    assert "PB" in r["pe_trap"]["interpretation"]
    print(f"✅ test_bank_quality_pb_valuation_anchor passed (pb={r['pe_trap']['pb']})")


def test_bank_quality_pb_missing():
    """PB 缺失时 pe_trap.available=False,但估值锚仍标注 PB + 股息率。"""
    from fundamental import analyze_fundamental
    financials = [
        {"period": "20231231", "net_profit": 100, "revenue": 1000, "roe": 15.0},
        {"period": "20241231", "net_profit": 110, "revenue": 1100, "roe": 15.2},
    ]
    r = analyze_fundamental("600036", "招商银行", "银行", pe=None, pb=None, financials=financials)
    assert r["pe_trap"]["available"] is False
    assert r["pe_trap"]["valuation_anchor"] == "PB + 股息率"
    print(f"✅ test_bank_quality_pb_missing passed")


# ===== 筹码换手率衰减 + 底仓追踪 + ASR/CYQK 测试 =====

def test_chip_decay_turnover_mode():
    """有流通股本时,使用换手率衰减模型(筹码守恒)。"""
    from chip_distribution import compute_chip_distribution
    pv = [(10.0, 1000)] * 60
    daily = make_chip_daily(pv)
    # 流通股本 100,000 股,每日 vol=1000 -> turnover=1%
    bins, chip, meta = compute_chip_distribution(daily, free_float_shares=100000)
    assert meta["decay_mode"] == "turnover"
    assert meta["avg_turnover"] is not None and 0 < meta["avg_turnover"] < 0.02
    print(f"✅ test_chip_decay_turnover_mode passed (mode={meta['decay_mode']}, avg_turnover={meta['avg_turnover']:.4f})")


def test_chip_decay_fixed_fallback():
    """无流通股本时,降级到固定 decay=0.05。"""
    from chip_distribution import compute_chip_distribution
    pv = [(10.0, 1000)] * 60
    daily = make_chip_daily(pv)
    bins, chip, meta = compute_chip_distribution(daily, free_float_shares=None)
    assert meta["decay_mode"] == "fixed"
    assert meta["avg_turnover"] is None
    print(f"✅ test_chip_decay_fixed_fallback passed (mode={meta['decay_mode']})")


def test_asr_indicator():
    """ASR(活动筹码):±10% 带内筹码占比。横盘时 ASR 应较高。"""
    from chip_distribution import compute_chip_distribution, compute_asr
    pv = [(10.0, 1000)] * 60
    daily = make_chip_daily(pv)
    bins, chip, _ = compute_chip_distribution(daily)
    asr = compute_asr(bins, chip, current_close=10.0)
    assert asr["value"] >= 50, f"横盘时 ASR 应高,got {asr['value']}"
    assert asr["label"] == "高"
    print(f"✅ test_asr_indicator passed (ASR={asr['value']:.1f}%, label={asr['label']})")


def test_cyqk_profit_ratio():
    """CYQK 获利比例:横盘时当前价等于筹码密集区,获利比例约 50%。"""
    from chip_distribution import compute_chip_distribution, compute_cyqk
    pv = [(10.0, 1000)] * 60
    daily = make_chip_daily(pv)
    bins, chip, _ = compute_chip_distribution(daily)
    cyqk = compute_cyqk(bins, chip, current_close=10.0)
    # 横盘 + 三角分布,close 在峰位,获利比例应接近 50%
    assert 30 <= cyqk["win_ratio"] <= 70, f"获利比例应在 30-70%, got {cyqk['win_ratio']}"
    print(f"✅ test_cyqk_profit_ratio passed (win_ratio={cyqk['win_ratio']:.1f}%, label={cyqk['label']})")


def test_bottom_chip_retention_signal():
    """底仓不动:价格涨 30%+ 但底仓保留率 ≥50% -> 主升浪续涨信号。"""
    from chip_distribution import compute_bottom_chip_retention
    # 60 天前在 10 元横盘吸筹(底仓),后 30 天价升到 13+(涨 30%+),但底仓价位筹码仍保留
    pv = [(10.0, 3000)] * 60 + [(11.0, 800), (12.0, 800), (13.0, 800), (13.5, 800)] * 7 + [(13.5, 500)] * 2
    daily = make_chip_daily(pv)
    # 流通股本较小,使换手率较高,但底仓因量大仍保留
    r = compute_bottom_chip_retention(daily, free_float_shares=200000, lookback_days=30)
    assert r["available"] is True
    assert r["price_rise_pct"] >= 0.30, f"价格应涨 30%+, got {r['price_rise_pct']}"
    # 信号应为"底仓不动"或"底仓部分转移"(依换手率,底仓大量沉淀应保留)
    assert r["signal"] in ("底仓不动", "底仓部分转移"), f"signal={r['signal']}, retention={r['retention_ratio']}"
    print(f"✅ test_bottom_chip_retention_signal passed (signal={r['signal']}, retention={r['retention_ratio']}, rise={r['price_rise_pct']:.2f})")


def test_bottom_chip_disappearance_signal():
    """底仓消失:价格涨 30%+,底仓大量转移 -> 见顶信号。"""
    from chip_distribution import compute_bottom_chip_retention
    # 60 天前 10 元横盘,后 30 天价升到 14,且 10 元附近持续高换手(底仓被消化)
    pv = [(10.0, 1000)] * 60 + [(11.0, 5000), (12.0, 5000), (13.0, 5000), (14.0, 5000)] * 7 + [(14.0, 3000)] * 2
    daily = make_chip_daily(pv)
    # 流通股本较小 -> 每日换手率高 -> 底仓被快速消化
    r = compute_bottom_chip_retention(daily, free_float_shares=30000, lookback_days=30)
    assert r["available"] is True
    assert r["price_rise_pct"] >= 0.30
    # 底仓应大量消失(保留率低)
    assert r["retention_ratio"] < 0.5, f"底仓保留率应低, got {r['retention_ratio']}"
    print(f"✅ test_bottom_chip_disappearance_signal passed (signal={r['signal']}, retention={r['retention_ratio']}, rise={r['price_rise_pct']:.2f})")


def test_enhanced_pattern_main_launch():
    """低位单峰 + 今日换手 <1.5%(无量突破)= 主升浪信号。"""
    from chip_distribution import analyze
    # 80 天 10 元横盘形成低位单峰,最后 1 天价小幅突破且量小
    pv = [(10.0, 1000)] * 80 + [(10.2, 300), (10.3, 250), (10.4, 200)]
    daily = make_chip_daily(pv)
    # 流通股本 100,000 -> 今日 vol=200 -> turnover=0.2%(<1.5%)
    r = analyze(daily, free_float_shares=100000)
    pat = r["pattern"]
    assert pat["pattern"] == "单峰"
    assert pat["position_label"] == "低位"
    assert pat["enhanced_signal"] == "主升浪信号", f"expected 主升浪信号, got {pat['enhanced_signal']}"
    print(f"✅ test_enhanced_pattern_main_launch passed (signal={pat['enhanced_signal']}, turnover={pat['today_turnover']})")


def test_enhanced_pattern_top_signal():
    """高位单峰 + 底仓消失 = 见顶信号。"""
    from chip_distribution import analyze
    # 60 天 10 元吸筹底仓,30 天 15 元高位横盘(形成高位单峰)
    # lookback_days=30 默认:30 天前价 10,现在 15,涨 50%;高换手让底仓消失
    pv = [(10.0, 1000)] * 60 + [(15.0, 5000)] * 30
    daily = make_chip_daily(pv)
    r = analyze(daily, free_float_shares=20000)
    pat = r["pattern"]
    assert pat["position_label"] == "高位", f"expected 高位, got {pat['position_label']}"
    assert pat["enhanced_signal"] == "见顶信号", f"expected 见顶信号, got {pat['enhanced_signal']} (pattern={pat['pattern']}, retention={r['bottom_retention']['retention_ratio']})"
    print(f"✅ test_enhanced_pattern_top_signal passed (signal={pat['enhanced_signal']}, retention={r['bottom_retention']['retention_ratio']})")


def test_chip_analyze_with_float():
    """端到端:analyze() 接收 free_float_shares,返回 ASR/CYQK/底仓追踪。"""
    pv = [(10.0, 1000)] * 60 + [(11.0, 1500), (12.0, 1800), (13.0, 2000)]
    daily = make_chip_daily(pv)
    r = chip_analyze(daily, free_float_shares=100000)
    assert "asr" in r
    assert "cyqk" in r
    assert "bottom_retention" in r
    assert r["decay_meta"]["decay_mode"] == "turnover"
    assert r["asr"]["value"] > 0
    assert 0 <= r["cyqk"]["win_ratio"] <= 100
    print(f"✅ test_chip_analyze_with_float passed (ASR={r['asr']['value']:.1f}%, CYQK_WIN={r['cyqk']['win_ratio']:.1f}%, decay={r['decay_meta']['decay_mode']})")


def test_recompute_chip_with_float():
    """端到端:recompute_chip_with_float 用流通股本重算筹码。"""
    from compute_indicators import compute, recompute_chip_with_float
    closes = [10 + i * 0.05 for i in range(120)]
    vols = [1000 + i * 5 for i in range(120)]
    daily = make_daily(closes, vols)
    indicators = compute(daily)
    # 原 compute 用固定 decay
    assert indicators["chip"]["decay_meta"]["decay_mode"] == "fixed"
    # 重算用换手率衰减
    indicators = recompute_chip_with_float(indicators, daily, free_float_shares=500000)
    assert indicators["chip"]["decay_meta"]["decay_mode"] == "turnover"
    assert "asr" in indicators["chip"]
    assert "cyqk" in indicators["chip"]
    print(f"✅ test_recompute_chip_with_float passed (decay_mode={indicators['chip']['decay_meta']['decay_mode']}, ASR={indicators['chip']['asr']['value']:.1f}%)")


if __name__ == "__main__":
    test_position_percentile()
    test_position_label()
    test_vol_ma_cross_golden()
    test_vol_ma_cross_death()
    test_vol_ratio()
    test_ground_vol()
    test_quadrant_vol_up_price_up()
    test_quadrant_vol_down_price_down()
    test_trend_5d_vol_up_price_up()
    test_trend_5d_vol_down_price_down()
    test_trend_5d_vol_up_price_down()
    test_trend_5d_vol_down_price_up()
    test_trend_5d_flat()
    test_trend_5d_strength_strong()
    test_trend_5d_consistency()
    test_trend_5d_in_compute()
    test_trend_5d_vs_daily_divergence()
    test_top_divergence_detected()
    test_top_divergence_not_detected()
    test_breakout_3day()
    test_breakout_failed()
    test_compute_full()
    test_wash_trade_detection()
    test_algo_no_vol_rise()
    test_chip_low_single_peak()
    test_chip_high_single_peak()
    test_chip_double_peak()
    test_chip_support_resistance()
    test_chip_concentration()
    test_chip_in_compute()
    test_classify_growth_stock()
    test_classify_cyclical_stock()
    test_classify_fake_growth()
    test_pe_trap_cyclical_low()
    test_pe_trap_cyclical_high()
    test_pe_trap_growth_high()
    test_investment_approach()
    test_growth_to_cyclical_risk()
    test_fundamental_end_to_end()
    test_classify_by_narrative_ai_compute()
    test_classify_by_narrative_robotics()
    test_classify_by_narrative_domestic_substitution()
    test_classify_by_narrative_none()
    test_gross_margin_trend_up()
    test_gross_margin_trend_down()
    test_gross_margin_trend_flat()
    test_revenue_growth_steady()
    test_revenue_growth_volatile()
    test_operating_profit_quality_good()
    test_operating_profit_quality_poor()
    test_cyclical_with_growth_narrative_zhaoyi()
    test_pe_trap_cyclical_with_narrative_high_pe()
    test_investment_approach_cyclical_with_narrative()
    test_investment_approach_growth_with_narrative()
    test_revenue_down_profit_up_warning()
    test_geopolitical_risk_overseas_mining()
    test_geopolitical_risk_oil_gas()
    test_geopolitical_risk_semiconductor_sanction()
    test_geopolitical_risk_policy_dependency()
    test_geopolitical_risk_cxo_double()
    test_geopolitical_risk_strategic_resource()
    test_geopolitical_risk_none()
    test_geopolitical_risk_in_analyze_fundamental()
    test_geopolitical_risk_overseas_business_power_equipment()
    test_geopolitical_risk_overseas_business_construction_machinery()
    test_geopolitical_risk_overseas_business_home_appliance()
    test_roic_stability_seasonal_adjustment()
    test_roic_stability_no_periods_fallback()
    test_roic_stability_only_quarterly_no_annual()
    test_siyuan_industry_fallback()
    test_is_bank_detection()
    test_bank_quality_growth_bank()
    test_bank_quality_ordinary_bank()
    test_bank_quality_weak_bank()
    test_bank_quality_pb_valuation_anchor()
    test_bank_quality_pb_missing()
    test_chip_decay_turnover_mode()
    test_chip_decay_fixed_fallback()
    test_asr_indicator()
    test_cyqk_profit_ratio()
    test_bottom_chip_retention_signal()
    test_bottom_chip_disappearance_signal()
    test_enhanced_pattern_main_launch()
    test_enhanced_pattern_top_signal()
    test_chip_analyze_with_float()
    test_recompute_chip_with_float()
    print("\n🎉 All tests passed!")
