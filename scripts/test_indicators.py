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
    compute_turnover, compute_volume_price_detail,
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
from fetch_news import (
    _classify_sentiment, _compute_sentiment_summary, _extract_key_events,
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


# ===== ROE 深度分析测试(巴菲特 + 杜邦 + 假高 ROE) =====

def test_roe_stability_buffett_pass():
    """巴菲特标准:5 年均 ROE > 15% + 单年 ≥ 12% -> 通过。"""
    from fundamental import compute_roe_stability
    # 茅台式:5 年 ROE 都 > 20%,稳定
    roes = [25, 28, 30, 27, 29]
    r = compute_roe_stability(roes)
    assert r["available"] is True
    assert r["mean"] > 15
    assert r["min"] >= 12
    assert r["buffett_filter"]["pass"] is True
    print(f"✅ test_roe_stability_buffett_pass passed (mean={r['mean']}%, min={r['min']}%)")


def test_roe_stability_buffett_fail_low_mean():
    """均 ROE < 15% -> 巴菲特标准未过。"""
    from fundamental import compute_roe_stability
    roes = [10, 11, 12, 13, 14]  # 均值 12,不达 15%
    r = compute_roe_stability(roes)
    assert r["buffett_filter"]["pass"] is False
    assert r["buffett_filter"]["mean_pass"] is False
    print(f"✅ test_roe_stability_buffett_fail_low_mean passed (mean={r['mean']}%)")


def test_roe_stability_buffett_fail_low_min():
    """单年 ROE < 12% -> 巴菲特标准未过(虽然均值可能 > 15%)。"""
    from fundamental import compute_roe_stability
    # 某年掉到 8%,均值仍 > 15%
    roes = [20, 22, 8, 24, 25]
    r = compute_roe_stability(roes)
    assert r["buffett_filter"]["pass"] is False
    assert r["buffett_filter"]["min_pass"] is False
    assert r["min"] < 12
    print(f"✅ test_roe_stability_buffett_fail_low_min passed (min={r['min']}%)")


def test_dupont_high_margin_mode():
    """杜邦分析:高净利率驱动(茅台式)。净利率 > 15% + 权益乘数低。"""
    from fundamental import compute_dupont_analysis
    # 茅台式:净利率 50%,周转 0.5,权益乘数 1.5
    financials = [
        {"period": "20241231", "net_profit": 50, "revenue": 100,
         "total_assets": 200, "net_assets": 130, "roe": 38.5},
    ]
    r = compute_dupont_analysis(financials)
    assert r["available"] is True
    assert r["net_margin"] > 15
    assert r["dominant_mode"] == "高净利率"
    print(f"✅ test_dupont_high_margin_mode passed (mode={r['dominant_mode']}, 净利率={r['net_margin']}%)")


def test_dupont_high_turnover_mode():
    """杜邦分析:高周转驱动(Costco 式)。周转 > 1.0。"""
    from fundamental import compute_dupont_analysis
    # Costco 式:净利率 2%,周转 3.5,权益乘数 2.5
    financials = [
        {"period": "20241231", "net_profit": 7, "revenue": 350,
         "total_assets": 100, "net_assets": 40, "roe": 17.5},
    ]
    r = compute_dupont_analysis(financials)
    assert r["available"] is True
    assert r["asset_turnover"] > 1.0
    assert r["dominant_mode"] == "高周转"
    print(f"✅ test_dupont_high_turnover_mode passed (mode={r['dominant_mode']}, 周转={r['asset_turnover']})")


def test_dupont_high_leverage_mode():
    """杜邦分析:高杠杆驱动(房企式)。权益乘数 > 5。"""
    from fundamental import compute_dupont_analysis
    # 房企式:净利率 8%,周转 0.3,权益乘数 8
    financials = [
        {"period": "20241231", "net_profit": 24, "revenue": 300,
         "total_assets": 1000, "net_assets": 125, "roe": 19.2},
    ]
    r = compute_dupont_analysis(financials)
    assert r["available"] is True
    assert r["equity_multiplier"] > 5
    assert r["dominant_mode"] == "高杠杆"
    print(f"✅ test_dupont_high_leverage_mode passed (mode={r['dominant_mode']}, 权益乘数={r['equity_multiplier']})")


def test_buffett_filter_all_pass():
    """巴菲特三步全过:ROE 达标 + 资产负债率 < 50% + 现金流 ≥ 净利润。"""
    from fundamental import compute_buffett_filter
    financials = [
        {"period": "20201231", "net_profit": 100, "revenue": 1000, "roe": 18,
         "operating_cf": 110, "debt_ratio_pct": 35},
        {"period": "20211231", "net_profit": 110, "revenue": 1100, "roe": 19,
         "operating_cf": 115, "debt_ratio_pct": 33},
        {"period": "20221231", "net_profit": 120, "revenue": 1200, "roe": 20,
         "operating_cf": 125, "debt_ratio_pct": 32},
        {"period": "20231231", "net_profit": 130, "revenue": 1300, "roe": 21,
         "operating_cf": 135, "debt_ratio_pct": 30},
        {"period": "20241231", "net_profit": 140, "revenue": 1400, "roe": 22,
         "operating_cf": 145, "debt_ratio_pct": 28},
    ]
    r = compute_buffett_filter(financials)
    assert r["step1_roe"]["pass"] is True
    assert r["step2_debt"]["pass"] is True
    assert r["step3_cashflow"]["pass"] is True
    assert r["all_pass"] is True
    print(f"✅ test_buffett_filter_all_pass passed (三步全过)")


def test_buffett_filter_fail_high_debt():
    """巴菲特筛选失败:资产负债率过高(>50%)。"""
    from fundamental import compute_buffett_filter
    financials = [
        {"period": "20201231", "net_profit": 100, "revenue": 1000, "roe": 20,
         "operating_cf": 110, "debt_ratio_pct": 65},
        {"period": "20211231", "net_profit": 110, "revenue": 1100, "roe": 21,
         "operating_cf": 115, "debt_ratio_pct": 68},
    ]
    r = compute_buffett_filter(financials)
    assert r["step1_roe"]["pass"] is True
    assert r["step2_debt"]["pass"] is False
    assert r["all_pass"] is False
    print(f"✅ test_buffett_filter_fail_high_debt passed (step2 fail: 资产负债率高)")


def test_buffett_filter_fail_cashflow():
    """巴菲特筛选失败:经营现金流 < 净利润(盈利质量差)。"""
    from fundamental import compute_buffett_filter
    financials = [
        {"period": "20231231", "net_profit": 100, "revenue": 1000, "roe": 20,
         "operating_cf": 50, "debt_ratio_pct": 30},
    ]
    r = compute_buffett_filter(financials)
    assert r["step3_cashflow"]["pass"] is False
    assert r["all_pass"] is False
    print(f"✅ test_buffett_filter_fail_cashflow passed (step3 fail: 现金流不匹配)")


def test_fake_roe_high_leverage_warning():
    """假高 ROE:高杠杆驱动(权益乘数 > 5 + 净利率 < 10%)。"""
    from fundamental import detect_fake_roe
    financials = [
        {"period": "20241231", "net_profit": 24, "revenue": 300,
         "total_assets": 1000, "net_assets": 125, "operating_profit": 23, "net_profit_prev": 20, "net_assets_prev": 120},
    ]
    r = detect_fake_roe(financials)
    assert r["is_fake"] is True
    types = [w["type"] for w in r["warnings"]]
    assert "high_leverage" in types
    print(f"✅ test_fake_roe_high_leverage_warning passed (warnings: {types})")


def test_fake_roe_one_shot_gain_warning():
    """假高 ROE:一次性收益(扣非/NI < 0.7)。"""
    from fundamental import detect_fake_roe
    # 净利润 100,扣非只有 50(比率 0.5 < 0.7) - 卖资产凑利润
    financials = [
        {"period": "20241231", "net_profit": 100, "operating_profit": 50,
         "revenue": 1000, "total_assets": 500, "net_assets": 300},
    ]
    r = detect_fake_roe(financials)
    assert r["is_fake"] is True
    types = [w["type"] for w in r["warnings"]]
    assert "one_shot_gain" in types
    print(f"✅ test_fake_roe_one_shot_gain_warning passed (warnings: {types})")


def test_fake_roe_buyback_shrink_warning():
    """假高 ROE:回购缩分母(净资产同比下降 + 净利润未下降)。"""
    from fundamental import detect_fake_roe
    # 上年净资产 200,今年 180(下降 10%);上年净利润 100,今年 110(上升)
    financials = [
        {"period": "20231231", "net_profit": 100, "net_assets": 200,
         "revenue": 1000, "total_assets": 500},
        {"period": "20241231", "net_profit": 110, "net_assets": 180,
         "revenue": 1100, "total_assets": 520},
    ]
    r = detect_fake_roe(financials)
    assert r["is_fake"] is True
    types = [w["type"] for w in r["warnings"]]
    assert "buyback_shrink" in types
    print(f"✅ test_fake_roe_buyback_shrink_warning passed (warnings: {types})")


def test_fake_roe_clean():
    """健康 ROE:无假高 ROE 信号。"""
    from fundamental import detect_fake_roe
    # 茅台式:高净利率、低杠杆、扣非匹配、净资产增长
    financials = [
        {"period": "20231231", "net_profit": 500, "net_assets": 1500,
         "revenue": 1000, "total_assets": 1800, "operating_profit": 490},
        {"period": "20241231", "net_profit": 550, "net_assets": 1700,
         "revenue": 1100, "total_assets": 2000, "operating_profit": 540},
    ]
    r = detect_fake_roe(financials)
    assert r["is_fake"] is False
    assert r["warning_count"] == 0
    print(f"✅ test_fake_roe_clean passed (无警告,ROE 健康)")


def test_analyze_fundamental_includes_roe_quality():
    """端到端:analyze_fundamental 输出包含 roe_quality 字段。"""
    from fundamental import analyze_fundamental
    financials = [
        {"period": "20201231", "net_profit": 100, "revenue": 1000, "roe": 18,
         "operating_cf": 110, "debt_ratio_pct": 35, "total_assets": 500, "net_assets": 325,
         "roic": 15, "fcf": 110, "gross_margin_pct": 60, "operating_profit": 95},
        {"period": "20211231", "net_profit": 110, "revenue": 1100, "roe": 19,
         "operating_cf": 115, "debt_ratio_pct": 33, "total_assets": 540, "net_assets": 365,
         "roic": 16, "fcf": 115, "gross_margin_pct": 61, "operating_profit": 105},
        {"period": "20221231", "net_profit": 120, "revenue": 1200, "roe": 20,
         "operating_cf": 125, "debt_ratio_pct": 32, "total_assets": 580, "net_assets": 400,
         "roic": 17, "fcf": 125, "gross_margin_pct": 62, "operating_profit": 115},
        {"period": "20231231", "net_profit": 130, "revenue": 1300, "roe": 21,
         "operating_cf": 135, "debt_ratio_pct": 30, "total_assets": 620, "net_assets": 435,
         "roic": 18, "fcf": 135, "gross_margin_pct": 63, "operating_profit": 125},
        {"period": "20241231", "net_profit": 140, "revenue": 1400, "roe": 22,
         "operating_cf": 145, "debt_ratio_pct": 28, "total_assets": 660, "net_assets": 475,
         "roic": 19, "fcf": 145, "gross_margin_pct": 64, "operating_profit": 135},
    ]
    r = analyze_fundamental("600519", "贵州茅台", "食品饮料", pe=30, pb=10, financials=financials)
    assert "roe_quality" in r
    rq = r["roe_quality"]
    assert "roe_stability" in rq
    assert "dupont" in rq
    assert "buffett_filter" in rq
    assert "fake_roe" in rq
    assert rq["roe_stability"]["buffett_filter"]["pass"] is True
    assert rq["buffett_filter"]["all_pass"] is True
    assert rq["fake_roe"]["is_fake"] is False
    print(f"✅ test_analyze_fundamental_includes_roe_quality passed (ROE 均 {rq['roe_stability']['mean']}%, 巴菲特全过)")


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


# ===== 机构研报评估 测试 =====

def test_is_foreign_broker_keywords():
    """外资/港资/台资券商识别:关键词匹配。"""
    from fetch_research_reports import is_foreign_broker
    assert is_foreign_broker("高盛高华证券") is True
    assert is_foreign_broker("瑞银证券") is True
    assert is_foreign_broker("摩根士丹利华鑫") is True
    assert is_foreign_broker("中银国际") is True
    assert is_foreign_broker("招银国际") is True
    assert is_foreign_broker("群益证券") is True
    assert is_foreign_broker("汇丰前海证券") is True
    assert is_foreign_broker("野村东方国际") is True
    # 内资券商
    assert is_foreign_broker("中信证券") is False
    assert is_foreign_broker("国泰君安") is False
    assert is_foreign_broker("华泰证券") is False
    assert is_foreign_broker("东方财富证券") is False
    # 空值
    assert is_foreign_broker("") is False
    assert is_foreign_broker(None) is False
    print("✅ test_is_foreign_broker_keywords passed")


def test_normalize_rating():
    """评级标准化:东财评级 -> 标准化标签。"""
    from fetch_research_reports import normalize_rating
    assert normalize_rating("买入") == "buy"
    assert normalize_rating("增持") == "overweight"
    assert normalize_rating("推荐") == "overweight"
    assert normalize_rating("中性") == "neutral"
    assert normalize_rating("持有") == "neutral"
    assert normalize_rating("减持") == "reduce"
    assert normalize_rating("卖出") == "sell"
    assert normalize_rating("回避") == "sell"
    assert normalize_rating("") == "unknown"
    assert normalize_rating("未知评级") == "unknown"
    assert normalize_rating(None) == "unknown"
    print("✅ test_normalize_rating passed")


def test_rating_consensus_strong():
    """评级共识:主导评级占多数 = 共识强。"""
    from fetch_research_reports import _compute_rating_consensus
    reports = [
        {"rating_norm": "buy"},
        {"rating_norm": "buy"},
        {"rating_norm": "buy"},
        {"rating_norm": "buy"},
        {"rating_norm": "buy"},
        {"rating_norm": "buy"},
        {"rating_norm": "buy"},
        {"rating_norm": "buy"},
        {"rating_norm": "overweight"},
        {"rating_norm": "overweight"},
        {"rating_norm": "overweight"},
    ]
    rc = _compute_rating_consensus(reports)
    assert rc["available"] is True
    assert rc["total"] == 11
    assert rc["dominant"] == "buy"
    assert rc["dominant_label"] == "买入"
    assert rc["dominant_pct"] == 72.7
    assert rc["label"] == "共识强"
    assert rc["score_mean"] == 1.73  # (8*2 + 3*1) / 11 = 19/11
    print(f"✅ test_rating_consensus_strong passed (主导 {rc['dominant_label']} {rc['dominant_pct']}%/{rc['label']})")


def test_rating_consensus_divergent():
    """评级共识:评级分散 = 分歧大。"""
    from fetch_research_reports import _compute_rating_consensus
    reports = [
        {"rating_norm": "buy"},
        {"rating_norm": "overweight"},
        {"rating_norm": "neutral"},
        {"rating_norm": "reduce"},
    ]
    rc = _compute_rating_consensus(reports)
    assert rc["available"] is True
    assert rc["label"] == "分歧大"
    assert rc["consensus_strength"] == 0.25
    print(f"✅ test_rating_consensus_divergent passed ({rc['label']}, 强度 {rc['consensus_strength']})")


def test_target_price_with_upside():
    """目标价统计:含上涨空间计算。"""
    from fetch_research_reports import _compute_target_price
    reports = [
        {"aim_price": 1500.0},
        {"aim_price": 1600.0},
        {"aim_price": 1700.0},
        {"aim_price": 1800.0},
        {"aim_price": 1900.0},
    ]
    tp = _compute_target_price(reports, current_price=1500.0)
    assert tp["available"] is True
    assert tp["count"] == 5
    assert tp["mean"] == 1700.0
    assert tp["max"] == 1900.0
    assert tp["min"] == 1500.0
    assert tp["upside_pct"] == 13.3
    assert tp["label"] == "空间中"
    print(f"✅ test_target_price_with_upside passed (mean {tp['mean']}, upside {tp['upside_pct']}%/{tp['label']})")


def test_target_price_no_current():
    """目标价统计:无当前价时 upside 为 None。"""
    from fetch_research_reports import _compute_target_price
    reports = [{"aim_price": 100.0}, {"aim_price": 120.0}]
    tp = _compute_target_price(reports, current_price=None)
    assert tp["available"] is True
    assert tp["mean"] == 110.0
    assert tp["upside_pct"] is None
    print(f"✅ test_target_price_no_current passed (mean {tp['mean']}, upside None)")


def test_eps_forecast_latest():
    """盈利预测:取最近有预测的研报。"""
    from fetch_research_reports import _compute_eps_forecast
    reports = [
        {"publish_date": "2026-05-25", "org": "诚通证券",
         "eps_this_year": 66.68, "pe_this_year": 19.8,
         "eps_next_year": 69.43, "pe_next_year": 19.1,
         "eps_year_after_next": 72.56, "pe_year_after_next": 18.2},
        {"publish_date": "2026-04-01", "org": "群益证券",
         "eps_this_year": 68.0, "pe_this_year": 20.0,
         "eps_next_year": 70.0, "pe_next_year": 19.5,
         "eps_year_after_next": None, "pe_year_after_next": None},
    ]
    ef = _compute_eps_forecast(reports)
    assert ef["available"] is True
    assert ef["current_year"]["eps"] == 66.68
    assert ef["current_year"]["org"] == "诚通证券"
    assert ef["next_year"]["eps"] == 69.43
    assert ef["year_after_next"]["eps"] == 72.56
    print(f"✅ test_eps_forecast_latest passed (今年 EPS {ef['current_year']['eps']}, 明年 {ef['next_year']['eps']})")


def test_foreign_summary():
    """外资汇总:外资研报数 + 共识 + 最近一条。"""
    from fetch_research_reports import _compute_foreign_summary
    foreign_reports = [
        {"org": "群益证券", "is_foreign": True, "publish_date": "2026-04-28",
         "rating": "持有", "rating_norm": "neutral", "aim_price": 1525.0,
         "title": "转型初见成效"},
        {"org": "群益证券", "is_foreign": True, "publish_date": "2026-01-09",
         "rating": "持有", "rating_norm": "neutral", "aim_price": None,
         "title": "春节前投放"},
    ]
    fs = _compute_foreign_summary(foreign_reports, current_price=1700.0)
    assert fs["available"] is True
    assert fs["count"] == 2
    assert fs["rating_consensus"]["dominant"] == "neutral"
    assert fs["latest"]["org"] == "群益证券"
    assert fs["latest"]["rating"] == "持有"
    print(f"✅ test_foreign_summary passed (count {fs['count']}, 共识 {fs['rating_consensus']['dominant_label']})")


def test_divergence_foreign_pessimistic():
    """分歧度:外资明显更悲观。"""
    from fetch_research_reports import _compute_divergence
    reports = [
        {"is_foreign": True, "rating_norm": "neutral"},
        {"is_foreign": True, "rating_norm": "neutral"},
        {"is_foreign": True, "rating_norm": "neutral"},
        {"is_foreign": False, "rating_norm": "buy"},
        {"is_foreign": False, "rating_norm": "buy"},
        {"is_foreign": False, "rating_norm": "buy"},
        {"is_foreign": False, "rating_norm": "buy"},
        {"is_foreign": False, "rating_norm": "buy"},
    ]
    foreign_reports = [r for r in reports if r.get("is_foreign")]
    div = _compute_divergence(reports, foreign_reports)
    assert div["available"] is True
    assert div["foreign_score"] == 0.0
    assert div["domestic_score"] == 2.0
    assert div["diff"] == -2.0
    assert div["label"] == "外资明显更悲观"
    print(f"✅ test_divergence_foreign_pessimistic passed ({div['label']}, diff {div['diff']})")


def test_divergence_consistent():
    """分歧度:内外资一致。"""
    from fetch_research_reports import _compute_divergence
    reports = [
        {"is_foreign": True, "rating_norm": "buy"},
        {"is_foreign": True, "rating_norm": "buy"},
        {"is_foreign": False, "rating_norm": "buy"},
        {"is_foreign": False, "rating_norm": "buy"},
    ]
    foreign_reports = [r for r in reports if r.get("is_foreign")]
    div = _compute_divergence(reports, foreign_reports)
    assert div["available"] is True
    assert div["diff"] == 0.0
    assert div["label"] == "内外资一致"
    print(f"✅ test_divergence_consistent passed ({div['label']})")


def test_summarize_research_report_empty():
    """研报评估摘要:空数据降级。"""
    from fundamental import summarize_research_report
    # None
    r = summarize_research_report(None)
    assert r["available"] is False
    assert r["quality_signal"] == "无覆盖"
    # error
    r = summarize_research_report({"error": "网络错误", "total_reports": 0})
    assert r["available"] is False
    assert r["quality_signal"] == "无覆盖"
    # total 0
    r = summarize_research_report({"total_reports": 0, "reports": []})
    assert r["available"] is False
    print("✅ test_summarize_research_report_empty passed")


def test_summarize_research_report_strong():
    """研报评估摘要:≥10 篇 + 共识强 = 机构认可度强。"""
    from fundamental import summarize_research_report
    reports_data = {
        "total_reports": 11,
        "rating_consensus": {
            "available": True, "total": 11, "dominant": "buy", "dominant_label": "买入",
            "dominant_pct": 72.7, "label": "共识强", "consensus_strength": 0.727, "score_mean": 1.73,
        },
        "target_price": {"available": False},
        "foreign_summary": {"available": False, "count": 0},
        "divergence": {"available": False},
        "eps_forecast": {
            "available": True,
            "current_year": {"eps": 1.73, "pe": 22.73},
            "next_year": {"eps": 2.01, "pe": 19.5},
            "year_after_next": {"eps": 2.32, "pe": 16.96},
        },
    }
    r = summarize_research_report(reports_data)
    assert r["available"] is True
    assert r["quality_signal"] == "强"
    assert r["total_reports"] == 11
    assert any("机构评级共识" in e for e in r["evidence"])
    assert any("盈利预测" in e for e in r["evidence"])
    print(f"✅ test_summarize_research_report_strong passed (quality={r['quality_signal']}, evidence {len(r['evidence'])} 条)")


def test_summarize_research_report_weak():
    """研报评估摘要:<5 篇 = 机构认可度弱。"""
    from fundamental import summarize_research_report
    reports_data = {
        "total_reports": 2,
        "rating_consensus": {
            "available": True, "total": 2, "dominant": "buy", "dominant_label": "买入",
            "dominant_pct": 100.0, "label": "共识强", "consensus_strength": 1.0, "score_mean": 2.0,
        },
        "target_price": {"available": False},
        "foreign_summary": {"available": False, "count": 0},
        "divergence": {"available": False},
        "eps_forecast": {"available": False},
    }
    r = summarize_research_report(reports_data)
    assert r["available"] is True
    assert r["quality_signal"] == "弱"
    print(f"✅ test_summarize_research_report_weak passed (quality={r['quality_signal']})")


def test_analyze_fundamental_with_research_reports():
    """端到端:analyze_fundamental 接收 research_reports 参数,输出 research_report 字段并写入 evidence。"""
    from fundamental import analyze_fundamental
    financials = [
        {"period": "20201231", "net_profit": 100, "revenue": 1000, "roe": 18,
         "operating_cf": 110, "debt_ratio_pct": 35, "total_assets": 500, "net_assets": 325,
         "roic": 15, "fcf": 110, "gross_margin_pct": 60, "operating_profit": 95},
        {"period": "20211231", "net_profit": 110, "revenue": 1100, "roe": 19,
         "operating_cf": 115, "debt_ratio_pct": 33, "total_assets": 540, "net_assets": 365,
         "roic": 16, "fcf": 115, "gross_margin_pct": 61, "operating_profit": 105},
        {"period": "20221231", "net_profit": 120, "revenue": 1200, "roe": 20,
         "operating_cf": 125, "debt_ratio_pct": 32, "total_assets": 580, "net_assets": 400,
         "roic": 17, "fcf": 125, "gross_margin_pct": 62, "operating_profit": 115},
        {"period": "20231231", "net_profit": 130, "revenue": 1300, "roe": 21,
         "operating_cf": 135, "debt_ratio_pct": 30, "total_assets": 620, "net_assets": 435,
         "roic": 18, "fcf": 135, "gross_margin_pct": 63, "operating_profit": 125},
        {"period": "20241231", "net_profit": 140, "revenue": 1400, "roe": 22,
         "operating_cf": 145, "debt_ratio_pct": 28, "total_assets": 660, "net_assets": 475,
         "roic": 19, "fcf": 145, "gross_margin_pct": 64, "operating_profit": 135},
    ]
    research_reports = {
        "total_reports": 15,
        "rating_consensus": {
            "available": True, "total": 15, "dominant": "buy", "dominant_label": "买入",
            "dominant_pct": 80.0, "label": "共识强", "consensus_strength": 0.8, "score_mean": 1.8,
        },
        "target_price": {
            "available": True, "count": 5, "mean": 1850.0, "median": 1800.0,
            "max": 2200.0, "min": 1500.0, "spread_pct": 37.8,
            "current_price": 1700.0, "upside_pct": 8.8, "label": "空间中",
        },
        "foreign_summary": {
            "available": True, "count": 3,
            "rating_consensus": {
                "available": True, "total": 3, "dominant": "neutral", "dominant_label": "中性",
                "dominant_pct": 100.0, "label": "共识强", "consensus_strength": 1.0, "score_mean": 0.0,
            },
            "target_price": {"available": False},
            "latest": {"title": "test", "org": "群益证券", "date": "2026-04-28",
                       "rating": "持有", "rating_norm": "neutral", "aim_price": 1525.0},
        },
        "divergence": {
            "available": True, "foreign_score": 0.0, "domestic_score": 2.0,
            "diff": -2.0, "label": "外资明显更悲观",
            "foreign_target_price": None, "domestic_target_price": 1900.0,
            "target_price_divergence_pct": None,
        },
        "eps_forecast": {
            "available": True,
            "current_year": {"eps": 66.68, "pe": 19.8},
            "next_year": {"eps": 69.43, "pe": 19.1},
            "year_after_next": {"eps": 72.56, "pe": 18.2},
        },
    }
    r = analyze_fundamental("600519", "贵州茅台", "食品饮料", pe=30, pb=10,
                            financials=financials, research_reports=research_reports)
    assert "research_report" in r
    rr = r["research_report"]
    assert rr["available"] is True
    assert rr["quality_signal"] == "强"
    assert rr["total_reports"] == 15
    # evidence 含研报共识
    ev = r["classification"]["evidence"]
    assert any("机构评级共识" in e for e in ev)
    assert any("目标价均值" in e for e in ev)
    assert any("外资" in e for e in ev)
    assert any("内外资分歧" in e for e in ev)
    # research_quality 字段
    rq = r["classification"].get("research_quality", {})
    assert rq.get("quality_signal") == "强"
    assert rq.get("total_reports") == 15
    print(f"✅ test_analyze_fundamental_with_research_reports passed (quality={rr['quality_signal']}, evidence {len(ev)} 条)")


def test_analyze_fundamental_no_research_reports():
    """端到端:不传 research_reports 时,research_report.available=False 但不报错。"""
    from fundamental import analyze_fundamental
    financials = [
        {"period": "20231231", "net_profit": 100, "revenue": 1000, "roe": 18,
         "operating_cf": 110, "debt_ratio_pct": 35, "total_assets": 500, "net_assets": 325,
         "roic": 15, "fcf": 110, "gross_margin_pct": 60, "operating_profit": 95},
        {"period": "20241231", "net_profit": 110, "revenue": 1100, "roe": 19,
         "operating_cf": 115, "debt_ratio_pct": 33, "total_assets": 540, "net_assets": 365,
         "roic": 16, "fcf": 115, "gross_margin_pct": 61, "operating_profit": 105},
    ]
    r = analyze_fundamental("600519", "贵州茅台", "食品饮料", pe=30, pb=10, financials=financials)
    assert "research_report" in r
    assert r["research_report"]["available"] is False
    print("✅ test_analyze_fundamental_no_research_reports passed")


# ===== 半导体行业特殊处理 + 短期筹码量价趋势 测试 =====

def test_is_semiconductor_by_industry():
    """半导体识别:行业关键词。"""
    from fundamental import is_semiconductor
    assert is_semiconductor("半导体", "", "") is True
    assert is_semiconductor("集成电路", "", "") is True
    assert is_semiconductor("EDA", "", "") is True
    assert is_semiconductor("半导体设备", "", "") is True
    assert is_semiconductor("功率半导体", "", "") is True
    assert is_semiconductor("第三代半导体", "", "") is True
    # 非半导体行业
    assert is_semiconductor("食品饮料", "", "") is False
    assert is_semiconductor("银行", "", "") is False
    assert is_semiconductor("房地产", "", "") is False
    print("✅ test_is_semiconductor_by_industry passed")


def test_is_semiconductor_by_name():
    """半导体识别:公司名称关键词。"""
    from fundamental import is_semiconductor
    assert is_semiconductor("", "301269", "华大九天") is True
    assert is_semiconductor("", "688981", "中芯国际") is True
    assert is_semiconductor("", "002049", "紫光国微") is True
    assert is_semiconductor("", "603986", "兆易创新") is True
    assert is_semiconductor("", "688256", "寒武纪") is True
    # 非半导体公司
    assert is_semiconductor("", "600519", "贵州茅台") is False
    assert is_semiconductor("", "000001", "平安银行") is False
    print("✅ test_is_semiconductor_by_name passed")


def test_semiconductor_special_handling():
    """端到端:半导体行业触发 special_handling,投资思路改写,基本面降级。"""
    from fundamental import analyze_fundamental
    financials = [
        {"period": "20201231", "net_profit": -50, "revenue": 500, "roe": 5,
         "operating_cf": -30, "debt_ratio_pct": 30, "total_assets": 800, "net_assets": 600,
         "roic": 3, "fcf": -30, "gross_margin_pct": 80, "operating_profit": -80},
        {"period": "20211231", "net_profit": -30, "revenue": 700, "roe": 4,
         "operating_cf": -20, "debt_ratio_pct": 28, "total_assets": 900, "net_assets": 700,
         "roic": 2, "fcf": -20, "gross_margin_pct": 82, "operating_profit": -50},
        {"period": "20221231", "net_profit": -20, "revenue": 900, "roe": 3,
         "operating_cf": -10, "debt_ratio_pct": 26, "total_assets": 1000, "net_assets": 800,
         "roic": 1, "fcf": -10, "gross_margin_pct": 85, "operating_profit": -30},
        {"period": "20231231", "net_profit": -10, "revenue": 1100, "roe": 2,
         "operating_cf": 0, "debt_ratio_pct": 24, "total_assets": 1100, "net_assets": 900,
         "roic": 1, "fcf": 0, "gross_margin_pct": 88, "operating_profit": -20},
        {"period": "20241231", "net_profit": -5, "revenue": 1300, "roe": 1,
         "operating_cf": 10, "debt_ratio_pct": 22, "total_assets": 1200, "net_assets": 1000,
         "roic": 2, "fcf": 10, "gross_margin_pct": 90, "operating_profit": -10},
    ]
    r = analyze_fundamental("301269", "华大九天", None, pe=1195, pb=15, financials=financials)
    assert "semiconductor_handling" in r
    sh = r["semiconductor_handling"]
    assert sh is not None
    assert sh["is_semiconductor"] is True
    assert sh["fundamental_weight"] == "low"
    assert sh["short_term_weight"] == "high"
    # 投资思路被改写
    appr = r["investment_approach"]
    assert "半导体" in appr["approach"]
    assert "短期" in appr["approach"]
    # evidence 含半导体提示
    ev = r["classification"]["evidence"]
    assert any("半导体行业:基本面权重降级" in e for e in ev)
    assert any("国产替代" in e for e in ev)
    print(f"✅ test_semiconductor_special_handling passed (approach={appr['approach']})")


def test_non_semiconductor_no_special_handling():
    """端到端:非半导体行业不触发 special_handling。"""
    from fundamental import analyze_fundamental
    financials = [
        {"period": "20231231", "net_profit": 100, "revenue": 1000, "roe": 18,
         "operating_cf": 110, "debt_ratio_pct": 35, "total_assets": 500, "net_assets": 325,
         "roic": 15, "fcf": 110, "gross_margin_pct": 60, "operating_profit": 95},
        {"period": "20241231", "net_profit": 110, "revenue": 1100, "roe": 19,
         "operating_cf": 115, "debt_ratio_pct": 33, "total_assets": 540, "net_assets": 365,
         "roic": 16, "fcf": 115, "gross_margin_pct": 61, "operating_profit": 105},
    ]
    r = analyze_fundamental("600519", "贵州茅台", "食品饮料", pe=30, pb=10, financials=financials)
    assert r.get("semiconductor_handling") is None
    print("✅ test_non_semiconductor_no_special_handling passed")


def test_short_term_chip_trend_basic():
    """短期筹码对比:5/10/20 日窗口都返回数据。"""
    from chip_distribution import compute_short_term_chip_trend
    import random
    random.seed(42)
    daily = []
    price = 100
    for i in range(120):
        op = price
        cl = price + random.uniform(-1, 1.5)
        hi = max(op, cl) + random.uniform(0, 0.8)
        lo = min(op, cl) - random.uniform(0, 0.8)
        vol = 100000 + i * 800 + random.randint(-10000, 20000)
        daily.append({"open": op, "high": hi, "low": lo, "close": cl, "volume": vol})
        price = cl
    r = compute_short_term_chip_trend(daily, free_float_shares=5000000, windows=[5, 10, 20])
    assert r["available"] is True
    assert 5 in r["windows"]
    assert 10 in r["windows"]
    assert 20 in r["windows"]
    # 每个窗口都有主峰和 CYQK
    for w in [5, 10, 20]:
        w_data = r["windows"][w]
        assert "dominant_peak" in w_data
        assert "cyqk_win_ratio" in w_data
        assert "concentration_5pct" in w_data
    # 趋势字段
    assert "trend" in r
    assert "peak_migration" in r["trend"]
    assert "concentration_trend" in r["trend"]
    assert "cyqk_trend" in r["trend"]
    print(f"✅ test_short_term_chip_trend_basic passed (trend={r['trend']})")


def test_short_term_chip_trend_concentration_rising():
    """短期筹码对比:5 日集中度 > 20 日,识别筹码集中。"""
    from chip_distribution import compute_short_term_chip_trend
    # 构造数据:前 110 天分散波动,后 10 天在 117-118 区间密集成交
    # 5 日窗口全是集中区,20 日窗口包含 10 天分散 + 10 天集中
    daily = []
    for i in range(110):
        p = 100 + (i % 10)
        daily.append({"open": p, "high": p+1, "low": p-1, "close": p, "volume": 50000})
    for i in range(10):
        p = 117 + (i % 3) * 0.5
        daily.append({"open": p, "high": p+1, "low": p-1, "close": p, "volume": 200000})
    r = compute_short_term_chip_trend(daily, free_float_shares=5000000, windows=[5, 10, 20])
    assert r["available"] is True
    short_conc = r["windows"][5]["concentration_5pct"]
    long_conc = r["windows"][20]["concentration_5pct"]
    # 5 日集中度应明显高于 20 日
    assert short_conc > long_conc
    assert r["trend"]["concentration_trend"] == "上升"
    print(f"✅ test_short_term_chip_trend_concentration_rising passed (5d={short_conc}% vs 20d={long_conc}%, {r['trend']['concentration_trend']})")


def test_short_term_trend_acceleration():
    """短期量价趋势:5 日斜率 > 20 日 = 加速上涨。"""
    from compute_indicators import compute_short_term_trend
    import pandas as pd
    # 构造数据:前 100 天震荡,后 15 天慢涨,最后 5 天加速上涨
    df_data = []
    for i in range(100):
        p = 100 + (i % 5)
        df_data.append({"open": p, "high": p+1, "low": p-1, "close": p, "volume": 50000, "amount": 50000*p})
    # 后 15 天慢涨(每天 +0.3)
    for i in range(15):
        p = 105 + i * 0.3
        df_data.append({"open": p, "high": p+1, "low": p-1, "close": p, "volume": 70000, "amount": 70000*p})
    # 最后 5 天加速(每天 +2,量也递增)
    for i in range(5):
        p = 110 + i * 2
        v = 100000 + i * 15000
        df_data.append({"open": p, "high": p+2, "low": p-0.5, "close": p+1, "volume": v, "amount": v*p})
    df = pd.DataFrame(df_data)
    r = compute_short_term_trend(df, windows=[5, 10, 20])
    assert r["available"] is True
    assert r["acceleration"] == "加速上涨"
    # 5 日象限应该是量增价涨
    assert "量增" in r["windows"][5]["quadrant"]
    assert "价涨" in r["windows"][5]["quadrant"]
    print(f"✅ test_short_term_trend_acceleration passed (acceleration={r['acceleration']}, 5d={r['windows'][5]['quadrant']})")


def test_short_term_trend_in_compute():
    """端到端:compute() 输出包含 short_term_trend 字段。"""
    from compute_indicators import compute
    closes = [10 + i * 0.05 for i in range(120)]
    vols = [1000 + i * 5 for i in range(120)]
    daily = make_daily(closes, vols)
    indicators = compute(daily)
    assert "short_term_trend" in indicators
    stt = indicators["short_term_trend"]
    assert stt["available"] is True
    assert "windows" in stt
    assert "acceleration" in stt
    print(f"✅ test_short_term_trend_in_compute passed (acceleration={stt['acceleration']})")


def test_short_term_chip_in_recompute():
    """端到端:compute() 默认用固定 decay 算 short_term_chip;recompute_chip_with_float 用流通股本重算(换手率衰减)。"""
    from compute_indicators import compute, recompute_chip_with_float
    closes = [10 + i * 0.05 for i in range(120)]
    vols = [1000 + i * 5 for i in range(120)]
    daily = make_daily(closes, vols)
    indicators = compute(daily)
    # compute() 默认用固定 decay 算 short_term_chip
    assert "short_term_chip" in indicators
    assert indicators["short_term_chip"]["available"] is True
    assert indicators["short_term_chip"]["windows"][5]["decay_mode"] == "fixed"
    # recompute_chip_with_float 用流通股本重算(换手率衰减)
    indicators = recompute_chip_with_float(indicators, daily, free_float_shares=500000)
    assert indicators["short_term_chip"]["available"] is True
    assert indicators["short_term_chip"]["windows"][5]["decay_mode"] == "turnover"
    print(f"✅ test_short_term_chip_in_recompute passed (trend={indicators['short_term_chip'].get('trend', {}).get('peak_migration')}, decay={indicators['short_term_chip']['windows'][5]['decay_mode']})")


# ==================== 主力资金流 + 筹码交叉验证 测试 ====================

def _make_flow_df(days: int = 30, base_main_net: float = 0.0,
                  trend_per_day: float = 0.0, base_close: float = 42.0) -> list:
    """合成 N 天资金流 DataFrame(用于测试,不走网络)。"""
    import pandas as pd
    rows = []
    for i in range(days):
        # 主力净额 = 基础 + 趋势*i + 小波动
        main_net = base_main_net + trend_per_day * i
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "main_net": main_net,
            "main_pct": main_net / 1e6,  # 简化:1百万 = 1%
            "super_large_net": main_net * 0.6,
            "large_net": main_net * 0.4,
            "mid_net": -main_net * 0.5,
            "small_net": -main_net * 0.5,
            "super_large_pct": main_net / 1e6 * 0.6,
            "large_pct": main_net / 1e6 * 0.4,
            "mid_pct": -main_net / 1e6 * 0.5,
            "small_pct": -main_net / 1e6 * 0.5,
            "close": base_close + i * 0.1,
            "pct_chg": 0.1,
        })
    return pd.DataFrame(rows)


def test_capital_flow_detect_secid():
    """secid 检测:SH=1.XXX, SZ/BJ=0.XXX。"""
    from fetch_capital_flow import _detect_secid
    assert _detect_secid("002472") == "0.002472"  # SZ
    assert _detect_secid("600519") == "1.600519"  # SH
    assert _detect_secid("300750") == "0.300750"  # SZ 创业板
    assert _detect_secid("830879") == "0.830879"  # BJ
    print("✅ test_capital_flow_detect_secid passed")


def test_capital_flow_parse_klines():
    """klines 字符串解析:date,main_net,small_net,...,close,pct_chg。"""
    from fetch_capital_flow import _parse_klines
    klines = [
        "2026-01-07,-80205389.0,130697523.0,-50492112.0,-46555824.0,-33649565.0,-7.05,11.48,-4.44,-4.09,-2.96,45.51,-1.13,0.00,0.00",
        "2026-01-08,-42617582.0,25539294.0,17078288.0,-15575600.0,-27041982.0,-3.99,2.39,1.60,-1.46,-2.53,45.48,-0.07,0.00,0.00",
    ]
    df = _parse_klines(klines)
    assert len(df) == 2
    assert df.iloc[0]["main_net"] == -80205389.0
    assert df.iloc[0]["small_net"] == 130697523.0
    assert df.iloc[0]["close"] == 45.51
    assert df.iloc[1]["main_pct"] == -3.99
    # 按日期升序
    assert df.iloc[0]["date"] < df.iloc[1]["date"]
    print(f"✅ test_capital_flow_parse_klines passed (rows={len(df)}, first_main_net={df.iloc[0]['main_net']})")


def test_capital_flow_cumulative():
    """5/10/20 日累计计算 + 流入/流出天数。"""
    from fetch_capital_flow import _compute_cumulative
    df = _make_flow_df(days=25, base_main_net=1e7)  # 全部流入
    cum = _compute_cumulative(df)
    assert cum["5d"]["available"] is True
    assert cum["5d"]["days_inflow"] == 5
    assert cum["5d"]["days_outflow"] == 0
    assert cum["5d"]["main_net_amount"] > 0
    assert cum["10d"]["days_inflow"] == 10
    assert cum["20d"]["days_inflow"] == 20
    print(f"✅ test_capital_flow_cumulative passed (5d_net={cum['5d']['main_net_amount']:.0f}, 5d_inflow_days={cum['5d']['days_inflow']})")


def test_capital_flow_consecutive_inflow():
    """连续流入天数:最近5日全流入 -> consecutive=5, label=持续流入。"""
    from fetch_capital_flow import _compute_trend
    df = _make_flow_df(days=25, base_main_net=1e7)  # 全部流入
    trend = _compute_trend(df)
    assert trend["available"] is True
    assert trend["consecutive_days"] == 25  # 全部25天都流入
    assert "流入" in trend["consecutive_label"]
    print(f"✅ test_capital_flow_consecutive_inflow passed (consecutive={trend['consecutive_days']}, label={trend['consecutive_label']})")


def test_capital_flow_consecutive_outflow():
    """连续流出天数:最近5日全流出 -> consecutive=-5, label=持续流出。"""
    from fetch_capital_flow import _compute_trend
    df = _make_flow_df(days=25, base_main_net=-1e7)  # 全部流出
    trend = _compute_trend(df)
    assert trend["consecutive_days"] == -25
    assert "流出" in trend["consecutive_label"]
    print(f"✅ test_capital_flow_consecutive_outflow passed (consecutive={trend['consecutive_days']}, label={trend['consecutive_label']})")


def test_capital_flow_ma_cross_golden():
    """MA5 上穿 MA20:前20天流出,今日大量流入 -> 金叉(交叉正好在今日)。"""
    from fetch_capital_flow import _compute_trend
    import pandas as pd
    rows = []
    # 前20天:小幅流出
    for i in range(20):
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "main_net": -1e6, "main_pct": -0.1, "super_large_net": 0,
                     "large_net": 0, "mid_net": 0, "small_net": 0,
                     "super_large_pct": 0, "large_pct": 0, "mid_pct": 0, "small_pct": 0,
                     "close": 42.0, "pct_chg": 0.0})
    # 第21天(今日):大幅流入,让 MA5 在今日上穿 MA20
    rows.append({"date": pd.Timestamp("2026-01-21"),
                 "main_net": 5e7, "main_pct": 5.0, "super_large_net": 0,
                 "large_net": 0, "mid_net": 0, "small_net": 0,
                 "super_large_pct": 0, "large_pct": 0, "mid_pct": 0, "small_pct": 0,
                 "close": 42.0, "pct_chg": 0.0})
    df = pd.DataFrame(rows)
    trend = _compute_trend(df)
    # 昨日 MA5=-1e6, MA20=-1e6(相等,prev_ma5 <= prev_ma20 成立)
    # 今日 MA5=(4*(-1e6)+5e7)/5=9.2e6, MA20=(19*(-1e6)+5e7)/20=1.55e6 -> MA5>MA20 -> 金叉
    assert trend["ma_cross"] == "金叉", f"expected 金叉, got {trend['ma_cross']} (MA5={trend['main_net_ma5']:.0f}, MA20={trend['main_net_ma20']:.0f})"
    print(f"✅ test_capital_flow_ma_cross_golden passed (ma_cross={trend['ma_cross']}, MA5={trend['main_net_ma5']:.0f}, MA20={trend['main_net_ma20']:.0f})")


def test_capital_flow_ma_cross_death():
    """MA5 下穿 MA20:前20天流入,今日大量流出 -> 死叉。"""
    from fetch_capital_flow import _compute_trend
    import pandas as pd
    rows = []
    for i in range(20):
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "main_net": 1e6, "main_pct": 0.1, "super_large_net": 0,
                     "large_net": 0, "mid_net": 0, "small_net": 0,
                     "super_large_pct": 0, "large_pct": 0, "mid_pct": 0, "small_pct": 0,
                     "close": 42.0, "pct_chg": 0.0})
    rows.append({"date": pd.Timestamp("2026-01-21"),
                 "main_net": -5e7, "main_pct": -5.0, "super_large_net": 0,
                 "large_net": 0, "mid_net": 0, "small_net": 0,
                 "super_large_pct": 0, "large_pct": 0, "mid_pct": 0, "small_pct": 0,
                 "close": 42.0, "pct_chg": 0.0})
    df = pd.DataFrame(rows)
    trend = _compute_trend(df)
    assert trend["ma_cross"] == "死叉", f"expected 死叉, got {trend['ma_cross']} (MA5={trend['main_net_ma5']:.0f}, MA20={trend['main_net_ma20']:.0f})"
    print(f"✅ test_capital_flow_ma_cross_death passed (ma_cross={trend['ma_cross']}, MA5={trend['main_net_ma5']:.0f}, MA20={trend['main_net_ma20']:.0f})")


def test_capital_flow_signal_accumulation():
    """吸筹信号:连续≥3日净流入 + 5日均主力净占比≥5%。"""
    from fetch_capital_flow import _compute_signals, _compute_trend, _compute_cumulative
    # 5天,每天主力净流入 6百万,占比 6%
    df = _make_flow_df(days=20, base_main_net=6e6)
    trend = _compute_trend(df)
    cum = _compute_cumulative(df)
    sig = _compute_signals(df, trend, cum)
    assert sig["main_force_action"] == "吸筹"
    print(f"✅ test_capital_flow_signal_accumulation passed (action={sig['main_force_action']}, strength={sig['strength']})")


def test_capital_flow_signal_distribution():
    """派发信号:连续≥3日净流出 + 5日均主力净占比≤-5%。"""
    from fetch_capital_flow import _compute_signals, _compute_trend, _compute_cumulative
    df = _make_flow_df(days=20, base_main_net=-6e6)
    trend = _compute_trend(df)
    cum = _compute_cumulative(df)
    sig = _compute_signals(df, trend, cum)
    assert sig["main_force_action"] == "派发"
    print(f"✅ test_capital_flow_signal_distribution passed (action={sig['main_force_action']}, strength={sig['strength']})")


def test_capital_flow_signal_strong_accumulation():
    """强吸筹:连续≥5日 + 5日均占比≥10%。"""
    from fetch_capital_flow import _compute_signals, _compute_trend, _compute_cumulative
    # 5天,每天主力净流入 1.2千万,占比 12%
    df = _make_flow_df(days=20, base_main_net=1.2e7)
    trend = _compute_trend(df)
    cum = _compute_cumulative(df)
    sig = _compute_signals(df, trend, cum)
    assert sig["main_force_action"] == "吸筹"
    assert sig["strength"] == "强"
    print(f"✅ test_capital_flow_signal_strong_accumulation passed (action={sig['main_force_action']}, strength={sig['strength']})")


def test_capital_flow_signal_neutral():
    """中性:净流入流出交替,信号不足。"""
    from fetch_capital_flow import _compute_signals, _compute_trend, _compute_cumulative
    import pandas as pd
    rows = []
    for i in range(20):
        # 交替正负
        main_net = 1e6 if i % 2 == 0 else -1e6
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "main_net": main_net, "main_pct": main_net / 1e6,
                     "super_large_net": 0, "large_net": 0, "mid_net": 0, "small_net": 0,
                     "super_large_pct": 0, "large_pct": 0, "mid_pct": 0, "small_pct": 0,
                     "close": 42.0, "pct_chg": 0.0})
    df = pd.DataFrame(rows)
    trend = _compute_trend(df)
    cum = _compute_cumulative(df)
    sig = _compute_signals(df, trend, cum)
    assert sig["main_force_action"] == "中性"
    print(f"✅ test_capital_flow_signal_neutral passed (action={sig['main_force_action']})")


def test_cross_validate_strong_accumulation():
    """强吸筹:主峰上移 + 资金流入 -> 强吸筹(高置信度)。"""
    from chip_distribution import cross_validate_chip_capital
    chip = {"available": True}
    short_term_chip = {
        "available": True,
        "trend": {
            "peak_migration": "向上迁移",
            "concentration_trend": "上升",
            "cyqk_trend": "上升",
        },
    }
    capital_flow = {
        "available": True,
        "today": {"main_net_amount": 5e7, "main_net_pct": 8.0},
        "cumulative": {"5d": {"main_net_amount": 2e8}},
        "trend": {"consecutive_days": 5, "ma_cross": "金叉"},
        "signals": {"main_force_action": "吸筹", "strength": "强"},
    }
    result = cross_validate_chip_capital(chip, short_term_chip, capital_flow)
    assert result["available"] is True
    assert result["main_force_intent"] == "强吸筹"
    assert result["confidence"] == "高"
    print(f"✅ test_cross_validate_strong_accumulation passed (intent={result['main_force_intent']}, confidence={result['confidence']})")


def test_cross_validate_distribution():
    """派发:主峰上移 + 资金流出 -> 派发(高位换手出货)。"""
    from chip_distribution import cross_validate_chip_capital
    chip = {"available": True}
    short_term_chip = {
        "available": True,
        "trend": {
            "peak_migration": "向上迁移",
            "concentration_trend": "下降",  # 集中度下降
            "cyqk_trend": "下降",
        },
    }
    capital_flow = {
        "available": True,
        "today": {"main_net_amount": -5e7, "main_net_pct": -8.0},
        "cumulative": {"5d": {"main_net_amount": -2e8}},
        "trend": {"consecutive_days": -5, "ma_cross": "死叉"},
        "signals": {"main_force_action": "派发", "strength": "强"},
    }
    result = cross_validate_chip_capital(chip, short_term_chip, capital_flow)
    assert result["available"] is True
    # 主峰上移 + 资金流出 -> 派发(高置信度)
    assert result["main_force_intent"] in ("派发", "强派发")
    assert result["confidence"] == "高"
    print(f"✅ test_cross_validate_distribution passed (intent={result['main_force_intent']}, confidence={result['confidence']})")


def test_cross_validate_contradiction():
    """矛盾:主峰上移 + 集中度下降(筹码信号内部矛盾)。"""
    from chip_distribution import cross_validate_chip_capital
    chip = {"available": True}
    short_term_chip = {
        "available": True,
        "trend": {
            "peak_migration": "向上迁移",
            "concentration_trend": "下降",  # 与主峰上移矛盾
            "cyqk_trend": "稳定",
        },
    }
    capital_flow = {
        "available": True,
        "today": {"main_net_amount": 1e6, "main_net_pct": 0.5},
        "cumulative": {"5d": {"main_net_amount": 5e6}},
        "trend": {"consecutive_days": 1, "ma_cross": "无交叉"},
        "signals": {"main_force_action": "中性", "strength": "弱"},
    }
    result = cross_validate_chip_capital(chip, short_term_chip, capital_flow)
    assert result["available"] is True
    # 主峰上移 + 集中度下降 = 矛盾
    assert result["main_force_intent"] == "矛盾"
    assert result["confidence"] == "低"
    print(f"✅ test_cross_validate_contradiction passed (intent={result['main_force_intent']}, confidence={result['confidence']})")


def test_cross_validate_unavailable():
    """任一输入缺失 -> available=False。"""
    from chip_distribution import cross_validate_chip_capital
    # 资金流不可用
    result = cross_validate_chip_capital(
        {"available": True},
        {"available": True, "trend": {}},
        {"available": False, "error": "rate limited"},
    )
    assert result["available"] is False

    # 短期筹码不可用
    result = cross_validate_chip_capital(
        {"available": True},
        {"available": False, "reason": "数据不足"},
        {"available": True, "today": {}, "trend": {}, "signals": {}},
    )
    assert result["available"] is False
    print("✅ test_cross_validate_unavailable passed")


def test_ths_realtime_parse():
    """THS 即时数据解析:中文金额"亿/万" -> 元,代码归一化 6 位。"""
    from fetch_capital_flow import _parse_ths_realtime
    import pandas as pd

    df = pd.DataFrame([
        {"code": "002472", "name": "双环传动", "price": 42.2, "pct_chg": -0.75,
         "turnover_rate": 3.22, "inflow": 483000000.0, "outflow": 545000000.0,
         "net": -61349600.0, "turnover": 1044000000.0},
        {"code": "600519", "name": "贵州茅台", "price": 1500, "pct_chg": 0.5,
         "turnover_rate": 0.2, "inflow": 1e9, "outflow": 8e8,
         "net": 2e8, "turnover": 2e9},
    ])
    r = _parse_ths_realtime("002472", df)
    assert r is not None
    assert r["source"] == "ths"
    assert r["available"] is True
    assert r["days_returned"] == 1
    assert r["today"]["main_net_amount"] == -61349600.0
    assert r["today"]["main_net_pct"] == -5.88  # -61349600 / 1044000000 * 100
    assert r["signals"]["main_force_action"] == "派发"
    assert r["signals"]["strength"] == "中"
    assert r["trend"]["available"] is False

    # 茅台(净流入 2 亿,占比 10%) -> 强吸筹
    r2 = _parse_ths_realtime("600519", df)
    assert r2["signals"]["main_force_action"] == "吸筹"
    assert r2["signals"]["strength"] == "强"

    # 不存在的代码
    r3 = _parse_ths_realtime("999999", df)
    assert r3 is None
    print("✅ test_ths_realtime_parse passed")


def test_ths_realtime_signal_thresholds():
    """THS 简化信号阈值:弱/中/强三档基于 |main_net_pct|。"""
    from fetch_capital_flow import _parse_ths_realtime
    import pandas as pd

    base = {"price": 10.0, "pct_chg": 0.0, "turnover_rate": 1.0}

    # 弱:占比 < 5%
    df_weak = pd.DataFrame([{
        "code": "000001", "name": "X", "inflow": 1.01e8, "outflow": 1.0e8,
        "net": 1e6, "turnover": 1e9, **base,
    }])
    r = _parse_ths_realtime("000001", df_weak)
    assert r["signals"]["strength"] == "弱"
    assert r["signals"]["main_force_action"] == "吸筹"

    # 中:占比 5-10%
    df_mid = pd.DataFrame([{
        "code": "000001", "name": "X", "inflow": 1.6e8, "outflow": 1.0e8,
        "net": 6e7, "turnover": 1e9, **base,
    }])
    r = _parse_ths_realtime("000001", df_mid)
    assert r["signals"]["strength"] == "中"
    assert r["signals"]["main_force_action"] == "吸筹"

    # 强:占比 >= 10%
    df_strong = pd.DataFrame([{
        "code": "000001", "name": "X", "inflow": 1.15e8, "outflow": 1.0e8,
        "net": 1.5e8, "turnover": 1e9, **base,
    }])
    r = _parse_ths_realtime("000001", df_strong)
    assert r["signals"]["strength"] == "强"
    assert r["signals"]["main_force_action"] == "吸筹"

    # 中性:净额为 0
    df_neutral = pd.DataFrame([{
        "code": "000001", "name": "X", "inflow": 1e8, "outflow": 1e8,
        "net": 0, "turnover": 1e9, **base,
    }])
    r = _parse_ths_realtime("000001", df_neutral)
    assert r["signals"]["main_force_action"] == "中性"
    print("✅ test_ths_realtime_signal_thresholds passed")


def test_cross_validate_ths_source_confidence_cap():
    """THS 降级源(只有今日数据)置信度上限为"中"。"""
    from chip_distribution import cross_validate_chip_capital

    short_term_chip = {
        "available": True,
        "trend": {
            "peak_migration": "向上迁移",
            "concentration_trend": "上升",
            "cyqk_trend": "稳定",
        },
    }
    # 东财源:强吸筹 + 筹码集中 -> 高置信度
    capital_em = {
        "available": True,
        "source": "eastmoney",
        "today": {"main_net_amount": 2e8, "main_net_pct": 15},
        "cumulative": {"5d": {"main_net_amount": 5e8}},
        "trend": {"available": True, "consecutive_days": 5, "ma_cross": "金叉"},
        "signals": {"main_force_action": "吸筹", "strength": "强"},
    }
    r_em = cross_validate_chip_capital({"available": True}, short_term_chip, capital_em)
    assert r_em["main_force_intent"] == "强吸筹"
    assert r_em["confidence"] == "高"

    # THS 源:同样信号 -> 置信度降为"中"
    capital_ths = {
        "available": True,
        "source": "ths",
        "today": {"main_net_amount": 2e8, "main_net_pct": 15},
        "cumulative": {"today": {"main_net_amount": 2e8}},
        "trend": {"available": False, "reason": "THS 即时数据无日线序列"},
        "signals": {"main_force_action": "吸筹", "strength": "强"},
    }
    r_ths = cross_validate_chip_capital({"available": True}, short_term_chip, capital_ths)
    assert r_ths["main_force_intent"] == "强吸筹"
    assert r_ths["confidence"] == "中"  # 被 THS 源限制
    # evidence 应标注 THS 降级
    assert any("THS" in e or "降级" in e for e in r_ths["evidence"])
    print(f"✅ test_cross_validate_ths_source_confidence_cap passed (em={r_em['confidence']}, ths={r_ths['confidence']})")


def test_cross_validate_ths_outflow_distribution():
    """THS 降级源 + 主峰上移 + 资金流出 -> 派发,置信度中。"""
    from chip_distribution import cross_validate_chip_capital

    short_term_chip = {
        "available": True,
        "trend": {
            "peak_migration": "向上迁移",
            "concentration_trend": "稳定",
            "cyqk_trend": "稳定",
        },
    }
    capital_ths = {
        "available": True,
        "source": "ths",
        "today": {"main_net_amount": -6e7, "main_net_pct": -6},
        "cumulative": {"today": {"main_net_amount": -6e7}},
        "trend": {"available": False, "reason": "THS"},
        "signals": {"main_force_action": "派发", "strength": "中"},
    }
    r = cross_validate_chip_capital({"available": True}, short_term_chip, capital_ths)
    assert r["main_force_intent"] == "派发"
    assert r["confidence"] == "中"  # THS 限制 + 中等强度
    print(f"✅ test_cross_validate_ths_outflow_distribution passed (intent={r['main_force_intent']}, confidence={r['confidence']})")


# ========== 换手率 + 量价细化测试 ==========

def test_turnover_calculation():
    """换手率计算正确性:266117手 / 8.5亿股 = 3.13%"""
    daily = make_daily([10.0] * 30, [266117] * 30)
    r = compute_turnover(daily, 8.5e8)
    assert r["available"] is True
    assert abs(r["today"] - 3.13) < 0.1, f"today={r['today']}"
    print(f"✅ test_turnover_calculation passed (today={r['today']}%)")


def test_turnover_label_cold():
    """换手率<1% -> 冷门"""
    # 8.5亿股本,<1% 需要 vol < 85000 手
    daily = make_daily([10.0] * 30, [50000] * 30)
    r = compute_turnover(daily, 8.5e8)
    assert r["label"] == "冷门", f"label={r['label']}, today={r['today']}"
    print(f"✅ test_turnover_label_cold passed (today={r['today']}%, label={r['label']})")


def test_turnover_label_normal():
    """换手率 1-3% -> 常态"""
    # 1.5% = 127500 手
    daily = make_daily([10.0] * 30, [127500] * 30)
    r = compute_turnover(daily, 8.5e8)
    assert r["label"] == "常态", f"label={r['label']}, today={r['today']}"
    print(f"✅ test_turnover_label_normal passed (today={r['today']}%, label={r['label']})")


def test_turnover_label_active():
    """换手率 5-10% -> 活跃"""
    # 7% = 595000 手
    daily = make_daily([10.0] * 30, [595000] * 30)
    r = compute_turnover(daily, 8.5e8)
    assert r["label"] == "活跃", f"label={r['label']}, today={r['today']}"
    print(f"✅ test_turnover_label_active passed (today={r['today']}%, label={r['label']})")


def test_turnover_label_warning():
    """换手率>15% -> 套现警戒"""
    # 18% = 1530000 手
    daily = make_daily([10.0] * 30, [1530000] * 30)
    r = compute_turnover(daily, 8.5e8)
    assert r["label"] == "套现警戒", f"label={r['label']}, today={r['today']}"
    print(f"✅ test_turnover_label_warning passed (today={r['today']}%, label={r['label']})")


def test_history_top_detection():
    """120日内2次天量(换手>=10%) -> is_top_signal=True"""
    # 120 天,平均 vol=100000(换手 1.18%),第 50、100 天 vol=900000(换手 10.59%)
    vols = [100000] * 120
    vols[49] = 900000
    vols[99] = 950000
    daily = make_daily([10.0] * 120, vols)
    r = compute_turnover(daily, 8.5e8)
    assert r["history_top_count"] == 2, f"history_top_count={r['history_top_count']}"
    assert r["is_top_signal"] is True, f"is_top_signal={r['is_top_signal']}"
    print(f"✅ test_history_top_detection passed (count={r['history_top_count']}, is_top={r['is_top_signal']})")


def test_volume_price_flat_low_position():
    """低位量增价平 -> 吸筹"""
    # 价格长期下跌后稳定在 10(低位),今日量增 + 价平
    closes = [20.0] * 10 + [15.0] * 10 + [10.0] * 10
    vols = [100000] * 30
    vols[-1] = 150000  # 今日量增 50%
    daily = make_daily(closes, vols)
    turnover = compute_turnover(daily, 8.5e8)
    r = compute_volume_price_detail(daily, turnover, "低位")
    assert r["available"] is True
    assert r["primary_signal"] == "量增价平-吸筹", f"signal={r['primary_signal']}"
    print(f"✅ test_volume_price_flat_low_position passed (signal={r['primary_signal']})")


def test_volume_price_flat_high_position():
    """高位量增价平 -> 出货"""
    # 价格从 5 涨到 10(高位),今日量增 + 价平
    closes = [5.0] * 10 + [7.0] * 10 + [10.0] * 10
    vols = [100000] * 30
    vols[-1] = 150000
    daily = make_daily(closes, vols)
    turnover = compute_turnover(daily, 8.5e8)
    r = compute_volume_price_detail(daily, turnover, "高位")
    assert r["available"] is True
    assert r["primary_signal"] == "量增价平-出货", f"signal={r['primary_signal']}"
    print(f"✅ test_volume_price_flat_high_position passed (signal={r['primary_signal']})")


def test_shrink_up_high_control():
    """均线陡峭 + 缩量上涨 -> 高控盘"""
    # 5日均线陡峭:5天内从 10 涨到 11.1(日均斜率 >2%)
    # ma5[-1] ≈ 10.92, ma5[-5] ≈ 10.5, slope = (10.92-10.5)/10.5/5 ≈ 0.008 -> 不够陡
    # 需要更陡:5天从 10 涨到 12(日均 4%)
    closes = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 11.8, 12.0]
    closes = [c * 0.9 for c in closes[:5]] + closes  # 前面加 5 天铺垫
    vols = [200000] * 10 + [180000, 170000, 160000, 150000, 100000]  # 最后一天量缩
    daily = make_daily(closes, vols)
    turnover = compute_turnover(daily, 8.5e8)
    r = compute_volume_price_detail(daily, turnover, "高位")
    # 今日价涨 + 量缩 + 均线陡峭 -> 高控盘
    signals = [s["signal"] for s in r["signals"]]
    assert "量缩价涨-高控盘" in signals, f"signals={signals}, ma5_slope={r['ma5_slope']}"
    print(f"✅ test_shrink_up_high_control passed (signals={signals}, slope={r['ma5_slope']})")


def test_shrink_up_capital_exhausted():
    """均线疲软 + 缩量上涨 -> 资金枯竭"""
    # 价格长期稳定 + 缓涨,均线疲软
    closes = [10.0] * 25 + [10.05, 10.06, 10.07, 10.08, 10.10]
    vols = [100000] * 25 + [95000, 90000, 85000, 80000, 70000]  # 量缩
    daily = make_daily(closes, vols)
    turnover = compute_turnover(daily, 8.5e8)
    r = compute_volume_price_detail(daily, turnover, "中位")
    signals = [s["signal"] for s in r["signals"]]
    assert "量缩价涨-资金枯竭" in signals, f"signals={signals}, ma5_slope={r['ma5_slope']}"
    print(f"✅ test_shrink_up_capital_exhausted passed (signals={signals}, slope={r['ma5_slope']})")


def test_extreme_ground_vol():
    """换手<1% + 量缩>30% -> 极致地量"""
    # 换手 <1%:vol < 85000 手
    # 量缩 >30%:今日 vol < 昨日 × 0.7
    closes = [10.0] * 30
    vols = [80000] * 29 + [50000]  # 最后一天量缩 37.5%
    daily = make_daily(closes, vols)
    turnover = compute_turnover(daily, 8.5e8)
    r = compute_volume_price_detail(daily, turnover, "低位")
    signals = [s["signal"] for s in r["signals"]]
    assert "极致地量" in signals, f"signals={signals}, today_turnover={turnover['today']}"
    print(f"✅ test_extreme_ground_vol passed (signals={signals}, turnover={turnover['today']}%)")


# ========== 新闻舆情测试 ==========

def test_news_classify_sentiment_positive():
    """标题命中"净利预增" -> 利好"""
    s, kw = _classify_sentiment("公司净利预增1099%", "业绩内容")
    assert s == "利好", f"sentiment={s}"
    assert "净利预增" in kw, f"keywords={kw}"
    print(f"✅ test_news_classify_sentiment_positive passed (sentiment={s}, kw={kw})")


def test_news_classify_sentiment_negative():
    """标题命中"减持" -> 利空"""
    s, kw = _classify_sentiment("股东减持计划", "内容")
    assert s == "利空", f"sentiment={s}"
    assert "减持" in kw, f"keywords={kw}"
    print(f"✅ test_news_classify_sentiment_negative passed (sentiment={s}, kw={kw})")


def test_news_classify_sentiment_neutral():
    """无关键词命中 -> 中性"""
    s, kw = _classify_sentiment("今日天气不错", "正常内容")
    assert s == "中性", f"sentiment={s}"
    assert kw == [], f"keywords={kw}"
    print(f"✅ test_news_classify_sentiment_neutral passed (sentiment={s})")


def test_news_classify_sentiment_title_weight():
    """标题权重 ×2:标题命中"涨停"(2分) vs 内容命中"减持"(1分) -> 利好(标题方胜)"""
    s, kw = _classify_sentiment("涨停", "减持")
    assert s == "利好", f"sentiment={s} (标题2分应胜内容1分)"
    assert "涨停" in kw and "减持" in kw, f"keywords={kw}"
    print(f"✅ test_news_classify_sentiment_title_weight passed (sentiment={s}, 标题权重×2生效)")


def test_news_sentiment_summary_dominant():
    """6 利好 + 2 利空 + 2 中性 -> 偏正面(60%)"""
    sentiments = ["利好"] * 6 + ["利空"] * 2 + ["中性"] * 2
    r = _compute_sentiment_summary(sentiments)
    assert r["dominant"] == "利好", f"dominant={r['dominant']}"
    assert r["dominant_pct"] == 60.0, f"pct={r['dominant_pct']}"
    assert r["label"] == "偏正面", f"label={r['label']}"
    print(f"✅ test_news_sentiment_summary_dominant passed (label={r['label']}, pct={r['dominant_pct']}%)")


def test_news_sentiment_summary_divergence():
    """3 利好 + 3 利空 + 4 中性 -> 分歧(无一方占 60%+)"""
    sentiments = ["利好"] * 3 + ["利空"] * 3 + ["中性"] * 4
    r = _compute_sentiment_summary(sentiments)
    # dominant 是 max,neutral=4 是最多,但 pct=40% < 60% -> 分歧
    assert r["label"] == "分歧", f"label={r['label']}, dominant={r['dominant']}, pct={r['dominant_pct']}"
    print(f"✅ test_news_sentiment_summary_divergence passed (label={r['label']})")


def test_news_key_events_extraction():
    """从标题提取关键事件 + 去重"""
    news_list = [
        {"title": "公司净利预增1099%,业绩大爆发"},
        {"title": "股东减持计划公布"},
        {"title": "公司净利预增1099%", "content": ""},  # 重复,应被去重
        {"title": "今日天气"},
    ]
    events = _extract_key_events(news_list, max_events=5)
    assert len(events) >= 2, f"events={events}"
    assert len(events) <= 3, f"events={events} (应去重)"
    print(f"✅ test_news_key_events_extraction passed (events={events})")


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
    test_roe_stability_buffett_pass()
    test_roe_stability_buffett_fail_low_mean()
    test_roe_stability_buffett_fail_low_min()
    test_dupont_high_margin_mode()
    test_dupont_high_turnover_mode()
    test_dupont_high_leverage_mode()
    test_buffett_filter_all_pass()
    test_buffett_filter_fail_high_debt()
    test_buffett_filter_fail_cashflow()
    test_fake_roe_high_leverage_warning()
    test_fake_roe_one_shot_gain_warning()
    test_fake_roe_buyback_shrink_warning()
    test_fake_roe_clean()
    test_analyze_fundamental_includes_roe_quality()
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
    # 机构研报评估
    test_is_foreign_broker_keywords()
    test_normalize_rating()
    test_rating_consensus_strong()
    test_rating_consensus_divergent()
    test_target_price_with_upside()
    test_target_price_no_current()
    test_eps_forecast_latest()
    test_foreign_summary()
    test_divergence_foreign_pessimistic()
    test_divergence_consistent()
    test_summarize_research_report_empty()
    test_summarize_research_report_strong()
    test_summarize_research_report_weak()
    test_analyze_fundamental_with_research_reports()
    test_analyze_fundamental_no_research_reports()
    # 半导体行业特殊处理 + 短期筹码量价趋势
    test_is_semiconductor_by_industry()
    test_is_semiconductor_by_name()
    test_semiconductor_special_handling()
    test_non_semiconductor_no_special_handling()
    test_short_term_chip_trend_basic()
    test_short_term_chip_trend_concentration_rising()
    test_short_term_trend_acceleration()
    test_short_term_trend_in_compute()
    test_short_term_chip_in_recompute()
    # 主力资金流 + 筹码交叉验证
    test_capital_flow_detect_secid()
    test_capital_flow_parse_klines()
    test_capital_flow_cumulative()
    test_capital_flow_consecutive_inflow()
    test_capital_flow_consecutive_outflow()
    test_capital_flow_ma_cross_golden()
    test_capital_flow_ma_cross_death()
    test_capital_flow_signal_accumulation()
    test_capital_flow_signal_distribution()
    test_capital_flow_signal_strong_accumulation()
    test_capital_flow_signal_neutral()
    test_cross_validate_strong_accumulation()
    test_cross_validate_distribution()
    test_cross_validate_contradiction()
    test_cross_validate_unavailable()
    test_ths_realtime_parse()
    test_ths_realtime_signal_thresholds()
    test_cross_validate_ths_source_confidence_cap()
    test_cross_validate_ths_outflow_distribution()
    # 换手率 + 量价细化
    test_turnover_calculation()
    test_turnover_label_cold()
    test_turnover_label_normal()
    test_turnover_label_active()
    test_turnover_label_warning()
    test_history_top_detection()
    test_volume_price_flat_low_position()
    test_volume_price_flat_high_position()
    test_shrink_up_high_control()
    test_shrink_up_capital_exhausted()
    test_extreme_ground_vol()
    # 新闻舆情
    test_news_classify_sentiment_positive()
    test_news_classify_sentiment_negative()
    test_news_classify_sentiment_neutral()
    test_news_classify_sentiment_title_weight()
    test_news_sentiment_summary_dominant()
    test_news_sentiment_summary_divergence()
    test_news_key_events_extraction()
    print("\n🎉 All tests passed!")
