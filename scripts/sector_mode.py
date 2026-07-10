"""行业分层模式:成长/周期/价值三模式动态判断。

不同行业用不同的位置阈值和核心信号:
- growth(成长):AI/半导体/创新药/新能源车等高景气赛道,位置<60%,量比+金叉(主升浪启动)
- cyclical(周期):煤炭/有色/化工等,位置<30%,量比+金叉+突破(低谷反转)
- value(价值):银行/白酒/红利等,位置<40%,量比+金叉(底部启动)

成长模式额外信号:
- 🔥 主升浪信号:高位(>50%) + 量比 + 金叉 + 吸筹 = 强主升浪标的
- 📍 底部启动信号:低位(<30%) + 量比 + 金叉 + 吸筹 = 成长股黄金坑
- ⚠️ 高位减仓信号:高位(>70%) + 派发 = 警惕
"""
from typing import Dict, Any, Optional


# 行业 -> 模式映射
SECTOR_MODE: Dict[str, str] = {
    # 成长模式(高景气赛道)
    'AI服务器': 'growth', 'AI芯片': 'growth', 'EDA-FPGA': 'growth',
    '存储': 'growth', '模拟芯片': 'growth', '半导体': 'growth',
    '光铜缆': 'growth', 'PCB': 'growth', '散热电源': 'growth',
    '通信': 'growth', '新能源车': 'growth', '光伏': 'growth',
    '军工': 'growth', '创新药': 'growth',
    # 周期模式
    '煤炭': 'cyclical', '钢铁': 'cyclical', '化工': 'cyclical',
    '有色': 'cyclical', '黄金': 'cyclical', '猪肉': 'cyclical',
    '养殖': 'cyclical', '航运': 'cyclical', '证券': 'cyclical',
    # 价值模式
    '银行': 'value', '白酒': 'value', '家电': 'value',
    '食品饮料': 'value', '红利': 'value', '保险': 'value',
}

# 位置阈值(按模式)
POSITION_THRESHOLD: Dict[str, int] = {
    'growth': 60,
    'cyclical': 30,
    'value': 40,
}

# industry字段 -> sector 映射(analyze.py单股票场景用,只有industry没有sector)
INDUSTRY_TO_SECTOR: Dict[str, str] = {
    '半导体': '半导体', '芯片': '半导体', '集成电路': '半导体',
    'AI': 'AI服务器', '服务器': 'AI服务器', '算力': 'AI服务器',
    '通信': '通信', '通信设备': '通信', '光模块': '光铜缆',
    '银行': '银行', '股份制银行': '银行', '商业银行': '银行',
    '食品饮料': '食品饮料', '白酒': '白酒', '啤酒': '白酒',
    '煤炭': '煤炭', '煤炭开采': '煤炭',
    '有色金属': '有色', '黄金': '黄金', '贵金属': '黄金',
    '化工': '化工', '化学制品': '化工',
    '医药生物': '创新药', '医疗器械': '创新药', '化学制药': '创新药',
    '生物制品': '创新药', '中药': '创新药',
    '新能源': '新能源车', '汽车': '新能源车', '汽车零部件': '新能源车',
    '电力设备': '光伏', '光伏': '光伏',
    '国防军工': '军工', '航空航天': '军工',
    '家用电器': '家电', '白色家电': '家电',
    '证券': '证券', '保险': '保险',
}


def classify_mode_by_sector(sector: str) -> str:
    """根据sector返回模式(growth/cyclical/value/unknown)。"""
    return SECTOR_MODE.get(sector, 'unknown')


def classify_sector_by_industry(industry: str, name: str = '') -> str:
    """根据industry字段推断sector。

    industry可能是"食品饮料(白酒)""半导体""煤炭开采"等格式。
    对于"大类(子类)"格式,优先匹配括号内子类(更具体)。
    name是股票名称,用于兜底(如"五粮液"->白酒)。
    """
    if not industry:
        # 名称兜底(白酒品牌+通用词)
        name_hints = {
            '酒': '白酒', '茅台': '白酒', '五粮液': '白酒', '老窖': '白酒',
            '银行': '银行', '煤': '煤炭',
            '半导体': '半导体', '芯片': '半导体',
        }
        for hint, sec in name_hints.items():
            if hint in name:
                return sec
        return ''

    # 按 key 长度降序匹配(更具体的优先)
    sorted_keys = sorted(INDUSTRY_TO_SECTOR.keys(), key=len, reverse=True)

    # 优先匹配括号内子类(如"食品饮料(白酒)" -> 先匹配"白酒")
    paren_content = ''
    if '(' in industry and ')' in industry:
        paren_content = industry[industry.index('(') + 1: industry.rindex(')')]
    if paren_content:
        for key in sorted_keys:
            if key in paren_content:
                return INDUSTRY_TO_SECTOR[key]

    # 整体匹配
    for key in sorted_keys:
        if key in industry:
            return INDUSTRY_TO_SECTOR[key]
    return ''


def get_position_threshold(mode: str) -> int:
    """返回模式对应的位置阈值,unknown默认30%。"""
    return POSITION_THRESHOLD.get(mode, 30)


def analyze_sector_mode(
    indicators: Dict[str, Any],
    fundamental: Optional[Dict[str, Any]] = None,
    sector: str = '',
) -> Dict[str, Any]:
    """行业分层模式分析。

    输入:
        indicators: 含 position/five_step/capital_flow/chip_cross_validation
        fundamental: 含 industry(单股票场景从industry推断sector)
        sector: 筛选场景直接传(优先于industry)

    输出:
        {
            sector, mode, position_threshold,
            pct_120d, position_pass,
            vr_pass, ma_pass, brk_pass,
            capital_flow_pass, intent,
            signal_pass,           # 核心信号(量比+金叉)
            funnel_count,          # 4步漏斗数
            is_buy_point,
            breakout_signal,       # 主升浪信号(成长模式)
            bottom_signal,         # 底部启动信号(成长模式)
            warn_signal,           # 高位减仓信号
            interpretation,        # 人话解读
        }
    """
    # 1. 判定sector和mode
    if not sector and fundamental:
        industry = (fundamental or {}).get('industry') or ''
        name = (fundamental or {}).get('name') or ''
        sector = classify_sector_by_industry(industry, name)
    mode = classify_mode_by_sector(sector)

    # 2. 位置
    pos = indicators.get('position', {})
    pct_120 = pos.get('pct_120d', 100)
    threshold = get_position_threshold(mode)
    position_pass = bool(pct_120 < threshold)

    # 3. 五步信号
    fs = indicators.get('five_step', {})
    vr_pass = bool(fs.get('3_vol_ratio', {}).get('pass', False))
    ma_pass = bool(fs.get('4_vol_ma_cross', {}).get('pass', False))
    brk_pass = bool(fs.get('5_breakout_3day', {}).get('pass', False))

    # 4. 资金流/主力意图
    cf = indicators.get('capital_flow', {})
    cf_signals = cf.get('signals', {}) if cf.get('available') else {}
    cf_action = cf_signals.get('main_force_action', '')
    cf_pass = cf_action in ('吸筹', '强吸筹')
    ccv = indicators.get('chip_cross_validation', {})
    intent = ccv.get('main_force_intent', '') if ccv else ''
    intent_absorb = '吸筹' in intent

    # 资金流确认:资金流pass OR 主力意图含吸筹(筹码×资金流交叉验证)
    capital_pass = cf_pass or intent_absorb

    # 5. 核心信号(step4,按模式调整)
    if mode == 'cyclical':
        signal_pass = vr_pass and ma_pass and brk_pass  # 量比+金叉+突破
    else:
        signal_pass = vr_pass and ma_pass  # growth/value:量比+金叉

    # 6. 漏斗4步
    step1 = mode != 'unknown'  # 质地(已分类行业)
    step2 = position_pass
    step3 = capital_pass
    step4 = signal_pass
    funnel_count = sum([step1, step2, step3, step4])
    is_buy_point = step1 and step2 and step3 and step4

    # 7. 特殊信号(成长模式特有)
    breakout_signal = False
    bottom_signal = False
    warn_signal = False
    if mode == 'growth':
        breakout_signal = (pct_120 > 50) and vr_pass and ma_pass and intent_absorb
        bottom_signal = (pct_120 < 30) and vr_pass and ma_pass and intent_absorb
        warn_signal = (pct_120 > 70) and ('派发' in intent)

    # 8. 解读
    signals_list = []
    if breakout_signal:
        signals_list.append('🔥 主升浪信号(高位+量比+金叉+吸筹)')
    if bottom_signal:
        signals_list.append('📍 底部启动信号(低位+量比+金叉+吸筹)')
    if warn_signal:
        signals_list.append('⚠️ 高位减仓信号(高位+派发)')

    interp_parts = [
        f"行业:{sector}({mode}模式)" if sector else f"模式:{mode}",
        f"位置阈值<{threshold}%(当前{pct_120}%)",
        f"漏斗{funnel_count}/4",
    ]
    if is_buy_point:
        interp_parts.append('✅ 达到买点')
    elif funnel_count >= 3:
        interp_parts.append('⚠️ 接近买点')
    else:
        interp_parts.append(f'差{4 - funnel_count}步')
    if signals_list:
        interp_parts.extend(signals_list)

    return {
        'sector': sector,
        'mode': mode,
        'position_threshold': threshold,
        'pct_120d': pct_120,
        'position_pass': position_pass,
        'vr_pass': vr_pass,
        'ma_pass': ma_pass,
        'brk_pass': brk_pass,
        'capital_flow_pass': capital_pass,
        'intent': intent,
        'signal_pass': signal_pass,
        'funnel_count': funnel_count,
        'is_buy_point': is_buy_point,
        'breakout_signal': breakout_signal,
        'bottom_signal': bottom_signal,
        'warn_signal': warn_signal,
        'interpretation': ' / '.join(interp_parts),
    }
