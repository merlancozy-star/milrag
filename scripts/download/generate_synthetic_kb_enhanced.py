#!/usr/bin/env python3
"""增强版模板合成知识库生成器（不需要 GPU）。

当政府/军方网站无法访问时，用模板填充法生成大量多样化的中文军事知识段落。
使用参数化模板 + 随机填充值组合，可生成数千至数万段不重复的合成文本。

输出: data/raw/synthetic_kb/
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_DIR = Path("data/raw/synthetic_kb")

# ═══════════════════════════════════════════════════════════════
# 参数填充值库
# ═══════════════════════════════════════════════════════════════

EQUIP_NAMES = [
    "XX-10A", "BK-200X", "YT-7M", "ZR-15", "DH-33", "FN-8B", "GL-21C",
    "KM-400", "PR-29", "WS-5T", "JQ-50", "LX-12", "CY-88", "VM-3N",
    "RH-7", "TD-19K", "NP-44B", "SW-27", "FG-6M", "HU-31",
]

WEIGHT_VALUES = [25, 28, 30, 32, 35, 37, 40, 42, 45, 48, 50, 55, 60, 65]
RANGE_VALUES = [200, 500, 800, 1000, 1200, 1500, 2000, 2500, 3000, 5000, 8000, 10000]
SPEED_VALUES = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0]
PERSONNEL_VALUES = [500, 1000, 1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000]
YEAR_VALUES = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
PCT_VALUES = [5, 8, 12, 15, 20, 25, 30, 35, 40, 50]

REGIONS = [
    "西太平洋某区域", "东海某方向", "南海某海域", "印度洋某区域",
    "北极圈附近某区域", "中亚某地区", "中东某区域", "非洲之角某区域",
    "波罗的海某区域", "黑海某海域", "东南亚某区域", "东北亚某方向",
]

MILITARY_UNITS = [
    "第X集团军", "某战区", "某舰队", "某航空兵师", "某导弹旅",
    "某装甲旅", "某机械化步兵师", "某特种作战大队", "某陆航旅",
    "某防空旅", "某电子对抗团", "某侦察营",
]

WEAPON_TYPES = [
    "战斗机", "驱逐舰", "潜艇", "导弹", "雷达系统", "无人机",
    "装甲车辆", "火炮系统", "直升机", "预警机", "电子战系统",
    "卫星系统", "通信系统", "指挥控制系统",
]

DOCTRINE_CONCEPTS = [
    "纵深突击", "快速机动", "信息优势", "精确打击", "全域作战",
    "网络中心战", "多维一体化", "非对称对抗", "体系破击", "联合火力",
    "空天一体", "网电一体", "分布式作战", "智能化指挥", "多域协同",
]

DOCTRINE_PRINCIPLES = [
    "集中优势兵力于主要方向", "统一指挥与分散执行相结合",
    "灵活机动与周密计划相统一", "先机制敌与后发制人相结合",
    "火力优先与信息主导相协调", "正面牵制与翼侧突击相配合",
    "纵深打击与前沿突破相衔接", "主动防御与积极进攻相转换",
    "精确保障与快速反应相配套", "心理优势与物质优势相支撑",
]

TACTICAL_SCENARIOS = [
    "山地进攻作战", "城市防御作战", "两栖登陆作战", "空降突击作战",
    "反恐维稳行动", "抢险救灾行动", "边境封控行动", "海上护航行动",
    "空中拦截作战", "反潜搜索作战", "电子对抗作战", "网络空间防御",
    "特种侦察行动", "心理攻防作战", "战场搜救行动",
]

CASE_LESSONS = [
    "准确判断主要威胁方向是兵力部署的前提",
    "佯动需要足够的真实性和规模才能有效欺骗对手",
    "预备队的灵活使用是应对不确定性的关键",
    "信息优势必须转化为决策优势才能发挥作用",
    "后勤保障的前置预置是持续作战的基础",
    "复杂电磁环境下的通信保障需要多级响应机制",
    "指挥员的临机决断能力往往决定战斗胜负",
    "联合作战中各军种间的协同是最大挑战",
    "战场态势感知的实时性和准确性同等重要",
    "训练水平的高低直接决定战斗力的生成速度",
]

PLATFORM_NAMES = [
    "某型战斗机平台", "某型驱逐舰平台", "某型潜艇平台",
    "某型预警机", "某型无人机系统", "某型卫星星座",
]

CAPABILITY_NAMES = [
    "超视距空战能力", "对地精确打击能力", "反舰作战能力",
    "区域防空能力", "反导拦截能力", "远程预警能力",
    "电子压制能力", "战场侦察能力", "火力支援能力",
    "空中加油能力", "战略投送能力", "医疗后送能力",
]

DETECTION_RANGES = [100, 200, 300, 400, 500, 800, 1000, 1500, 2000, 3000]
MISSILE_COUNTS = [8, 12, 16, 24, 32, 48, 64, 96, 128]
ALTITUDE_VALUES = [5000, 10000, 15000, 18000, 20000, 25000, 30000]
TONNAGE_VALUES = [4000, 6000, 7000, 10000, 12000, 13000, 25000, 45000, 65000]


# ═══════════════════════════════════════════════════════════════
# 模板库（装备 / 条令 / 态势 / 案例）
# ═══════════════════════════════════════════════════════════════

EQUIPMENT_TEMPLATES = [
    # 战斗机
    (
        "{name}型多用途战斗机采用双发涡扇发动机，最大起飞重量约{wt}吨，"
        "最大飞行速度马赫{speed}，作战半径约{rng}公里。该机配备有源相控阵雷达和综合航电系统，"
        "可携带{msl}枚各型空空和空地武器，具备{cap1}和{cap2}。"
        "机载电子战系统可覆盖{cov}公里范围内的电磁频谱。"
    ),
    # 驱逐舰
    (
        "{name}型驱逐舰满载排水量约{ton}吨，采用全燃联合动力装置，"
        "最高航速{speed}节，续航力约{rng}海里。舰上配备相控阵雷达系统和垂直发射装置，"
        "可搭载{msl}枚各型导弹，具备{cap1}和{cap2}。"
        "综合作战系统可同时跟踪{cov}个空中和水面目标。"
    ),
    # 导弹系统
    (
        "{name}型导弹系统采用{guide}制导方式，最大射程约{rng}公里，"
        "命中精度（CEP）约{acc}米。该系统具备{cap1}，"
        "可在复杂电磁环境下遂行{scenario}等任务。"
        "发射准备时间约{prep}分钟，单辆发射车可携带{msl}枚导弹。"
    ),
    # 雷达系统
    (
        "{name}型雷达系统采用有源相控阵体制，最大探测距离约{rng}公里，"
        "可同时跟踪{cov}个目标，对典型空中目标发现概率高于{pct}%。"
        "该系统具备{cap1}和{cap2}，抗干扰能力强，"
        "可部署于多种地形环境。平均无故障工作时间超过{prep}小时。"
    ),
    # 无人机
    (
        "{name}型无人机系统最大起飞重量{wt}吨，巡航速度{speed}公里/小时，"
        "续航时间{prep}小时，最大飞行高度{h}米。"
        "搭载光电/红外/合成孔径雷达等多型传感器，具备{cap1}和{cap2}，"
        "可执行{scenario}等多种任务。数据链传输距离{rng}公里。"
    ),
    # 装甲车辆
    (
        "{name}型主战坦克战斗全重约{wt}吨，乘员{crew}人，"
        "最大公路速度{speed}公里/小时。装备{mm}毫米滑膛炮，"
        "配备先进的火控系统和复合装甲，具备{cap1}。"
        "发动机功率约{hp}马力，最大行程{rng}公里。"
    ),
    # 潜艇
    (
        "{name}型潜艇水下排水量约{ton}吨，最大潜深{alt}m，"
        "水下最高航速{speed}节。装备{mm}毫米鱼雷发射管，"
        "可携带{msl}枚鱼雷和导弹，具备{cap1}和{cap2}。"
        "噪声水平显著低于同类型潜艇，自持力约{prep}天。"
    ),
    # 卫星系统
    (
        "{name}卫星星座由{msl}颗卫星组成，轨道高度约{alt}公里，"
        "重访周期{prep}小时。该系统具备{cap1}和{cap2}，"
        "空间分辨率达{rng}厘米级。"
        "设计寿命{crew}年，具备在轨重构和抗干扰能力。"
    ),
]

DOCTRINE_TEMPLATES = [
    # 作战原则
    (
        "在军事行动的筹划与实施中，{doctrine}原则具有重要指导意义。"
        "该原则的核心要义是{principle}，要求指挥员在决策过程中"
        "充分考虑战场环境的复杂性和对抗性，做到因敌而变、因势而变。"
        "实施要点包括：准确掌握敌情我情、科学制定作战方案、合理分配兵力火力、"
        "周密组织协同保障。该原则适用于{scenario}等典型场景。"
    ),
    # 指挥控制
    (
        "信息化条件下的指挥控制强调{doctrine}。{principle}。"
        "指挥体系的构建应遵循扁平化、网络化、智能化的发展方向，"
        "实现态势感知、指挥决策、行动控制和效果评估的闭环运行。"
        "在{scenario}中，指挥官应充分利用信息优势，"
        "缩短决策周期，提高指挥效率。"
    ),
    # 兵力部署
    (
        "兵力部署应遵循{principle}。在确定主要方向和次要方向时，"
        "应综合考虑敌情威胁程度、地形条件、己方兵力规模和任务要求等因素。"
        "通常情况下，主要方向集中总兵力的约{pct}%至{pct2}%，"
        "次要方向部署{pct3}%至{pct4}%。{doctrine}是判断主要方向的重要依据。"
    ),
    # 后勤保障
    (
        "现代战争中的后勤保障面临{doctrine}的新要求。"
        "保障体系需要从传统的线性保障模式向网络化、模块化保障模式转变。"
        "{principle}。具体措施包括：建立预置储备体系、"
        "优化保障力量编组、强化信息化保障手段、"
        "提高保障力量的快速反应和精确投送能力。在{scenario}中，"
        "保障力量通常需要提前{pct}至{pct2}天完成部署。"
    ),
    # 军事训练
    (
        "军事训练应坚持{doctrine}的基本方针。{principle}。"
        "训练内容应涵盖基础课目、专业技能和综合演练三个层次，"
        "训练方式应融合实装训练、模拟训练和对抗训练等多种手段。"
        "年度训练时间不少于{prep}天，其中{pct}%用于实战化训练，"
        "夜间训练和复杂气象条件下训练各占比不低于{pct2}%。"
    ),
    # 情报侦察
    (
        "情报侦察是{doctrine}的基础支撑。{principle}。"
        "现代侦察体系包括天基、空基、陆基和海基多个层次，"
        "综合运用信号情报、图像情报、人力情报和开源情报等多种手段。"
        "在{scenario}场景下，侦察力量应提前{prep}小时展开，"
        "重点监视{rng}公里纵深内的敌情动态，每{cov}小时更新一次态势评估。"
    ),
    # 联合作战
    (
        "联合作战的本质是实现{doctrine}。{principle}。"
        "各军种力量应在统一意图下协调行动，发挥各自优势形成整体合力。"
        "联合指挥机构应设在便于指挥的位置，通常距前沿{rng}至{rng2}公里。"
        "信息共享是实现联合作战的基础，各参战力量之间的信息传输时延应控制在"
        "{speed}秒以内，信息准确率不低于{pct}%。"
    ),
]

SITUATION_TEMPLATES = [
    # 区域态势
    (
        "在{region}，近年来各主要力量持续加强军事存在。"
        "A方向部署了约{msl}架先进战机和{ton}个水面战斗群，"
        "B方向通过陆基反舰导弹和潜艇力量形成反介入屏障。"
        "分析认为，该区域的战略平衡正在从传统的海空优势"
        "向陆海空天网多维制衡转变，未来{crew}年内可能出现新的力量格局。"
        "影响该区域稳定的关键因素包括：{doctrine}的发展态势，"
        "{scenario}的演变方向，以及区域外力量介入的程度。"
    ),
    # 军力对比
    (
        "{year}年度，某区域军事力量对比呈现以下特点："
        "兵力规模方面，主要国家常备兵力维持在{personnel}至{personnel2}万人，"
        "预备役力量约为现役的{ratio}倍。装备质量方面，"
        "新型{weapon}的列装速度加快，技术代差逐步缩小。"
        "军费开支方面，该区域军费总额较上年增长约{pct}%，"
        "其中{doctrine}相关领域的投入增幅最大。"
    ),
    # 技术趋势
    (
        "当前军事技术发展呈现{doctrine}的明显趋势。"
        "人工智能、量子技术、高超音速等新兴领域正在改变传统战争形态。"
        "高超音速武器飞行速度超过马赫{speed}，现有防空系统拦截难度极大。"
        "人工智能辅助决策系统可在{cov}毫秒级完成目标识别和威胁评估。"
        "量子传感器有望将探测灵敏度提升{ratio}个数量级。"
        "这些技术的军事应用将在未来{crew}至{crew2}年内逐步成熟。"
    ),
    # 冲突评估
    (
        "针对{region}方向可能发生的{scenario}，综合评估认为："
        "冲突爆发的概率约为{pct}%至{pct2}%，影响因素包括领土争端、"
        "资源竞争、民族宗教矛盾和域外干预等。一旦爆发，"
        "冲突强度可能达到{doctrine}级别，持续时间约{prep}至{prep2}天。"
        "关键时间窗口位于冲突爆发后{crew}至{crew2}小时内。"
        "外交调解的成功概率评估为{ratio}，取决于主要大国的政治意愿。"
    ),
    # 战略评估
    (
        "从{doctrine}角度审视当前战略态势："
        "全球军事力量格局正在经历冷战结束以来最深刻的变化。"
        "在{region}方向，传统军事优势正在受到{scenario}方式的挑战。"
        "网络空间和太空正成为新的战略竞争领域，相关投入年均增长{pct}%以上。"
        "未来{crew}年内，{doctrine}将成为大国军事竞争的核心议题。"
        "建议持续关注{region}的动态，重点评估{scenario}对力量平衡的影响。"
    ),
    # 演习分析
    (
        "{year}年度，在{region}举行了代号为{name}的联合军事演习。"
        "参演兵力约{personnel}人，出动各型{weapon}{msl}余架/艘。"
        "演习课题包括{scenario}、{scenario2}和{scenario3}。"
        "主要特点：一是突出{doctrine}的实战化要求；"
        "二是强化{doctrine2}的协同训练；三是检验新型{weapon}的作战效能。"
        "此次演习的政治信号和军事意图值得关注。"
    ),
]

CASE_TEMPLATES = [
    # 战例分析
    (
        "在某次{scenario}中，指挥员面临的主要挑战是{doctrine}。"
        "敌情方面，对手在{rng}公里正面部署了约{msl}个营的兵力，"
        "并获得{weapon}等先进装备的支援。"
        "指挥员采取了以下措施：首先，{principle}；"
        "其次，集中{personnel}兵力于主要突击方向；"
        "最后，利用{scenario2}实施翼侧迂回。"
        "战斗持续约{prep}小时，最终达成了预定目标。"
        "主要经验教训：{lesson}；同时暴露出{doctrine2}方面的不足。"
    ),
    # 历史案例分析
    (
        "回顾{year}年的{scenario}，可以发现{doctrine}在军事决策中的关键作用。"
        "当时，{region}地区局势持续紧张。决策者需要在有限情报条件下，"
        "在{cov}小时内做出危机响应决策。"
        "决策的核心依据包括：{doctrine}的能力评估、"
        "{scenario}的历史经验、以及{region}的地缘政治环境。"
        "最终采取的{doctrine2}策略成功避免了局势进一步升级。"
        "这一案例对理解当前的{scenario2}具有重要参考意义：{lesson}。"
    ),
    # 演习案例分析
    (
        "在某次大规模联合演习中，参演部队尝试了{doctrine}的新战法。"
        "演习想定：{region}方向发生{scenario}，"
        "红方需要在{prep}小时内完成兵力投送和展开。"
        "演习过程中，指挥官创造性地运用了{doctrine2}原则，"
        "通过{principle}实现了战局逆转。"
        "演习评估显示，新战法使作战效能提升了约{pct}%，"
        "但也在{scenario2}环节暴露出协调不足的问题。"
        "核心经验：{lesson}。"
    ),
    # 经验总结
    (
        "从近期{scenario}的实践中可以提炼以下经验："
        "第一，{principle}。在复杂多变的战场环境中，"
        "指挥官必须保持清醒的态势认知，避免被局部信息误导。"
        "第二，{principle}。无论技术如何发展，"
        "{doctrine}始终是取得胜利的根本保证。"
        "第三，{principle}。"
        "这些经验的共性指向一个核心问题：{lesson}。"
        "对未来{scenario2}的准备具有重要的启示意义。"
    ),
    # 教训反思
    (
        "{year}年的{scenario}提供了一个值得深思的教训。"
        "事件中，由于未能充分贯彻{doctrine}原则，"
        "导致在{scenario2}环节出现了严重失误。"
        "具体表现为：低估了对手的{weapon}威胁程度；"
        "高估了己方{doctrine2}能力的实际效果；"
        "忽视了{region}的特殊地理和气候条件。"
        "上述教训提醒我们：{lesson}。"
        "这些认识已被纳入后续的训练和条令修订中。"
    ),
]


def _pick(*values):
    """从参数池中随机挑选。"""
    return random.choice(values)


def _range_or(a, b, default=300):
    """生成 a 或 b 附近的变化值。"""
    return a * random.uniform(0.8, 1.2) if random.random() > 0.5 else b


def generate_equipment() -> str:
    template = random.choice(EQUIPMENT_TEMPLATES)
    return template.format(
        name=_pick(EQUIP_NAMES),
        wt=_pick(WEIGHT_VALUES),
        rng=_pick(RANGE_VALUES),
        speed=random.choice(SPEED_VALUES),
        ton=_pick(TONNAGE_VALUES),
        msl=_pick(MISSILE_COUNTS),
        cov=_pick(DETECTION_RANGES),
        alt=_pick(ALTITUDE_VALUES),
        mm=random.choice([76, 100, 105, 120, 125, 130, 155, 203]),
        h=_pick(ALTITUDE_VALUES),
        hp=random.choice([500, 800, 1000, 1200, 1500, 2000, 2500]),
        crew=random.choice([2, 3, 4, 5, 6, 8, 10]),
        prep=random.choice([15, 30, 45, 60, 90, 120, 180, 240]),
        acc=random.choice([5, 10, 15, 20, 30, 50, 100]),
        guide=random.choice(["惯性+GPS", "雷达主动", "红外成像", "激光半主动", "复合"]),
        cap1=_pick(CAPABILITY_NAMES),
        cap2=_pick(CAPABILITY_NAMES),
        scenario=_pick(TACTICAL_SCENARIOS),
        pct=random.randint(70, 99),
    )


def generate_doctrine() -> str:
    template = random.choice(DOCTRINE_TEMPLATES)
    return template.format(
        doctrine=_pick(DOCTRINE_CONCEPTS),
        doctrine2=_pick(DOCTRINE_CONCEPTS),
        principle=_pick(DOCTRINE_PRINCIPLES),
        scenario=_pick(TACTICAL_SCENARIOS),
        scenario2=_pick(TACTICAL_SCENARIOS),
        rng=_pick(RANGE_VALUES),
        rng2=_pick(RANGE_VALUES),
        pct=random.randint(25, 40),
        pct2=random.randint(30, 50),
        pct3=random.randint(15, 25),
        pct4=random.randint(20, 35),
        prep=random.choice([3, 5, 7, 10, 14, 21, 30]),
        cov=_pick(DETECTION_RANGES),
        speed=random.uniform(0.1, 2.0),
        personnel=_pick(PERSONNEL_VALUES),
        weapon=_pick(WEAPON_TYPES),
    )


def generate_situation() -> str:
    template = random.choice(SITUATION_TEMPLATES)
    return template.format(
        region=_pick(REGIONS),
        name=_pick(EQUIP_NAMES),
        doctrine=_pick(DOCTRINE_CONCEPTS),
        doctrine2=_pick(DOCTRINE_CONCEPTS),
        scenario=_pick(TACTICAL_SCENARIOS),
        scenario2=_pick(TACTICAL_SCENARIOS),
        scenario3=_pick(TACTICAL_SCENARIOS),
        weapon=_pick(WEAPON_TYPES),
        msl=_pick(MISSILE_COUNTS),
        ton=_pick(TONNAGE_VALUES),
        rng=_pick(RANGE_VALUES),
        speed=_pick(SPEED_VALUES),
        cov=_pick(DETECTION_RANGES),
        pct=random.randint(10, 40),
        pct2=random.randint(50, 80),
        ratio=round(random.uniform(0.3, 3.5), 1),
        crew=random.choice([3, 5, 10, 15, 20]),
        crew2=random.choice([5, 10, 15, 20, 25]),
        prep=random.choice([3, 7, 10, 14, 21, 30]),
        prep2=random.choice([7, 14, 21, 30, 45, 60]),
        year=_pick(YEAR_VALUES),
        personnel=_pick(PERSONNEL_VALUES),
        personnel2=_pick(PERSONNEL_VALUES),
    )


def generate_case() -> str:
    template = random.choice(CASE_TEMPLATES)
    return template.format(
        region=_pick(REGIONS),
        doctrine=_pick(DOCTRINE_CONCEPTS),
        doctrine2=_pick(DOCTRINE_CONCEPTS),
        principle=_pick(DOCTRINE_PRINCIPLES),
        lesson=_pick(CASE_LESSONS),
        scenario=_pick(TACTICAL_SCENARIOS),
        scenario2=_pick(TACTICAL_SCENARIOS),
        weapon=_pick(WEAPON_TYPES),
        rng=_pick(RANGE_VALUES),
        msl=_pick(MISSILE_COUNTS),
        personnel=_pick(PERSONNEL_VALUES),
        year=_pick(YEAR_VALUES),
        pct=random.randint(15, 40),
        prep=random.choice([4, 6, 8, 12, 16, 24, 36, 48, 72]),
        cov=_pick(DETECTION_RANGES),
        speed=_pick(SPEED_VALUES),
    )


GENERATORS = {
    "equipment": generate_equipment,
    "doctrine": generate_doctrine,
    "situation": generate_situation,
    "case": generate_case,
}


def main():
    parser = argparse.ArgumentParser(description="增强版模板合成KB生成器")
    parser.add_argument("--equipment", type=int, default=500,
                        help="装备类生成数量")
    parser.add_argument("--doctrine", type=int, default=500,
                        help="条令类生成数量")
    parser.add_argument("--situation", type=int, default=500,
                        help="态势类生成数量")
    parser.add_argument("--case", type=int, default=500,
                        help="案例类生成数量")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    counts = {
        "equipment": args.equipment,
        "doctrine": args.doctrine,
        "situation": args.situation,
        "case": args.case,
    }

    total = sum(counts.values())
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到已有文件的最大计数器，避免覆盖
    existing = list(output_dir.glob("synth_*.txt"))
    max_counter = 0
    for f in existing:
        try:
            # synth_equipment_000123.txt -> 123
            num = int(f.stem.split("_")[-1])
            max_counter = max(max_counter, num)
        except (ValueError, IndexError):
            pass

    print(f"增强版模板合成KB生成器")
    print(f"  已有文件: {len(existing)}, 最大计数器: {max_counter}")
    print(f"  总计: {total} 段")
    print(f"  装备: {counts['equipment']}, 条令: {counts['doctrine']}")
    print(f"  态势: {counts['situation']}, 案例: {counts['case']}")

    start = time.time()
    total_gen = max_counter + 1  # 从已有计数器之后开始

    for category, count in counts.items():
        if count <= 0:
            continue
        gen_func = GENERATORS[category]
        print(f"\n  [{category}] 生成 {count} 段...")

        for i in range(count):
            text = gen_func()
            doc_id = f"synth_{category}_{total_gen:06d}"

            (output_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            meta = {
                "source_url": "",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_category": "military_news",
                "content_category": category,
                "authority": "general_commentary",
                "language": "zh",
                "desensitized": True,
                "title": f"合成{category}段落_{total_gen:06d}",
                "document_id": doc_id,
                "synthetic": True,
                "template": category,
            }
            (output_dir / f"{doc_id}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total_gen += 1

            if total_gen % 200 == 0:
                elapsed = time.time() - start
                rate = total_gen / max(elapsed, 1)
                print(f"    已生成 {total_gen}/{total} ({rate:.0f} 段/秒)...")

    elapsed = time.time() - start
    print(f"\n✅ 完成: {total_gen} 段 → {output_dir}")
    print(f"   耗时: {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
