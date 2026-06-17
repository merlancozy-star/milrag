#!/usr/bin/env python3
"""将 WebSearch/WebFetch 采集的军事语料保存到 data/raw/web_collected/"""
import json
from pathlib import Path
from datetime import datetime, timezone

OUT = Path("data/raw/web_collected")
OUT.mkdir(parents=True, exist_ok=True)
NOW = datetime.now(timezone.utc).isoformat()

DOCS = []

# ── 1. 中国军事战略白皮书 2015 ──
DOCS.append({
    "id": "cn_military_strategy_2015",
    "title": "中国的军事战略 (2015) — China's Military Strategy White Paper",
    "source_category": "doctrine",
    "content_category": "doctrine",
    "authority": "official_bulletin",
    "lang": "zh",
    "source_url": "http://eng.mod.gov.cn/2025xb/M/P_251591/16415093.html",
    "desensitized": True,
    "text": (
        "中国的军事战略（2015年5月，国务院新闻办公室）\n\n"
        "积极防御战略方针：中国奉行积极防御的军事战略方针，核心是'人不犯我，我不犯人；"
        "人若犯我，我必犯人'。积极防御将战略上的防御与战役战术上的进攻有机统一，"
        "强调在战略上坚持防御和自卫立场，在战役战术上采取积极进攻行动以争取主动权。\n\n"
        "军事斗争准备基点：将军事斗争准备基点放在打赢信息化局部战争上。"
        "突出海上军事斗争准备，有效控制重大危机，妥善应对连锁反应，"
        "坚决捍卫国家领土主权、统一和安全。\n\n"
        "战略任务（8项）：\n"
        "1. 应对各种突发事件和军事威胁，有效维护国家领土、领空、领海主权和安全\n"
        "2. 坚决捍卫祖国统一\n"
        "3. 维护新型领域安全和利益\n"
        "4. 维护海外利益安全\n"
        "5. 保持战略威慑，组织核反击行动\n"
        "6. 参加地区和国际安全合作，维护地区和世界和平\n"
        "7. 加强反渗透、反分裂、反恐怖斗争，维护国家政治安全和社会大局稳定\n"
        "8. 担负抢险救灾、维护权益、安保警戒和支援国家经济社会建设等任务\n\n"
        "军种发展战略：\n"
        "- 陆军：按照机动作战、立体攻防的战略要求，实现区域防卫型向全域机动型转变\n"
        "- 海军：按照近海防御与远海护卫相结合的战略要求，逐步实现由近海防御型向"
        "近海防御与远海护卫型结合转变\n"
        "- 空军：按照空天一体、攻防兼备的战略要求，实现由国土防空型向攻防兼备型转变\n"
        "- 火箭军：按照核常兼备、全域慑战的战略要求，增强可信可靠的核威慑和核反击能力\n"
        "- 战略支援部队：加快新型作战力量建设，增强战场综合保障能力\n\n"
        "核政策：中国始终奉行不首先使用核武器的政策，坚持自卫防御的核战略，"
        "无条件不对无核武器国家和无核武器区使用或威胁使用核武器。"
        "中国的核力量始终维持在国家安全所需的最低水平。\n\n"
        "海外利益攸关区：首次明确提出此概念，强调随着国家利益拓展，"
        "维护海外能源资源安全、战略通道安全以及海外机构、人员和资产安全"
        "成为军事战略的重要组成部分。"
    ),
})

# ── 2. PLA 力量结构 (2013白皮书) ──
DOCS.append({
    "id": "cn_armed_forces_2013",
    "title": "中国武装力量的多样化运用 (2013) — PLA Force Structure Disclosure",
    "source_category": "doctrine",
    "content_category": "equipment",
    "authority": "official_bulletin",
    "lang": "zh",
    "source_url": "https://guofang.tsinghua.edu.cn/info/1017/1514.htm",
    "desensitized": True,
    "text": (
        "中国武装力量的多样化运用（2013年4月，国务院新闻办公室）\n\n"
        "首次公开发布军种兵力数据：\n"
        "- 陆军：85万人\n"
        "- 海军：23.5万人\n"
        "- 空军：39.8万人\n"
        "- 第二炮兵（现火箭军）：编制不详，配备东风系列弹道导弹和长剑巡航导弹\n\n"
        "18个集团军部署（2013年体制，后于2017年重组为13个）：\n"
        "- 沈阳军区：第16、39、40集团军\n"
        "- 北京军区：第27、38、65集团军\n"
        "- 兰州军区：第21、47集团军\n"
        "- 济南军区：第20、26、54集团军\n"
        "- 南京军区：第1、12、31集团军\n"
        "- 广州军区：第41、42集团军\n"
        "- 成都军区：第13、14集团军\n\n"
        "陆军机动作战部队包括18个集团军和部分独立合成作战师（旅），现有85万人。"
        "集团军由师、旅编成，分别隶属于7个军区。\n\n"
        "海军现有23.5万人，下辖北海、东海和南海3个舰队。"
        "海军编有水面舰艇部队、潜艇部队、航空兵、岸防部队和陆战队。\n\n"
        "空军现有39.8万人，下辖沈阳、北京、兰州、济南、南京、"
        "广州、成都7个军区空军和1个空降兵军。\n\n"
        "武装力量的多样化运用：保卫边海防安全、保卫空防安全、维护社会稳定、"
        "参加抢险救灾、维护海洋权益、维护海外利益（亚丁湾护航等）、"
        "参加联合国维和行动、国际人道主义救援。"
    ),
})

# ── 3. 新时代中国国防 2019 ──
DOCS.append({
    "id": "cn_defense_new_era_2019",
    "title": "新时代的中国国防 (2019) — China National Defense in the New Era",
    "source_category": "doctrine",
    "content_category": "doctrine",
    "authority": "official_bulletin",
    "lang": "zh",
    "source_url": "http://english.scio.gov.cn/2019-07/24/content_75026800_5.htm",
    "desensitized": True,
    "text": (
        "新时代的中国国防（2019年7月，国务院新闻办公室）\n\n"
        "军事改革成果（2015-2016年深化国防和军队改革）：\n"
        "- 原四总部（总参谋部、总政治部、总后勤部、总装备部）改为15个军委职能部门\n"
        "- 7大军区调整为东、南、西、北、中5个战区\n"
        "- 18个集团军重组为13个集团军（第71集团军至第83集团军）\n"
        "- 成立陆军领导机构、火箭军和战略支援部队\n"
        "- 组建军委联勤保障部队和军委国防动员部\n\n"
        "裁军：裁减军队员额30万，现役总员额减至200万。"
        "军官数量减少约25%，非战斗单位压减近50%。\n\n"
        "军种新体制：\n"
        "- 陆军：由'大陆军'体制向各军种平等体制转变，组建独立的陆军领导机构\n"
        "- 海军：加快推进由近海防御型向远海防卫型转变\n"
        "- 空军：加快实现由国土防空型向攻防兼备型转变\n"
        "- 火箭军：增强战略威慑与核反击能力，核常兼备\n"
        "- 战略支援部队：整合情报、技术侦察、电子对抗、网络攻防、心理战等新型作战力量\n"
        "- 联勤保障部队：实施联勤保障和战略战役支援保障\n\n"
        "国防费：2012年至2017年，中国国防费占GDP比重平均约1.3%，"
        "远低于美国的3.5%和俄罗斯的4.4%。中国国防费始终维持在合理适度水平。\n\n"
        "三个'永远不'：中国永远不称霸，永远不扩张，永远不谋求势力范围。\n\n"
        "军事政策制度：坚持党对军队的绝对领导，军委主席负责制是根本制度和最高政治要求。"
        "构建'军委管总、战区主战、军种主建'的新格局。"
    ),
})

# ── 4. FM 3-0 Operations (US Army Doctrine) ──
DOCS.append({
    "id": "fm3_0_doctrine",
    "title": "FM 3-0 Operations — US Army Capstone Doctrine (2011, Change 1)",
    "source_category": "doctrine",
    "content_category": "doctrine",
    "authority": "official_bulletin",
    "lang": "en",
    "source_url": "https://www.globalsecurity.org/military/library/policy/army/fm/3-0/index_2011.html",
    "desensitized": True,
    "text": (
        "FM 3-0: OPERATIONS — Headquarters, Department of the Army, Washington, DC, 22 February 2011\n"
        "Distribution: Approved for public release; unlimited distribution\n\n"
        "FM 3-0 is one of the Army's two capstone doctrinal publications. "
        "It presents overarching doctrinal guidance and direction for conducting operations.\n\n"
        "CHAPTER 1 — Operational Context: The global environment of persistent conflict, "
        "the operational environment, and unified action. Covers expeditionary and campaign capabilities. "
        "Soldiers and leaders remain the Army's most important advantage.\n\n"
        "CHAPTER 2 — Spectrum of Conflict: Describes a spectrum of conflict extending from stable peace "
        "to general war. Establishes five operational themes into which joint operations fit.\n\n"
        "CHAPTER 3 — Full Spectrum Operations: The most important chapter. Full spectrum operations "
        "seize, retain, and exploit the initiative and achieve decisive results through combinations "
        "of offense, defense, and stability or civil support. Establishes mission command as the "
        "preferred battle command method.\n\n"
        "CHAPTER 4 — Combat Power: Six warfighting functions bound by leadership and employing "
        "information as the elements of combat power: Mission Command, Movement and Maneuver, "
        "Intelligence, Fires, Sustainment, and Protection.\n\n"
        "CHAPTER 5 — The Operations Process: plan, prepare, execute, and assess. "
        "Commanders understand, visualize, describe, direct, lead, and continually assess.\n\n"
        "MISSION COMMAND: The exercise of authority and direction by the commander using mission orders "
        "to enable disciplined initiative within the commander's intent to empower agile and adaptive leaders "
        "in the conduct of unified land operations.\n\n"
        "PRINCIPLES OF WAR: Objective, Offensive, Mass, Economy of Force, Maneuver, "
        "Unity of Command, Security, Surprise, Simplicity."
    ),
})

# ── 5. Fighter Jet Specs ──
DOCS.append({
    "id": "fighter_jet_specs",
    "title": "Fighter Jet Specifications Comparison — Public Open-Source Data",
    "source_category": "encyclopedia",
    "content_category": "equipment",
    "authority": "mainstream_media",
    "lang": "en",
    "source_url": "https://www.wionews.com/photos/6-fighter-jets-compared-by-speed-range-and-climb-rate-1760342658761",
    "desensitized": True,
    "text": (
        "Fighter Jet Specifications Comparison (Public Domain Data)\n\n"
        "5th Generation Fighters:\n"
        "- F-22 Raptor (USA): Mach 2.25 (~2,414 km/h), Combat Range ~1,100 nmi, "
        "Service Ceiling ~65,000 ft (19,800 m), Introduced 2005, Unit Cost ~$360M\n"
        "- F-35 Lightning II (USA): Mach 1.6 (~1,931 km/h), Combat Range >1,000 km, "
        "Service Ceiling ~50,000 ft (15,240 m), Introduced 2015, Unit Cost ~$80-110M\n"
        "- Su-57 Felon (Russia): Mach 2.0 (~2,136 km/h), Combat Range ~1,900 km, "
        "Service Ceiling ~65,600 ft (20,000 m), Introduced 2020\n"
        "- Chengdu J-20 (China): Mach 2.0 (~2,470 km/h), Combat Range ~2,000 km, "
        "Service Ceiling ~65,000 ft (20,000 m), Introduced 2017\n\n"
        "4th/4.5 Generation Fighters (US):\n"
        "- F-15 Eagle/EX: Mach 2.5 (~2,655 km/h), Combat Radius ~1,770 km, "
        "Ferry Range ~5,550 km, MTOW 36,741 kg (81,000 lb), Payload ~13,400 kg\n"
        "- F-16 Fighting Falcon: Mach 2.0 (~2,120 km/h), Combat Radius ~550 km, "
        "Ferry Range ~4,220 km, MTOW 19,200 kg, Payload ~7,700 kg\n"
        "- F/A-18E/F Super Hornet: Mach 1.6 (~1,915 km/h), Combat Radius ~1,160 km, "
        "MTOW 29,937 kg (66,000 lb), Payload ~8,050 kg\n\n"
        "4th/4.5 Generation Fighters (European):\n"
        "- Eurofighter Typhoon: Mach 2.35 (~2,495 km/h), Combat Radius ~1,389 km, "
        "Ferry Range ~2,900 km, MTOW 23,500 kg, Payload ~9,000 kg\n"
        "- Dassault Rafale: Mach 1.8 (~1,915 km/h), Combat Radius ~1,850 km, "
        "Ferry Range ~3,700 km, MTOW 24,500 kg, Payload ~9,500 kg\n"
        "- Saab JAS 39 Gripen E: Mach 2.0 (~2,100 km/h), Combat Radius ~800 km, "
        "MTOW 14,000 kg, Payload ~5,300 kg\n\n"
        "Russian Fighters:\n"
        "- MiG-25 Foxbat: Mach 2.83 (~3,000 km/h) — Fastest operational fighter\n"
        "- Su-27 Flanker: Mach 2.35 (~2,500 km/h), Combat Radius ~1,340 km\n"
        "- MiG-29 Fulcrum: Mach 2.3 (~2,450 km/h), Combat Radius ~650 km\n\n"
        "Supercruise Capable: F-22 Raptor, Su-57 Felon, Dassault Rafale, Eurofighter Typhoon\n"
        "All figures from publicly available defense publications and manufacturer specifications."
    ),
})

# ── 6. Strategic Analysis ──
DOCS.append({
    "id": "strategic_analysis_2024",
    "title": "Strategic Analysis — Global Military Balance and Defense Strategy (2024-2025)",
    "source_category": "commentary",
    "content_category": "situation",
    "authority": "mainstream_media",
    "lang": "en",
    "source_url": "https://press.armywarcollege.edu/parameters/vol55/iss1/7/",
    "desensitized": True,
    "text": (
        "Strategic Analysis — Global Military Balance and Defense Strategy (2024-2025)\n\n"
        "US Defense Strategy Assessment:\n"
        "The US defense strategy has been described as 'insolvent' — tasked missions exceed "
        "available means. Recommendations include deploying additional forces in the Western "
        "Pacific and Europe, developing robust sensing and targeting grids, acquiring larger "
        "quantities of standoff precision munitions, and adopting innovative operational "
        "concepts without necessarily increasing budgets (RAND Corporation, 2024).\n\n"
        "Great-Power Competition:\n"
        "China is identified as the primary strategic challenge across multiple analyses. "
        "The PLA's thinking on escalation dynamics has become significantly more risk-tolerant. "
        "China could initiate conflict activities if it judges political risk of inaction "
        "greater than military risk (RAND, June 2024).\n\n"
        "Japan's Defense Evolution:\n"
        "Japan established a new Joint Operations Command (March 2025) for centralized command "
        "of ground, maritime, air, space, and cyber forces. Japan is accelerating development "
        "of long-range strike weapons including upgraded Type 12 anti-ship missiles, hypersonic "
        "glide vehicles, and Tomahawk/JASSM-ER imports. The 2024 Defense White Paper labels "
        "China as 'an unprecedented and the greatest strategic challenge.'\n\n"
        "Maritime Strategy Framework:\n"
        "Contemporary maritime military strategy is built on three core concepts: area denial, "
        "sea control, and power projection. Advances in C4ISRT, long-range precision munitions, "
        "and reconnaissance capabilities have fundamentally transformed naval warfare. "
        "Land-based forces can now effectively strike maritime forces at extended ranges.\n\n"
        "Force Planning Challenges:\n"
        "Wargames suggest the US would likely run out of critical precision munitions in less "
        "than one week in a Taiwan Strait contingency scenario. The $36 trillion federal debt "
        "constrains defense spending, while growing shipbuilding and force disparities with "
        "China continue to widen. Mission-based force planning is proposed as an alternative "
        "to the traditional two-major-theater-war construct."
    ),
})

# ── 7. US Army FM Directory ──
DOCS.append({
    "id": "us_army_fm_directory",
    "title": "US Army Field Manuals Complete Public Directory (200+ Manuals)",
    "source_category": "doctrine", "content_category": "doctrine",
    "authority": "official_bulletin", "lang": "en",
    "source_url": "https://www.globalsecurity.org/military/library/policy/army/fm/index.html",
    "desensitized": True,
    "text": (
        "US Army Field Manuals — Complete Public Directory\n"
        "All documents approved for public release; distribution unlimited\n\n"
        "CAPSTONE: FM 1 The Army / FM 3-0 Operations / FM 5-0 Operations Process / FM 6-0 Mission Command / FM 7-0 Training\n\n"
        "INTELLIGENCE: FM 2-0 Intelligence / FM 2-22.3 Human Intelligence Collector Operations\n\n"
        "OPERATIONS (selected): FM 3-06 Urban Operations / FM 3-07 Stability Operations / FM 3-09 Fire Support / "
        "FM 3-11 CBRN Defense / FM 3-12 Cyberspace and EW / FM 3-13 Information Operations / FM 3-14 Space Support / "
        "FM 3-16 Multinational Operations / FM 3-21.8 Infantry Rifle Platoon and Squad / FM 3-24 Counterinsurgency / "
        "FM 3-34 Engineer Operations / FM 3-39 Military Police / FM 3-52 Airspace Control / "
        "FM 3-55 Information Collection / FM 3-57 Civil Affairs / FM 3-60 Targeting Process / "
        "FM 3-90 Tactics / FM 3-96 Brigade Combat Team / FM 3-98 Reconnaissance and Security / FM 3-99 Airborne and Air Assault\n\n"
        "SUSTAINMENT: FM 4-0 Sustainment / FM 4-01 Transportation / FM 4-02 Army Health System\n\n"
        "LEADERSHIP: FM 6-22 Army Leadership / FM 6-22.5 Combat Stress / FM 7-22 Physical Readiness\n\n"
        "TRAINING: FM 7-0 Training the Force / FM 7-8 Infantry Rifle Platoon and Squad / FM 7-15 Army Universal Task List / FM 7-21.13 Soldiers Guide"
    ),
})

# ── 8. USAF AFDD Directory ──
DOCS.append({
    "id": "usaf_afdd_directory",
    "title": "US Air Force Doctrine Documents Public Directory",
    "source_category": "doctrine", "content_category": "doctrine",
    "authority": "official_bulletin", "lang": "en",
    "source_url": "https://www.globalsecurity.org/military/library/policy/usaf/afdd/",
    "desensitized": True,
    "text": (
        "US Air Force Doctrine Documents (AFDD) Public Directory\n"
        "All approved for public release by LeMay Center/CC\n\n"
        "FOUNDATIONAL: AFDD 1 Air Force Basic Doctrine / AFDD 1-1 Leadership and Force Development\n\n"
        "OPERATIONS: AFDD 3-01 Counterair / AFDD 3-12 Cyberspace / AFDD 3-13 Information Operations / "
        "AFDD 3-13.1 Electronic Warfare / AFDD 3-14 Space Operations / AFDD 3-14.1 Counterspace / "
        "AFDD 3-17 Air Mobility / AFDD 3-27 Homeland Operations / AFDD 3-40 Counter-CBRN / "
        "AFDD 3-60 Targeting / AFDD 3-70 Strategic Attack / AFDD 3-72 Nuclear Operations\n\n"
        "COMMAND AND SUPPORT: AFDD 6-0 Command and Control / AFDD 4-0 Combat Support\n\n"
        "Standard release: There are no releasability restrictions on this publication."
    ),
})

# ── 9. Mission Command ──
DOCS.append({
    "id": "mission_command_principles",
    "title": "Mission Command Core Principles (FM 6-0)",
    "source_category": "doctrine", "content_category": "doctrine",
    "authority": "official_bulletin", "lang": "en",
    "source_url": "https://www.globalsecurity.org/military/library/policy/army/fm/6-0/",
    "desensitized": True,
    "text": (
        "Mission Command — Core Principles and Doctrine (FM 6-0)\n\n"
        "Mission Command is the exercise of authority and direction by the commander using mission orders "
        "to enable disciplined initiative within the commanders intent to empower agile and adaptive leaders "
        "in the conduct of unified land operations.\n\n"
        "Six Principles of Mission Command:\n"
        "1. Build cohesive teams through mutual trust\n"
        "2. Create shared understanding\n"
        "3. Provide a clear commanders intent\n"
        "4. Exercise disciplined initiative\n"
        "5. Use mission orders\n"
        "6. Accept prudent risk\n\n"
        "Commander Tasks: Understand / Visualize / Describe / Direct / Lead / Assess\n\n"
        "Operations Process: Plan / Prepare / Execute / Assess\n\n"
        "Staff Tasks: Conduct operations process / Knowledge and information management / "
        "Inform and influence activities / Cyber electromagnetic activities"
    ),
})

# ── 10. Warfighting Functions ──
DOCS.append({
    "id": "warfighting_functions",
    "title": "Six Warfighting Functions US Army Doctrine",
    "source_category": "doctrine", "content_category": "doctrine",
    "authority": "official_bulletin", "lang": "en",
    "source_url": "https://www.globalsecurity.org/military/library/policy/army/fm/3-0/index_2011.html",
    "desensitized": True,
    "text": (
        "Six Warfighting Functions — US Army Doctrine (FM 3-0)\n\n"
        "1. MISSION COMMAND: Develops and integrates activities enabling commanders to balance "
        "the art of command and the science of control.\n"
        "2. MOVEMENT AND MANEUVER: Moves forces to achieve positions of advantage relative to the enemy.\n"
        "3. INTELLIGENCE: Synchronizes collection, processing, analysis, and dissemination of information.\n"
        "4. FIRES: Creates effects to support operations through targeting and fire support.\n"
        "5. SUSTAINMENT: Provides logistics, personnel services, and health service support.\n"
        "6. PROTECTION: Preserves the force through air defense, CBRN defense, security, and information protection.\n\n"
        "Combined arms and mutual support represent the payoff of integration across all functions. "
        "These six functions, bound by leadership, replace the older Battlefield Operating Systems (BOS)."
    ),
})

# ── 11. Missile Systems Specs ──
DOCS.append({
    "id": "missile_systems_specs",
    "title": "Missile Systems Global Specifications Comparison",
    "source_category": "encyclopedia", "content_category": "equipment",
    "authority": "mainstream_media", "lang": "en",
    "source_url": "https://www.globalmilitary.net/missiles/",
    "desensitized": True,
    "text": (
        "Missile Systems Global Specifications Comparison\n\n"
        "ICBMs:\n"
        "- RS-28 Sarmat (Russia): Range 18,000km, Speed Mach 20, 8-10 MIRV, up to 10 tons payload\n"
        "- DF-41 (China): Range 15,000km, Speed Mach 25, up to 10 MIRV, inertial+stellar+BeiDou guidance\n"
        "- Hwasong-17 (North Korea): Range 15,000km, Speed Mach 22, inertial guidance\n"
        "- LGM-30 Minuteman III (USA): Range 13,000km, Speed Mach 23, 1-3 MIRV, inertial+GPS\n"
        "- RS-24 Yars (Russia): Range 11,250km, Speed Mach 20, 4-6 MIRV, inertial+GLONASS\n"
        "- M51 (France): Range 9,000km, Speed Mach 25, 6-10 MIRV\n\n"
        "SLBMs:\n"
        "- Trident II D5 (USA/UK): Range 12,000km, 8-12 MIRV, inertial+stellar\n"
        "- JL-2 Julang-2 (China): Range 7,600km, 1-3 MIRV, inertial+stellar+BeiDou\n"
        "- K-4 (India): Range 3,500km, 1 warhead\n\n"
        "Cruise Missiles:\n"
        "- BGM-109 Tomahawk (USA): Range 2,500km, subsonic, INS+GPS+TERCOM+DSMAC, 450kg warhead\n"
        "- 3M-54 Kalibr (Russia): Range 1,500-2,500km, terminal Mach 2.9, 450kg warhead\n"
        "- CJ-10/DF-10 (China): Range 1,500km+, subsonic, INS+BeiDou+TERCOM\n"
        "- AGM-158B JASSM-ER (USA): Range 1,000km, stealth, INS+GPS+IIR, 450kg\n"
        "- AGM-158C LRASM (USA): Range 926km, stealth, passive RF+IIR+AI targeting\n"
        "- BrahMos (India/Russia): Range 450-800km, Mach 2.8-3.0, 200-300kg\n"
        "- Storm Shadow/SCALP-EG (UK/France): Range 560km, 450kg\n\n"
        "Air-to-Air Missiles:\n"
        "- R-37 (Russia): Range 400km, Mach 7.3, active radar\n"
        "- PL-17 (China): Range 400km, Mach 7.4, active radar+datalink\n"
        "- Meteor (Europe): Range 200km+, Mach 4.9, ramjet propulsion\n"
        "- AIM-260 JATM (USA): Range 200km, Mach 6.1, active radar+datalink\n"
        "- PL-15 (China): Range 200km, Mach 6.1, AESA radar\n"
        "- AIM-120D AMRAAM (USA): Range 180km, Mach 4.9, active radar+datalink\n\n"
        "Surface-to-Air Missiles:\n"
        "- S-500 Prometheus (Russia): Range 600km, anti-ICBM capable\n"
        "- S-400 Triumf (Russia): Range 400km, engagement altitude 30km\n"
        "- Davids Sling (Israel/USA): Range 300km, active radar+EO/IR\n"
        "- HQ-9 (China): Range 200km, 180kg warhead\n"
        "- Patriot PAC-3 (USA): Range 160km, hit-to-kill interceptor\n"
        "- Aster 30 (France): Range 120km, active radar\n"
        "- Iron Dome Tamir (Israel): Range 70km, counter-rocket/artillery/mortar"
    ),
})

# ── 12. Chaco War Operational Art ──
DOCS.append({
    "id": "chaco_war_case_study",
    "title": "Operational Art and the Chaco War (1932-1935) Case Study",
    "source_category": "case", "content_category": "case",
    "authority": "mainstream_media", "lang": "en",
    "source_url": "https://www.armyupress.army.mil/Journals/Military-Review/English-Edition-Archives/November-December-2025/Chaco-War/",
    "desensitized": True,
    "text": (
        "Operational Art and the Chaco War (1932-1935) — Military Review Case Study\n\n"
        "The Chaco War was fought between Bolivia and Paraguay (1932-1935) over the Gran Chaco region. "
        "Paraguay defeated a larger, better-equipped Bolivian army through superior operational art. "
        "This case study analyzes how a materially inferior force achieved decisive victory.\n\n"
        "CENTER OF GRAVITY ANALYSIS:\n"
        "Paraguay identified Bolivia's logistical vulnerability as its center of gravity. Bolivian forces "
        "operated at the end of an extremely long and fragile supply line stretching from the Andean highlands "
        "through nearly impassable Chaco terrain. Paraguay targeted this vulnerability by interdicting supply "
        "routes rather than seeking decisive battle against the main Bolivian force.\n\n"
        "DECISIVE POINTS:\n"
        "Key geographic decisive points included water sources, forts, and road junctions. Paraguay captured "
        "these sequentially, each success further degrading Bolivia's logistical position while improving "
        "Paraguay's own. The forts of Boqueron, Nanawa, and Campo Via became pivotal operational objectives.\n\n"
        "OPERATIONAL REACH:\n"
        "Bolivia had greater strategic depth but inadequate operational reach due to poor infrastructure. "
        "Paraguay's shorter supply lines from the Paraguay River allowed sustained operations despite "
        "smaller overall resources. The campaign demonstrated that operational reach is not merely "
        "geographic distance but a function of logistics, organization, and leadership.\n\n"
        "TEMPO:\n"
        "Paraguay maintained higher operational tempo through better logistics, decentralized command, "
        "and superior tactical proficiency in Chaco conditions. Rapid movement between objectives prevented "
        "Bolivia from reconstituting defenses. The encirclement operations at Campo Via (1933) trapped "
        "two Bolivian divisions, demonstrating how tempo creates decisive opportunities.\n\n"
        "RISK AND COMMAND:\n"
        "Paraguayan commanders accepted prudent risk, operating with extended flanks in low-density terrain. "
        "Trust in subordinate initiative allowed rapid exploitation of fleeting opportunities. The Bolivian "
        "high command conversely suffered from excessive centralization and slow decision cycles.\n\n"
        "LESSONS FOR CONTEMPORARY PLANNERS:\n"
        "1. Operational art can offset numerical and materiel inferiority\n"
        "2. Logistics and operational reach often determine campaign outcomes more than tactical skill\n"
        "3. Decentralized command with clear intent enables higher tempo than centralized control\n"
        "4. Environmental mastery (terrain, climate, disease) is a force multiplier\n"
        "5. The identification and exploitation of the enemy center of gravity is the essence of operational art"
    ),
})

# ── 13. Joint Functions Doctrine ──
DOCS.append({
    "id": "joint_functions_doctrine",
    "title": "The Joint Functions — Theory Doctrine and Practice (JFQ 2024)",
    "source_category": "doctrine", "content_category": "doctrine",
    "authority": "mainstream_media", "lang": "en",
    "source_url": "https://digitalcommons.ndu.edu/joint-force-quarterly/",
    "desensitized": True,
    "text": (
        "The Joint Functions: Theory, Doctrine, and Practice — JFQ 115, 4th Quarter 2024\n"
        "Author: Matthew J. Tackett, Professor, Joint Military Operations, US Naval War College\n\n"
        "The Seven Joint Functions (JP 3-0):\n"
        "1. Command and Control (C2) — Authority and direction over forces\n"
        "2. Information — Added as seventh function in 2017, reflecting information environment importance\n"
        "3. Intelligence — Collection, processing, analysis, dissemination\n"
        "4. Fires — Lethal and non-lethal effects creation\n"
        "5. Movement and Maneuver — Gaining positional advantage\n"
        "6. Protection — Force preservation\n"
        "7. Sustainment — Logistics, personnel, health services\n\n"
        "HISTORICAL EVOLUTION:\n"
        "- Clausewitz Principles of War (1812): Foundation for modern operational thinking\n"
        "- French staff system under Thiebault: First systematic operational planning\n"
        "- Goldwater-Nichols Act (1986): Mandated joint operations, forced Service integration\n"
        "- JP 3-0 (2001): No listed joint functions\n"
        "- JP 3-0 (2006): Six joint functions\n"
        "- JP 3-0 (2017): Seven joint functions (Information added)\n\n"
        "THEORY VS DOCTRINE TENSION:\n"
        "Milan Vego (Joint Operational Warfare) proposes theoretical operational functions that differ "
        "from current US joint doctrine. The article examines whether joint doctrine represents satisficing "
        "(Herbert Simon concept of good enough) rather than optimization. Service parochialism still "
        "affects joint operations despite decades of joint doctrine development.\n\n"
        "MODERN VIGNETTES:\n"
        "- USS Carney TLAM strikes (February 2024): Operational fires in practice\n"
        "- Army field artillery exercises in Iraq (May 2024): Joint fires coordination\n"
        "- Marine Corps training at Twentynine Palms (June 2024): Maneuver and fires integration"
    ),
})

# ── 14. Okinawa Joint Planning ──
DOCS.append({
    "id": "okinawa_joint_planning",
    "title": "Joint Planning and the Battle of Okinawa (1945) Case Study",
    "source_category": "case", "content_category": "case",
    "authority": "mainstream_media", "lang": "en",
    "source_url": "https://www.armyupress.army.mil/Journals/NCO-Journal/Archives/2025/July/The-Battle-of-Okinawa/",
    "desensitized": True,
    "text": (
        "Joint Planning and the Battle of Okinawa (1945) Case Study\n\n"
        "The 82-day Okinawa campaign (April-June 1945) was the largest amphibious assault "
        "of the Pacific War and the last major battle of World War II. It remains a premier "
        "case study in joint operational planning.\n\n"
        "CENTER OF GRAVITY: Japanese defense relied on deeply fortified positions in southern "
        "Okinawa (Shuri Line). The joint force identified and systematically reduced these through "
        "combined naval gunfire, aerial bombardment, and ground assault.\n\n"
        "JOINT COMMAND STRUCTURE: Admiral Nimitz (naval) held overall command with General Buckner "
        "(ground) leading Tenth Army. The joint command integrated US Navy, Marine Corps, Army, "
        "and Allied naval forces under unified operational control.\n\n"
        "SUSTAINMENT: The operation sustained 183,000 troops across 1,200 miles of ocean for 82 "
        "continuous days of combat. The logistics pipeline required coordination between Army service "
        "forces and Navy supply chains, demonstrating the criticality of joint sustainment planning.\n\n"
        "TEMPO AND CASUALTIES: Highest single-battle casualties of the Pacific campaign. US losses: "
        "over 12,000 killed, 38,000 wounded. Japanese losses: approximately 110,000 troops, plus "
        "an estimated 100,000+ civilian casualties. The kamikaze campaign inflicted heavy naval losses.\n\n"
        "LESSONS:\n"
        "1. Joint command relationships must be clearly defined before operations commence\n"
        "2. Sustainment planning is as critical as maneuver planning in joint operations\n"
        "3. Physical terrain can be shaped (island fortification) to offset maneuver disadvantage\n"
        "4. Civilian population considerations must be integrated into operational planning\n"
        "5. Joint fires coordination between naval gunfire, air, and ground requires dedicated liaison"
    ),
})

# ── Save all ──
for d in DOCS:
    txt_path = OUT / f"{d['id']}.txt"
    meta_path = OUT / f"{d['id']}.meta.json"
    txt_path.write_text(d.pop("text"), encoding="utf-8")
    d["collected_at"] = NOW
    d["document_id"] = d["id"]
    meta_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Saved {len(DOCS)} documents to {OUT}")
for d in DOCS:
    size = (OUT / f"{d['id']}.txt").stat().st_size
    print(f"  {d['id']}: {size} bytes")
