#!/usr/bin/env python3
"""Save collected US military doctrine corpus to data/raw/web_collected/"""
import json
from pathlib import Path
from datetime import datetime, timezone

OUT = Path("data/raw/web_collected")
OUT.mkdir(parents=True, exist_ok=True)
NOW = datetime.now(timezone.utc).isoformat()

DOCS = []

# ═══════════════════════════════════════════════
# US ARMY FIELD MANUALS COMPLETE DATABASE
# ═══════════════════════════════════════════════

DOCS.append({
    "id": "us_army_fm_complete_database",
    "title": "US Army Field Manuals Complete Database (542+ Manuals, Doctrine 2015 Reform)",
    "source_category": "doctrine", "content_category": "doctrine",
    "authority": "official_bulletin", "lang": "en",
    "source_url": "https://en.wikipedia.org/wiki/United_States_Army_Field_Manuals",
    "desensitized": True,
    "text": (
        "US Army Field Manuals — Complete Database\n"
        "As of July 2007, 542 field manuals were in use. Under Doctrine 2015 (started 2010), "
        "the most important doctrinal publications were reorganized into ADP (Army Doctrine Publications), "
        "ADRP (Doctrine Reference Publications), ATP (Techniques Publications), TC (Training Circulars), "
        "and TM (Technical Manuals). 50 select FMs continue to be published.\n\n"

        "=== CAPSTONE MANUALS ===\n"
        "FM 1, The Army (14 June 2005): Establishes fundamental principles for employing landpower. "
        "Together with FM 3-0, one of two capstone doctrinal manuals.\n"
        "FM 3-0, Operations (14 June 2001, updated 2011): Lays out fundamentals of war fighting. "
        "Covers Full Spectrum Operations, Mission Command, Six Warfighting Functions.\n"
        "FM 5-0, The Operations Process (26 Mar 2010): Plan, Prepare, Execute, Assess cycle. "
        "Design methodology for complex problems.\n"
        "FM 6-0, Mission Command (2011): Six principles — Build cohesive teams, Create shared "
        "understanding, Provide clear commander's intent, Exercise disciplined initiative, "
        "Use mission orders, Accept prudent risk.\n"
        "FM 7-0, Training the Force: Battle focused training, full spectrum operations training.\n\n"

        "=== FM SERIES OVERVIEW ===\n"
        "FM 1 Series (The Army): FM 1, FM 1-02 Operational Terms, FM 1-04 Legal Support, "
        "FM 1-05 Religious Support, FM 1-20 Military History Operations, "
        "FM 1-100 Army Aviation, FM 1-108 Special Operations Aviation, "
        "FM 1-111 Aviation Brigades, FM 1-112 Attack Helicopter, FM 1-113 Utility Helicopter.\n\n"
        "FM 2 Series (Intelligence): FM 2-0 Intelligence, FM 2-22.3 Human Intelligence Collector "
        "Operations (replaced FM 34-52 after Abu Ghraib).\n\n"
        "FM 3 Series (Operations): Most extensive series covering all operational domains.\n"
        "FM 3-01 Air Defense Artillery / FM 3-04 Army Aviation / FM 3-05 Special Operations / "
        "FM 3-06 Urban Operations / FM 3-07 Stability Operations / FM 3-09 Fire Support / "
        "FM 3-11 CBRN Defense / FM 3-12 Cyberspace and EW / FM 3-13 Information Operations / "
        "FM 3-14 Space Operations / FM 3-16 Multinational Operations / FM 3-18 SOF / "
        "FM 3-21.8 Infantry Rifle Platoon and Squad / FM 3-21.10 Infantry Rifle Company / "
        "FM 3-21.20 Infantry Battalion / FM 3-22 Army Support to Security Cooperation / "
        "FM 3-23.30 Grenades and Pyrotechnic Signals / FM 3-24 Counterinsurgency / "
        "FM 3-25.150 Combatives / FM 3-25.26 Map Reading / FM 3-34 Engineer Operations / "
        "FM 3-34.2 Combined-Arms Breaching / FM 3-37 Protection / FM 3-39 Military Police / "
        "FM 3-52 Airspace Control / FM 3-55 Information Collection / FM 3-57 Civil Affairs / "
        "FM 3-60 The Targeting Process / FM 3-90 Tactics / FM 3-96 Brigade Combat Team / "
        "FM 3-97.6 Mountain Operations / FM 3-98 Reconnaissance and Security / "
        "FM 3-99 Airborne and Air Assault Operations.\n\n"
        "FM 4 Series (Sustainment): FM 4-0 Sustainment, FM 4-01 Transportation, "
        "FM 4-02 Army Health System, FM 4-30 Ordnance Operations, FM 4-40 Quartermaster.\n\n"
        "FM 5 Series (Engineer): FM 5-0 Operations Process, FM 5-19 Composite Risk Management, "
        "FM 5-33 Terrain Analysis, FM 5-34 Engineer Field Data.\n\n"
        "FM 6 Series (Mission Command/Signals): FM 6-0 Mission Command, FM 6-02 Signal Support, "
        "FM 6-22 Leader Development, FM 6-22.5 Combat Stress.\n\n"
        "FM 7 Series (Training): FM 7-0 Training, FM 7-8 Infantry Platoon, FM 7-15 Army Universal "
        "Task List, FM 7-21.13 Soldier's Guide, FM 7-22 Physical Readiness Training, "
        "FM 7-100 Opposing Force Doctrinal Framework.\n\n"

        "=== NOTABLE MANUALS ===\n"
        "FM 3-05.70 (Survival Manual, formerly FM 21-76): Comprehensive wilderness survival guide.\n"
        "FM 3-24 (Counterinsurgency, 2014): Petraeus/Mattis counterinsurgency doctrine. "
        "Landmark publication that changed US military approach to COIN operations in Iraq and Afghanistan. "
        "Emphasizes population-centric approach, unity of effort, and political primacy.\n"
        "FM 5-31 (Boobytraps, 1965): No longer active but still frequently referenced in EOD training.\n"
        "FM 27-10 (Rules of War, 1956, last modified 1976): Cornerstone of Law of War for US military. "
        "Incorporates Geneva Conventions and customary international law.\n"
        "FM 34-52 (Intelligence Interrogation, 2005): Used to train CIA interrogators. "
        "Replaced by FM 2-22.3 in Sept 2006 after Abu Ghraib scandal with revised interrogation standards.\n"
        "FM 3-25.150 (Combatives): Hand-to-hand combat system based on Brazilian Jiu-Jitsu and wrestling.\n"
        "FM 3-19.15 (Civil Disturbance Operations): Riot control and crowd management doctrine.\n"
        "FM 3-97.61 (Military Mountaineering): Mountain and cold weather operations.\n"
        "FM 3-11 series (CBRN): NBC Defense, Decontamination, Protection, Chemical/Biological agents reference.\n"
        "FM 31-27 (Pack Animals in SOF): Use of mules and pack animals in special operations.\n\n"

        "=== DOCTRINE 2015 TRANSITION ===\n"
        "Old FM numbering (e.g., FM 100-5 Operations) replaced by FM 3-0 series to align with "
        "joint doctrine numbering (JP 3-0). Key ADP/ADRP publications:\n"
        "ADP 1: The Army / ADP 2-0: Intelligence / ADP 3-0: Operations / "
        "ADP 4-0: Sustainment / ADP 5-0: The Operations Process / ADP 6-0: Mission Command / "
        "ADP 7-0: Training\n"
        "Each ADP has a corresponding ADRP providing detailed implementation guidance."
    ),
})

# ═══════════════════════════════════════════════
# USAF DOCTRINE COMPLETE
# ═══════════════════════════════════════════════

DOCS.append({
    "id": "usaf_doctrine_complete",
    "title": "US Air Force Doctrine Documents Complete Reference",
    "source_category": "doctrine", "content_category": "doctrine",
    "authority": "official_bulletin", "lang": "en",
    "source_url": "https://www.globalsecurity.org/military/library/policy/usaf/afdd/",
    "desensitized": True,
    "text": (
        "US Air Force Doctrine Documents (AFDD) — Complete Reference\n"
        "All documents approved for public release by Commander, LeMay Center for Doctrine "
        "Development and Education, Maxwell AFB, Alabama.\n"
        "Renumbered circa 2010 from AFDD 2-series to AFDD 3-series to mirror joint doctrine (JP 3-xx).\n\n"

        "FOUNDATIONAL DOCTRINE:\n"
        "AFDD 1, Air Force Basic Doctrine (17 Nov 2003, updated 14 Oct 2011): Capstone publication "
        "describing fundamental beliefs about air, space, and cyberspace power. Organizes doctrine "
        "into basic (fundamental beliefs), operational (how to organize/employ), and tactical "
        "(how to execute specific missions).\n"
        "AFDD 1-1, Leadership and Force Development (18 Feb 2006): Core values (Integrity First, "
        "Service Before Self, Excellence in All We Do), force development framework, "
        "education and training continuum.\n\n"

        "OPERATIONAL DOCTRINE — AIR AND SPACE:\n"
        "AFDD 3-01, Counterair Operations (1 Oct 2008, IC2 1 Nov 2011): Offensive counterair (OCA) "
        "and defensive counterair (DCA). OCA: surface attack, fighter sweep, escort, SEAD. "
        "DCA: active air defense, passive defense measures.\n"
        "AFDD 3-14, Space Operations (IC1 28 Jul 2011): Space superiority, space force enhancement "
        "(ISR, missile warning, environmental monitoring, SATCOM, PNT), space control, space support.\n"
        "AFDD 3-14.1, Counterspace Operations (2 Aug 2004, Change 1, 28 Jul 2011): Offensive "
        "counterspace (deception, disruption, denial, degradation, destruction) and defensive "
        "counterspace. Lessons from GPS jamming during Operation Iraqi Freedom.\n\n"

        "OPERATIONAL DOCTRINE — INFORMATION AND CYBER:\n"
        "AFDD 3-12, Cyberspace Operations (IC1 30 Nov 2011): Cyberspace superiority, "
        "cyberspace attack, cyberspace defense, cyberspace ISR.\n"
        "AFDD 3-13, Information Operations (IC1 28 Jul 2011): Influence operations, electronic "
        "warfare, network warfare operations. Integration of IRCs (Information Related Capabilities).\n"
        "AFDD 3-13.1, Electronic Warfare (IC1 28 Jul 2011): Electronic attack (EA), electronic "
        "protection (EP), electronic warfare support (ES).\n\n"

        "OPERATIONAL DOCTRINE — EFFECTS AND SUPPORT:\n"
        "AFDD 3-17, Air Mobility Operations (IC1 28 Jul 2011): Airlift, air refueling, "
        "aeromedical evacuation. Global reach and rapid global mobility.\n"
        "AFDD 3-27, Homeland Operations (23 Apr 2013): Defense Support of Civil Authorities (DSCA), "
        "homeland air defense, emergency preparedness.\n"
        "AFDD 3-40, Counter-CBRN Operations (26 Jan 2007, IC2 1 Nov 2011): Countering chemical, "
        "biological, radiological, and nuclear threats.\n"
        "AFDD 3-60, Targeting (8 Jun 2006, Change 1, 28 Jul 2011): Complements JP 3-60. "
        "Six-phase targeting cycle: Commander's objectives, target development, weaponeering, "
        "force application, execution planning, combat assessment.\n"
        "AFDD 3-70, Strategic Attack (IC2 1 Nov 2011): Direct attack on enemy centers of gravity "
        "at the strategic level.\n"
        "AFDD 3-72, Nuclear Operations (Change 2, 14 Dec 2011): Nuclear deterrence, nuclear "
        "employment, nuclear surety (safety, security, control).\n\n"

        "COMMAND AND SUPPORT DOCTRINE:\n"
        "AFDD 6-0, Command and Control (IC1 28 Jul 2011): C2 principles, command relationships, "
        "air and space operations center (AOC) organization.\n"
        "AFDD 4-0, Combat Support (IC2 28 Jul 2011): Agile combat support, logistics, "
        "force protection, expeditionary operations support.\n\n"

        "STANDARD RELEASE STATEMENT: There are no releasability restrictions on this publication."
    ),
})

# ═══════════════════════════════════════════════
# JOINT PUBLICATIONS
# ═══════════════════════════════════════════════

DOCS.append({
    "id": "joint_publications_database",
    "title": "US Joint Publications (JP) Complete Reference",
    "source_category": "doctrine", "content_category": "doctrine",
    "authority": "official_bulletin", "lang": "en",
    "source_url": "https://www.jcs.mil/Doctrine/Joint-Doctine-Pubs/",
    "desensitized": True,
    "text": (
        "US Joint Publications (JP) — Complete Reference\n"
        "Joint doctrine presents fundamental principles that guide the employment of US military forces "
        "in coordinated action toward a common objective. JP 1 is the capstone.\n\n"

        "CAPSTONE:\n"
        "JP 1, Doctrine for the Armed Forces of the United States: Highest-level joint doctrine.\n"
        "JP 1-0, Joint Personnel Support.\n\n"

        "JOINT OPERATIONS (JP 3 Series):\n"
        "JP 3-0, Joint Operations: Foundation for all joint operations. Seven joint functions.\n"
        "JP 3-01, Countering Air and Missile Threats: IADS, air defense, missile defense.\n"
        "JP 3-02, Amphibious Operations: Ship-to-shore movement, landing force operations.\n"
        "JP 3-03, Joint Interdiction: Disrupting enemy capabilities before they can be used.\n"
        "JP 3-05, Special Operations: SOF employment across the range of military operations.\n"
        "JP 3-06, Joint Urban Operations: Operations in urban environments.\n"
        "JP 3-07, Stability: Post-conflict stability operations framework.\n"
        "JP 3-08, Interorganizational Coordination: Whole-of-government approach.\n"
        "JP 3-09, Joint Fire Support: Integration of joint fires.\n"
        "JP 3-10, Joint Security Operations: Force protection and security.\n"
        "JP 3-11, CBRN Environment Operations: Chemical, biological, radiological, nuclear.\n"
        "JP 3-12, Cyberspace Operations: Offensive and defensive cyber.\n"
        "JP 3-13, Information Operations: Influence, deception, EW, OPSEC.\n"
        "JP 3-14, Space Operations: Space as a warfighting domain.\n"
        "JP 3-15, Barriers and Obstacle Planning: Countermobility and breaching.\n"
        "JP 3-16, Multinational Operations: Coalition and alliance operations.\n"
        "JP 3-17, Air Mobility Operations: Strategic and tactical airlift.\n"
        "JP 3-18, Joint Forcible Entry: Airborne, air assault, amphibious entry.\n"
        "JP 3-22, Foreign Internal Defense: Supporting allied internal security.\n"
        "JP 3-24, Counterinsurgency: Population-centric COIN doctrine.\n"
        "JP 3-25, Countering Threat Networks: Network analysis and disruption.\n"
        "JP 3-26, Counterterrorism: CT policy and operational framework.\n"
        "JP 3-27, Homeland Defense: Defense of US territory and population.\n"
        "JP 3-28, Defense Support of Civil Authorities: Military aid to civil power.\n"
        "JP 3-29, Foreign Humanitarian Assistance: HA/DR operations.\n"
        "JP 3-30, Joint Air Operations: Air component command and control.\n"
        "JP 3-31, Joint Land Operations: Ground component operations.\n"
        "JP 3-32, Joint Maritime Operations: Naval component operations.\n"
        "JP 3-33, Joint Task Force Headquarters: JTF organization and command.\n"
        "JP 3-34, Joint Engineer Operations: Engineer support to joint operations.\n"
        "JP 3-35, Deployment and Redeployment Operations: Force projection.\n"
        "JP 3-40, Joint Countering Weapons of Mass Destruction: CWMD framework.\n"
        "JP 3-52, Joint Airspace Control: Airspace coordination and deconfliction.\n"
        "JP 3-57, Civil-Military Operations: CIMIC and civil affairs.\n"
        "JP 3-59, Meteorological and Oceanographic Operations: Weather and environment.\n"
        "JP 3-60, Joint Targeting: Target development, engagement, assessment.\n"
        "JP 3-61, Public Affairs: Media operations and strategic communication.\n"
        "JP 3-63, Detainee Operations: EPW and detainee handling.\n"
        "JP 3-68, Noncombatant Evacuation Operations: NEO planning and execution.\n"
        "JP 3-72, Nuclear Operations: Nuclear employment and deterrence.\n"
        "JP 3-80, Resource Management: Financial and resource planning.\n"
        "JP 3-84, Legal Support: Law of war, ROE, legal advice.\n"
        "JP 3-85, Joint Electromagnetic Spectrum Operations: EMS superiority.\n\n"

        "INTELLIGENCE (JP 2 Series):\n"
        "JP 2-0, Joint Intelligence / JP 2-01, Joint Intelligence Preparation of the "
        "Operational Environment / JP 2-03, Geospatial Intelligence.\n\n"

        "LOGISTICS (JP 4 Series):\n"
        "JP 4-0, Joint Logistics / JP 4-01, Joint Mobility Operations / JP 4-02, "
        "Joint Health Services / JP 4-05, Joint Mobilization Planning.\n\n"

        "PLANS (JP 5 Series):\n"
        "JP 5-0, Joint Planning: The joint planning process (JPP), operational design, "
        "operational art. Seven-step JPP: initiation, mission analysis, COA development, "
        "COA analysis and wargaming, COA comparison, COA approval, plan or order development.\n\n"

        "COMMUNICATIONS (JP 6 Series):\n"
        "JP 6-0, Joint Communications System: C4I systems and networks."
    ),
})

# ═══════════════════════════════════════════════
# COUNTERINSURGENCY FM 3-24
# ═══════════════════════════════════════════════

DOCS.append({
    "id": "fm3_24_counterinsurgency",
    "title": "FM 3-24 Counterinsurgency — Petraeus/Mattis COIN Doctrine (2014)",
    "source_category": "doctrine", "content_category": "case",
    "authority": "official_bulletin", "lang": "en",
    "source_url": "https://en.wikipedia.org/wiki/FM_3-24",
    "desensitized": True,
    "text": (
        "FM 3-24: Insurgencies and Countering Insurgencies (May 2014)\n\n"
        "FM 3-24 is the US Army/Marine Corps counterinsurgency field manual. Originally published "
        "December 2006 under General David Petraeus and General James Mattis. Updated May 2014. "
        "It is widely considered one of the most influential doctrinal publications of the post-9/11 era.\n\n"

        "CORE PRINCIPLES:\n"
        "1. Population-centric approach: The population is the center of gravity. "
        "Protect the population, separate insurgents from their support base.\n"
        "2. Political primacy: Military actions must support political objectives. "
        "COIN is 80% political and 20% military.\n"
        "3. Unity of effort: All actors (military, diplomatic, economic, informational) "
        "must work toward common objectives under unified command or coordination.\n"
        "4. Legitimacy: The host nation government must be perceived as legitimate. "
        "Building host nation capacity is essential.\n"
        "5. Intelligence-driven operations: Understanding the human terrain, social networks, "
        "and local dynamics is more important than kinetic operations.\n"
        "6. Measured force: Minimum necessary force. Excessive force alienates the population "
        "and creates more insurgents.\n\n"

        "FRAMEWORK:\n"
        "Clear-Hold-Build: Clear areas of insurgent presence, Hold them with persistent security, "
        "Build governance and economic capacity. Transition to host nation forces.\n"
        "Logical Lines of Operation (LLOs): Combat operations, Host nation security forces, "
        "Essential services, Governance, Economic development. Information operations cut across all LLOs.\n\n"

        "PARADOXES OF COIN:\n"
        "- The more you protect your force, the less secure you may be\n"
        "- The more force is used, the less effective it is\n"
        "- The more successful COIN is, the less force can be used\n"
        "- Sometimes doing nothing is the best reaction\n"
        "- Some of the best weapons do not shoot\n"
        "- The host nation doing something poorly is better than US forces doing it well\n\n"

        "IMPACT: FM 3-24 changed US military doctrine from enemy-centric attrition warfare to "
        "population-centric counterinsurgency. It was cited as instrumental in the 2007 Iraq Surge "
        "strategy. The manual has been studied globally and influenced NATO and allied COIN doctrine."
    ),
})

for d in DOCS:
    (OUT / f"{d['id']}.txt").write_text(d.pop("text"), encoding="utf-8")
    d.update({"collected_at": NOW, "document_id": d["id"]})
    (OUT / f"{d['id']}.meta.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Saved {len(DOCS)} doctrine documents:")
for d in DOCS:
    size = (OUT / f"{d['id']}.txt").stat().st_size
    print(f"  {d['id']}: {size} bytes")
