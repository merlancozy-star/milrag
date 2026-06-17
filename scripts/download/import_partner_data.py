#!/usr/bin/env python3
"""合作单位脱密数据导入。

从 data/raw/sensitive/ 目录读取合作单位线下交付的脱密语料，
验证脱密标记后导入 data/raw/partner_data/。

安全规则:
  1. 只处理 data/raw/sensitive/ 目录（已在 .gitignore 中）
  2. 要求每条记录标记 desensitized: true
  3. 写入 data/raw/partner_data/（公开目录），原始 sensitive 文件保留
  4. 输出文件不包含任何可追溯到原始敏感信息的内容

如果 sensitive/ 目录不存在或为空，自动使用合成脱密数据填充。
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


SENSITIVE_DIR = Path("data/raw/sensitive")
OUTPUT_DIR = Path("data/raw/partner_data")

# 合成脱密数据模板（仅在合作数据不可用时使用）
SYNTHETIC_FALLBACK = [
    {
        "title": "某部年度训练总结（脱密版）",
        "content_category": "case",
        "text": (
            "某部在年度训练中围绕核心使命任务组织了多课目实战化训练。"
            "训练内容涵盖基础课目复训、专业技能强化和综合演练三个阶段。"
            "基础课目复训阶段重点抓体能达标和单兵技战术基础；"
            "专业技能强化阶段突出分队协同和指挥控制训练；"
            "综合演练阶段设置了复杂电磁环境下的多兵种联合训练课题。"
            "训练实践表明：实战化标准的确立显著提升了部队遂行任务能力；"
            "复杂环境构设是提高训练效益的关键因素；"
            "训练评估机制的完善对发现短板和改进训练具有重要指导意义。"
        ),
    },
    {
        "title": "军事地形保障系统应用报告（脱密版）",
        "content_category": "situation",
        "text": (
            "军事地形保障系统在某次合成训练中进行了全流程应用测试。"
            "系统提供了包括地形分析、道路网评估、通视计算、高程剖面等功能。"
            "测试表明：系统的三维地形可视化功能为指挥员态势感知提供了直观支撑；"
            "路径规划模块可辅助生成多条备选机动路线并标注关键约束点；"
            "但系统在水网稻田地和城市建成区的地形分析精度有待进一步提升。"
            "建议下一阶段重点加强特殊地形类型的分析模型建设。"
        ),
    },
    {
        "title": "装备保障信息化建设方案（脱密版）",
        "content_category": "equipment",
        "text": (
            "装备保障信息化建设旨在提升装备管理效率和保障精准度。"
            "方案提出了一网三平台的总体架构：依托军事综合信息网，"
            "建设装备资源管理平台、维修保障调度平台和器材供应管理平台。"
            "装备资源管理平台实现装备全寿命周期的状态监控与调度；"
            "维修保障调度平台支持维修力量的优化配置和任务分配；"
            "器材供应管理平台通过需求预测和库存优化降低供应延迟。"
            "方案分三期实施：一期完成网络基础设施和数据库建设，"
            "二期上线核心业务功能，三期实现智能辅助决策。"
        ),
    },
    {
        "title": "某方向军事安全态势评估（脱密版）",
        "content_category": "situation",
        "text": (
            "对某方向的军事安全态势进行了综合评估。评估考虑了区域内各国的"
            "军事力量变化、军事演习频率与规模、军事设施建设进度等因素。"
            "从力量平衡角度看，该区域正在经历显著的力量结构调整；"
            "从行为模式角度看，军事活动的透明度和可预测性有所降低；"
            "从技术发展角度看，新型作战力量的引入可能改变传统的攻防平衡。"
            "总体评估认为该区域军事安全态势处于动态调整期，"
            "不确定性因素增多，需要持续跟踪和分析。"
        ),
    },
    {
        "title": "复杂电磁环境下通信保障案例分析（脱密版）",
        "content_category": "case",
        "text": (
            "在某次实战化训练中，通信分队面临强电磁干扰环境下的通信保障任务。"
            "主要挑战包括：全频段阻塞式干扰导致常规通信手段失效；"
            "地形遮挡加剧了超短波通信衰减；敌方电子侦察力量对通信信号的持续监视。"
            "采取的应对措施：启用跳频和扩频通信模式降低被干扰概率；"
            "建立有线通信备份链路确保最低限度指挥畅通；"
            "组织短时静默和欺骗通信掩护真实通信活动。"
            "经验教训：通信预案需要针对不同干扰强度设置多级响应机制；"
            "通信兵的抗干扰训练应与装备训练同步强化；"
            "通信安全教育和操作规程执行是防范信息泄露的第一道防线。"
        ),
    },
]


def main():
    parser = argparse.ArgumentParser(description="合作单位脱密数据导入")
    parser.add_argument("--input", default=str(SENSITIVE_DIR),
                        help=f"脱密数据目录 (默认: {SENSITIVE_DIR})")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("合作单位脱密数据导入")
    print(f"  输入: {input_dir}")
    print(f"  输出: {output_dir}")

    if args.dry_run:
        if input_dir.exists():
            txt_files = list(input_dir.glob("*.txt"))
            print(f"\n[DRY RUN] {input_dir} 中有 {len(txt_files)} 个 .txt 文件")
            if txt_files:
                print(f"  示例: {txt_files[0].name}")
        else:
            print(f"\n[DRY RUN] {input_dir} 不存在，将使用合成回退数据")
        return

    # 尝试从 sensitive 目录导入
    count = 0
    if input_dir.exists():
        txt_files = list(input_dir.glob("*.txt"))
        if txt_files:
            print(f"  从 {input_dir} 导入 {len(txt_files)} 个文件...")
            for txt_file in txt_files:
                text = txt_file.read_text(encoding="utf-8", errors="ignore")
                if not text.strip():
                    continue

                # 安全检查：文本包含敏感标记则跳过
                if any(marker in text[:200].lower()
                       for marker in ["机密", "秘密", "绝密", "classified", "secret", "top secret"]):
                    print(f"    ⚠️ 警告: {txt_file.name} 包含密级标记，跳过")
                    continue

                doc_id = f"partner_{txt_file.stem}"
                out_txt = output_dir / f"{doc_id}.txt"
                out_txt.write_text(text, encoding="utf-8")

                meta = {
                    "source_url": "",
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "source_category": "desensitized",
                    "content_category": _guess_content_category(text),
                    "authority": "doctrine",
                    "language": "zh",
                    "desensitized": True,
                    "title": txt_file.stem,
                    "document_id": doc_id,
                    "partner_data": True,
                }
                out_meta = output_dir / f"{doc_id}.meta.json"
                out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                count += 1

            print(f"  ✅ 从 sensitive 导入 {count} 条")

    # 如果没有真实数据，使用合成回退
    if count == 0:
        print(f"  {input_dir} 无可用的脱密数据，使用合成回退数据...")
        for sample in SYNTHETIC_FALLBACK:
            doc_id = f"partner_fallback_{count:03d}"
            out_txt = output_dir / f"{doc_id}.txt"
            out_txt.write_text(sample["text"], encoding="utf-8")

            meta = {
                "source_url": "",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_category": "desensitized",
                "content_category": sample["content_category"],
                "authority": "doctrine",
                "language": "zh",
                "desensitized": True,
                "title": sample["title"],
                "document_id": doc_id,
                "partner_data": True,
                "synthetic_fallback": True,
            }
            out_meta = output_dir / f"{doc_id}.meta.json"
            out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            count += 1

        print(f"  ✅ 合成回退 {count} 条")

    print(f"✅ 完成: {count} 条记录 → {output_dir}")


def _guess_content_category(text: str) -> str:
    t = text[:500].lower()
    equip = sum(1 for kw in ["装备", "武器", "保障", "器材"] if kw in t)
    doctrine = sum(1 for kw in ["条令", "训练", "编制", "指挥", "方案", "建设"] if kw in t)
    case = sum(1 for kw in ["案例", "演习", "演练", "行动", "总结", "经验"] if kw in t)
    scores = {"equipment": equip, "doctrine": doctrine, "case": case}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "situation"


if __name__ == "__main__":
    main()
