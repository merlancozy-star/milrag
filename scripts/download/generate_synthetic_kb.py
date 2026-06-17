#!/usr/bin/env python3
"""Qwen3 合成军事知识库段落生成。

当真实数据源（CMNEE、维基、条令等）不足以达到 314,759 段目标时，
用 Qwen3-8B 生成合成的、不涉密的军事知识段落来填充缺口。

四类内容：装备 / 条令 / 态势 / 案例。
所有生成内容为虚构的训练用伪数据，不引用真实军事机密。

⚠️ 需要 GPU（Qwen3-8B vLLM 推理）。
生成 10,000 段约需 2-3 小时 GPU 时间。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_DIR = Path("data/raw/synthetic_kb")

# 四类合成模板
TEMPLATES = {
    "equipment": {
        "prompt": """生成一段虚构的军事装备技术参数描述（约200字）。

格式要求：
- 装备名称：虚构的型号名称
- 技术参数：虚构的尺寸、重量、速度、射程等物理参数
- 作战能力：虚构的作战用途和功能描述

示例风格：
"XX型多用途战斗机采用双发涡扇发动机，最大起飞重量约30吨，作战半径约1200公里。该机配备有源相控阵雷达和综合航电系统，可携带多种空空和空地武器，具备超视距空战和对地精确打击能力。"

重要：所有数据、型号、参数均为虚构，不引用任何真实武器装备。
仅输出正文，不要加前缀说明。""",
        "min_chars": 100,
        "max_chars": 400,
    },
    "doctrine": {
        "prompt": """生成一段虚构的军事条令原则描述（约200字）。

格式要求：
- 原则名称：虚构的军事原则或条令概念
- 适用场景：虚构的战场或任务场景
- 实施要点：虚构的战术指导原则

示例风格：
"纵深突击原则强调在敌防御体系中快速形成突破并持续向纵深发展。该原则要求指挥员在选定主要突击方向后，集中优势兵力和火力于狭窄正面，以高速连续突击方式摧毁敌防御体系。实施要点包括：准确判断敌防御薄弱点、合理编组突击梯队、确保后续梯队及时跟进、保持火力支援不间断。"

重要：所有原则、场景、要点均为虚构的通用军事概念，不引用任何国家真实条令。
仅输出正文。""",
        "min_chars": 100,
        "max_chars": 400,
    },
    "situation": {
        "prompt": """生成一段虚构的战略态势分析（约200字）。

格式要求：
- 地区：虚构的地理区域名称
- 力量对比：虚构的军事力量部署描述
- 趋势评估：虚构的战略态势走向分析

示例风格：
"在远东某假想区域，近年来主要大国持续加强军事存在。A国在该区域部署了约200架先进战机和两个航母战斗群，B国则通过陆基反舰导弹和潜艇力量形成反介入屏障。分析认为，该区域的战略平衡正在从传统的海空优势向陆海空天网多维制衡转变，未来5年可能出现新的力量格局。"

重要：所有地名、数据、力量描述均为虚构，不涉及任何真实国家的实际部署。
仅输出正文。""",
        "min_chars": 100,
        "max_chars": 400,
    },
    "case": {
        "prompt": """生成一段虚构的军事历史案例分析（约200字）。

格式要求：
- 背景：虚构的冲突或军事行动时间地点
- 关键决策：虚构的指挥官决策要点
- 经验教训：虚构的军事经验总结

示例风格：
"在某次联合军事行动中，指挥官面临着多方向敌情威胁和有限兵力的矛盾。通过将预备队集中于最可能的主要威胁方向，并以佯动牵制次要方向敌军，成功达成了战役目的。此案例表明：准确判断主要威胁方向是兵力部署的前提；佯动需要足够的真实性和规模才能有效欺骗对手；预备队的灵活使用是应对不确定性的关键。"

重要：所有事件、人物、地点均为虚构，不引用任何真实历史战例或军事行动。
仅输出正文。""",
        "min_chars": 100,
        "max_chars": 400,
    },
}


def generate_with_qwen3(prompt: str, backbone, max_new_tokens: int = 256, temperature: float = 0.8) -> str:
    """使用 Qwen3-8B 生成合成段落。"""
    try:
        result = backbone.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return result.strip()
    except Exception as e:
        print(f"    生成失败: {e}")
        return ""


def generate_with_template(template_set: dict) -> str:
    """不依赖 GPU 的模板填充生成（用于 QA 数值验证等简单场景）。"""
    # 这是 LLM 生成的备用，当 GPU 不可用时回退到简单模板
    fillers = {
        "equipment_names": [
            "XX-10A", "BK-200X", "YT-7M", "ZR-15", "DH-33",
            "FN-8B", "GL-21C", "KM-400", "PR-29", "WS-5T",
        ],
        "weights": [25, 28, 30, 32, 35, 37, 40, 42, 45],
        "ranges": [800, 1000, 1200, 1500, 2000, 2500],
        "regions": ["西北某区域", "东方某海峡", "南部某群岛", "北极圈附近", "中部某高原"],
        "years": ["2015", "2018", "2020", "2022", "2024"],
    }
    # 简单填充 — 这只是应急回退
    return f"虚构军事知识段落（模板生成，仅供系统验证用）。该段落不包含任何真实军事情报数据。"


def _content_category_from_template(template_name: str) -> str:
    mapping = {
        "equipment": "equipment",
        "doctrine": "doctrine",
        "situation": "situation",
        "case": "case",
    }
    return mapping.get(template_name, "situation")


def main():
    parser = argparse.ArgumentParser(description="合成军事知识库段落生成")
    parser.add_argument("--count", type=int, default=100,
                        help="每类生成数量 (默认: 100)")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--model-path", default="/models/Qwen3-8B-Instruct",
                        help="Qwen3 模型路径")
    parser.add_argument("--use-vllm", action="store_true",
                        help="使用 vLLM 后端（推荐，吞吐更高）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示将生成的数量，不实际生成")
    parser.add_argument("--template-only", action="store_true",
                        help="仅使用模板（不需要 GPU，但质量低）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = args.count * len(TEMPLATES)
    print(f"合成军事知识库段落生成")
    print(f"  每类: {args.count} 段")
    print(f"  类别数: {len(TEMPLATES)}")
    print(f"  总数: {total}")

    if args.dry_run:
        print("\n[DRY RUN] 预估:")
        print(f"  生成 {total} 段 × ~200 字 = ~{total * 200 // 10000} 万字")
        print(f"  GPU 时间估算: {total / 150:.1f} 小时 (假设 150 段/分钟, Qwen3-8B vLLM)")
        return

    start_time = time.time()

    # 加载模型（如果需要）
    backbone = None
    if not args.template_only:
        try:
            from milrag.llm.backbone import Backbone

            backend = "vllm" if args.use_vllm else "hf_eager"
            print(f"  加载模型: {args.model_path} (backend={backend})...")
            backbone = Backbone(args.model_path, backend=backend)
            print(f"  ✅ 模型加载完成")
        except ImportError:
            print("  ⚠️ 无法导入 milrag.llm.backbone，回退到模板模式")
        except Exception as e:
            print(f"  ⚠️ 模型加载失败: {e}，回退到模板模式")

    total_generated = 0
    for template_name, template_set in TEMPLATES.items():
        print(f"\n  [{template_name}] 生成 {args.count} 段...")

        for i in range(args.count):
            if backbone is not None:
                text = generate_with_qwen3(
                    template_set["prompt"],
                    backbone,
                    max_new_tokens=256,
                    temperature=0.8,
                )
            else:
                text = generate_with_template(template_set)

            if len(text) < template_set["min_chars"]:
                continue

            doc_id = f"synth_{template_name}_{total_generated:06d}"
            txt_path = output_dir / f"{doc_id}.txt"
            txt_path.write_text(text, encoding="utf-8")

            meta = {
                "source_url": "",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_category": "military_news",
                "content_category": _content_category_from_template(template_name),
                "authority": "general_commentary",
                "language": "zh",
                "desensitized": True,
                "title": f"合成{template_name}段落_{total_generated:06d}",
                "document_id": doc_id,
                "synthetic": True,
                "template": template_name,
            }
            meta_path = output_dir / f"{doc_id}.meta.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            total_generated += 1

            if total_generated % 20 == 0:
                elapsed = time.time() - start_time
                rate = total_generated / max(elapsed, 1) * 60
                print(f"    已生成 {total_generated}/{total} ({rate:.0f} 段/分钟)...")

    elapsed = time.time() - start_time
    print(f"\n✅ 完成: {total_generated} 段 → {output_dir}")
    print(f"   耗时: {elapsed / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
