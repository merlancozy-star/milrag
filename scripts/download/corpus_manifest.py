#!/usr/bin/env python3
"""生成语料清单 corpus_manifest.json。

遍历 data/raw/ 下的所有采集目录，汇总统计信息，
写入 data/raw/corpus_manifest.json（仅元数据，不含语料正文）。
该文件提交到 git，作为实验可追溯性基础。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# 来源类别 → 内容类别 默认映射
SOURCE_TO_CONTENT = {
    "cmnee": "situation",
    "zhwiki_military": "equipment",
    "enwiki_military": "equipment",
    "us_army_fm": "doctrine",
    "us_joint_pubs": "doctrine",
    "declassified": "case",
    "cn_white_papers": "doctrine",
    "cn_doctrine": "doctrine",
    "baidu_baike_military": "equipment",
    "professional_military": "situation",
    "academic_military": "situation",
    "81cn": "situation",
    "mod_gov": "situation",
    "xinhua_military": "situation",
    "international_defense": "situation",
    "synthetic_kb": "situation",  # 按比例分配
    "partner_data": "case",
}

SOURCE_TO_CATEGORY = {
    "cmnee": "military_news",
    "zhwiki_military": "military_news",
    "enwiki_military": "military_news",
    "us_army_fm": "doctrine",
    "us_joint_pubs": "doctrine",
    "declassified": "doctrine",
    "cn_white_papers": "doctrine",
    "cn_doctrine": "doctrine",
    "baidu_baike_military": "encyclopedia",
    "professional_military": "encyclopedia",
    "academic_military": "encyclopedia",
    "81cn": "commentary",
    "mod_gov": "commentary",
    "xinhua_military": "commentary",
    "international_defense": "commentary",
    "synthetic_kb": "military_news",
    "partner_data": "desensitized",
}


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def estimate_paragraphs(text: str) -> int:
    """按双换行估计段落数。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paragraphs)


def estimate_content_category(text: str, source_name: str) -> str:
    """根据文本内容和来源推断内容类别。"""
    text_lower = text.lower()

    # 装备关键词
    equip_keywords = [
        "战斗机", "驱逐舰", "坦克", "导弹", "雷达", "潜艇", "航母",
        "fighter", "destroyer", "tank", "missile", "radar", "submarine",
        "射程", "排水量", "起飞重量", "口径", "发动机",
    ]
    # 条令关键词
    doctrine_keywords = [
        "条令", "作战原则", "指挥控制", "兵力部署", "战术",
        "doctrine", "operations", "command", "tactics", "field manual",
        "joint publication", "作战纲要",
    ]
    # 案例关键词
    case_keywords = [
        "案例", "战例", "冲突", "战争", "军事行动", "演习",
        "case study", "conflict", "war", "operation", "exercise",
        "历史战役",
    ]

    equip_score = sum(1 for kw in equip_keywords if kw in text_lower)
    doctrine_score = sum(1 for kw in doctrine_keywords if kw in text_lower)
    case_score = sum(1 for kw in case_keywords if kw in text_lower)

    scores = {
        "equipment": equip_score,
        "doctrine": doctrine_score,
        "case": case_score,
        "situation": 0,  # 默认
    }

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return SOURCE_TO_CONTENT.get(source_name, "situation")


def build_manifest(raw_dir: Path, output_path: Path) -> dict:
    """扫描 data/raw/ 生成语料清单。"""
    if not raw_dir.exists():
        print(f"错误: {raw_dir} 不存在")
        return {}

    sources = {}
    total_paras = 0
    total_files = 0
    category_counts: dict[str, int] = {}
    content_counts: dict[str, int] = {}
    lang_counts: dict[str, int] = {"zh": 0, "en": 0}

    for src_dir in sorted(raw_dir.iterdir()):
        if not src_dir.is_dir():
            continue
        if src_dir.name == "sensitive":
            continue

        source_name = src_dir.name
        txt_files = list(src_dir.glob("*.txt"))
        if not txt_files:
            continue

        # 从 .meta.json 文件聚合类别统计
        meta_files_sample = list(src_dir.glob("*.meta.json"))[:500]
        source_meta_counts: dict[str, dict[str, int]] = {}  # {content_cat: count, source_cat: count}
        source_lang = "zh"
        for mf in meta_files_sample:
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
                source_lang = meta.get("language", source_lang)
                sc = meta.get("source_category", "military_news")
                cc = meta.get("content_category", "situation")
                source_meta_counts.setdefault(cc, 0)
                source_meta_counts[cc] = source_meta_counts.get(cc, 0) + 1
            except Exception:
                pass

        # 总段落数估算（采样 200 个文件）
        source_paras = 0
        for tf in txt_files[:200]:
            try:
                text = tf.read_text(encoding="utf-8", errors="ignore")
                source_paras += estimate_paragraphs(text)
            except Exception:
                continue

        # 从文件名模式推断内容类别分布（快速，不读文件）
        import re as _re
        cat_pattern = _re.compile(r"synth_(equipment|doctrine|situation|case)_\d+\.txt$")
        content_cat_counts: dict[str, int] = {}

        # 合成 KB 使用已知的生成比例（避免对 300K+ 文件的低效计数）
        if source_name == "synthetic_kb":
            # 生成参数: equipment 25%, doctrine 17%, situation 46.3%, case 11.7%
            synth_total = len(txt_files)
            content_cat_counts = {
                "equipment": int(synth_total * 0.25),
                "doctrine": int(synth_total * 0.17),
                "situation": int(synth_total * 0.463),
                "case": int(synth_total * 0.117),
            }
        else:
            for tf in txt_files:
                m = cat_pattern.match(tf.name)
                if m:
                    cat = m.group(1)
                    content_cat_counts[cat] = content_cat_counts.get(cat, 0) + 1

        if len(txt_files) > 200:
            source_paras = int(source_paras * len(txt_files) / 200)

        # 内容类别：按文件名模式中的比例分配段落数
        total_cat = sum(content_cat_counts.values()) if content_cat_counts else len(txt_files)
        dominant_cat = max(content_cat_counts, key=content_cat_counts.get) if content_cat_counts else "situation"
        source_cat = SOURCE_TO_CATEGORY.get(source_name, "military_news")

        # 按比例分配段落数到各内容类别
        for cat, cat_count in content_cat_counts.items():
            ratio = cat_count / max(total_cat, 1)
            content_counts[cat] = content_counts.get(cat, 0) + int(source_paras * ratio)

        sources[source_name] = {
            "files": len(txt_files),
            "paragraphs_est": source_paras,
            "lang": source_lang,
            "source_category": source_cat,
            "dominant_content_category": dominant_cat,
        }

        total_paras += source_paras
        total_files += len(txt_files)
        category_counts[source_cat] = category_counts.get(source_cat, 0) + source_paras
        lang_counts[source_lang] = lang_counts.get(source_lang, 0) + source_paras

    # 计算百分比
    total = max(total_paras, 1)
    category_pct = {k: round(v / total * 100, 1) for k, v in category_counts.items()}
    content_pct = {k: round(v / total * 100, 1) for k, v in content_counts.items()}
    lang_pct = {k: round(v / total, 2) for k, v in lang_counts.items()}

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "total_raw_files": total_files,
        "total_estimated_paragraphs": total_paras,
        "sources": sources,
        "source_categories": {k: {"paragraphs_est": v, "pct": category_pct.get(k, 0)}
                              for k, v in category_counts.items()},
        "content_categories": {k: {"paragraphs_est": v, "pct": content_pct.get(k, 0)}
                               for k, v in content_counts.items()},
        "language_distribution": lang_pct,
        "fallbacks_applied": [],
        "sources_unavailable": [],
    }

    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"语料清单已写入 {output_path}")
    print(f"  总文件数: {total_files}")
    print(f"  预估段落数: {total_paras}")
    print(f"  来源类别: {category_pct}")
    print(f"  内容类别: {content_pct}")
    print(f"  语言分布: {lang_pct}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="生成语料清单")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="原始语料目录 (默认: data/raw)")
    parser.add_argument("--output", default="data/raw/corpus_manifest.json",
                        help="输出路径 (默认: data/raw/corpus_manifest.json)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    build_manifest(raw_dir, output_path)


if __name__ == "__main__":
    main()
