#!/usr/bin/env python3
"""批量军事实体识别。
读取 data/kb/cleaned/ 下的清洗后文本，
调用 ner.MilitaryNER 进行实体识别（默认规则模式），
输出实体标注到 data/kb/entities.json。

用法: python scripts/run_ner.py [--mode rule|model]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milrag.data.ner import MilitaryNER


def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量军事实体识别")
    parser.add_argument("--mode", default="rule", choices=["rule", "model"],
                        help="NER 模式: rule(规则) 或 model(BERT-CRF)")
    args = parser.parse_args()

    cleaned_dir = Path("data/kb/cleaned")
    out_path = Path("data/kb/entities.json")

    if not cleaned_dir.exists():
        print(f"错误: {cleaned_dir} 不存在，请先运行 scripts/run_clean.py")
        sys.exit(1)

    ner = MilitaryNER(mode=args.mode)

    txt_files = list(cleaned_dir.glob("**/*.txt"))
    print(f"实体识别 (mode={args.mode}): 找到 {len(txt_files)} 个文件")

    records = []
    for i, txt_file in enumerate(txt_files):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        meta_path = txt_file.with_suffix(".meta.json")
        lang = "zh"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                lang = meta.get("language", "zh")
            except Exception:
                pass

        if lang != "zh":
            # 英文文本跳过中文 NER
            records.append({
                "source": str(txt_file), "entity_count": 0,
                "entities": [], "lang": lang, "ner_skipped": True,
            })
            continue

        entities = ner.extract_entities(text)
        records.append({
            "source": str(txt_file),
            "char_count": len(text),
            "entity_count": len(entities),
            "lang": lang,
            "entities": [
                {"text": e.text, "normalized": e.normalized,
                 "label": e.label, "start": e.start, "end": e.end}
                for e in entities
            ],
        })

        if (i + 1) % 5000 == 0:
            print(f"  已处理 {i+1}/{len(txt_files)}...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total = sum(r["entity_count"] for r in records)
    print(f"✅ NER 完成: {len(records)} 个文档, {total} 个实体 → {out_path}")


if __name__ == "__main__":
    main()
