#!/usr/bin/env python3
"""CMNEE 数据集导入。

将已有的 CMNEE（Chinese Military Named Entity Evaluation）数据集
转换为 milrag 的标准 raw 格式，写入 data/raw/cmnee/。

假设 CMNEE 数据以 JSONL 格式提供，每行:
  {"text": "...", "source": "...", "entities": [...], ...}
或纯文本目录，每个文件一个文档。

用法:
  python import_cmnee.py --input /path/to/cmnee/data
  python import_cmnee.py --input /path/to/cmnee --format jsonl
  python import_cmnee.py --input /path/to/cmnee --format txt
"""
from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("data/raw/cmnee")


def detect_format(input_path: Path) -> str:
    """自动检测数据格式。"""
    if input_path.is_file():
        first_line = input_path.read_text(encoding="utf-8", errors="ignore").strip()[:500]
        if first_line.startswith("{") or first_line.startswith("["):
            return "jsonl"
        return "txt"

    # 目录：检查第一个文件
    for child in input_path.iterdir():
        if child.is_file() and child.suffix in (".json", ".jsonl"):
            return "jsonl"
        if child.is_file() and child.suffix in (".txt", ".text", ""):
            return "txt"
    return "txt"


def process_jsonl(input_path: Path) -> int:
    """处理 JSONL 格式 CMNEE 数据。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = OUTPUT_DIR / "checkpoint.json"

    # 恢复进度
    processed = set()
    if checkpoint_file.exists():
        processed = set(json.loads(checkpoint_file.read_text()).get("processed_ids", []))

    count = 0
    files_to_read = [input_path] if input_path.is_file() else sorted(input_path.glob("*.json*"))

    for data_file in files_to_read:
        with open(data_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = record.get("text", "") or record.get("content", "") or record.get("body", "")
                if not text.strip():
                    continue

                doc_id = record.get("id", "") or record.get("doc_id", "") or hashlib.md5(text.encode()).hexdigest()[:12]
                if doc_id in processed:
                    continue

                # 写文本
                txt_path = OUTPUT_DIR / f"cmnee_{doc_id}.txt"
                txt_path.write_text(text, encoding="utf-8")

                # 写元信息
                meta = {
                    "source_url": record.get("source_url", record.get("url", "")),
                    "collected_at": record.get("collected_at",
                                                datetime.now(timezone.utc).isoformat()),
                    "source_category": "military_news",
                    "content_category": _guess_content_category(text),
                    "authority": record.get("authority", "mainstream_media"),
                    "language": "zh",
                    "desensitized": record.get("desensitized", False),
                    "title": record.get("title", ""),
                    "document_id": doc_id,
                    "cmnee_entities": record.get("entities", []),
                }
                meta_path = OUTPUT_DIR / f"cmnee_{doc_id}.meta.json"
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

                processed.add(doc_id)
                count += 1

                if count % 500 == 0:
                    print(f"  已导入 {count} 条...")

    # 保存进度
    checkpoint_file.write_text(json.dumps({"processed_ids": list(processed)}, ensure_ascii=False))

    return count


def process_txt(input_path: Path) -> int:
    """处理纯文本目录格式 CMNEE 数据。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = []
    if input_path.is_file():
        txt_files = [input_path]
    else:
        txt_files = sorted(input_path.glob("**/*.txt")) + sorted(input_path.glob("**/*.text"))

    count = 0
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue

        doc_id = txt_file.stem
        # 写入
        out_txt = OUTPUT_DIR / f"cmnee_{doc_id}.txt"
        out_txt.write_text(text, encoding="utf-8")

        meta = {
            "source_url": "",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "source_category": "military_news",
            "content_category": _guess_content_category(text),
            "authority": "mainstream_media",
            "language": "zh",
            "desensitized": False,
            "title": doc_id,
            "document_id": doc_id,
        }
        out_meta = OUTPUT_DIR / f"cmnee_{doc_id}.meta.json"
        out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        count += 1
        if count % 500 == 0:
            print(f"  已导入 {count} 条...")

    return count


def _guess_content_category(text: str) -> str:
    """根据文本内容猜测内容类别。"""
    text_lower = text.lower()
    equip_score = sum(1 for kw in ["战斗机", "驱逐舰", "坦克", "导弹", "雷达", "潜艇", "航母"] if kw in text_lower)
    doctrine_score = sum(1 for kw in ["条令", "作战原则", "兵力部署", "指挥", "战术"] if kw in text_lower)
    case_score = sum(1 for kw in ["案例", "战例", "演习", "冲突", "军事行动"] if kw in text_lower)

    scores = {"equipment": equip_score, "doctrine": doctrine_score, "case": case_score}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "situation"


def main():
    parser = argparse.ArgumentParser(description="CMNEE 数据集导入")
    parser.add_argument("--input", required=True, help="CMNEE 数据集路径（文件或目录）")
    parser.add_argument("--format", choices=["jsonl", "txt", "auto"], default="auto",
                        help="数据格式 (默认: 自动检测)")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {args.input}")
        raise SystemExit(1)

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(input_path)

    print(f"CMNEE 数据导入")
    print(f"  输入: {input_path}")
    print(f"  格式: {fmt}")
    print(f"  输出: {OUTPUT_DIR}")

    if fmt == "jsonl":
        count = process_jsonl(input_path)
    else:
        count = process_txt(input_path)

    print(f"✅ 导入完成: {count} 条记录")


if __name__ == "__main__":
    main()
