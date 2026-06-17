#!/usr/bin/env python3
"""下载 PLA 公开条令教材。

来源：公开军事教育材料、公开的条令类文献。
这些是已公开发行的军事教育和理论研究材料。

输出: data/raw/cn_doctrine/
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_DIR = Path("data/raw/cn_doctrine")

# 公开可获取的 PLA 条令教材类文献（编号/名称均为公开信息）
DOCTRINE_TEMPLATES = [
    {
        "title": "中国人民解放军共同条令",
        "category": "doctrine",
        "note": "公开版本，已公开发行",
    },
    {
        "title": "中国人民解放军内务条令",
        "category": "doctrine",
        "note": "公开版本",
    },
    {
        "title": "中国人民解放军纪律条令",
        "category": "doctrine",
        "note": "公开版本",
    },
    {
        "title": "中国人民解放军队列条令",
        "category": "doctrine",
        "note": "公开版本",
    },
    {
        "title": "中国人民解放军政治工作条例",
        "category": "doctrine",
        "note": "公开版本",
    },
]


def main():
    parser = argparse.ArgumentParser(description="PLA 公开条令教材下载")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--input-dir", default=None,
                        help="如有本地已准备的条令文本目录，直接导入")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("PLA 公开条令教材下载")
    print(f"  输出: {output_dir}")

    if args.input_dir:
        # 从本地目录导入
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"  错误: 输入目录不存在 {args.input_dir}")
            raise SystemExit(1)

        count = 0
        for txt_file in sorted(input_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue

            doc_id = f"cn_doctrine_{txt_file.stem}"
            out_txt = output_dir / f"{doc_id}.txt"
            out_txt.write_text(text, encoding="utf-8")

            meta = {
                "source_url": "",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "source_category": "doctrine",
                "content_category": "doctrine",
                "authority": "official_bulletin",
                "language": "zh",
                "desensitized": True,
                "title": txt_file.stem,
                "document_id": doc_id,
            }
            out_meta = output_dir / f"{doc_id}.meta.json"
            out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            count += 1

        print(f"✅ 已导入 {count} 份条令文档")
    else:
        # 生成占位索引
        print("  注意: PLA 公开条令教材需要手动获取文本（PDF 转 TXT）后放入此目录")
        print(f"  当前生成了 {len(DOCTRINE_TEMPLATES)} 个条目作为索引参考")
        index_path = output_dir / "doctrine_index.json"
        index_path.write_text(
            json.dumps(DOCTRINE_TEMPLATES, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  索引文件: {index_path}")
        print("  请将对应的 .txt 文件放入此目录后再运行: python scripts/download/run_clean.py")


if __name__ == "__main__":
    main()
