#!/usr/bin/env python3
"""批量文本清洗流水线。
从 data/raw/ 下所有 .txt 文件读取原始文本，
调用 clean.clean_document_with_meta 根据语言自动配置清洗参数，
输出到 data/kb/cleaned/。

用法: python scripts/run_clean.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milrag.data.clean import clean_document_with_meta


def main():
    raw_dir = Path("data/raw")
    out_dir = Path("data/kb/cleaned")
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(raw_dir.glob("**/*.txt"))
    # 排除 sensitive 目录
    txt_files = [f for f in txt_files if "sensitive" not in str(f)]

    print(f"文本清洗: 找到 {len(txt_files)} 个文件")

    count = 0
    for txt_file in txt_files:
        # 尝试读取对应的 .meta.json
        meta_path = txt_file.with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        raw_text = txt_file.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_document_with_meta(raw_text, meta)

        # 保持相对路径结构
        rel_path = txt_file.relative_to(raw_dir)
        out_path = out_dir / rel_path.with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned, encoding="utf-8")

        count += 1
        if count % 5000 == 0:
            print(f"  已清洗 {count}/{len(txt_files)}...")

    print(f"✅ 清洗完成: {count} 个文件 → {out_dir}")


if __name__ == "__main__":
    main()
