#!/usr/bin/env python3
"""批量语义分块流水线。
读取 data/kb/cleaned/ 下的清洗后文本，
调用 chunk.chunk_document_with_meta 进行双语分块（512/64），
输出到 data/kb/chunks.json。

用法: python scripts/run_chunk.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milrag.data.chunk import chunk_document_with_meta


def main():
    cleaned_dir = Path("data/kb/cleaned")
    out_path = Path("data/kb/chunks.json")

    if not cleaned_dir.exists():
        print(f"错误: {cleaned_dir} 不存在，请先运行 scripts/run_clean.py")
        sys.exit(1)

    txt_files = list(cleaned_dir.glob("**/*.txt"))
    print(f"语义分块: 找到 {len(txt_files)} 个清洗后文件")

    all_chunks = []
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue

        # 读取 meta
        meta_path = txt_file.with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        doc_id = txt_file.stem
        chunks = chunk_document_with_meta(
            text, meta=meta, window=512, overlap=64, doc_id=doc_id
        )
        for c in chunks:
            all_chunks.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "meta": c.meta,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ 分块完成: {len(all_chunks)} chunks → {out_path}")


if __name__ == "__main__":
    main()
