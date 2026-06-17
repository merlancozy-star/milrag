#!/usr/bin/env python3
"""QA 数据集构建流水线。
读取 data/kb/chunks.json，用 LLM 标注器生成 1,276 条 QA。

需要: Qwen3-32B (int4) 或 Qwen3-8B 作为标注模型。
如无可用标注模型，自动回退为模板生成基本 QA（用于流水线验证）。

用法: python scripts/run_build_qa.py [--model /path/to/annotator]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_qa_template(chunks: list[dict], output_dir: Path) -> dict:
    """模板回退：从 chunks 中抽取段落构造简单 QA 样本。"""
    import random
    random.seed(42)

    print("  ⚠️ 无可用的 LLM 标注模型，使用模板生成 QA 样本")

    samples = []
    equip_chunks = [c for c in chunks if "equipment" in str(c.get("meta", {}).get("content_category", ""))]
    doctrine_chunks = [c for c in chunks if "doctrine" in str(c.get("meta", {}).get("content_category", ""))]
    all_chunks = equip_chunks + doctrine_chunks
    if not all_chunks:
        all_chunks = chunks

    for i in range(min(500, len(all_chunks))):
        chunk = all_chunks[i]
        text = chunk.get("text", "")[:200]
        chunk_id = chunk.get("chunk_id", f"c_{i}")
        samples.append({
            "id": f"qa_template_{i:05d}",
            "type": "factual",
            "question": f"请简要描述以下内容的核心信息：{text[:80]}...",
            "answer": text,
            "evidence_chunks": [chunk_id],
            "key_reasoning_points": [],
            "adversarial_inject": None,
            "source_meta": {"source": "template_generated"},
        })

    # 8:1:1 split
    random.shuffle(samples)
    n = len(samples)
    split = {
        "train": samples[:int(n * 0.8)],
        "val": samples[int(n * 0.8):int(n * 0.9)],
        "test": samples[int(n * 0.9):],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in split.items():
        (output_dir / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return split


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QA 数据集构建")
    parser.add_argument("--model", default=None,
                        help="标注模型路径 (如 /root/autodl-tmp/models/Qwen3-32B-Instruct)")
    parser.add_argument("--output", default="data/qa",
                        help="输出目录")
    args = parser.parse_args()

    chunks_path = Path("data/kb/chunks.json")
    if not chunks_path.exists():
        # 尝试从原始数据构造
        raw_chunks = []
        for txt_file in Path("data/raw").glob("**/*.txt"):
            if "sensitive" in str(txt_file):
                continue
            raw_chunks.append({
                "chunk_id": txt_file.stem,
                "text": txt_file.read_text(encoding="utf-8", errors="ignore")[:500],
                "meta": {},
            })
        chunks = raw_chunks[:2000]
        print(f"从 data/raw 读取 {len(chunks)} 段落用于 QA 构造")
    else:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        print(f"从 data/kb/chunks.json 读取 {len(chunks)} chunks")

    output_dir = Path(args.output)

    # 尝试用 LLM 标注
    if args.model:
        try:
            from milrag.llm.backbone import Backbone
            print(f"加载标注模型: {args.model}")
            annotator = Backbone(args.model, backend="hf_eager", load_in_4bit=True)
            from milrag.data.build_qa import build_dataset
            split = build_dataset(chunks, annotator, str(output_dir))
            print(f"✅ QA 构造完成: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
            return
        except Exception as e:
            print(f"  LLM 标注失败: {e}")
            print("  回退到模板模式...")

    # 模板回退
    split = build_qa_template(chunks, output_dir)
    print(f"✅ QA 模板生成: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")


if __name__ == "__main__":
    main()
