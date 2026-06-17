#!/usr/bin/env python3
"""一站式索引构建流水线。
读取 data/kb/chunks.json，构建 FAISS HNSW + ES 倒排 + PG 元信息。

支持两种模式：
  - 本地模式: python scripts/run_index.py
  - API 模式:  python scripts/run_index.py --api-base http://localhost:8001/v1

AutoDL 注意: ES 和 PG 可能不可用，失败不阻断、自动降级为纯 FAISS。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="一站式索引构建")
    parser.add_argument("--api-base", default=None,
                        help="Embedding API 地址 (如 http://localhost:8001/v1)，用于远程编码")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="API 编码批大小 (默认 32)")
    args = parser.parse_args()

    chunks_path = Path("data/kb/chunks.json")
    if not chunks_path.exists():
        print(f"错误: {chunks_path} 不存在，请先运行 scripts/run_chunk.py")
        sys.exit(1)

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"索引构建: {len(chunks)} chunks")

    # ── FAISS（必须）────────────────────────────
    print("\n[1/3] 构建 FAISS HNSW 索引...")
    try:
        from milrag.retrieval.embedding import Embedder
        from milrag.data.index_build import build_faiss

        if args.api_base:
            print(f"  使用 API 模式: {args.api_base}")
            embedder = Embedder(
                "qwen3-embedding-4b",
                backend="api",
                api_base=args.api_base,
                normalize=True,
            )
        else:
            # 本地模式
            try:
                import yaml
                cfg = yaml.safe_load(Path("config/base.yaml").read_text())
                emb_path = cfg.get("models", {}).get("embedding", "/models/Qwen3-Embedding-4B")
            except Exception:
                emb_path = "/root/autodl-tmp/models/Qwen3-Embedding-4B"
            print(f"  使用本地模型: {emb_path}")
            embedder = Embedder(emb_path, device="cuda", max_seq_len=512, normalize=True)

        index, chunk_ids, embeddings = build_faiss(
            chunks, embedder, save_dir="data/kb"
        )
        print(f"  ✅ FAISS: {len(chunk_ids)} vectors, dim={embeddings.shape[1]}")
    except Exception as e:
        print(f"  ❌ FAISS 失败: {e}")
        sys.exit(1)

    # ── ES（可选，失败不阻断）──────────────────
    print("\n[2/3] 构建 ES 倒排索引...")
    try:
        from milrag.data.index_build import build_es
        build_es(chunks, index_name="mil_kb", es_host="http://localhost:9200")
        print("  ✅ ES: 索引完成")
    except Exception as e:
        print(f"  ⚠️ ES 跳过: {e}")
        print("  （混合检索降级为纯 FAISS，不影响实验主体）")

    # ── PG（可选，失败不阻断）──────────────────
    print("\n[3/3] 初始化 PG 元信息...")
    try:
        from milrag.data.index_build import init_metadata_db
        init_metadata_db(chunks)
        print("  ✅ PG: 元信息完成")
    except Exception as e:
        print(f"  ⚠️ PG 跳过: {e}")
        print("  （元信息已保存在 chunk.meta 字段，不影响检索）")

    print(f"\n✅ 索引构建完成 → data/kb/")


if __name__ == "__main__":
    main()
