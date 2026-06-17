#!/usr/bin/env python3
"""通过百炼 API 构建嵌入向量，支持断点续传。

用法:
  python scripts/build_embeddings_api.py
  python scripts/build_embeddings_api.py --resume    # 断点续传
  python scripts/build_embeddings_api.py --status     # 查看进度

嵌入模型: text-embedding-v4 (1024d, 百炼最新嵌入, 对应Qwen3-Embedding)
API: 阿里云百炼 (OpenAI 兼容)
预估: 496K chunks, ~8小时, ~37元
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

API_KEY = "sk-6652845ae9d748129811775b2a75aace"
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BATCH_SIZE = 10  # text-embedding-v3 最大 batch size
CHUNKS_PATH = "data/kb/chunks.json"
EMB_PATH = "data/kb/embeddings.npy"
IDS_PATH = "data/kb/chunk_ids.json"
CKPT_PATH = "data/kb/embeddings_checkpoint.npy"


def show_status():
    if not Path(CKPT_PATH).exists():
        total = len(json.loads(open(CHUNKS_PATH, encoding="utf-8").read()))
        print(f"状态: 未开始, 共 {total} chunks 待编码")
        return
    embs = np.load(CKPT_PATH)
    total = len(json.loads(open(CHUNKS_PATH, encoding="utf-8").read()))
    pct = len(embs) / total * 100
    print(f"状态: {len(embs)}/{total} ({pct:.1f}%)")


def build():
    chunks = json.loads(open(CHUNKS_PATH, encoding="utf-8").read())
    total = len(chunks)

    if Path(EMB_PATH).exists():
        embs = np.load(EMB_PATH)
        print(f"✅ 已完成: {len(embs)} embeddings → {EMB_PATH}")
        return

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"][:2000] for c in chunks]

    all_embs = []
    start_idx = 0

    if Path(CKPT_PATH).exists():
        all_embs = list(np.load(CKPT_PATH))
        start_idx = len(all_embs)
        print(f"从断点恢复: {start_idx} embeddings")

    print(f"开始编码: {total} chunks, batch={BATCH_SIZE}")
    print(f"预估: {total // BATCH_SIZE} 次API调用, ~{total / BATCH_SIZE * 0.57 / 3600:.1f}h")
    start_time = time.time()
    errors = 0
    last_save = time.time()

    for i in range(start_idx, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for retry in range(5):
            try:
                resp = client.embeddings.create(
                    model="text-embedding-v4", input=batch, dimensions=1024
                )
                all_embs.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if retry < 4:
                    time.sleep(2**retry)
                else:
                    errors += 1
                    all_embs.extend([[0.0] * 1024] * len(batch))

        # 每 200 批保存 checkpoint 并报告进度
        if (i // BATCH_SIZE) % 200 == 0 and i > start_idx:
            elapsed = time.time() - start_time
            done = len(all_embs)
            rate = (done - start_idx) / max(elapsed, 1)
            eta = (total - done) / max(rate, 0.01) / 60
            print(
                f"  {done}/{total} ({done / total * 100:.1f}%) "
                f"{rate:.0f} emb/s ETA:{eta:.0f}min errors:{errors}"
            )
            np.save(CKPT_PATH, np.array(all_embs, dtype=np.float32))
            last_save = time.time()

        # 每 5 分钟至少保存一次
        if time.time() - last_save > 300:
            np.save(CKPT_PATH, np.array(all_embs, dtype=np.float32))
            last_save = time.time()

    embeddings = np.array(all_embs, dtype=np.float32)
    np.save(EMB_PATH, embeddings)
    Path(IDS_PATH).write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")
    Path(CKPT_PATH).unlink(missing_ok=True)

    elapsed = time.time() - start_time
    print(f"✅ 完成: {len(embeddings)} embeddings, {elapsed / 60:.1f}min, {errors} errors")
    print(f"   嵌入已保存: {EMB_PATH}")


def main():
    parser = argparse.ArgumentParser(description="API 嵌入编码 + FAISS 索引构建")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--status", action="store_true", help="查看进度")
    parser.add_argument("--build-index", action="store_true", help="仅构建 FAISS 索引（嵌入已存在时）")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.build_index:
        _build_faiss_index()
        return

    build()
    _build_faiss_index()


def _build_faiss_index():
    """从已有 embeddings 构建 FAISS HNSW 索引。"""
    import faiss

    emb_path = Path(EMB_PATH)
    ids_path = Path(IDS_PATH)
    if not emb_path.exists():
        print(f"错误: {emb_path} 不存在，请先运行编码")
        sys.exit(1)

    embeddings = np.load(emb_path).astype(np.float32)
    chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))

    dim = embeddings.shape[1]
    M, ef_construction, ef_search = 32, 200, 64

    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)
    index.hnsw.efSearch = ef_search

    save_dir = Path("data/kb")
    save_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_dir / "faiss_hnsw.index"))

    print(f"✅ FAISS HNSW 索引已构建: {len(chunk_ids)} vectors, dim={dim}")


if __name__ == "__main__":
    main()
