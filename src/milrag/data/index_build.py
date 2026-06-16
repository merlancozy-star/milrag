"""data/index_build.py — 向量化索引构建（论文 3.2.2）。

FAISS HNSW (M=32, efConstruction=200) 稠密检索 + Elasticsearch 稀疏倒排。
元信息（来源/采集时间/权威性等级/脱敏标识）入 PostgreSQL，三方靠 chunk_id 关联。
检索时 EF search 单独调 efSearch（默认 64）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


# ── FAISS 索引 ──────────────────────────────────────────────
def build_faiss(
    chunks: list[dict],
    embedder,
    *,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    save_dir: str = "data/kb",
) -> tuple:
    """构建 FAISS HNSW 索引。

    Args:
        chunks: 分块列表，每块 {"chunk_id": str, "text": str, ...}。
        embedder: 嵌入模型（有 .encode(texts) -> np.ndarray 方法）。
        M: HNSW 连接数。
        ef_construction: 构建时探索因子。
        ef_search: 查询时探索因子。
        save_dir: 索引保存目录。

    Returns:
        (index, chunk_ids, embeddings): FAISS 索引、doc_id 列表、嵌入矩阵。
    """
    import faiss

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    # 编码
    embeddings = embedder.encode(texts)
    if isinstance(embeddings, list):
        embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]

    # HNSW 索引
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)
    index.hnsw.efSearch = ef_search

    # 持久化
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "faiss_hnsw.index"))
    np.save(out / "embeddings.npy", embeddings)
    (out / "chunk_ids.json").write_text(
        json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8"
    )

    return index, chunk_ids, embeddings


def load_faiss(index_path: str, ef_search: int = 64) -> tuple:
    """加载 FAISS 索引 + 元信息。"""
    import faiss

    index_dir = Path(index_path)
    index = faiss.read_index(str(index_dir / "faiss_hnsw.index"))
    index.hnsw.efSearch = ef_search
    embeddings = np.load(index_dir / "embeddings.npy")
    chunk_ids = json.loads((index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    return index, chunk_ids, embeddings


# ── ES 索引 ─────────────────────────────────────────────────
def build_es(
    chunks: list[dict],
    *,
    index_name: str = "mil_kb",
    analyzer: str = "ik_smart",
    es_host: str = "http://localhost:9200",
) -> None:
    """构建 Elasticsearch 稀疏倒排索引。

    Args:
        chunks: 分块列表。
        index_name: ES 索引名。
        analyzer: 中文分词器（默认 ik_smart）。
        es_host: ES 地址。
    """
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        raise ImportError("请安装 elasticsearch: pip install elasticsearch>=8.13")

    es = Elasticsearch(es_host)

    # 删除已有索引
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # 创建索引映射
    mapping = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 0,
            "analysis": {"analyzer": {"default": {"type": analyzer}}},
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "text": {"type": "text", "analyzer": analyzer},
                "title": {"type": "text", "analyzer": analyzer},
                "category": {"type": "keyword"},
                "authority": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "desensitized": {"type": "boolean"},
            }
        },
    }
    es.indices.create(index=index_name, body=mapping)

    # 批量索引
    from elasticsearch.helpers import bulk
    actions = [
        {
            "_index": index_name,
            "_id": c["chunk_id"],
            "_source": {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "title": c.get("meta", {}).get("title", ""),
                "category": c.get("meta", {}).get("category", ""),
                "authority": c.get("meta", {}).get("authority", "general_commentary"),
                "timestamp": c.get("meta", {}).get("timestamp", ""),
                "desensitized": c.get("meta", {}).get("desensitized", False),
            },
        }
        for c in chunks
    ]
    bulk(es, actions)
    es.indices.refresh(index=index_name)

    return es


# ── PostgreSQL 元信息 ────────────────────────────────────────
def init_metadata_db(
    chunks: list[dict],
    *,
    db_url: str = "postgresql://localhost:5432/milrag",
) -> None:
    """初始化 PostgreSQL 元信息表并写入分块元数据。

    三方关联：FAISS / ES / PG 靠同一 chunk_id。
    """
    try:
        import psycopg2
    except ImportError:
        raise ImportError("请安装 psycopg2: pip install psycopg2-binary")

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunk_meta (
            chunk_id TEXT PRIMARY KEY,
            source TEXT,
            category TEXT,
            authority TEXT,
            collected_at TEXT,
            desensitized BOOLEAN DEFAULT FALSE,
            token_count INTEGER,
            metadata JSONB
        )
    """)

    for c in chunks:
        meta = c.get("meta", {})
        cur.execute(
            """INSERT INTO chunk_meta (chunk_id, source, category, authority,
               collected_at, desensitized, token_count, metadata)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (chunk_id) DO NOTHING""",
            (
                c["chunk_id"],
                meta.get("source_doc", ""),
                meta.get("category", ""),
                meta.get("authority", "general_commentary"),
                meta.get("timestamp", ""),
                meta.get("desensitized", False),
                len(c["text"]),
                json.dumps(meta, ensure_ascii=False),
            ),
        )
    conn.commit()
    cur.close()
    conn.close()
