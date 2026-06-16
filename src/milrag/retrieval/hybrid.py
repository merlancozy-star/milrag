"""retrieval/hybrid.py — 混合检索 RRF（论文 2.2.3 公式 2-6）。

Reciprocal Rank Fusion (RRF)，k=60。
融合稠密检索（FAISS HNSW）与稀疏检索（Elasticsearch BM25 / ik_smart 分词）的结果。
"""
from __future__ import annotations

from collections import OrderedDict


def rrf_fuse(
    dense_ranking: list[str],
    sparse_ranking: list[str],
    k: int = 60,
) -> list[str]:
    """RRF 混合融合（论文公式 2-6）。

    Args:
        dense_ranking: 稠密检索的 doc_id 排序列表（best first）。
        sparse_ranking: 稀疏检索的 doc_id 排序列表（best first）。
        k: RRF 常数，默认 60（论文值）。

    Returns:
        融合后的 doc_id 排序列表。
    """
    scores: dict[str, float] = {}

    for rank, did in enumerate(dense_ranking, start=1):
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)

    for rank, did in enumerate(sparse_ranking, start=1):
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)

    # 按 RRF 分降序排列
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [did for did, _ in ranked]


def dense_retrieve(
    query: str,
    embedder,
    faiss_index,
    chunk_ids: list[str],
    topk: int = 20,
) -> list[tuple[str, float]]:
    """FAISS 稠密检索。

    Returns:
        [(chunk_id, cosine_similarity), ...] 按相似度降序。
    """
    import numpy as np

    q_emb = embedder.encode([query])[0].reshape(1, -1).astype(np.float32)
    distances, indices = faiss_index.search(q_emb, topk)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunk_ids):
            # FAISS HNSW 返回 L2 距离，转近似余弦相似度
            sim = float(1.0 / (1.0 + dist))
            results.append((chunk_ids[idx], sim))
    return results


def sparse_retrieve(
    query: str,
    es_client,
    index_name: str = "mil_kb",
    topk: int = 20,
) -> list[str]:
    """Elasticsearch BM25 稀疏检索。"""
    body = {
        "query": {"match": {"text": query}},
        "size": topk,
        "_source": ["chunk_id"],
    }
    resp = es_client.search(index=index_name, body=body)
    return [hit["_source"]["chunk_id"] for hit in resp["hits"]["hits"]]


def hybrid_retrieve(
    query: str,
    embedder,
    faiss_index,
    chunk_ids: list[str],
    es_client,
    *,
    dense_topk: int = 20,
    sparse_topk: int = 20,
    rrf_k: int = 60,
    es_index: str = "mil_kb",
) -> list[tuple[str, float]]:
    """完整的混合检索流水线：稠密 + 稀疏 → RRF 融合。

    Returns:
        [(chunk_id, rrf_score), ...] 按 RRF 分数降序。
    """
    # 稠密检索
    dense_results = dense_retrieve(query, embedder, faiss_index, chunk_ids, topk=dense_topk)
    dense_ids = [did for did, _ in dense_results]

    # 稀疏检索
    sparse_ids = sparse_retrieve(query, es_client, index_name=es_index, topk=sparse_topk)

    # RRF 融合
    fused_ids = rrf_fuse(dense_ids, sparse_ids, k=rrf_k)

    # 附加分数（从稠密结果复原）
    dense_score_map = dict(dense_results)
    return [(did, dense_score_map.get(did, 0.0)) for did in fused_ids]
