"""dynamic/fuse.py — 证据融合（论文 4.5.2）。

三步流水线：
  1. 语义相似度去重：余弦 > 0.85 视为重复，保留权威性高者
  2. 来源可信度加权：条令 > 官方公报 > 主流媒体 > 一般评论
  3. 多版本排序：按时间倒序，在 prompt 中标注版本

# Adapted from TrustRAG similarity_filtering（0.85 去重阈值）。
冲突的进一步处理交给 defense/consistency.py（论文 4.5.2 末）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval.metrics import _cosine

# ── 权威性等级（降序）────────────────────────────────────────
_AUTHORITY_ORDER = [
    "doctrine",            # 条令/法规 — 最高权威
    "official_bulletin",   # 官方公报
    "mainstream_media",    # 主流媒体
    "general_commentary",  # 一般评论
    "unknown",
]
_AUTHORITY_RANK = {a: i for i, a in enumerate(_AUTHORITY_ORDER)}


@dataclass
class EvidenceItem:
    chunk_id: str
    content: str
    embedding: np.ndarray | None = None
    authority: str = "unknown"
    timestamp: str = ""
    similarity: float = 0.0


def _authority_score(auth: str) -> int:
    """越高级越小（排名值越小越好）"""
    return _AUTHORITY_RANK.get(auth, len(_AUTHORITY_ORDER))


def fuse_evidence(
    existing: list[dict],
    new: list[dict],
    cfg: dict,
    embedder=None,
) -> list[dict]:
    """证据融合主流程。

    Args:
        existing: 当前已注入上下文的证据 [{chunk_id, content, embedding, authority, timestamp}, ...]。
        new: 新检索到的证据（同格式）。
        cfg: config/dynamic.yaml → evidence_fusion 段。
        embedder: 嵌入模型（用于计算去重所需的余弦相似度）。

    Returns:
        融合后的证据列表（去重 + 加权排序后）。
    """
    fg = cfg["evidence_fusion"]
    dedup_sim: float = fg["dedup_sim"]              # 0.85
    auth_order: list[str] = fg["authority_order"]
    version_sort: str = fg.get("version_sort", "time_desc")

    # 合并 existing + new
    all_items = []
    seen_ids = set()
    for e in existing + new:
        if e["chunk_id"] not in seen_ids:
            seen_ids.add(e["chunk_id"])
            all_items.append(EvidenceItem(
                chunk_id=e["chunk_id"],
                content=e["content"],
                embedding=np.asarray(e["embedding"]) if e.get("embedding") is not None else None,
                authority=e.get("authority", "unknown"),
                timestamp=e.get("timestamp", ""),
                similarity=e.get("similarity", 0.0),
            ))

    # 步骤 1：语义去重（余弦 > dedup_sim 视为重复）
    deduped = _deduplicate(all_items, threshold=dedup_sim)

    # 步骤 2：权威性加权排序
    deduped.sort(key=lambda x: (_authority_score(x.authority), -x.similarity))

    # 步骤 3：同权威等级内按时间倒序
    if version_sort == "time_desc":
        _sort_by_time_within_authority(deduped)

    # 转回 dict 并标注版本
    result = []
    for item in deduped:
        d = {
            "chunk_id": item.chunk_id,
            "content": item.content,
            "authority": item.authority,
            "timestamp": item.timestamp,
            "similarity": item.similarity,
        }
        if item.timestamp:
            d["version_note"] = f"[来源:{item.authority}, 时间:{item.timestamp}]"
        result.append(d)

    return result


def _deduplicate(items: list[EvidenceItem], threshold: float = 0.85) -> list[EvidenceItem]:
    """余弦去重：保留权威性高者。若无嵌入，按精确内容匹配去重。"""
    if not items:
        return []

    # 若无嵌入，按内容哈希去重
    has_embeddings = any(item.embedding is not None for item in items)
    if not has_embeddings:
        seen_content: set[str] = set()
        deduped = []
        for item in items:
            key = item.content[:200]  # 前 200 字指纹
            if key not in seen_content:
                seen_content.add(key)
                deduped.append(item)
        return deduped

    kept = []
    for item in items:
        is_dup = False
        for k in kept:
            if k.embedding is not None and item.embedding is not None:
                sim = _cosine(item.embedding, k.embedding)
                if sim > threshold:
                    # 重复 → 保留权威性高者
                    is_dup = True
                    if _authority_score(item.authority) < _authority_score(k.authority):
                        # 当前更权威，替换
                        kept[kept.index(k)] = item
                    break
        if not is_dup:
            kept.append(item)

    return kept


def _sort_by_time_within_authority(items: list[EvidenceItem]) -> None:
    """同一权威等级内按时间倒序排列（新版本在前）。"""
    # group by authority rank
    from itertools import groupby
    items.sort(key=lambda x: _authority_score(x.authority))
    result = []
    for _, group in groupby(items, key=lambda x: _authority_score(x.authority)):
        group_list = list(group)
        group_list.sort(key=lambda x: x.timestamp if x.timestamp else "", reverse=True)
        result.extend(group_list)
    items[:] = result
