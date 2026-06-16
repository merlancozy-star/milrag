"""src/milrag/defense/cluster_filter.py — 阶段一：语义聚类前置过滤。

论文对应：5.3 基于语义聚类的文档过滤机制。
    5.3.2 K-means(K=2) + 离群检测（公式 5-2）
    5.3.3 领域特征增强：实体匹配度 Ent / 关键词匹配度 Kwd / 注入指令检测 Inj
    5.3.4 级联打分 Score_trust（公式 5-3），取 Top-N
    5.3.5 阈值由验证集网格搜索（目标 F1_Robust，公式 5-4）

# Adapted from HuichiZhou/TrustRAG defend_module.{k_mean_filtering, group_n_gram_filtering,
#   similarity_filtering}. 改造点：
#   1. prompt/特征换中文军事域；TrustRAG 阈值(0.88/0.85/0.25)为开放域，军事验证集需重搜。
#   2. 增加论文的领域特征加权 Score_trust（TrustRAG 无此项）。
#   3. 阈值/权重进 config/defense.yaml，不沿用 TrustRAG 硬编码当结论。

超参从 config/defense.yaml: cluster_filter 读。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FilterResult:
    kept_ids: list[str]
    kept_contents: list[str]
    removed_ids: list[str]
    scores: dict[str, float]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def outlier_scores(embeddings: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """离群得分（公式 5-2）：标准化的「到所属簇中心距离」。"""
    dist = np.array([np.linalg.norm(embeddings[i] - centers[labels[i]]) for i in range(len(embeddings))])
    mu, sigma = dist.mean(), dist.std() + 1e-12
    return (dist - mu) / sigma


# ----- 领域特征（5.3.3）-----
def entity_match(content: str, entity_dict: set[str]) -> float:
    """Ent(d) = 实体命中数 / 文档长度（按字符近似）。"""
    if not content:
        return 0.0
    hits = sum(content.count(e) for e in entity_dict)
    return hits / max(len(content), 1)


def keyword_match(content: str, query_content_words: set[str]) -> float:
    """Kwd(d) = 与查询实词的重叠度。"""
    if not query_content_words:
        return 0.0
    return sum(1 for w in query_content_words if w in content) / len(query_content_words)


def injection_score(content: str, patterns: list[str]) -> float:
    """Inj(d) ∈ [0,1]：注入指令检测（正则/关键词命中 + 可选轻分类器）。命中越多越接近 1。"""
    hits = sum(1 for p in patterns if p in content.lower())
    # TODO: 叠加一个轻量二分类器输出，与规则命中取 max
    return min(1.0, hits / 2.0)


class ClusterFilter:
    def __init__(self, cfg: dict, entity_dict: set[str]):
        c = cfg["cluster_filter"]
        self.km = c["kmeans"]
        self.theta_out: float = c["outlier_threshold"]          # 1.85
        self.lam = c["score_trust_weights"]                     # λ1..λ4
        self.keep_n: int = c["keep_top_n"]                      # 8
        self.inj_patterns: list[str] = c["injection_patterns"]
        self.entity_dict = entity_dict

    def filter(
        self,
        ids: list[str],
        contents: list[str],
        embeddings: np.ndarray,
        query_content_words: set[str],
    ) -> FilterResult:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        emb = _l2_normalize(StandardScaler().fit_transform(embeddings))
        labels = KMeans(
            n_clusters=self.km["n_clusters"], n_init=self.km["n_init"],
            max_iter=self.km["max_iter"], random_state=self.km["random_state"],
        ).fit_predict(emb)
        centers = np.array([emb[labels == k].mean(axis=0) for k in range(self.km["n_clusters"])])
        outl = outlier_scores(emb, labels, centers)

        scores: dict[str, float] = {}
        for i, did in enumerate(ids):
            o = 1.0 if outl[i] > self.theta_out else 0.0          # 离群标志（也可用连续值）
            ent = entity_match(contents[i], self.entity_dict)
            kwd = keyword_match(contents[i], query_content_words)
            inj = injection_score(contents[i], self.inj_patterns)
            # Score_trust（公式 5-3）：Σλ=1
            scores[did] = (
                self.lam["lambda1_outlier"] * (1 - o)
                + self.lam["lambda2_entity"] * ent
                + self.lam["lambda3_keyword"] * kwd
                + self.lam["lambda4_inject"] * (1 - inj)
            )
        ranked = sorted(ids, key=lambda d: scores[d], reverse=True)
        kept = ranked[: self.keep_n]
        removed = ranked[self.keep_n :]
        id2content = dict(zip(ids, contents))
        return FilterResult(
            kept_ids=kept,
            kept_contents=[id2content[d] for d in kept],
            removed_ids=removed,
            scores=scores,
        )


# TODO(Claude Code):
#   - 接 ngram_filter.py（rouge-L>0.25 去重，对齐 TrustRAG group_n_gram_filtering）。
#   - 阈值/λ 网格搜索（Exp 5-4），目标 F1_Robust；写回 config，θ_out=1.85, λ=(0.45,0.22,0.13,0.20)。
#   - 聚类算法对照 Exp 5-3：K-means vs HDBSCAN/谱/DEC。
#   - 可视化 t-SNE（论文图 5-2）投毒簇/离群。
