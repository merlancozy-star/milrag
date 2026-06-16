"""src/milrag/defense/consistency.py — 阶段二：内外知识一致性 / 冲突检测。

论文对应：5.4 基于内外部知识一致性的冲突检测。
    5.4.1 内部知识先验提取（prior.py 产出 y_prior + 结构化事实声明 F_prior）
    5.4.2 一致性判定器（公式 5-5）：c = α·sim_sem + β·nli_score
          nli_score = P(entail) - P(contradict)；默认 α=0.4, β=0.6（NLI 优先）
    5.4.3 证据关系图 + Louvain 社区 -> 可信簇 / 冲突簇 / 孤立证据

# Adapted from HuichiZhou/TrustRAG defend_module.conflict_query（三阶段：内部知识 ->
#   排除操纵性指令/PIA -> 内外消解）。改造点：prompt 换中文军事；增加证据关系图+Louvain
#   社区（TrustRAG 仅两两比对，无图聚类）。

超参从 config/defense.yaml: consistency 读。NLI 模型由外部注入。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import networkx as nx
    import community as community_louvain  # python-louvain
except ImportError:  # pragma: no cover
    nx = None
    community_louvain = None

from eval.metrics import NLIModel, _cosine  # 复用接口与余弦


@dataclass
class ConsistencyResult:
    trust_cluster: list[str]       # 可信簇：互证度高且与先验一致
    conflict_cluster: list[str]    # 冲突簇：互证但与先验/其他簇系统性矛盾（疑似投毒）
    isolated: list[str]            # 孤立证据
    has_conflict: bool


def nli_score(nli: NLIModel, premise: str, hypothesis: str) -> float:
    """P(entail) - P(contradict)，∈ [-1, 1]。正=蕴含，负=矛盾。"""
    p = nli.predict(premise, hypothesis)
    return p["entail"] - p["contradict"]


def pairwise_consistency(
    nli: NLIModel,
    evidence: str,
    prior_claim: str,
    sim_sem: float,
    alpha: float,
    beta: float,
) -> float:
    """综合一致性分 c(d_i, f_j)（公式 5-5）。"""
    return alpha * sim_sem + beta * nli_score(nli, evidence, prior_claim)


class ConsistencyChecker:
    def __init__(self, cfg: dict, nli: NLIModel):
        c = cfg["consistency"]
        self.alpha: float = c["alpha_sim"]      # 0.4
        self.beta: float = c["beta_nli"]        # 0.6
        self.nli = nli

    def build_graph_and_cluster(
        self,
        evidence_ids: list[str],
        evidence_texts: list[str],
        embeddings: np.ndarray,
        prior_claims: list[str],
    ) -> ConsistencyResult:
        """构建证据关系图（边权=证据间 NLI 蕴含度，正=互证/负=冲突），Louvain 社区，
        再按「与内部先验平均一致性」给簇分类。
        """
        assert nx is not None, "需要 networkx + python-louvain"
        g = nx.Graph()
        g.add_nodes_from(evidence_ids)
        n = len(evidence_ids)
        for i in range(n):
            for j in range(i + 1, n):
                w = nli_score(self.nli, evidence_texts[i], evidence_texts[j])
                if abs(w) > 1e-3:
                    g.add_edge(evidence_ids[i], evidence_ids[j], weight=w)

        # Louvain 只能用非负权重 -> 用 |w| 做社区发现，符号另存用于簇分类
        pos_g = nx.Graph()
        pos_g.add_nodes_from(evidence_ids)
        for u, v, d in g.edges(data=True):
            pos_g.add_edge(u, v, weight=abs(d["weight"]))
        partition = community_louvain.best_partition(pos_g) if pos_g.number_of_edges() else {i: idx for idx, i in enumerate(evidence_ids)}

        # 每个簇与内部先验的平均一致性
        id2text = dict(zip(evidence_ids, evidence_texts))
        clusters: dict[int, list[str]] = {}
        for did, cid in partition.items():
            clusters.setdefault(cid, []).append(did)

        trust, conflict, isolated = [], [], []
        for cid, members in clusters.items():
            if len(members) == 1:
                isolated.extend(members)
                continue
            prior_agree = np.mean([
                max(nli_score(self.nli, id2text[m], pc) for pc in prior_claims) if prior_claims else 0.0
                for m in members
            ])
            (trust if prior_agree >= 0 else conflict).extend(members)

        return ConsistencyResult(
            trust_cluster=trust,
            conflict_cluster=conflict,
            isolated=isolated,
            has_conflict=len(conflict) > 0,
        )


# TODO(Claude Code):
#   - 三阶段消解 prompt（中文军事版，对齐 TrustRAG conflict_query）：
#       stage1 不看检索生成内部知识 -> stage2 排除「操纵指令/预设答案/PIA 模式」
#       -> stage3 综合内外知识给最终答案。放到 pipeline/orchestrator.py 的双路。
#   - 一致性判定消融 Exp 5-5：仅 sim / 仅 NLI / 融合。
#   - 冲突簇 -> 触发第 4 章靶向补检索（协同，论文 5.6）。
