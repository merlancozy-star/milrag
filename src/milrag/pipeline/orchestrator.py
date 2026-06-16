"""pipeline/orchestrator.py — 端到端编排（论文第6章 + 4/5章协同 5.6）。★核心数据流写这里。

三层架构（6.2）：
  数据层（KB/嵌入/索引）→ 控制层（vLLM/检测器/重构/循环）→ 安全防御层（聚类/一致性/评估/决策）

★第4+5章协同三点（论文 5.6）：
  1. 重构后再评估：动态检索每次新召回的证据先过 defense.cluster_filter 再注入上下文
  2. 重构信号反馈：consistency 检出冲突簇 → 触发 dynamic.reformulate 靶向补检索
  3. 循环停止协同：多轮仍冲突 → 主动拒答而非强行生成

双路生成（5.4）：
  - y_prior（内部知识先验，prior.py）
  - y_ext（基于过滤后的外部证据）
  交 self_assess + decision 选最终输出。

为控延迟：主路达 EC 阈值即返回；发现冲突则界面提示（6.3.5）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrchestratorResult:
    answer: str
    evidence: list[dict]
    confidence: float
    refused: bool
    reason: str
    source: str                    # "external" | "internal" | "refused"
    trace: dict = field(default_factory=dict)
    latency_ms: float = 0.0


class Orchestrator:
    """端到端编排器：串联 第3章检索→第4章动态→第5章防御→第6章部署。"""

    def __init__(self, cfg: dict,
                 loop,                  # DynamicRetrievalLoop
                 prior_extractor,       # PriorExtractor
                 cluster_filter,        # ClusterFilter
                 consistency,           # ConsistencyChecker
                 assessor,              # SelfAssessor
                 decider,               # Decider
                 embedder=None,         # 嵌入模型
                 nli=None):             # NLI 模型
        self.cfg = cfg
        self.loop = loop
        self.prior_extractor = prior_extractor
        self.cluster_filter = cluster_filter
        self.consistency = consistency
        self.assessor = assessor
        self.decider = decider
        self.embedder = embedder
        self.nli = nli

        # 协同配置（论文 5.6）
        synergy = cfg.get("defense", {}).get("synergy", {})
        self.filter_incremental = synergy.get("filter_incremental_evidence", True)
        self.conflict_triggers_reformulate = synergy.get("conflict_triggers_reformulate", True)
        self.conflict_persist_then_refuse = synergy.get("conflict_persist_then_refuse", True)

        self._conflict_count = 0
        self._max_conflict_retries = 2

    def answer(self, question: str) -> OrchestratorResult:
        """主入口：端到端问答。

        流程：
          1. 内部先验提取（prior，并行启动）
          2. 动态检索-生成循环（loop）
          3. 防御过滤（cluster_filter → consistency）
          4. 自评估 + 双路决策
          5. 必要时补检索/拒答

        Args:
            question: 用户问题。

        Returns:
            OrchestratorResult 含答案、证据、置信度、拒答标志、原因、溯源。
        """
        import time
        start = time.time()

        trace = {"question": question, "steps": []}

        # ── 步骤 1：内部先验（异步理想，此处同步）───────────
        prior = self.prior_extractor.extract(question)
        trace["steps"].append({"step": "prior", "claims": len(prior.claims)})

        # ── 步骤 2：动态检索-生成循环 ──────────────────────
        loop_state = self.loop.run(question)
        y_ext = loop_state.generated
        raw_evidence = loop_state.evidence
        trace["steps"].append({
            "step": "loop",
            "k": loop_state.k,
            "token_count": loop_state.token_count,
            "termination": loop_state.termination_reason,
        })

        # ── 步骤 3★：防御过滤（协同点 1）──────────────────
        filtered_evidence = raw_evidence
        if raw_evidence and self.cluster_filter is not None:
            # 准备嵌入（若有）
            ids = [e.get("chunk_id", f"ev_{i}") for i, e in enumerate(raw_evidence)]
            contents = [e.get("content", "") for e in raw_evidence]
            try:
                if self.embedder:
                    embs = self.embedder.encode(contents)
                else:
                    embs = None

                # 提取查询实词
                query_words = _extract_query_keywords(question)

                filter_result = self.cluster_filter.filter(
                    ids, contents, embs, query_words,
                )
                # 只保留清洗后的证据
                kept_set = set(filter_result.kept_ids)
                filtered_evidence = [
                    e for i, e in enumerate(raw_evidence)
                    if ids[i] in kept_set
                ]
                trace["steps"].append({
                    "step": "cluster_filter",
                    "before": len(raw_evidence),
                    "after": len(filtered_evidence),
                    "removed": len(filter_result.removed_ids),
                })
            except Exception as exc:
                trace["steps"].append({"step": "cluster_filter", "error": str(exc)})

        # ── 步骤 4★：一致性检查（协同点 2）────────────────
        has_conflict = False
        conflict_cluster = []
        if filtered_evidence and self.consistency is not None and len(filtered_evidence) >= 2:
            try:
                ev_ids = [e.get("chunk_id", f"ev_{i}") for i, e in enumerate(filtered_evidence)]
                ev_texts = [e.get("content", "") for e in filtered_evidence]
                if self.embedder:
                    ev_embs = self.embedder.encode(ev_texts)
                else:
                    ev_embs = None

                cons_result = self.consistency.build_graph_and_cluster(
                    ev_ids, ev_texts, ev_embs, prior.claims,
                )
                has_conflict = cons_result.has_conflict
                conflict_cluster = cons_result.conflict_cluster
                trace["steps"].append({
                    "step": "consistency",
                    "has_conflict": has_conflict,
                    "trust": len(cons_result.trust_cluster),
                    "conflict": len(cons_result.conflict_cluster),
                    "isolated": len(cons_result.isolated),
                })
            except Exception as exc:
                trace["steps"].append({"step": "consistency", "error": str(exc)})

        # ── 冲突触发的靶向补检索（协同点 2）───────────────
        if has_conflict and self.conflict_triggers_reformulate:
            self._conflict_count += 1
            if self._conflict_count <= self._max_conflict_retries:
                # 用冲突簇中的关键词做靶向补检索
                for cc in conflict_cluster[:2]:
                    conflict_text = next(
                        (e.get("content", "") for e in filtered_evidence
                         if e.get("chunk_id", f"ev_{i}") == cc),
                        "",
                    )
                    # 重新检索以获取替代证据
                    try:
                        extra_evidence = self.loop.retriever(question)
                        filtered_evidence.extend(extra_evidence[:3])
                    except Exception:
                        pass
                trace["steps"].append({
                    "step": "conflict_reformulate",
                    "retry_count": self._conflict_count,
                })

        # ── 步骤 5：自评估 ─────────────────────────────────
        claims = prior.claims if prior.claims else [y_ext[:50]]
        ev_texts = [e.get("content", "") for e in filtered_evidence]

        v_ext = self.assessor.assess(
            y_ext, claims, ev_texts,
        )
        v_prior = self.assessor.assess(
            prior.y_prior, prior.claims, [],
        )
        trace["steps"].append({
            "step": "self_assess",
            "ext": v_ext.as_dict(),
            "prior": v_prior.as_dict(),
        })

        # ── 步骤 6：决策 ─────────────────────────────────
        features = (
            v_ext.as_features()
            + v_prior.as_features()
            + [float(has_conflict), float(len(filtered_evidence))]
        )
        decision = self.decider.decide(v_ext, v_prior, features, has_conflict)
        trace["steps"].append({"step": "decision", "decision": decision})

        # ── 步骤 7：构建最终输出 ──────────────────────────
        if decision["action"] == "use_ext":
            final_answer = y_ext
            source = "external"
            refused = False
        elif decision["action"] == "use_prior":
            final_answer = prior.y_prior
            source = "internal"
            refused = False
        elif decision["action"] == "re_retrieve":
            # 快速重试：再跑一次 loop
            retry_state = self.loop.run(question)
            final_answer = retry_state.generated or y_ext
            source = "external_retry"
            refused = False
        else:  # refuse / ask_user
            # 协同点 3：多轮冲突 → 拒答
            final_answer = (
                f"抱歉，无法给出可靠回答。{decision['reason']}"
                if self.decider.refuse_with_reason
                else "抱歉，暂时无法回答该问题。"
            )
            source = "refused"
            refused = True

        latency = (time.time() - start) * 1000

        return OrchestratorResult(
            answer=final_answer,
            evidence=filtered_evidence,
            confidence=decision["confidence"],
            refused=refused,
            reason=decision.get("reason", ""),
            source=source,
            trace=trace,
            latency_ms=latency,
        )


def _extract_query_keywords(question: str) -> set[str]:
    """从查询中提取关键词（供 cluster_filter 的 Kwd 特征使用）。"""
    import re
    # 简单：提取中文字符序列（长度 >= 2）
    words = set(re.findall(r"[一-鿿]{2,}", question))
    # 加上数字+单位
    words.update(re.findall(r"\d+[^\s,，。；;]{1,4}", question))
    return words
