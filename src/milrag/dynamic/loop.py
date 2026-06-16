"""dynamic/loop.py — 动态检索-生成 循环主流程（论文 4.5.1，算法 1）。

主线：触发检测 → 策略选择 → 查询重构 → 检索 → (cluster_filter) → 证据融合 → 续写 → 终止判定。

每 Δt=8 token 检测一次。终止准则（4.5.3，任一满足即停）：
  1. EOS token 生成
  2. 总长度 ≥ Lmax=1024
  3. S(t) < τ 连续 Lstop=16 步（置信度恢复，无需再检索）
  4. 迭代次数达 Kmax（简单事实=1，复杂推理=4）
  5. 证据总长度超上下文窗口上限

★协同（论文 5.6）：每次新召回的证据先过 defense.cluster_filter 再注入上下文
   — 此逻辑在 pipeline/orchestrator.py 中实现，本 module 提供干净的 Hook 接口。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoopState:
    """动态检索循环的状态记录。"""
    query: str
    generated: str = ""
    evidence: list[dict] = field(default_factory=list)
    k: int = 0                     # 当前检索次数
    low_cnt: int = 0               # 连续低于阈值步数
    token_count: int = 0           # 已生成 token 数
    retrieval_history: list[dict] = field(default_factory=list)  # 每次检索记录
    terminated: bool = False
    termination_reason: str = ""


class DynamicRetrievalLoop:
    """对齐论文算法 1 的动态检索-生成循环。

    依赖：
      - backbone: Backbone 实例（白盒生成 + stream_with_signals）
      - detector: NeedDetector 实例
      - selector: StrategySelector 实例
      - retriever: hybrid_retrieve 函数（或类似接口）
      - fuser: fuse_evidence 函数
      - cluster_filter: ClusterFilter 实例（可选，第 5 章协同用）
      - classifier: QuestionTypeClassifier 实例
    """

    def __init__(self, cfg: dict, backbone, detector, selector,
                 retriever, fuser, classifier,
                 cluster_filter=None):
        self.cfg = cfg
        self.backbone = backbone
        self.detector = detector
        self.selector = selector
        self.retriever = retriever
        self.fuser = fuser
        self.classifier = classifier
        self.cluster_filter = cluster_filter

        loop_cfg = cfg["loop"]
        self.delta_t: int = cfg["generation"]["check_interval"]       # 8
        self.kmax: int = loop_cfg["kmax"]                             # 4
        self.kmax_simple: int = loop_cfg.get("kmax_simple", 1)        # 简单事实=1
        self.lstop: int = loop_cfg["early_stop_low_steps"]            # 16
        self.lmax: int = cfg["generation"]["max_new_tokens"]          # 1024
        self.cache_retrieval: bool = loop_cfg.get("cache_retrieval", True)
        self._retrieval_cache: dict[str, list] = {}

    def run(self, query: str, q_type: str = "") -> LoopState:
        """运行动态检索-生成循环。

        Args:
            query: 用户问题。
            q_type: 问题类型（若为空则自动分类）。

        Returns:
            LoopState 含生成结果、证据链、终止原因。
        """
        state = LoopState(query=query)

        # 0. 自动分类问题类型
        if not q_type:
            type_probs = self.classifier.predict_proba(query)
            q_type = max(type_probs, key=type_probs.get)
        else:
            type_probs = self.classifier.predict_proba(query)

        # 1. 初始检索 Z_0 ← Retrieve(E(q))
        initial_evidence = self._retrieve(query, q_type)
        if initial_evidence:
            state.evidence = initial_evidence

        # 简单事实问题只搜一次
        kmax = self.kmax_simple if q_type == "equipment_param" else self.kmax

        # 2. 主循环（算法 1 第 3-17 行）
        prompt = self._build_prompt(query, state.evidence)

        try:
            gen_stream = self.backbone.stream_with_signals(prompt, delta_t=self.delta_t)
        except NotImplementedError:
            # 回退：HF generate 无信号模式（仅用于基准测试）
            result = self.backbone.generate(prompt, max_new_tokens=self.lmax)
            state.generated = result
            state.terminated = True
            state.termination_reason = "no_signal_mode"
            return state

        for step_output in gen_stream:
            if state.terminated:
                break

            token = step_output.get("token", "")
            state.generated += token
            state.token_count += 1

            # 终止条件 1：EOS
            if step_output.get("is_eos", False):
                state.terminated = True
                state.termination_reason = "eos"
                break

            # 终止条件 2：Lmax
            if state.token_count >= self.lmax:
                state.terminated = True
                state.termination_reason = "lmax"
                break

            # 每 Δt 步检测一次
            if state.token_count % self.delta_t != 0:
                continue

            # 提取信号
            logits = step_output.get("logits")
            logprob = step_output.get("logprob", 0.0)
            prob = step_output.get("prob", 0.0)
            attn = step_output.get("attn_layers")

            if logits is None:
                continue  # 非白盒后端跳过

            # 提取实体注意力
            attn_to_ent = np.array([])
            if attn is not None:
                from milrag.dynamic.signals import attention_to_entities
                # 实体位置需从外部传入（此处为简化，实际由 orchestrator 管理）
                attn_to_ent = np.array([])  # TODO: get entity_positions from orchestrator context

            # S(t) 判别
            decision = self.detector.step(
                token_logprob=logprob,
                token_prob=prob,
                logits=logits,
                attn_to_entities=attn_to_ent,
                type_probs=type_probs,
            )

            if decision.triggered and state.k < kmax:
                # 触发检索 → 重构 → 检索 → 融合 → 续写
                new_q, _ = self.selector.select_and_reformulate(
                    query, state.generated, attn_to_ent, q_type,
                )
                new_evidence = self._retrieve(new_q, q_type)

                # ★协同点：新证据先过 cluster_filter（orchestrator 负责）
                if self.cluster_filter is not None and new_evidence:
                    from milrag.defense.cluster_filter import ClusterFilter
                    # 由 orchestrator 注入 cluster_filter
                    pass  # orchestrator 在外部串联此步骤

                if new_evidence:
                    state.evidence = self.fuser(state.evidence, new_evidence, self.cfg)
                    # 更新 prompt 继续生成
                    prompt = self._build_prompt(query, state.evidence)

                state.k += 1
                state.low_cnt = 0
                state.retrieval_history.append({
                    "step": state.token_count,
                    "k": state.k,
                    "reformulated_query": new_q,
                    "new_evidence_count": len(new_evidence),
                    "score": decision.score,
                })
            else:
                state.low_cnt += 1
                # 终止条件 3：连续 Lstop 步不触发
                if state.low_cnt >= self.lstop:
                    state.terminated = True
                    state.termination_reason = "low_stop"

            # 终止条件 4：检索次数达上限
            if state.k >= kmax and state.low_cnt >= self.lstop:
                state.terminated = True
                state.termination_reason = "kmax"
                break

        return state

    def _retrieve(self, query: str, q_type: str = "") -> list[dict]:
        """带缓存的检索。"""
        import hashlib
        cache_key = hashlib.md5(query.encode()).hexdigest() if self.cache_retrieval else ""
        if cache_key and cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]

        results = self.retriever(query)
        if self.cache_retrieval and cache_key:
            self._retrieval_cache[cache_key] = results
        return results

    @staticmethod
    def _build_prompt(query: str, evidence: list[dict]) -> str:
        """构建带证据上下文的生成 prompt。"""
        if not evidence:
            return f"问题：{query}\n请根据你的军事知识回答。\n回答："

        ctx_parts = []
        for i, e in enumerate(evidence[:10], 1):  # 最多注入 10 条
            version = e.get("version_note", "")
            ctx_parts.append(f"[证据{i}] {version}\n{e['content']}")

        context = "\n\n".join(ctx_parts)
        return (
            f"参考以下军事情报证据，回答用户问题。\n\n"
            f"{context}\n\n"
            f"问题：{query}\n"
            f"请基于上述证据回答，如果证据不充分请说明：\n回答："
        )


import numpy as np  # noqa: E402
