"""pipeline/rag_base.py — 朴素/进阶 RAG 基线（论文 2.3.2）。

所有提升都相对朴素 RAG 报告。进阶 RAG 含 HyDE（查询改写）+ Cross-Encoder 重排。

另含对照基线（用于 Exp 4-1 / 6-1）：
  - FLARE: Forward-Looking Active Retrieval
  - Self-RAG: Self-Reflective RAG
  - CRAG: Corrective Retrieval Augmented Generation
  - IRCoT: Interleaving Retrieval with Chain-of-Thought
  - DRAGIN: Dynamic Retrieval Augmented Generation

这些基线通过 prompt 模拟实现，不依赖原始代码库。
"""
from __future__ import annotations

# ── 提示模板（中文军事）─────────────────────────────────────

# 朴素 RAG：检索 + 直接生成
_NAIVE_RAG_PROMPT = """参考以下军事情报证据，回答用户问题。

证据：
{evidence}

问题：{question}
请基于上述证据回答："""

# HyDE（Hypothetical Document Embedding）：先假设答案再检索
_HYDE_PROMPT = """你是一个军事分析专家。请根据问题，先写一个假设的参考答案片段（用于辅助检索），
再给出最终答案。

问题：{question}

假设参考："""

# Self-RAG：自我反思式生成与评估
_SELF_RAG_PROMPT = """你是军事分析专家。请在回答时逐段自评。

证据：
{evidence}

问题：{question}

请按以下格式回答：
[检索相关度评判：高/中/低]
[回答]...
[信息支持度：充分/部分/不足]"""

# CRAG：纠正式检索增强
_CRAG_PROMPT = """你是军事分析专家。首先判断检索到的证据是否足以回答问题。
如不足，请说明需要补充哪类信息，并基于现有证据尽力回答。

证据：
{evidence}

问题：{question}

[证据充分性评估]
[回答]"""

# IRCoT：链式思维 + 交错检索
_IRCOT_PROMPT = """你是军事分析专家。请逐步推理，每步判断是否需要额外信息。

证据：
{evidence}

问题：{question}

逐步推理：
步骤1: ...
[是否需要更多信息？是/否]
步骤2: ...
...
最终回答："""

# DRAGIN：动态检索增强生成
_DRAGIN_PROMPT = """你是军事分析专家。请在生成过程中标注不确定点，并主动要求补充证据。

证据：
{evidence}

问题：{question}

回答（若不确定，用[需检索:xxx]标注）："""

# FLARE：前瞻性主动检索
_FLARE_PROMPT = """你是军事分析专家。请预测回答中可能需要检索的下一句
（用[检索:关键词]标注预测的检索需求）。

证据：
{evidence}

问题：{question}

回答："""


def _build_evidence_text(evidence: list[dict]) -> str:
    """将证据列表格式化为 prompt 文本。"""
    if not evidence:
        return "（无额外证据）"
    parts = []
    for i, e in enumerate(evidence[:10], 1):
        parts.append(f"[{i}] {e.get('content', str(e))}")
    return "\n".join(parts)


class NaiveRAG:
    """朴素 RAG：检索 Top-K 证据 → 拼接 prompt → LLM 生成。"""

    def __init__(self, backbone, retriever):
        self.backbone = backbone
        self.retriever = retriever

    def answer(self, question: str) -> dict:
        evidence = self.retriever(question)
        ev_text = _build_evidence_text(evidence)
        prompt = _NAIVE_RAG_PROMPT.format(evidence=ev_text, question=question)
        response = self.backbone.generate(prompt)
        return {"answer": response, "evidence": evidence}


class AdvancedRAG:
    """进阶 RAG：HyDE 查询改写 + Cross-Encoder 重排。"""

    def __init__(self, backbone, retriever, reranker=None):
        self.backbone = backbone
        self.retriever = retriever
        self.reranker = reranker

    def answer(self, question: str) -> dict:
        # 1. HyDE：用 LLM 生成假设答案，用于改进检索
        hyde_prompt = _HYDE_PROMPT.format(question=question)
        hypothetical = self.backbone.generate(hyde_prompt, max_new_tokens=128)
        # 将假设答案附加到查询
        augmented_query = f"{question}\n{hypothetical[:200]}"
        evidence = self.retriever(augmented_query)

        # 2. 重排序（可选）
        if self.reranker:
            from milrag.retrieval.reranker import rerank
            contents = [e.get("content", str(e)) for e in evidence]
            reranked = rerank(question, contents, keep=10, reranker=self.reranker)
            # 重建 evidence 列表
            ev_map = {e.get("content", str(e)): e for e in evidence}
            evidence = [ev_map.get(c, {"content": c}) for c in reranked]

        ev_text = _build_evidence_text(evidence)
        prompt = _NAIVE_RAG_PROMPT.format(evidence=ev_text, question=question)
        response = self.backbone.generate(prompt)
        return {"answer": response, "evidence": evidence, "hyde_hypothesis": hypothetical}


# ── 对照基线 ──────────────────────────────────────────────────
_BASELINE_PROMPTS = {
    "flare": _FLARE_PROMPT,
    "self_rag": _SELF_RAG_PROMPT,
    "crag": _CRAG_PROMPT,
    "ircot": _IRCOT_PROMPT,
    "dragin": _DRAGIN_PROMPT,
}


class BaselineRAG:
    """对照基线封装（FLARE / Self-RAG / CRAG / IRCoT / DRAGIN）。

    来源标注：prompt 从对应论文的设计理念重实现为中文军事版。
    """

    def __init__(self, backbone, retriever, baseline_type: str):
        if baseline_type not in _BASELINE_PROMPTS:
            raise ValueError(f"未知基线类型: {baseline_type}，可选: {list(_BASELINE_PROMPTS)}")
        self.baseline_type = baseline_type
        self.backbone = backbone
        self.retriever = retriever
        self.prompt_template = _BASELINE_PROMPTS[baseline_type]

    def answer(self, question: str) -> dict:
        evidence = self.retriever(question)
        ev_text = _build_evidence_text(evidence)
        prompt = self.prompt_template.format(evidence=ev_text, question=question)
        response = self.backbone.generate(prompt)
        return {"answer": response, "evidence": evidence, "baseline": self.baseline_type}
