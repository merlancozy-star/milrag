"""defense/prior.py — 内部知识先验提取（论文 5.4.1）。

不看检索结果，引导 LLM 输出初步回答 y_prior + 结构化事实声明 F_prior={f1..fm}，
每条带内部置信度（log-prob + 多次采样一致性，SelfCheckGPT 思路）。

# 对齐 TrustRAG conflict_query stage_one：
  "Generate concise text... else say I don't know"

双路生成中的"内部路"。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


# ── 内部先验提取 prompt（中文军事版）─────────────────────────
_PRIOR_PROMPT = """你是一个军事分析专家。请在不参考任何外部资料的情况下，仅依据你自身的军事知识回答以下问题。

要求：
1. 如果确信，直接给出简洁回答；如果不确定，请明确说"无法确定"
2. 将回答中的关键事实逐条列出，格式为：[事实1] ... [事实2] ...
3. 对每条事实标注置信度（高/中/低）

问题：{question}

请回答："""


@dataclass
class InternalPrior:
    """内部知识先验。"""
    y_prior: str                          # 初步回答
    claims: list[str]                     # 结构化事实声明 F_prior
    confidences: list[float]              # 每条事实的内部置信度 [0,1]
    prior_logprob: float = 0.0            # 回答的整体 log-prob
    source: str = "internal_knowledge"


class PriorExtractor:
    """内部知识先验提取器。"""

    def __init__(self, cfg: dict, backbone):
        """
        Args:
            cfg: config/defense.yaml → prior 段。
            backbone: Backbone 实例（LLM 推理后端）。
        """
        self.cfg = cfg["prior"]
        self.backbone = backbone
        self.generate_without_retrieval = self.cfg.get("generate_without_retrieval", True)
        self.conf_method = self.cfg.get("confidence", "logprob_plus_sampling_consistency")

    def extract(self, question: str) -> InternalPrior:
        """提取内部知识先验。

        Args:
            question: 用户问题。

        Returns:
            InternalPrior 含 y_prior、claims、confidences。
        """
        prompt = _PRIOR_PROMPT.format(question=question)
        response = self.backbone.generate(prompt, temperature=0.3, max_new_tokens=512)

        # 解析回答
        y_prior, claims = self._parse_response(response)

        # 计算置信度（SelfCheckGPT 风格）
        confidences = self._compute_confidences(question, y_prior, claims)

        # 整体 log-prob（如果能获取）
        prior_logprob = self._estimate_logprob(y_prior)

        return InternalPrior(
            y_prior=y_prior,
            claims=claims,
            confidences=confidences,
            prior_logprob=prior_logprob,
        )

    def _parse_response(self, response: str) -> tuple[str, list[str]]:
        """解析 LLM 输出，分离回答与事实声明。"""
        import re

        # 提取 [事实N] 标记的内容
        fact_pattern = re.compile(r"\[事实\d+\]\s*(.+?)(?=\[事实\d+\]|$)", re.DOTALL)
        facts = fact_pattern.findall(response)

        # 清理事实声明
        claims = [f.strip() for f in facts if f.strip()]

        # 移除事实标记后的纯文本回答
        y_prior = re.sub(r"\[事实\d+\].*", "", response, flags=re.DOTALL).strip()
        if not y_prior and claims:
            y_prior = "；".join(claims)

        return y_prior, claims

    def _compute_confidences(
        self,
        question: str,
        y_prior: str,
        claims: list[str],
    ) -> list[float]:
        """SelfCheckGPT 风格的多重采样一致性打分。

        对每个事实声明，采样 N 次并计算与首次声明的一致性（余弦相似度均值）。

        Args:
            question: 原问题。
            y_prior: 初次回答。
            claims: 事实声明列表。

        Returns:
            [confidence_f1, confidence_f2, ...] ∈ [0, 1]。
        """
        if not claims:
            return []

        # 多次采样（2 次额外采样，节约成本）
        n_samples = 2
        extra_responses = []
        for _ in range(n_samples):
            prompt = _PRIOR_PROMPT.format(question=question)
            resp = self.backbone.generate(prompt, temperature=0.7, max_new_tokens=256)
            extra_responses.append(resp)

        confidences = []
        for claim in claims:
            # 检查在额外采样中是否一致出现
            consistency_count = 0
            for resp in extra_responses:
                # 简化：用字符级近似匹配
                if _fuzzy_contains(resp, claim, threshold=0.6):
                    consistency_count += 1
            conf = consistency_count / max(1, n_samples)
            confidences.append(conf)

        return confidences

    def _estimate_logprob(self, text: str) -> float:
        """估算文本的整体 log-prob（若后端支持）。"""
        # vLLM 后端不暴露逐步 logprob，返回占位值
        return 0.0


def _fuzzy_contains(text: str, claim: str, threshold: float = 0.6) -> bool:
    """模糊检查 claim 是否在 text 中有近似匹配。"""
    # 简化：字符集重叠度
    claim_chars = set(claim.replace(" ", ""))
    text_chars = set(text.replace(" ", ""))
    if not claim_chars:
        return False
    overlap = len(claim_chars & text_chars) / len(claim_chars)
    return overlap > threshold


def extract_claims_from_response(response: str) -> list[str]:
    """从 LLM 回答中提取事实声明（辅助函数）。

    按句号/分号切分，过滤过短和纯疑问句。
    """
    import re

    sentences = re.split(r"[。；\n]", response)
    claims = []
    for s in sentences:
        s = s.strip()
        # 过滤：长度 < 5 字或包含问号
        if len(s) < 5 or "?" in s or "？" in s:
            continue
        # 移除列表标记
        s = re.sub(r"^[\d\-\*\.、\)）]\s*", "", s)
        if s:
            claims.append(s)
    return claims
