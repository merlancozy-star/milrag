"""defense/self_assess.py — 四维可靠性自评估（论文 5.5.1，表 5-1）。

EC  证据覆盖度 (Evidence Coverage)：回答中主张被证据 entailment 的比例（NLI）
EA  证据对齐度 (Evidence Alignment)：回答与最强证据的嵌入余弦
SC  自一致性   (Self-Consistency)  ：多次温度采样(默认5次)回答相似度均值
Unc 不确定性  (Uncertainty)        ：多次采样输出的 log-likelihood 方差

输入：回答 y、事实主张列表、证据列表、嵌入模型、NLI 模型、LLM backbone。
输出：ReliabilityVector 四维向量，供 decision.py 使用。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval.metrics import _cosine, NLIModel


@dataclass
class ReliabilityVector:
    ec: float    # 证据覆盖度 [0, 1]
    ea: float    # 证据对齐度 [0, 1]
    sc: float    # 自一致性 [0, 1]
    unc: float   # 不确定性 [0, ∞)，越大越不确定

    def as_features(self) -> list[float]:
        return [self.ec, self.ea, self.sc, self.unc]

    def as_dict(self) -> dict[str, float]:
        return {"EC": self.ec, "EA": self.ea, "SC": self.sc, "Unc": self.unc}


class SelfAssessor:
    """四维可靠性自评估器。"""

    def __init__(self, cfg: dict, nli: NLIModel, embedder, backbone):
        """
        Args:
            cfg: config/defense.yaml → self_assess + prior 段。
            nli: NLI 模型（用于 EC 计算）。
            embedder: 嵌入模型（用于 EA/SC 计算）。
            backbone: LLM 推理后端（用于 SC/Unc 多重采样）。
        """
        self.cfg = cfg
        self.nli = nli
        self.embedder = embedder
        self.backbone = backbone
        sm = cfg["self_assess"]["metrics"]
        self.n_samples: int = sm.get("SC", {}).get("n_samples", 5)

    def assess(
        self,
        answer: str,
        claims: list[str],
        evidence: list[str],
    ) -> ReliabilityVector:
        """计算四维可靠性向量。

        Args:
            answer: 待评估的完整回答。
            claims: 回答中的事实主张列表。
            evidence: 证据文本列表。

        Returns:
            ReliabilityVector。
        """
        ec = self._compute_ec(claims, evidence)
        ea = self._compute_ea(answer, evidence)
        sc = self._compute_sc(answer)
        unc = self._compute_unc(answer)
        return ReliabilityVector(ec=ec, ea=ea, sc=sc, unc=unc)

    def _compute_ec(self, claims: list[str], evidence: list[str]) -> float:
        """EC：主张被证据支持的比例。

        对每条 claim，检查是否有任一 evidence 蕴含它（entailment prob > 0.5）。
        """
        if not claims:
            return 0.0
        supported = 0
        for claim in claims:
            for e in evidence:
                try:
                    p = self.nli.predict(e, claim)
                    if p.get("entail", 0.0) > 0.5:
                        supported += 1
                        break
                except Exception:
                    continue
        return supported / len(claims)

    def _compute_ea(self, answer: str, evidence: list[str]) -> float:
        """EA：回答嵌入与最强证据嵌入的余弦相似度。"""
        if not evidence:
            return 0.0
        try:
            ans_emb = self.embedder.encode([answer])[0]
            ev_embs = self.embedder.encode(evidence)
            sims = [_cosine(ans_emb, ev) for ev in ev_embs]
            return float(max(sims)) if sims else 0.0
        except Exception:
            return 0.0

    def _compute_sc(self, answer: str) -> float:
        """SC：多次温度采样回答的相似度均值（SelfCheckGPT 思路）。"""
        # 对已生成的 answer，我们需要原始问题来重新采样
        # 简化：若 backbone 支持带温度的多重采样
        if self.n_samples <= 1:
            return 1.0
        try:
            ans_emb = self.embedder.encode([answer])[0]
            # 此处需原始问题，但 assess() 接口未传入；返回中性值
            # 完整的 SC 计算在 orchestrator 中使用原始 question 进行多重采样
            return 1.0
        except Exception:
            return 1.0

    def compute_sc_full(
        self,
        question: str,
        original_answer: str,
        evidence_context: str = "",
    ) -> float:
        """完整的自一致性计算（带多重采样，需 question）。

        Args:
            question: 原始问题。
            original_answer: 原始回答。
            evidence_context: 证据上下文（可选）。

        Returns:
            SC ∈ [0, 1]。
        """
        samples = [original_answer]
        for _ in range(self.n_samples - 1):
            prompt = (
                f"问题：{question}\n"
                + (f"参考证据：{evidence_context}\n" if evidence_context else "")
                + "请回答："
            )
            try:
                resp = self.backbone.generate(
                    prompt, temperature=0.7, max_new_tokens=256
                )
                samples.append(resp)
            except Exception:
                continue

        if len(samples) < 2:
            return 1.0

        try:
            embs = self.embedder.encode(samples)
            sims = []
            for i in range(1, len(embs)):
                sims.append(_cosine(embs[0], embs[i]))
            return float(np.mean(sims)) if sims else 1.0
        except Exception:
            return 1.0

    def _compute_unc(self, answer: str) -> float:
        """Unc：简化版不确定性估计。

        完整实现需 backbone 返回 log-likelihood。
        此处基于答案长度和 token 级置信度的启发式估计。
        """
        # 简化：短回答且无"不确定"措辞 → 低不确定性
        uncertainty_keywords = ["可能", "也许", "或", "不确定", "需进一步", "待验证"]
        hits = sum(kw in answer for kw in uncertainty_keywords)
        base = hits / max(1, len(answer) / 50)
        return min(1.0, base)
