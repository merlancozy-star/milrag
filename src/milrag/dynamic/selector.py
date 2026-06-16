"""dynamic/selector.py — 策略选择器（论文 4.4.4）。

按问题类型选主策略：
  装备参数查询 → 实体增强 (entity_enhance)
  战略态势分析 → 上下文融合 (context_fusion)
  对抗环境验证 → 实词聚焦 (content_word_focus)

检索质量不佳（首条结果相似度 < 0.55）时回退另一策略。
"""
from __future__ import annotations


# 策略↔类型映射
_STRATEGY_MAP = {
    "equipment_param": "entity_enhance",
    "strategic_status": "context_fusion",
    "adversarial_check": "content_word_focus",
}

# 回退顺序：当主策略检索质量差时尝试的策略
_FALLBACK_ORDER = {
    "entity_enhance": "content_word_focus",
    "content_word_focus": "context_fusion",
    "context_fusion": "entity_enhance",
}


class StrategySelector:
    """根据问题类型和检索质量选择重构策略。"""

    def __init__(self, cfg: dict, reformulator):
        """
        Args:
            cfg: config/dynamic.yaml → reformulate.selector 段。
            reformulator: QueryReformulator 实例。
        """
        s = cfg["reformulate"]["selector"]
        self.fallback_threshold: float = s["fallback_when_low_sim"]  # 0.55
        self.by_question_type: bool = s.get("by_question_type", True)
        self.reformulator = reformulator

    def select_and_reformulate(
        self,
        query: str,
        generated: str,
        attn_to_entities: "np.ndarray | None",
        q_type: str,
        top1_similarity: float = 1.0,
        trigger_pos: int = -1,
    ) -> tuple[str, str]:
        """选择策略并执行重构。

        Args:
            query: 原始查询。
            generated: 当前已生成文本。
            attn_to_entities: 对实体位置的注意力（供实体增强/实词聚焦使用）。
            q_type: 问题类型 (equipment_param / strategic_status / adversarial_check)。
            top1_similarity: 首条检索结果的余弦相似度。
            trigger_pos: 触发时的生成位置（供上下文融合使用）。

        Returns:
            (strategy_name, reformulated_query)
        """
        primary = _STRATEGY_MAP.get(q_type, "entity_enhance")

        # 若检索质量太差，尝试回退
        if top1_similarity < self.fallback_threshold:
            primary = _FALLBACK_ORDER.get(primary, "entity_enhance")

        # 执行对应策略
        if primary == "entity_enhance":
            return primary, self.reformulator.entity_enhance(
                query, generated, attn_to_entities
            )
        elif primary == "context_fusion":
            return primary, self.reformulator.context_fusion(
                query, generated, trigger_pos
            )
        else:  # content_word_focus
            return primary, self.reformulator.content_word_focus(
                query, generated, attn_to_entities
            )
