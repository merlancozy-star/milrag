"""dynamic/classifier.py — 问题类型分类器（论文 4.3.3）。

短文本嵌入 → MLP → 三类概率：equipment_param / strategic_status / adversarial_check。
验证集目标 acc 92.8%。top1-top2 概率差 < 0.15 时由 detector 退回三权重均值。

架构：嵌入（BGE 或可训练短文本编码器）+ 2 层 MLP（256→128→3）+ softmax。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ── 军事关键词提示（辅助分类，不替代 MLP）───────────────────
_EQUIP_KEYWORDS = [
    "参数", "性能", "指标", "重量", "速度", "射程", "口径", "载荷",
    "雷达截面", "推力", "油耗", "载弹量", "编制", "数量", "装备",
]
_STRATEGIC_KEYWORDS = [
    "态势", "部署", "动向", "意图", "关系", "联盟", "演习", "对峙",
    "战略", "战术", "评估", "预测", "趋势", "演变", "对峙",
]
_ADVERSARIAL_KEYWORDS = [
    "验证", "可信", "是否", "矛盾", "确认", "据称", "可能", "疑似",
    "来源", "可靠", "质疑", "反驳", "争议",
]


class QuestionTypeClassifier:
    """问题类型三分类器（论文表 4-1）。"""

    LABELS = ["equipment_param", "strategic_status", "adversarial_check"]

    def __init__(self, cfg: dict, embedder=None):
        """
        Args:
            cfg: config/dynamic.yaml → classifier 段。
            embedder: 嵌入模型（用于编码问题文本）。
        """
        c = cfg["classifier"]
        self.flatten_gap = c.get("flatten_fallback_gap", 0.15)
        self.target_acc = c.get("target_acc", 0.928)
        self.embedder = embedder
        self._model: nn.Module | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_model(self, input_dim: int = 1024):
        self._model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        ).to(self._device)

    def load(self, checkpoint_path: str):
        """加载训练好的 MLP 权重。"""
        import torch
        self._build_model()
        state = torch.load(checkpoint_path, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.eval()

    def predict_proba(self, question: str) -> dict[str, float]:
        """返回三类概率字典。

        若模型未加载，回退到关键词规则分类（基于关键词命中率）。
        """
        if self._model is not None and self.embedder is not None:
            return self._mlp_predict(question)
        return self._rule_predict(question)

    def _mlp_predict(self, question: str) -> dict[str, float]:
        """MLP 推理。"""
        import torch
        emb = self.embedder.encode([question])[0]
        x = torch.tensor(emb, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return dict(zip(self.LABELS, probs.astype(float)))

    def _rule_predict(self, question: str) -> dict[str, float]:
        """关键词规则回退（无模型时）。"""
        q = question.lower()

        equip_hits = sum(1 for kw in _EQUIP_KEYWORDS if kw in q)
        strat_hits = sum(1 for kw in _STRATEGIC_KEYWORDS if kw in q)
        adv_hits = sum(1 for kw in _ADVERSARIAL_KEYWORDS if kw in q)

        total = equip_hits + strat_hits + adv_hits
        if total == 0:
            # 无匹配时退回均匀分布
            return {k: 1.0 / 3 for k in self.LABELS}

        # 加平滑
        raw = np.array([equip_hits + 0.1, strat_hits + 0.1, adv_hits + 0.1])
        probs = raw / raw.sum()
        return dict(zip(self.LABELS, probs.astype(float)))

    def predict_type(self, question: str) -> str:
        """预测问题类型（单标签）。"""
        probs = self.predict_proba(question)
        return max(probs, key=probs.get)
