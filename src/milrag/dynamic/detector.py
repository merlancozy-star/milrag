"""src/milrag/dynamic/detector.py — 实时信息需求检测器 D。

论文对应：4.3 实时信息需求检测机制。
    三类内部信号（4.3.1）：s_p 概率 / s_a 注意力 / s_h 熵（趋势增强见 entropy_trend.py）
    综合判别（4.3.2，公式 4-6）：S(t) = w_p·s̃_p + w_a·s̃_a + w_h·s̃_h , Σw=1 ; S(t)>τ 触发
    任务自适应权重（4.3.3，表 4-1）：按问题类型选权重，分类器置信度低时退回均值
    触发阈值（4.3.4）：τ=0.62，验证集 PR 曲线 F1 最大点

约束：白盒路径。s_p/s_a 依赖生成时 logits 与 attention（eager 实现）。
超参全部从 config/dynamic.yaml 读。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .entropy_trend import EntropyTrendTracker, token_entropy


@dataclass
class TriggerDecision:
    triggered: bool
    score: float
    s_p: float
    s_a: float
    s_h: float
    weights: tuple[float, float, float]


class SlidingZScore:
    """对单个信号在最近 W 步做滑窗 z-score 标准化（论文 4.3.2）。"""

    def __init__(self, window: int = 16):
        self.buf: deque[float] = deque(maxlen=window)

    def normalize(self, x: float) -> float:
        self.buf.append(x)
        if len(self.buf) < 2:
            return 0.0
        arr = np.asarray(self.buf)
        std = arr.std()
        return float((x - arr.mean()) / std) if std > 1e-8 else 0.0


# 任务类型 -> 权重键（对齐 config/dynamic.yaml: detector.task_weights）
TASK_KEYS = {"equipment_param", "strategic_status", "adversarial_check"}


class NeedDetector:
    """综合判别函数 S(t)，决定是否触发检索。"""

    def __init__(self, cfg: dict):
        d = cfg["detector"]
        self.threshold: float = d["threshold"]                 # τ = 0.62
        self.task_weights: dict = d["task_weights"]
        self.fallback_weights: dict = d["fallback_weights"]
        self.flatten_gap: float = cfg["classifier"]["flatten_fallback_gap"]  # 0.15
        self.entropy_topk: int = cfg["signals"]["entropy"]["topk_truncate"]
        self.last_layers: int = cfg["signals"]["attention"]["last_layers"]   # L=4
        # 每个信号一个滑窗标准化器
        w = d["zscore_window"]
        self._z = {"p": SlidingZScore(w), "a": SlidingZScore(w), "h": SlidingZScore(w)}
        self._trend = EntropyTrendTracker(
            smoothing_alpha=cfg["signals"]["entropy_trend"]["smoothing_alpha"],
            use_first=cfg["signals"]["entropy_trend"]["first_order"],
            use_second=cfg["signals"]["entropy_trend"]["second_order"],
        )

    # ----- 三类原始信号 -----
    @staticmethod
    def s_p(token_logprob: float) -> float:
        """s_p = -log p(y_t)，公式 4-3。"""
        return -token_logprob

    def s_a(self, attn_to_entities: np.ndarray, token_prob: float) -> float:
        """s_a：对问题中军事实体位置的多头平均依赖 × (1 - p)，公式 4-4。

        attn_to_entities: 最后 L 层、多头平均后、对实体位置 E 的注意力依赖向量。
        """
        if attn_to_entities.size == 0:
            return 0.0
        return float(attn_to_entities.mean() * (1.0 - token_prob))

    def s_h(self, logits: np.ndarray) -> tuple[float, float]:
        """返回 (单点熵, 趋势增强熵信号)。"""
        h = token_entropy(logits, topk_truncate=self.entropy_topk)
        trend = self._trend.update(h)
        return h, trend

    # ----- 权重选择（任务自适应，4.3.3）-----
    def select_weights(self, type_probs: dict[str, float]) -> tuple[float, float, float]:
        ordered = sorted(type_probs.values(), reverse=True)
        if len(ordered) >= 2 and (ordered[0] - ordered[1]) < self.flatten_gap:
            w = self.fallback_weights
        else:
            top_type = max(type_probs, key=type_probs.get)
            w = self.task_weights.get(top_type, self.fallback_weights)
        return w["w_p"], w["w_a"], w["w_h"]

    # ----- 综合判别 S(t) -----
    def step(
        self,
        token_logprob: float,
        token_prob: float,
        logits: np.ndarray,
        attn_to_entities: np.ndarray,
        type_probs: dict[str, float],
    ) -> TriggerDecision:
        sp = self.s_p(token_logprob)
        sa = self.s_a(attn_to_entities, token_prob)
        _, sh = self.s_h(logits)            # 用趋势增强熵作为 s_h（ETC）
        # 滑窗 z-score 标准化
        zp = self._z["p"].normalize(sp)
        za = self._z["a"].normalize(sa)
        zh = self._z["h"].normalize(sh)
        wp, wa, wh = self.select_weights(type_probs)
        score = wp * zp + wa * za + wh * zh
        return TriggerDecision(
            triggered=score > self.threshold,
            score=score, s_p=sp, s_a=sa, s_h=sh, weights=(wp, wa, wh),
        )

    def reset(self) -> None:
        for z in self._z.values():
            z.buf.clear()
        self._trend.reset()


# TODO(Claude Code):
#   - τ 的确定：在验证集上画 PR 曲线（触发召回 vs 触发精确），取 F1 最大点，写回 config（Exp 4-3）。
#   - 信号消融 Exp 4-2：分别只开 s_p / s_a / s_h 及两两组合，复现表 4-3。
#   - 注意：简单事实问题上三信号融合可能误触发，故触发判断与问题类型绑定（论文 4.6.4）。
