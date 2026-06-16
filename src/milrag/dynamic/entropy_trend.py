"""src/milrag/dynamic/entropy_trend.py — 熵趋势信号（ETC 增强版 s_h）。

论文对应：4.3.1 输出熵信号 s_h（单点熵）。
增强来源：ETC (WisdomShell/ETC, AAAI'26 Oral)。
    核心洞见：触发不应只看单 token 置信度，而应看 token 级熵的「趋势」——
    一阶差分、二阶差分 + 动态平滑，能更早、更准检测不稳定解码状态，并减少冗余检索。

# Adapted from WisdomShell/ETC (entropy trend modeling). 改造点：
#   1. 中文军事域（停用词/分词换中文方案，不用 spaCy en_core_web_sm）。
#   2. 不替换注意力信号 s_a 与查询重构——它们是本论文增量。
#   3. transformers 4.40 eager attention 路径（ETC 用 4.30，钩子 API 不同）。

接入：本模块产出 trend 信号，作为 detector.py 中 s_h 的趋势版参与加权 S(t)。
所有超参从 config/dynamic.yaml: signals.entropy_trend 读，禁止硬编码。
"""
from __future__ import annotations

from collections import deque

import numpy as np


def token_entropy(logits: np.ndarray, topk_truncate: int = 50, eps: float = 1e-12) -> float:
    """单步输出熵 s_h = -Σ p log p。

    论文公式 4-5。topk 截断：只在概率最高的 top-k token 上算熵，避免长尾噪声污染
    （全词表算熵慢且不稳）。logits 输入，内部走 log_softmax 数值稳定。
    """
    x = logits - logits.max()
    log_p = x - np.log(np.exp(x).sum() + eps)
    p = np.exp(log_p)
    if topk_truncate and topk_truncate < p.shape[-1]:
        idx = np.argpartition(p, -topk_truncate)[-topk_truncate:]
        p, log_p = p[idx], log_p[idx]
        p = p / (p.sum() + eps)
        log_p = np.log(p + eps)
    return float(-(p * log_p).sum())


class EntropyTrendTracker:
    """维护生成过程中的熵序列，输出趋势信号。

    流程（ETC）：
        1. 累积 token 熵序列 H = [h_0, h_1, ...]
        2. 一阶差分 ΔH_t = h_t - h_{t-1}
        3. 二阶差分 Δ²H_t = ΔH_t - ΔH_{t-1}
        4. EMA 动态平滑，抑制单点抖动造成的噪声触发
        5. 趋势信号 = 平滑后的 (一阶 + 二阶) 组合，反映「不确定性是否在急剧上升」
    """

    def __init__(self, smoothing_alpha: float = 0.3, use_first: bool = True, use_second: bool = True):
        self.alpha = smoothing_alpha
        self.use_first = use_first
        self.use_second = use_second
        self._hist: deque[float] = deque(maxlen=4096)
        self._ema: float | None = None

    def update(self, entropy: float) -> float:
        """喂入新一步的熵，返回当前趋势信号（越大=不确定性上升越急）。"""
        self._hist.append(entropy)
        first = self._hist[-1] - self._hist[-2] if len(self._hist) >= 2 else 0.0
        second = 0.0
        if len(self._hist) >= 3:
            prev_first = self._hist[-2] - self._hist[-3]
            second = first - prev_first
        raw = (first if self.use_first else 0.0) + (second if self.use_second else 0.0)
        # EMA 平滑
        self._ema = raw if self._ema is None else self.alpha * raw + (1 - self.alpha) * self._ema
        return self._ema

    @property
    def sequence(self) -> list[float]:
        return list(self._hist)

    def reset(self) -> None:
        self._hist.clear()
        self._ema = None


# TODO(Claude Code):
#   - 单测 tests/test_entropy_trend.py：构造已知 logits 序列，断言一阶/二阶差分与 EMA 正确；
#     断言「熵阶跃上升」时趋势信号显著为正，「熵平稳」时趋势信号 ~0。
#   - 与 ETC 原实现做对照实验（Exp 4-2）：单点 s_h vs 趋势 s_h，验证趋势版触发更及时、N_R 更低。
