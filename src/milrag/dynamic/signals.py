"""dynamic/signals.py — 三类内部信号的底层提取（白盒）。

论文 4.3.1。s_p / s_a / s_h 的"数值定义"见 detector.py；本文件负责从模型 forward
产物里取原料：token logprob、词表 logits、最后 L 层对实体位置的注意力。

约束：必须 eager attention（output_attentions=True），不能与 FlashAttention-2 共存。
实体位置索引 E 来自 data/ner.py（论文 4.3.1：E 为问题中军事实体位置集）。

使用方式：
  1. 生成时每步调用 extract_per_step 提取当前步的 logits/attn/layer_attns
  2. attention_to_entities 从多层注意力中提取对实体位置的依赖向量
"""
from __future__ import annotations

import numpy as np


def attention_to_entities(
    attn_layers: list[np.ndarray],
    entity_positions: list[int],
    last_layers: int = 4,
) -> np.ndarray:
    """取最后 last_layers 层、多头平均后，当前生成位置对实体位置 E 的依赖向量。

    Args:
        attn_layers: 每层的注意力矩阵列表。
                     每层 shape: [batch=1, num_heads, q_len, k_len]
                     当前生成步 q_len=1。
        entity_positions: 问题中军事实体位置的 token 索引列表 E。
        last_layers: 取最后 L 层（默认 4）。

    Returns:
        shape [|E|,] — 对每个实体位置的平均注意力依赖值（跨层+多头平均）。
    """
    if not entity_positions or not attn_layers:
        return np.array([], dtype=np.float32)

    sel = attn_layers[-last_layers:]

    all_vals = []
    for layer_attn in sel:
        # layer_attn: [batch=1, num_heads, 1, k_len]
        # 多头平均
        head_mean = layer_attn[0].mean(axis=0)  # [q_len=1, k_len]
        # 取实体位置的注意值
        entity_vals = head_mean[0, entity_positions]  # [|E|]
        all_vals.append(entity_vals)

    # 跨层平均
    stacked = np.stack(all_vals, axis=0)  # [L, |E|]
    return stacked.mean(axis=0)  # [|E|]


def extract_step_logprob(logits: np.ndarray, token_id: int) -> tuple[float, float]:
    """从当前步 logits 中提取选中 token 的 log-prob 与 prob。

    Args:
        logits: [vocab_size] 原始 logits。
        token_id: 被选中的 token id。

    Returns:
        (log_prob, prob): log(p) 和 p（均已数值稳定处理）。
    """
    # log_softmax
    logits_f = logits.astype(np.float64)
    shifted = logits_f - logits_f.max()
    log_softmax = shifted - np.log(np.exp(shifted).sum() + 1e-12)

    log_prob = float(log_softmax[token_id])
    prob = float(np.exp(np.clip(log_prob, -50, 0)))
    return log_prob, prob


def compute_s_a(
    attn_to_entities: np.ndarray,
    token_prob: float,
) -> float:
    """s_a = mean(attn_to_entities) * (1 - p)，论文公式 4-4。

    Args:
        attn_to_entities: attention_to_entities 的输出 [|E|,]。
        token_prob: 当前 token 概率 p(y_t | ...)。

    Returns:
        s_a 信号值。
    """
    if attn_to_entities.size == 0:
        return 0.0
    return float(attn_to_entities.mean() * (1.0 - token_prob))
