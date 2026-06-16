"""llm/hooks.py — 白盒钩子：采集 logits 与 attention（论文 6.3.3）。

实现：HuggingFace LogitsProcessor + forward hook，每步取下一 token logits 与最后 L 层
注意力矩阵；滑动窗口缓冲做 z-score。

★ transformers 4.40：output_attentions=True 且 attn_implementation="eager"（FA2 不返回 attn）。
ETC 的钩子是 4.30 写的，搬过来要按 4.40 API 改。

约束（CLAUDE.md §5.1）：白盒信号路径必须用 eager attention。
"""
from __future__ import annotations

from typing import Callable

import torch
from transformers import LogitsProcessor


class SignalCaptureProcessor(LogitsProcessor):
    """逐步缓存 logits/logprob/entropy，供 detector 读取。

    用法：传给 model.generate(logits_processor=[processor], ...)
    每步调用后，processor 内部 buffer 更新。
    """

    def __init__(self):
        super().__init__()
        self.buffer: list[dict] = []

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """HF generate 每步回调。

        Args:
            input_ids: 截止当前的 token ids [batch, seq_len]。
            scores: 下一步的原始 logits [batch, vocab_size]。

        Returns:
            未修改的 scores（不改变采样分布）。
        """
        # 保存当前步的 logits（batch=1 取第一个）
        logits = scores[0].detach().float().cpu().numpy()
        self.buffer.append({"logits": logits})
        return scores

    def latest(self) -> dict | None:
        """获取最近一步的信号原料。"""
        return self.buffer[-1] if self.buffer else None

    def clear(self):
        self.buffer.clear()


class AttentionCaptureHook:
    """forward hook 捕获每层 attention 矩阵。

    ★ 只能在 eager attention 下工作（FlashAttention-2 不返回 attention 权重）。
    注册到模型后，生成时每步自动记录 attention。

    用法：
        model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")
        hook = AttentionCaptureHook(model, last_layers=4)
        # 生成...
        attention_data = hook.attention_buffer  # 读取
        hook.remove()
    """

    def __init__(self, model, last_layers: int = 4):
        """
        Args:
            model: HF CausalLM 模型（已设置 output_attentions=True）。
            last_layers: 保留最后 L 层的注意力（默认 4）。
        """
        self.model = model
        self.last_layers = last_layers
        self.attention_buffer: list[list[torch.Tensor]] = []  # [step][layer] -> tensor
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register()

    def _register(self):
        """为每个 transformer 层注册 forward hook。"""
        # 查找模型中的 attention 模块
        try:
            # 路径 1：model.transformer.h[].attn (GPT-2 系)
            layers = self.model.transformer.h
        except AttributeError:
            try:
                # 路径 2：model.model.layers[] (LLaMA/Qwen 系)
                layers = self.model.model.layers
            except AttributeError:
                # 路径 3：遍历所有模块查找 decoder layer
                layers = self._find_decoder_layers()

        if layers:
            total_layers = len(layers)
            start_layer = max(0, total_layers - self.last_layers)
            for i in range(start_layer, total_layers):
                handle = layers[i].register_forward_hook(
                    self._make_hook(i - start_layer)
                )
                self._handles.append(handle)

    def _make_hook(self, buffer_idx: int):
        def hook(module, input, output):
            # output 是 attention 模块的输出，通常是 (hidden_states, attention_weights, ...)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # attention weights
                if attn_weights is not None:
                    # 确保 buffer 有足够的位置
                    while len(self.attention_buffer) <= buffer_idx:
                        self.attention_buffer.append([])
                    self.attention_buffer[buffer_idx].append(
                        attn_weights.detach().float().cpu()
                    )
        return hook

    def _find_decoder_layers(self):
        """自动查找 decoder layers。"""
        for name, module in self.model.named_modules():
            if name.endswith(".layers") or name.endswith(".h"):
                return list(module.children())
        return []

    def get_attention(self) -> list[np.ndarray] | None:
        """获取最新生成步的注意力矩阵列表 [layer] -> np.ndarray。

        Returns:
            最后 L 层的注意力矩阵（numpy），shape 各层不同。
            若尚无数据返回 None。
        """
        if not self.attention_buffer:
            return None
        result = []
        for layer_buf in self.attention_buffer:
            if layer_buf:
                result.append(layer_buf[-1].numpy())  # 取最新一步
        return result if result else None

    def clear(self):
        self.attention_buffer.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def register_attention_hook(model, last_layers: int = 4) -> AttentionCaptureHook:
    """便捷：注册注意力捕获 hook。

    Args:
        model: HF CausalLM（必须 attn_implementation="eager"）。
        last_layers: 保留最后 L 层注意力。

    Returns:
        AttentionCaptureHook 实例（用完记得 .remove()）。
    """
    return AttentionCaptureHook(model, last_layers=last_layers)


import numpy as np  # noqa: E402
