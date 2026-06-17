"""llm/backbone.py — LLM 推理封装（三后端，统一接口）。

★关键架构（CLAUDE.md §5）：
  - 白盒信号路径（第4章逐 token logits/attention）：HF generate + LogitsProcessor + forward hook，
    必须 attn_implementation="eager"。
  - 吞吐/部署路径（基线、第6章）：vLLM。
  - 远程API路径（AutoDL部署）：OpenAI兼容API调用。

generate()       — 标准文本生成（三后端均可）
stream_with_signals() — 逐 Δt token 产出信号数据（仅 HF eager 后端）
"""
from __future__ import annotations

from typing import Iterator


class Backbone:
    """LLM 推理后端统一封装。

    三种后端：
      - "hf_eager": HuggingFace generate + eager attention，支持白盒信号。
      - "vllm":     vLLM 推理引擎，高吞吐但不暴露内部信号。
      - "api":      OpenAI 兼容远程 API（AutoDL 部署 vLLM 服务）。
    """

    def __init__(self, model_path: str, backend: str = "hf_eager",
                 device: str = "cuda", **kwargs):
        """
        Args:
            model_path: 模型本地路径 或 API endpoint URL。
            backend: "hf_eager" | "vllm" | "api".
            device: 推理设备（api 模式忽略）。
            **kwargs:
              api_base:  API 后端地址（backend="api" 时使用）
              api_key:   API 密钥（可选，默认 "not-needed"）
        """
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.kwargs = kwargs
        self._model = None
        self._tokenizer = None
        # API 模式配置
        self._api_base = kwargs.get("api_base", "http://localhost:8000/v1")
        self._api_key = kwargs.get("api_key", "not-needed")

    def _ensure_loaded(self):
        if self._model is not None:
            return

        if self.backend == "vllm":
            self._load_vllm()
        elif self.backend == "api":
            self._load_api()
        else:
            self._load_hf_eager()

    def _load_hf_eager(self):
        """加载 HuggingFace 模型（eager attention，白盒信号路径）。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",               # ★必须 eager
            output_attentions=True,                     # ★需要注意力
            local_files_only=True,
            trust_remote_code=True,
        )
        self._model.eval()

        # 设置 pad_token（若缺失）
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _load_vllm(self):
        """加载 vLLM 推理引擎（高吞吐，无内部信号）。"""
        try:
            from vllm import LLM, SamplingParams
            self._vllm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.kwargs.get("tensor_parallel_size", 1),
                gpu_memory_utilization=self.kwargs.get("gpu_memory_utilization", 0.90),
                max_model_len=self.kwargs.get("max_model_len", 4096),
            )
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True,
            )
            self._model = "vllm"
        except ImportError:
            raise ImportError("vLLM 未安装或不可用，请使用 'hf_eager' 后端")

    def _load_api(self):
        """API 模式：验证连接，不加载模型。"""
        self._model = "api"
        print(f"[backbone] API 模式: {self._api_base}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """标准文本生成（两种后端统一接口）。

        Args:
            prompt: 输入 prompt。
            temperature: 采样温度。
            max_new_tokens: 最大生成 token 数。
            top_p: nucleus sampling。
            **kwargs: 额外参数传给底层 generate。

        Returns:
            生成文本。
        """
        self._ensure_loaded()

        if self.backend == "vllm":
            return self._vllm_generate(prompt, temperature, max_new_tokens, top_p)
        elif self.backend == "api":
            return self._api_generate(prompt, temperature, max_new_tokens, top_p)
        else:
            return self._hf_generate(prompt, temperature, max_new_tokens, top_p, **kwargs)

    def _hf_generate(
        self, prompt: str, temperature: float, max_new_tokens: int, top_p: float, **kwargs
    ) -> str:
        """HuggingFace generate。"""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=top_p,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                **kwargs,
            )
        # 只取新生成的部分
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _vllm_generate(
        self, prompt: str, temperature: float, max_new_tokens: int, top_p: float
    ) -> str:
        """vLLM 生成。"""
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            max_tokens=max_new_tokens,
            top_p=top_p,
        )
        outputs = self._vllm.generate([prompt], params)
        return outputs[0].outputs[0].text if outputs[0].outputs else ""

    def _api_generate(
        self, prompt: str, temperature: float, max_new_tokens: int, top_p: float
    ) -> str:
        """OpenAI 兼容 API 调用。百炼优先用 chat 端点。"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        client = OpenAI(base_url=self._api_base, api_key=self._api_key)
        model = self.model_path if "/" in str(self.model_path) else "default"

        # 百炼/DashScope 优先用 chat 端点（qwen 是 chat 模型）
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return resp.choices[0].message.content if resp.choices else ""
        except Exception:
            # 回退 completions 端点（vLLM 等自部署场景）
            resp = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return resp.choices[0].text if resp.choices else ""

    def stream_with_signals(
        self,
        prompt: str,
        delta_t: int = 8,
        temperature: float = 0.3,
        max_new_tokens: int = 1024,
    ) -> Iterator[dict]:
        """逐 Δt 步产出 {token, logprob, prob, logits, attn_layers, is_eos}。

        仅 hf_eager 后端支持。vLLM 后端会抛出 NotImplementedError。

        用法（对齐 loop.py）：
            for step in backbone.stream_with_signals(prompt):
                logits = step["logits"]
                # ... detector.step(...)
        """
        self._ensure_loaded()

        if self.backend == "vllm":
            raise NotImplementedError(
                "vLLM 后端不暴露逐步 logits/attention。请使用 hf_eager 后端"
            )

        import torch
        from milrag.llm.hooks import SignalCaptureProcessor, register_attention_hook

        sig_proc = SignalCaptureProcessor()
        attn_hook = register_attention_hook(self._model, last_layers=4)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    logits_processor=[sig_proc],
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            # 后处理：将 buffer 与生成 token 对齐
            full_ids = generated_ids[0]
            new_ids = full_ids[input_len:]

            for step_idx, token_id in enumerate(new_ids):
                token = self._tokenizer.decode([token_id], skip_special_tokens=False)

                output = {
                    "token": token,
                    "token_id": int(token_id),
                    "is_eos": bool(token_id == self._tokenizer.eos_token_id),
                    "logits": None,
                    "logprob": 0.0,
                    "prob": 1.0,
                    "attn_layers": None,
                }

                # 从 buffer 取对应步的 logits
                if step_idx < len(sig_proc.buffer):
                    buf_entry = sig_proc.buffer[step_idx]
                    logits = buf_entry["logits"]
                    output["logits"] = logits
                    # log_softmax
                    logits_f = logits.astype("float64")
                    shifted = logits_f - logits_f.max()
                    log_softmax = shifted - torch.logsumexp(
                        torch.tensor(shifted), dim=-1
                    ).item()
                    log_prob = log_softmax[int(token_id)]
                    output["logprob"] = float(log_prob)
                    output["prob"] = float(torch.exp(torch.tensor(log_prob)).item())

                # 注意力（每 Δt 步才取，节省显存）
                if step_idx % delta_t == 0:
                    attn = attn_hook.get_attention()
                    if attn:
                        output["attn_layers"] = attn

                yield output

        finally:
            attn_hook.remove()
