"""retrieval/embedding.py — 嵌入模型封装（论文 3.4.1）。

★主线：Qwen3-Embedding-4B（1024d，8192 ctx）。对照：Qwen3-Embedding-8B、BGE-large-zh、BGE-M3、E5-large-zh。
支持 query / document 双编码器 + LoRA 适配器加载。
完全离线加载（local_files_only=True）。
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


class Embedder:
    """嵌入模型统一封装。

    支持：
      - sentence-transformers 模型
      - HuggingFace 原生模型 + mean pooling
      - LoRA 适配器注入（通过 PEFT / sentence-transformers adapter）
    """

    def __init__(self, model_path: str, lora_path: str | None = None,
                 device: str = "cuda", max_seq_len: int = 512,
                 normalize: bool = True):
        """
        Args:
            model_path: 模型本地路径。
            lora_path: LoRA 适配器路径（可选）。
            device: 推理设备。
            max_seq_len: 最大序列长度。
            normalize: 是否 L2 归一化输出嵌入。
        """
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.max_seq_len = max_seq_len
        self.normalize_output = normalize
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_path,
                device=self.device,
            )
            self._model.max_seq_length = self.max_seq_len
            # 加载 LoRA 适配器（PEFT 格式）
            if self.lora_path and Path(self.lora_path).exists():
                from peft import PeftModel
                # sentence-transformers 内部是 Transformer 包装
                base = self._model._first_module()
                if hasattr(base, "auto_model"):
                    base.auto_model = PeftModel.from_pretrained(
                        base.auto_model, self.lora_path
                    )
        except ImportError:
            # 回退：HuggingFace 原生
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            self._hf_model = AutoModel.from_pretrained(
                self.model_path, local_files_only=True
            ).to(self.device)
            self._hf_model.eval()
            self._model = "hf"  # 标记用 HF 后端

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        instruction: str = "",
        show_progress: bool = False,
    ) -> np.ndarray:
        """编码文本列表为嵌入向量。

        Args:
            texts: 输入文本列表。
            batch_size: 批大小。
            instruction: query 端指令前缀（BGE 风格："为这个句子生成表示以用于检索相关文章："）。
            show_progress: 是否显示进度条。

        Returns:
            [len(texts), dim] float32 数组。
        """
        self._ensure_loaded()

        if instruction and texts:
            texts = [f"{instruction}{t}" for t in texts]

        if self._model == "hf":
            return self._hf_encode(texts, batch_size)
        else:
            emb = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize_output,
            )
            if isinstance(emb, np.ndarray):
                return emb.astype(np.float32)
            return np.asarray(emb, dtype=np.float32)

    def _hf_encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        """HuggingFace 原生 mean pooling 编码。"""
        import torch
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_seq_len, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self._hf_model(**inputs)
            # mean pooling（考虑 attention mask）
            attention_mask = inputs["attention_mask"]
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1)
            if self.normalize_output:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embs.append(pooled.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    @property
    def dim(self) -> int:
        """嵌入维度。"""
        self._ensure_loaded()
        if self._model == "hf":
            return self._hf_model.config.hidden_size
        return self._model.get_sentence_embedding_dimension() or 1024
