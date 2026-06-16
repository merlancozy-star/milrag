"""retrieval/reranker.py — Cross-Encoder 重排序（论文 3.5.1）。

★模型：Qwen3-Reranker-8B（Cross-Encoder，替代 BGE-Reranker-large）。
仅终轮/最终展示启用（+~2.7pt R@10，+65ms/次额外延迟）。
动态检索中间轮不启用——节约 ~65ms/次（CLAUDE.md §6）。

流程：top-50 候选 → Cross-Encoder 逐对打分 → 保留 top-10。
"""
from __future__ import annotations

from typing import Sequence


class Reranker:
    """Cross-Encoder 重排序器（本地离线加载）。"""

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Args:
            model_path: BGE-Reranker 模型本地路径。
            device: 推理设备。
        """
        self.model_path = model_path
        self.device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_path,
                device=self.device,
            )
        except ImportError:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            self._hf_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, local_files_only=True,
            ).to(self.device)
            self._hf_model.eval()
            self._model = "hf"

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        keep: int = 10,
    ) -> list[tuple[str, float]]:
        """重排序 candidate 列表。

        Args:
            query: 查询文本。
            candidates: [(chunk_id, content), ...] 候选列表（top-50）。
            keep: 保留 top-k（默认 10）。

        Returns:
            [(chunk_id, relevance_score), ...] 按相关性降序，保留前 keep。
        """
        self._ensure_loaded()

        if not candidates:
            return []

        ids, contents = zip(*candidates) if candidates else ([], [])

        if self._model == "hf":
            scores = self._hf_rerank(query, list(contents))
        else:
            pairs = [(query, c) for c in contents]
            scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(did, float(score)) for did, score in ranked[:keep]]

    def _hf_rerank(self, query: str, contents: list[str]) -> list[float]:
        """HuggingFace 原生 Cross-Encoder 推理。"""
        import torch
        scores = []
        for content in contents:
            inputs = self._tokenizer(
                query, content,
                truncation=True, max_length=512, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._hf_model(**inputs).logits
                score = float(torch.sigmoid(logits).cpu().numpy().flatten()[0])
            scores.append(score)
        return scores


def rerank(
    query: str,
    candidates: list[str],
    model_path: str = "",
    keep: int = 10,
    reranker: Reranker | None = None,
) -> list[str]:
    """便捷函数：Cross-Encoder 重排序。

    Args:
        query: 查询文本。
        candidates: 候选段落文本列表（top-50）。
        model_path: 模型路径（若 reranker 为 None）。
        keep: 保留 top-k。
        reranker: 已有 Reranker 实例（复用）。

    Returns:
        重排后的段落文本列表（top-keep）。
    """
    if reranker is None and model_path:
        reranker = Reranker(model_path)
    if reranker is None:
        return candidates[:keep]

    # 生成临时 id
    indexed = [(f"tmp_{i}", c) for i, c in enumerate(candidates)]
    ranked = reranker.rerank(query, indexed, keep=keep)
    id2content = dict(indexed)
    return [id2content[did] for did, _ in ranked]
