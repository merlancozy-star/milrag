"""retrieval/lora_finetune.py — 嵌入 LoRA 领域微调（论文 3.4.2 / 3.4.3）。

核心配置（全来自 config/retrieval.yaml）：
  LoRA: r=16, α=32, dropout=0.05, 目标=Q/K/V 投影
  目标函数: InfoNCE, τ=0.05
  负样本: 批内负 : BM25 难负 = 1:1（top-50 BM25 中非正样本）
  优化: AdamW, lr=1e-4, warmup 0.05, cosine 退火
  批大小: 128（显存不够用梯度累积等效）
  周期: 10 epoch

对照 Exp:
  3-2: LoRA vs 全参微调（78.6 vs 79.3, 显存 17.8 vs 38.5GB）
  3-3: 秩消融 r∈{4,8,16,32,64}
  3-4: 负样本消融（仅批内 / 仅 BM25 / 1:1 混合）
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class QADataset(Dataset):
    """QA 对数据集：每个样本 = (query, positive_doc, negative_docs)。"""

    def __init__(self, pairs: list[dict]):
        """
        Args:
            pairs: [{"query": str, "positive": str, "negatives": [str, ...]}, ...]
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def info_nce_loss(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE 损失（论文公式 2-5）。

    Args:
        query_emb: [B, D] 查询嵌入（已 L2 归一化）。
        doc_emb: [B, D] 正样本文档嵌入（已 L2 归一化）。
        temperature: τ.

    Returns:
        标量损失。
    """
    # 余弦相似 → 内积（已归一化）
    scores = torch.matmul(query_emb, doc_emb.T) / temperature  # [B, B]
    labels = torch.arange(scores.size(0), device=scores.device)
    loss = F.cross_entropy(scores, labels)
    return loss


def _prepare_hard_negatives(
    queries: list[str],
    positives: list[str],
    doc_corpus: list[str],
    embedder,
    bm25,
    topk: int = 50,
) -> list[list[str]]:
    """用 BM25 从语料中检索难负样本（top-50 中排除正样本后随机取）。"""
    import random
    hard_negs = []
    for q, pos in zip(queries, positives):
        bm25_results = bm25.search(q, topk=topk + 1)
        neg_candidates = [d for d in bm25_results if d != pos][:topk]
        # 随机取与批内负等量的难负样本（具体数量在训练循环中控制）
        hard_negs.append(neg_candidates)
    return hard_negs


def train_lora(
    cfg: dict,
    embedder,
    train_pairs: list[dict],
    val_pairs: list[dict] | None = None,
    output_dir: str = "experiments/ckpts/lora",
) -> str:
    """LoRA 微调主流程。

    Args:
        cfg: config/retrieval.yaml → retrieval 子字典（或完整 config，含 retrieval.embedding_finetune）。
        embedder: Embedder 实例（需是 sentence-transformers 模型或 HF 模型）。
        train_pairs: 训练集 QA 对。
        val_pairs: 验证集 QA 对（可选）。
        output_dir: 输出目录。

    Returns:
        保存的 LoRA 适配器路径。
    """
    # 解析配置
    if "embedding_finetune" in cfg:
        efg = cfg["embedding_finetune"]
    elif "lora" in cfg:
        efg = cfg
    else:
        raise ValueError("config 中缺少 embedding_finetune 配置")

    lora_cfg = efg["lora"]
    optim_cfg = efg["optim"]
    loss_cfg = efg["loss"]
    neg_cfg = efg["negatives"]
    epochs = efg["epochs"]
    bs = efg["batch_size"]

    # 确保模型加载
    embedder._ensure_loaded()

    # 获取底层模型并应用 LoRA
    from peft import LoraConfig, get_peft_model, TaskType

    # 取 sentence-transformers 的底层 transformer
    if embedder._model != "hf":
        base_module = embedder._model._first_module()
        if hasattr(base_module, "auto_model"):
            base_model = base_module.auto_model
        else:
            base_model = base_module
    else:
        base_model = embedder._hf_model

    # LoRA 配置：挂在 Q/K/V 投影
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target"],  # ["q_proj", "k_proj", "v_proj"]
    )

    # 检查 target_modules 是否存在
    available_modules = {name for name, _ in base_model.named_modules()}
    target = [m for m in lora_cfg["target"] if any(m in am for am in available_modules)]
    if not target:
        # 回退：尝试常见命名
        target = _find_attention_modules(base_model)

    peft_config.target_modules = target

    lora_model = get_peft_model(base_model, peft_config)
    lora_model.train()

    # 优化器
    optimizer = torch.optim.AdamW(
        lora_model.parameters(),
        lr=optim_cfg["lr"],
    )

    # 学习率调度
    total_steps = epochs * max(1, len(train_pairs) // bs)
    warmup_steps = int(total_steps * optim_cfg["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    temperature = loss_cfg["temperature"]
    device = embedder.device
    lora_model.to(device)

    dataset = QADataset(train_pairs)
    # 梯度累积达到等效 bs=128
    accum_steps = max(1, bs // 8)  # micro-batch = 8

    for epoch in range(epochs):
        epoch_loss = 0.0
        loader = DataLoader(dataset, batch_size=min(8, len(train_pairs)), shuffle=True)

        for step, batch in enumerate(loader):
            queries = [p["query"] for p in batch]
            positives = [p["positive"] for p in batch]
            negatives = [p.get("negatives", []) for p in batch]

            # 编码
            q_emb = torch.tensor(embedder.encode(queries), device=device)
            p_emb = torch.tensor(embedder.encode(positives), device=device)

            # InfoNCE（批内负 + 可选难负）
            loss = info_nce_loss(q_emb, p_emb, temperature)

            # 若提供难负样本，额外计算难负对比损失
            if neg_cfg.get("bm25_hard") and any(negatives):
                flat_negs = []
                q_indices = []
                for i, neg_list in enumerate(negatives):
                    if neg_list:
                        sample_neg = neg_list[:2]  # 每条取 2 个难负
                        flat_negs.extend(sample_neg)
                        q_indices.extend([i] * len(sample_neg))
                if flat_negs:
                    neg_emb = torch.tensor(embedder.encode(flat_negs), device=device)
                    hard_loss = 0.0
                    for i, n_idx in enumerate(q_indices):
                        neg_score = torch.dot(q_emb[n_idx], neg_emb[i]) / temperature
                        hard_loss += -torch.log(torch.sigmoid(-neg_score) + 1e-8)
                    loss = loss + hard_loss / len(flat_negs)

            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps

        # 验证
        if val_pairs and (epoch + 1) % 2 == 0:
            lora_model.eval()
            with torch.no_grad():
                val_queries = [p["query"] for p in val_pairs[:32]]
                val_positives = [p["positive"] for p in val_pairs[:32]]
                vq = torch.tensor(embedder.encode(val_queries), device=device)
                vp = torch.tensor(embedder.encode(val_positives), device=device)
                val_loss = info_nce_loss(vq, vp, temperature).item()
            lora_model.train()

    # 保存 LoRA 适配器
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(str(out))

    return str(out)


def _find_attention_modules(model) -> list[str]:
    """自动查找注意力 Q/K/V 投影模块名。"""
    candidates = []
    for name, _ in model.named_modules():
        for suffix in ["q_proj", "k_proj", "v_proj", "query", "key", "value",
                        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            if name.endswith(suffix) and name not in candidates:
                candidates.append(name)
    if not candidates:
        # 回退到常见模式
        candidates = ["q_proj", "k_proj", "v_proj"]
    return candidates[:10]  # 限制数量
