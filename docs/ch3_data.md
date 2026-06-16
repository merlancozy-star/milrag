# 第3章 design doc — 军事情报知识库构建与领域嵌入微调

## 目标
数据与检索表示层的地基。论文 3.1：系统效果取决于知识库质量与嵌入对军事语义的匹配能力。

## 论文小节 ↔ 模块
| 论文 | 模块 |
|---|---|
| 3.2.2 清洗 | data/clean.py |
| 3.2.2 NER(F1 91.2%) | data/ner.py |
| 3.2.2 分块(语义+滑窗512/64) | data/chunk.py |
| 3.2.2 索引(FAISS HNSW M32/efC200 + ES) | data/index_build.py |
| 3.3 QA构造(1276,机器+人工,κ0.82) | data/build_qa.py |
| 5.7.1 投毒变体 | data/build_adversarial.py |
| 3.4 LoRA微调 | retrieval/lora_finetune.py |
| 3.5 混合/重排 | retrieval/hybrid.py, reranker.py |

## 数据规模（验收）
知识库 314,759 段 / 86.4万 token。四类：装备78,349(24.9%)/条令53,472(17.0%)/态势145,826(46.3%)/案例37,112(11.8%)。
QA 1276 = factual510 + reasoning446 + adv320；划分 1020/128/128。

## 关键超参
LoRA r16/α32/dropout0.05，Q/K/V；InfoNCE τ0.05；负样本 批内:BM25难负=1:1；lr1e-4,warmup0.05,cosine,bs128,10ep。
chunk 512/64；FAISS HNSW M32/efC200；dense topk20；rerank top50->top10（仅终轮）。

## 实验与验收（论文表 3-4~3-7）
- Exp3-1：BGE-large-zh R@10 71.4→78.6（主线）；BGE-M3 72.3→79.1
- Exp3-2：LoRA 78.6 vs 全参 79.3（差0.7pt，显存17.8 vs 38.5GB）
- Exp3-6：+CrossEncoder → 81.3
## 局限
数据量~1000条制约微调增益；对笼统/口语化态势查询改善有限。
