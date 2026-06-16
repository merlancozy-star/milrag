# 第5章 design doc — 对抗环境 两阶段可信增强与噪声过滤

## 威胁模型（论文 5.1）
白盒检索-黑盒生成。关注「进入知识库的数据来源被污染」，非内网入侵。
ρ∈{1,5,10,20}%（曲线到30%）。三策略：直接矛盾/部分替换(最难防)/提示注入。

## 参考实现：TrustRAG（复用边界见 CLAUDE.md §4.2）
k_mean_filtering / similarity_filtering / conflict_query 三阶段。改造：中文军事 prompt；
增加领域特征加权 Score_trust、证据关系图+Louvain、四维自评估、显式拒答（TrustRAG 无）。

## 论文小节 ↔ 模块
| 论文 | 模块 |
|---|---|
| 5.3 聚类前置过滤 | defense/cluster_filter.py + ngram_filter.py + inject_detect.py |
| 5.4.1 内部先验 | defense/prior.py |
| 5.4.2/5.4.3 一致性+图 | defense/consistency.py |
| 5.5.1 四维自评估 | defense/self_assess.py |
| 5.5.2/5.5.3 决策+拒答 | defense/decision.py |
| 5.6 协同 | pipeline/orchestrator.py |

## 关键超参
K-means K2/n_init10/max500；θ_out1.85；同簇0.88；rouge0.25；去重0.85；TopN8；
Score_trust λ=(0.45,0.22,0.13,0.20)；一致性 α0.4/β0.6；自评估采样5次。

## 实验与验收（论文表 5-2~5-4）
- Exp5-1(ρ10%)：ASR 47.5→32.5；RP 71.0→84.6；Recall 略降至91.0
- Exp5-2：三模块全开32.5（单模块~40+）；推理 +15%
- Exp5-8：第4+5协同 ASR→28.3
- Exp5-7：拒答 精确85.2%/召回61.7%
## 局限
ASR 仍32.5%（1/3得逞）；高投毒率明显变差(30%→~45%)；过滤误删；阈值敏感；部分替换最难防。
