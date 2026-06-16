# CLAUDE.md — 军事情报问答 鲁棒动态 RAG 实验代码库

> 本文件是 Claude Code 在本仓库工作的最高优先级约定。每次开工前先读本文件 + `docs/` 下对应章节的 design doc，再动代码。
> 项目代号：`milrag`（Military Intelligence Dynamic & Robust RAG）。论文题目：《面向军事情报问答的检索增强生成方法研究》。

---

## 0. 一句话项目定义

复现并扩展一篇硕士论文的实验：在**中文军事情报问答**场景下，做三件事——(1) 领域知识库 + 嵌入 LoRA 微调；(2) 多信号**动态检索**触发与查询重构；(3) 对抗环境下的**两阶段可信过滤与拒答**。最终集成为可本地离线运行的端到端原型，并跑通论文中全部编号实验（Exp 3-1 ~ Exp 6-6）。

★ **2026-06 Qwen3 迁移**：主干 LLM 从 Qwen3-8B → **Qwen3-8B**；嵌入从 BGE-large → **Qwen3-Embedding-4B**；重排 → **Qwen3-Reranker-8B**。旧版 Qwen2.5 数值作为对照基线保留在 config 中。

两个外部参考实现，复用边界见 §4：
- **ETC**（`WisdomShell/ETC`, AAAI'26 Oral）— 熵趋势触发，对应论文第 4 章。
- **TrustRAG**（`HuichiZhou/TrustRAG`, arXiv 2501.00879）— 聚类过滤 + 内外知识冲突消解，对应论文第 5 章。

---

## 1. 不可违背的硬约束（HARD CONSTRAINTS）

写任何代码前先确认不违反以下任意一条，违反则停下来问我：

1. **白盒约束**：触发检测依赖模型生成时的 `logits` 与 `attention`。所有动态检索逻辑只能假设开源白盒模型（Qwen3-8B 主线）。**禁止**写任何依赖闭源 API 内部状态的触发路径。API 模型只能作为"无内部信号的退化基线"出现。
2. **离线约束**：运行时**零外网请求**。所有模型（LLM/嵌入/NLI/NER/Reranker）、索引、依赖必须可本地加载。任何 `from_pretrained` 默认走本地路径或 `HF_HUB_OFFLINE=1`。联网只允许出现在一次性的数据/权重下载脚本里，且单独放 `scripts/download/`。
3. **数据合规**：仓库内**不得提交**任何真实涉密/未脱密语料。`data/raw/sensitive/` 永远在 `.gitignore` 里。示例与单测只用合成或公开样本（CMNEE 公开部分）。
4. **复现性**：所有实验固定随机种子，**报告 5 个种子的均值±标准差**（论文口径）。任何带随机性的结果不允许只跑一次。配置即真相——超参只能来自 `config/`，禁止硬编码在逻辑里。
5. **不夸大**：这是学术实验，不是产品。代码注释、日志、README 里描述效果时如实写，不替论文吹结论。指标该是多少是多少。

---

## 2. 论文章节 ↔ 代码模块映射（核心地图）

| 论文 | 方法要点 | 代码位置 | 对应实验 |
|---|---|---|---|
| 第 3 章 数据/嵌入 | 多源清洗→NER→分块→索引；1276条QA构造；BGE LoRA 微调 | `src/milrag/data/`, `src/milrag/retrieval/` | Exp 3-1 ~ 3-7 |
| 第 4 章 动态检索 | token概率+注意力+熵 三信号融合触发；三类查询重构；循环控制 | `src/milrag/dynamic/` | Exp 4-1 ~ 4-8 |
| 第 5 章 对抗防御 | 聚类前置过滤；内外知识一致性；四维自评估+拒答 | `src/milrag/defense/` | Exp 5-1 ~ 5-10 |
| 第 6 章 系统集成 | 三层架构端到端编排；vLLM/FastAPI/Streamlit | `src/milrag/pipeline/` | Exp 6-1 ~ 6-6 |

**第 4、5 章协同**（论文 5.6）：动态检索每次新召回的证据，必须先过 `defense.cluster_filter` 再注入上下文；检测到冲突簇时反向触发 `dynamic.reformulate` 做靶向补检索。这条数据流写在 `pipeline/orchestrator.py`，不要散落在各模块里。

---

## 3. 目录结构（按此组织，不要随意新增顶层目录）

```
milrag/
├── CLAUDE.md                 # 本文件
├── README.md
├── pyproject.toml            # 用 uv / pip-tools 锁版本
├── docs/                     # design doc，按章节：ch3_data.md ch4_dynamic.md ...
├── config/
│   ├── base.yaml             # 全局：路径、种子、设备
│   ├── retrieval.yaml        # 第3章超参
│   ├── dynamic.yaml          # 第4章超参
│   ├── defense.yaml          # 第5章超参
│   └── experiments/          # exp3_1.yaml ... exp6_6.yaml（每个实验一份，继承 base）
├── data/
│   ├── raw/                  # 原始语料（sensitive/ 不入库）
│   ├── kb/                   # 四类分块知识库 (装备/条令/态势/案例)
│   ├── qa/                   # train(1020)/val(128)/test(128)
│   └── adversarial/          # 投毒变体，按 ρ 分目录
├── src/milrag/
│   ├── data/
│   │   ├── clean.py          # HTML去除/全半角/繁简/标点
│   │   ├── ner.py            # BERT-CRF 军事实体 + 归一化
│   │   ├── chunk.py          # 语义边界优先 + 滑窗(512/64)
│   │   ├── index_build.py    # FAISS HNSW + ES 倒排
│   │   └── build_qa.py       # 机器初标(Qwen3-32B (int4 量化)本地) + 人工校验接口
│   ├── retrieval/
│   │   ├── embedding.py      # BGE-large-zh-v1.5 / BGE-M3 封装
│   │   ├── lora_finetune.py  # InfoNCE + 难负样本
│   │   ├── hybrid.py         # RRF(k=60) 稠密+稀疏
│   │   └── reranker.py       # Cross-Encoder（仅终轮启用）
│   ├── dynamic/              # 参考 ETC
│   │   ├── signals.py        # s_p / s_a / s_h 三类原始信号
│   │   ├── entropy_trend.py  # ★ETC: 一阶/二阶差分 + 动态平滑
│   │   ├── detector.py       # 滑窗z-score标准化 + 加权 S(t)
│   │   ├── classifier.py     # 问题类型 → 权重配置
│   │   ├── reformulate.py    # 实体增强/实词聚焦/上下文融合
│   │   ├── selector.py       # 策略选择 + 回退(sim<0.55)
│   │   ├── fuse.py           # 证据融合：去重(0.85)/权威加权/版本
│   │   └── loop.py           # 循环主流程 + 终止/早停
│   ├── defense/             # 参考 TrustRAG
│   │   ├── cluster_filter.py # ★K-means(K=2) + 离群 + 领域特征
│   │   ├── ngram_filter.py   # rouge-L>0.25 重叠去重（TrustRAG group_n_gram）
│   │   ├── inject_detect.py  # 提示注入正则 + 轻量分类器
│   │   ├── prior.py          # 内部知识先验 + 置信度(SelfCheck式)
│   │   ├── consistency.py    # ★NLI 判定 + 证据关系图 + Louvain
│   │   ├── self_assess.py    # 四维 EC/EA/SC/Unc
│   │   └── decision.py       # 阈值/学习式MLP + 拒答/追问
│   ├── pipeline/
│   │   ├── rag_base.py       # 朴素/进阶 RAG 基线
│   │   ├── orchestrator.py   # ★端到端编排（4+5 协同写这里）
│   │   └── serve.py          # vLLM 后端 + FastAPI + Streamlit 前端
│   ├── llm/
│   │   ├── backbone.py       # vLLM 推理封装（生成路径）
│   │   └── hooks.py          # ★白盒钩子：取 logits/attention（训练路径用 HF）
│   └── utils/                # 日志、种子、计时、IO
├── scripts/                  # 一键复现，命名对齐实验号
├── experiments/              # 产出：logs/ results/ ckpts/（git 忽略大文件）
├── eval/
│   ├── metrics.py            # 全部指标实现
│   └── run_eval.py
└── tests/                    # 每个核心算法配最小单测
```

---

## 4. 参考实现的复用边界（重要，别盲抄）

### 4.1 ETC（第 4 章）
**直接借鉴**：熵趋势建模思想。论文当前的 `s_h` 是**单点输出熵**，ETC 证明了用熵序列的**一阶差分、二阶差分 + 动态平滑**触发能更早、更准、且减少冗余检索。在 `entropy_trend.py` 里实现这套趋势信号，作为 `s_h` 的增强版接入 `detector.py` 的加权融合。

**必须改造**：
- ETC 是纯英文 + Wikipedia + Elasticsearch + spaCy(`en_core_web_sm`) 停用词。本项目是**中文军事**域，停用词/分词换中文方案，实词聚焦用中文词性 + 军事词典（见 §6 实词聚焦）。
- ETC 只有"何时触发"，**没有**注意力信号 `s_a`、没有查询重构、没有任务自适应权重——这些是本论文的增量，不要被 ETC 的精简结构带偏而砍掉它们。
- ETC 用 `transformers==4.30.2`，本项目用 `4.40`，注意力输出 API 有差异（见 §5 钩子坑）。

**不要借鉴**：ETC 的数据加载/评测脚本（数据集完全不同）。

### 4.2 TrustRAG（第 5 章）
**直接借鉴**：
- `k_mean_filtering`：K=2 KMeans + 组内 rouge-L>0.25 n-gram 检测 + 余弦阈值 0.88 判同簇——直接作为 `cluster_filter.py` + `ngram_filter.py` 的起点。
- `conflict_query` 三阶段范式：①不看检索生成 internal knowledge → ②consolidate 阶段显式排除"操纵性指令/预设答案/PIA 模式" → ③综合内外知识给最终答案。直接对应论文双路生成（`prior.py` + `orchestrator.py` 的内外双路）。
- `similarity_filtering`：余弦>0.85 去重，复用到 `dynamic/fuse.py`。

**必须改造**：
- TrustRAG 的 prompt 全英文、面向开放域 QA（Christian name / 热力学那种 few-shot 例子）。**全部换成中文军事情报 prompt**，few-shot 例子换成装备/条令/态势样例。
- TrustRAG 只做了聚类+冲突消解，**没有**论文的：领域特征增强（实体匹配度/关键词匹配度/注入检测打分的加权 `Score_trust`）、证据关系图 + Louvain 社区、四维自评估（EC/EA/SC/Unc）、显式拒答规则。这些是本论文增量，按论文 5.3.3 / 5.4.3 / 5.5 实现。
- TrustRAG 默认阈值（0.88/0.85/0.25）是开放域调出来的，本项目要在**军事验证集**上重新网格搜索（论文：λ=(0.45,0.22,0.13,0.20), θ_out=1.85）。所有阈值进 `config/defense.yaml`，不许沿用 TrustRAG 硬编码值当结论。
- TrustRAG 用 `lmdeploy` 或 OpenAI API。本项目生成统一走 `llm/backbone.py`（vLLM），NLI 走独立本地模型。

**不要借鉴**：InstructRAG / AstuteRAG 那两个 baseline 的具体 prompt（可作为对照基线另写，但归到 `pipeline/rag_base.py` 并标注来源）。

---

## 5. 技术栈与环境（含已知坑）

```
Python 3.10
torch 2.4.x  |  transformers 4.51  |  peft 0.14
vllm 0.7      |  faiss-cpu/gpu 1.8.0  |  elasticsearch 8.13
sentence-transformers (Qwen3-Embedding / BGE) | postgresql 16 (元信息)
streamlit (前端) | fastapi + grpc (防御微服务)
硬件：vGPU-48GB 单卡即可覆盖全部实验（Qwen3-14B 需 device_map="auto" offload）
```

**已知坑，写代码时主动规避：**

1. **注意力钩子的版本敏感性（最高频坑）**。`hooks.py` 取 attention 在 `transformers 4.51` 上需要 `output_attentions=True` 且不能与 FlashAttention-2 共存（FA2 不返回 attention 权重）。结论：**白盒信号采集路径用 eager attention 实现**（`attn_implementation="eager"`），生成吞吐路径才用 vLLM。这是两条独立路径，不要混。Qwen3 的 eager attention 路径已验证兼容。

2. **vLLM 不暴露逐步 logits/attention**。vLLM 适合最终基线吞吐/部署（第 6 章），**不适合**第 4 章逐 token 信号采集。第 4 章主实验用 HF `generate` + 自定义 `LogitsProcessor` + forward hook 拿信号；vLLM 只用于对照"无动态检索"的快速基线。Qwen3-8B 在 vLLM 0.7 上原生支持。

3. **熵/概率的数值稳定**。`s_p = -log p`、`s_h = -Σ p log p` 都要在 `log_softmax` 空间算，加 `eps`，对词表做 top-k 截断再算熵（全词表算熵慢且被长尾噪声污染）。

4. **FAISS HNSW 构建参数**：`M=32, efConstruction=200`，查询 `efSearch` 单独调。chunk=512/overlap=64。别用默认值。

5. **ES 与 FAISS 的 doc_id 必须一致**，用 PostgreSQL 存元信息（来源/时间/权威性等级/脱敏标识），三方靠同一 `chunk_id` 关联。混合检索 RRF `k=60`。

---

## 6. 关键算法的实现规格（直接照此写，超参来自论文）

### 第 3 章
- **LoRA 微调**：挂在 query/doc encoder 的注意力 Q/K/V 投影；`r=16, α=32, dropout=0.05`；`lr=1e-4`, warmup 0.05, cosine 退火, `AdamW`, bs 128（显存不够用梯度累积等效到 128），10 epoch。★ Qwen3-Embedding-4B 主线（替代 BGE-large）；BGE 保留为对照基线。
- **训练目标**：InfoNCE，`τ=0.05`。负样本 = 批内负 : BM25 难负（top-50 中非正样本）= 1:1 混合。
- **检索**：dense top-k=20 → rerank top-50 打分留 top-10。**Cross-Encoder 只在终轮/最终展示启用**，动态检索中间轮不启用。★重排用 Qwen3-Reranker-8B（替代 BGE-Reranker）。

### 第 4 章
- **三信号**（`signals.py`）：
  - `s_p(t) = -log pθ(y_t | ·)`
  - `s_a(t) = mean_{i∈E}( Ā_t,i ) · (1 - p_θ(y_t))`，`E` = 问题中军事实体位置（来自 NER），取最后 `L=4` 层多头平均。
  - `s_h(t) = -Σ_v p log p`（top-k 截断）。**用 `entropy_trend.py` 的趋势版增强**（ETC）。
- **判别**（`detector.py`）：各信号在最近 `W` 步滑窗 z-score 标准化 → `S(t)=w_p·s̃_p + w_a·s̃_a + w_h·s̃_h`，`Σw=1`。`S(t)>τ` 触发，`τ=0.62`（验证集 PR 曲线 F1 最大点，进 config）。
- **任务自适应权重**（`classifier.py` 输出类型）：
  | 类型 | w_p | w_a | w_h |
  |---|---|---|---|
  | 装备参数查询 | 0.6 | 0.2 | 0.2 |
  | 战略态势分析 | 0.2 | 0.5 | 0.3 |
  | 对抗环境验证 | 0.2 | 0.2 | 0.6 |
  
  分类器 top1−top2 概率差 <0.15 时退回三权重均值。
- **查询重构**（`reformulate.py`）：
  - 实体增强：`α_i = β·att(e_i) + (1-β)·IDF(e_i)`，`β=0.6`，取 Top-3~5 实体。→ 装备参数类。
  - 实词聚焦：中文词性标注 + 军事词典，保留名词/动词/专名/数词 + 全部军事术语，按注意力排序取 Top-K。→ 对抗环境类（配过滤噪声）。**坑：短问题别把限定词删掉导致语义漂移。**
  - 上下文融合：取触发点前 ~100 字已生成片段 `c_t`，`q' = q + "当前推理：" + c_t`。→ 战略态势类。
  - 选择器回退：首条结果相似度 <0.55 时换另一策略。
- **循环控制**（`loop.py`）：每 `Δt=8` token 检测一次。终止条件任一满足即停：EOS / `Lmax=1024`；`S(t)<τ` 连续 `Lstop=16` 步；迭代达 `Kmax`（简单事实=1，复杂推理=4）；证据超上下文上限。
- **证据融合**（`fuse.py`）：去重(余弦>0.85，留权威性高者) → 权威加权(条令>官方公报>主流媒体>一般评论) → 多版本按时间倒序并在 prompt 标注版本。

### 第 5 章
- **聚类过滤**（`cluster_filter.py`）：K-means(K=2, n_init=10, max_iter=500) on 标准化+L2归一嵌入；离群分 `outlier(d)=(‖e_d−μ_c‖−μ̄_dist)/σ_dist > θ_out`，`θ_out=1.85`。
- **领域特征 + 级联打分**：`Score_trust(d)=λ1(1−outlier)+λ2·Ent(d)+λ3·Kwd(d)+λ4(1−Inj(d))`，`λ=(0.45,0.22,0.13,0.20)`，按分排序取 Top-N（默认 N=8）。`Ent`=实体命中/长度，`Kwd`=查询实词重叠，`Inj`=注入指令检测(正则+轻分类器, 0~1)。
- **内部先验**（`prior.py`）：不看检索生成 `y_prior` + 结构化事实声明 `F_prior`，每条带内部置信度（log-prob + 多次采样一致性，SelfCheckGPT 思路）。
- **一致性**（`consistency.py`）：`c(d_i,f_j)=α·sim_sem + β·nli_score`，`α=0.4, β=0.6`，`nli_score=P(entail)−P(contradict)`。证据关系图边权用证据间 NLI 蕴含度（正=互证/负=冲突），Louvain 社区 → 可信簇/冲突簇/孤立证据。
- **四维自评估**（`self_assess.py`）：EC(主张被证据 entailment 比例)、EA(与最强证据余弦)、SC(多次采样回答相似度均值, 默认 5 次)、Unc(多次采样 log-likelihood 方差)。
- **决策**（`decision.py`）：阈值式（`EC(y_ext)>τ_EC 且 Unc(y_ext)<τ_Unc` 则用 `y_ext`）；学习式（10~12 维特征 MLP → 5 类：用ext/用prior/补检索/追问/拒答）。**拒答原则：宁可拒答不可错答**，拒答带原因。

### 投毒构造（`data/adversarial`）
三策略：直接矛盾 / 部分替换（只改数字实体最难防）/ 提示注入。`ρ ∈ {1%,5%,10%,20%,30%}`，每条原样本生成对应变体。

---

## 7. 评测指标（`eval/metrics.py` 全部实现，口径对齐论文）

- 检索：Recall@{1,5,10,20}, MRR, nDCG@10
- 生成：EM, F1, Faithfulness(NLI 对每条主张-证据蕴含打分), Answer Relevance
- 鲁棒：ASR↓, PDR↓, RP(召回纯净度)↑, Recall_clean↑, FPR↓, 拒答精确率/召回率, `F1_Robust=2·R_clean·(1−FPR)/(R_clean+(1−FPR))`
- 效率：平均检索次数 N̄_R, 平均延迟, Token 消耗

**所有数值实验跑 5 个种子报均值±标准差。** 单测里用固定小样本断言指标实现正确性（不是断言论文数值）。

---

## 8. 实验复现入口（`scripts/`，命名对齐论文实验号）

每个实验 = 一个 `config/experiments/expX_Y.yaml`（继承 base）+ 一个 `scripts/run_expX_Y.sh`。约定：

| 脚本 | 内容 |
|---|---|
| `run_exp3_1.sh` | 5 种嵌入模型 零样本 vs LoRA 微调（核心：BGE-large-zh R@10 71.4→78.6） |
| `run_exp3_2.sh` | LoRA vs 全参微调 |
| `run_exp3_6.sh` | 重排序消融（+Cross-Encoder ~81.3） |
| `run_exp4_1.sh` | 动态检索总体对比（vs FLARE/Self-RAG/CRAG/IRCoT/DRAGIN，F1 ~61.5） |
| `run_exp4_2.sh` | 三信号消融 |
| `run_exp5_1.sh` | 对抗鲁棒总体（ρ=10%，ASR 47.5→32.5） |
| `run_exp5_2.sh` | 防御模块消融 |
| `run_exp5_8.sh` | 第4+5章协同（ASR→28.3） |
| `run_exp6_1.sh` | 端到端总体 |
| `run_exp6_2.sh` | 模块级消融 |

脚本统一：读 config → 跑 → 把 `results/*.json` + 日志落到 `experiments/<exp_id>/<timestamp>/`，自动记录 git commit hash、config 快照、种子。

---

## 9. Claude Code 协作规范（你的工作方式）

1. **先读后写**：动某章代码前，先读 `docs/chX_*.md` 和本文件对应段。design doc 缺失就先帮我补 design doc 再写实现，不要直接堆代码。
2. **小步提交**：一个模块一个 PR 粒度。每个核心算法函数（signals/detector/cluster_filter/consistency 等）**配最小单测**再算完成。
3. **配置即真相**：任何超参、阈值、路径只能从 `config/` 读，禁止函数里写魔法数字。新增超参同步进对应 yaml 并在 design doc 记一笔。
4. **实验可追溯**：跑实验必须落 commit hash + config 快照 + 种子到 `experiments/`。不允许"跑完就丢"。
5. **不偷改超参对结果**：复现对不上论文数值时，**先排查实现 bug / 数据 / 种子**，不要为了凑数悄悄改阈值。改了任何默认值必须在 commit message 说明原因。
6. **复用要标注来源**：从 ETC / TrustRAG 借鉴的函数，文件头注释标 `# Adapted from TrustRAG defend_module.k_mean_filtering` 并说明改了什么。许可证：两者均 MIT，保留 LICENSE 声明。
7. **中文优先**：注释、日志、design doc 用中文；变量名/类名/术语用英文。prompt 模板用中文军事语料。
8. **不确定就问**：涉及数据合规、威胁模型边界、论文里写"详见某节"但 design doc 没展开的地方，停下来问，不要自行脑补设定。

---

## 10. 推荐推进顺序（增量可验证）

1. `utils` + `config` + `eval/metrics.py`（先有度量和脚手架）
2. 第 3 章：`data/` → `retrieval/`（先把检索基线和 Recall 跑通，这是一切的地基）
3. `pipeline/rag_base.py`（朴素/进阶 RAG 基线 —— 所有提升都相对它报告）
4. 第 4 章：`signals` → `entropy_trend`(ETC) → `detector` → `reformulate` → `loop`
5. 第 5 章：`cluster_filter`+`ngram_filter`(TrustRAG) → `prior` → `consistency` → `self_assess` → `decision`
6. `pipeline/orchestrator.py`（4+5 协同）→ `serve.py`
7. 逐个跑 `scripts/run_expX_Y.sh`，对齐论文数值，记录偏差与原因。

> 每完成一阶段，更新 `docs/PROGRESS.md`：完成项、与论文数值的差距、已知问题、下一步。
