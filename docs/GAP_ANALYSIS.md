# milrag 系统完备性分析报告

> 生成日期：2026-06-16 | 版本：Qwen3 迁移后 (v0.2.0)
> 用途：提交给 Claude Web 进行论文修改参考

---

## 一、当前系统状态总览

### 1.1 代码规模

| 层 | 文件数 | 代码行数 | 实现状态 |
|---|---|---|---|
| `src/milrag/data/` | 6 | ~1,123 | ✅ 完整 |
| `src/milrag/retrieval/` | 4 | ~629 | ✅ 完整 |
| `src/milrag/dynamic/` | 7 | ~945 | ✅ 完整 |
| `src/milrag/defense/` | 7 | ~1,015 | ✅ 完整 |
| `src/milrag/llm/` | 2 | ~402 | ✅ 完整 |
| `src/milrag/pipeline/` | 3 | ~606 | ✅ 完整 |
| `src/milrag/utils/` | 3 | ~68 | ✅ 完整 |
| `eval/` | 2 | ~480 | ✅ 完整（含 31 实验 dispatch） |
| `config/` | 36 | ~1,500 | ✅ 完整 |
| `tests/` | 11 | ~500 | ✅ 46/47 pass |
| **总计** | **~80+** | **~6,000** | **全部源码已实现** |

### 1.2 论文实验覆盖

| 论文章节 | 实验数量 | 配置文件 | 运行脚本 | 评测函数 |
|---|---|---|---|---|
| 第 3 章 | 7 (Exp 3-1 ~ 3-7) | ✅ 7 | ✅ 7 | ✅ 7 |
| 第 4 章 | 8 (Exp 4-1 ~ 4-8) | ✅ 8 | ✅ 8 | ✅ 8 |
| 第 5 章 | 10 (Exp 5-1 ~ 5-10) | ✅ 10 | ✅ 10 | ✅ 10 |
| 第 6 章 | 6 (Exp 6-1 ~ 6-6) | ✅ 6 | ✅ 6 | ✅ 6 |
| **合计** | **31** | **31** | **31** | **31** |

---

## 二、当前缺失项（需要外部获取/准备）

### 2.1 🔴 模型权重（全部缺失 — 需下载/挂载）

| 模型 | 用途 | 大小（约） | 获取方式 | 优先级 |
|---|---|---|---|---|
| **Qwen3-8B-Instruct** | 主线 LLM（第4/5/6章） | ~16GB fp16 | HuggingFace / ModelScope 下载 | 🔴 必须 |
| **Qwen3-14B-Instruct** | 大模型对照（Exp 4-7, 6-3） | ~28GB fp16 | 同上 | 🟡 对照实验 |
| **Qwen3-32B-Instruct** | QA 机器标注（第3章） | ~64GB fp16 / ~18GB int4 | 同上（建议 int4） | 🟡 数据集构造用 |
| **Qwen3-Embedding-4B** | 嵌入主线（第3章） | ~8GB fp16 | 同上 | 🔴 必须 |
| **Qwen3-Embedding-8B** | 嵌入对照 | ~16GB fp16 | 同上 | 🟡 对照实验 |
| **Qwen3-Embedding-0.6B** | 轻量对照 | ~1.2GB fp16 | 同上 | 🟢 可选 |
| **Qwen3-Reranker-8B** | Cross-Encoder 重排 | ~16GB fp16 | 同上 | 🟡 终轮检索 |
| **NLI 模型（中文）** | 一致性判定/EC 计算（第5章） | ~2GB | 需选型（见 §2.2） | 🔴 必须 |
| **BERT-CRF 军事 NER** | 军事实体识别（第3/4章） | ~1GB | 需训练或选替代（见 §2.2） | 🔴 必须 |
| **DeepSeek-R1-Distill-Qwen-7B** | 外部对照基线（Exp 4-7） | ~14GB fp16 | 保留 Qwen2.5 时代权重 | 🟢 可选 |

### 2.2 🟡 模型选型待决策项

#### NLI 模型
- 当前 config 中写的是 `/models/nli-zh`，但未指定具体模型
- **建议**：`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`（支持中文）
- 或使用 Qwen3-8B 本身做 NLI（zero-shot prompt: "前提A 假设B 是否蕴含？"）
- **影响**：Exp 5-5（一致性方法消融）、第5章的 EC 指标

#### 军事 NER 模型
- 论文要求 BERT-CRF 达到 F1 91.2%
- 当前 `ner.py` 实现了两种模式：规则匹配（rule）和模型推理（model）
- 规则模式可用但 F1 不满论文指标
- **建议**：使用 `Qwen3-8B` 做 few-shot NER，或使用通用中文 NER 模型 + 军事词典增强
- **影响**：s_a 注意力信号的实体位置 E（第4章）、实体增强查询重构

### 2.3 🔴 数据集（全部缺失 — 需构造或获取）

| 数据集 | 规模 | 当前状态 | 构造方式 |
|---|---|---|---|
| **四类知识库（KB）** | 314,759 段 / 86.4 万 token | ❌ 不存在 | `data/clean.py` → `ner.py` → `chunk.py` → `index_build.py` |
| **QA 训练集** | 1,020 条 | ❌ 不存在 | `data/build_qa.py`（需 32B 标注器） |
| **QA 验证集** | 128 条 | ❌ 不存在 | 同上 |
| **QA 测试集** | 128 条 | ❌ 不存在 | 同上 |
| **对抗变体** | 按 ρ ∈ {1%,5%,10%,20%,30%} × 3 策略 | ❌ 不存在 | `data/build_adversarial.py`（需干净 QA） |
| **FAISS 索引** | — | ❌ 未构建 | `data/index_build.py` |
| **ES 倒排索引** | — | ❌ 未构建 | `data/index_build.py` |
| **PG 元信息库** | — | ❌ 未建表 | `data/index_build.py` |

**知识库来源要求**（论文表 3-1）：
- CMNEE 等开源军事情报：174,328 段
- 公开条令/解密手册：53,472 段
- 军事百科/专业网站：31,854 段
- 权威防务媒体：47,690 段
- 合作单位脱密语料：7,415 段

**数据合规提醒**（CLAUDE.md §1.3）：`data/raw/sensitive/` 永远在 `.gitignore` 中——不得提交真实涉密语料。QA 构造和测试仅用合成/公开样本。

### 2.4 🟡 基础设施服务

| 服务 | 版本 | 用途 | 当前状态 |
|---|---|---|---|
| Elasticsearch | 8.13 | 稀疏倒排检索（BM25） | ❌ 需在服务器启动 |
| PostgreSQL | 16 | 知识库元信息存储 | ❌ 需在服务器启动/建库 |
| FAISS | 1.8.0 | 稠密向量检索（HNSW） | ✅ Python 包已就绪 |

---

## 三、相对原论文的修改汇总

### 3.1 模型迁移（最大修改）— Qwen2.5 → Qwen3

| 维度 | 原论文（Qwen2.5 时代） | 当前系统（Qwen3 迁移后） | 论文影响 |
|---|---|---|---|
| 主线 LLM | Qwen2.5-7B-Instruct | Qwen3-8B | 所有 F1/EM/ASR 数值预期变化 |
| 大模型对照 | Qwen2.5-14B-Instruct | Qwen3-14B | Exp 4-7/6-3 对比基线更强 |
| 机器标注 | Qwen2.5-72B-Instruct | Qwen3-32B (int4) | QA 质量预期持平或更好 |
| 嵌入模型 | BGE-large-zh-v1.5 | Qwen3-Embedding-4B | R@10 基线改变，LoRA 微调增益可能不同 |
| 嵌入对照 | BGE-M3 | Qwen3-Embedding-8B | 多语言对照更强 |
| 重排序 | BGE-Reranker-large | Qwen3-Reranker-8B | 重排增量预期不同 |
| 软件栈 | torch 2.1 / tf 4.40 / vLLM 0.4 | torch 2.4 / tf 4.51 / vLLM 0.7 | 论文 §6.4 环境描述需更新 |
| 硬件要求 | A100 40GB 单卡 / A100 80GB×2 | vGPU-48GB 单卡 | 论文 §6.4 降低部署门槛 |

**论文修改要点**：
- 第 3 章 §3.4.1：嵌入模型从"BGE-large-zh-v1.5"改为"Qwen3-Embedding-4B"
- 第 4 章 §4.6.1：生成主干从"Qwen2.5-7B"改为"Qwen3-8B"
- 第 6 章 §6.4：部署环境从"A100 80GB×2"改为"vGPU-48GB 单卡"
- 所有实验表中的 Qwen2.5 数值可作为"旧版基线"列保留，Qwen3 数值单独测后填入

### 3.2 架构层面修改

| 修改项 | 原设计 | 当前实现 | 原因 |
|---|---|---|---|
| 动态检索权重 τ | 0.62（Qwen2.5 验证集确定） | 0.62（暂保留，标注"Qwen3 上需重扫"） | Qwen3 概率分布不同 |
| 聚类过滤阈值 θ_out | 1.85（BGE 嵌入空间） | 1.85（暂保留，标注需重搜） | Qwen3-Embedding 嵌入空间不同 |
| 一致性 α/β | 0.4/0.6 | 0.4/0.6（保留） | 权重是领域无关的相对权重 |
| 投毒构造 | 原文三策略 | 完全对齐 | 无变化 |

### 3.3 代码实现层面的增量（超出论文原文描述）

论文是高层次方法描述，以下实现细节是代码层面的工程补充：

1. **ETC 熵趋势**（`entropy_trend.py`）：一阶/二阶差分 + EMA 平滑的具体实现
2. **TrustRAG 聚类过滤**（`cluster_filter.py`）：K-means(K=2) + 领域特征级联打分 `Score_trust` 的完整流水线
3. **证据关系图 + Louvain**（`consistency.py`）：成对 NLI 边构建 → 社区发现 → 可信/冲突/孤立分类
4. **四维自评估**（`self_assess.py`）：EC/EA/SC/Unc 的具体 NLI/嵌入/多次采样实现
5. **双路决策**（`decision.py`）：阈值式 + 学习式（MLP 12维 → 5类）双策略
6. **白盒信号钩子**（`llm/hooks.py`）：LogitsProcessor + forward hook 的 transformers 4.51 适配
7. **双后端抽象**（`llm/backbone.py`）：HF eager（信号采集）vs vLLM（吞吐部署）统一接口
8. **31 个实验的统一评测框架**（`eval/run_eval.py`）：config 继承链 + task dispatch + 5 种子聚合 + expected 对账

---

## 四、论文数值对账状态

| 实验 | 核心指标 | 论文原值 (Qwen2.5) | 当前实测 | 备注 |
|---|---|---|---|---|
| Exp 3-1 | BGE-large R@10 (LoRA) | 78.6 | N/A | Qwen3-Embedding 待测 |
| Exp 3-1 | Qwen3-Emb-4B R@10 (LoRA) | N/A | **TBD** | 新主线 |
| Exp 3-2 | LoRA vs 全参差距 | 0.7pt | **TBD** | Qwen3 微调特性待验证 |
| Exp 3-6 | +CrossEncoder R@10 | 81.3 | **TBD** | Qwen3-Reranker |
| Exp 4-1 | 动态检索 F1 | 61.5 | **TBD** | Qwen3-8B 应更高 |
| Exp 4-2 | 三信号消融 | 61.5/51.2/48.7/50.1 | **TBD** | 信号相对强度可能变化 |
| Exp 5-1 | 对抗鲁棒 ASR↓ (ρ=10%) | 32.5 | **TBD** | Qwen3 应更难被欺骗 |
| Exp 5-8 | 第4+5章协同 ASR↓ | 28.3 | **TBD** | 协同效果待验证 |
| Exp 6-1 | 端到端 F1/ASR | 61.5/28.3 | **TBD** | 全系统集成指标 |
| Exp 6-3 | Qwen3-14B F1 | N/A | **TBD** | 原为 Qwen2.5-14B 的 65.2 |

**所有 TBD 项需在 GPU 服务器上跑通实验后填入。**

---

## 五、运行前的准备清单（在 vGPU-48GB 上）

### 必须完成（否则无法实验）

```
[ ] 1. 下载 Qwen3-8B-Instruct → /models/Qwen3-8B-Instruct
[ ] 2. 下载 Qwen3-Embedding-4B → /models/Qwen3-Embedding-4B
[ ] 3. 准备/获取中文军事语料 → data/raw/（CMNEE 公开部分）
[ ] 4. 安装 Python 依赖：pip install -r requirements.txt
[ ] 5. 启动 ES + PostgreSQL 服务
[ ] 6. 编辑 config/base.yaml → models.* 路径
[ ] 7. 运行数据清洗 → 分块 → 索引构建（第3章流水线）
[ ] 8. 构造 QA 数据集（如无可直接用的数据）
```

### 建议完成（提升实验完整性）

```
[ ] 9. 下载 Qwen3-14B-Instruct（对照实验用）
[ ] 10. 下载 NLI 模型（第5章一致性判定）
[ ] 11. 准备/微调军事 NER 模型（或使用 Qwen3 替代方案）
[ ] 12. 下载 Qwen3-Reranker-8B（重排消融实验）
[ ] 13. 构造对抗变体数据集（从干净 QA 出发）
```

---

## 六、给论文修改的具体建议

### 6.1 可直接声明的修改点

1. **"基于 Qwen3 系列的最新实验"**：在论文摘要/引言中加入一句"实验基于 Qwen3 系列模型进行了复现与扩展"
2. **环境降级**：从"A100 80GB×2"改为"vGPU-48GB 单卡"，大幅降低复现门槛
3. **嵌入升级**：从"BGE-large-zh"改为"Qwen3-Embedding-4B + BGE 基线对照"
4. **版本号更新**：第 6 章 §6.4 的软件栈版本全部更新

### 6.2 实验表中需要新增的列

| 表格 | 建议新增列 | 说明 |
|---|---|---|
| 表 3-4（嵌入模型对比） | Qwen3-Embedding-4B (Zero/LoRA) | 新主线 |
| 表 4-2（动态检索对比） | Qwen3-8B 列 | 与 Qwen2.5-7B 对比 |
| 表 4-4（主干迁移） | Qwen3-14B | 替代原 Qwen2.5-14B |
| 表 5-2（对抗鲁棒对比） | Qwen3-8B ASR | 预期更低 |
| 表 6-3（端到端总体） | Qwen3-8B 全链路 | 最终系统指标 |

### 6.3 可讨论的"未来方向"

1. **Qwen3 GQA 注意力**对 s_a 信号的影响（Qwen3 使用 Grouped Query Attention vs Qwen2.5 的多头注意力）
2. **Qwen3 的更长上下文**（32K vs Qwen2.5 的 8K）对动态检索循环预算的影响
3. **Qwen3-Embedding 的 Matryoshka 表示**对存储/检索效率的优化空间
