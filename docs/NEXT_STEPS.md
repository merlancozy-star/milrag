# 后续工作清单

> 生成日期：2026-06-16 | 当前状态：全部源码已实现（33 模块 / 46 单测通过），实验数据待跑
> 用途：毕业前需要完成的所有事项，按优先级排列

---

## 总览

当前系统状态：**代码 100%、数据 0%、实验 0%、论文修改未启动**。以下按依赖关系分为六个阶段，逐个完成即可达到答辩条件。

```
阶段 0 环境准备 ──→ 阶段 1 数据构造 ──→ 阶段 2 阈值标定
                                          ↓
阶段 4 论文修改 ←── 阶段 3 跑通 31 实验 ──→ 阶段 5 答辩准备
```

---

## 阶段 0：环境准备（1 天）

> 在 AutoDL vGPU-48GB 或同等 GPU 服务器上完成。这是所有后续工作的前提。

### 0.1 租用 GPU 服务器

| 要求 | 规格 |
|---|---|
| GPU 显存 | ≥ 48GB（单卡 vGPU-48GB 或 A100 40GB） |
| 内存 | ≥ 64GB |
| 磁盘 | ≥ 200GB（模型权重 ~80GB + 数据 ~20GB + 产出 ~10GB） |
| 推荐平台 | AutoDL（按量计费 vGPU-48GB ~3 元/时） |

### 0.2 安装系统依赖

```bash
# 基础工具
sudo apt-get update && sudo apt-get install -y git curl wget build-essential

# Docker（ES + PG 用）
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Python 环境
conda create -n milrag python=3.10 -y
conda activate milrag
pip install -r requirements.txt

# 验证关键包
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
python -c "import vllm; print(vllm.__version__)"
```

### 0.3 启动基础设施

```bash
# Elasticsearch 8.13
docker run -d --name es-milrag \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.0

# PostgreSQL 16
docker run -d --name pg-milrag \
  -p 5432:5432 \
  -e POSTGRES_DB=milrag \
  -e POSTGRES_PASSWORD=milrag123 \
  postgres:16

# 验证
curl http://localhost:9200
psql -h localhost -U postgres -d milrag -c "SELECT 1"
```

### 0.4 下载模型权重

**🔴 必须下载（否则无法实验）**：

| 模型 | 大小 | 用途 | 下载命令 |
|---|---|---|---|
| Qwen3-8B-Instruct | ~16GB | 主线 LLM，第 4/5/6 章 | `huggingface-cli download Qwen/Qwen3-8B-Instruct --local-dir /models/Qwen3-8B-Instruct` |
| Qwen3-Embedding-4B | ~8GB | 嵌入主线，第 3 章 | `huggingface-cli download Qwen/Qwen3-Embedding-4B --local-dir /models/Qwen3-Embedding-4B` |

**🟡 建议下载（提升实验完整性）**：

| 模型 | 大小 | 用途 |
|---|---|---|
| Qwen3-14B-Instruct | ~28GB | 大模型对照 Exp 4-7, 6-3 |
| Qwen3-Reranker-8B | ~16GB | 重排消融 Exp 3-6 |
| Qwen3-Embedding-8B | ~16GB | 嵌入对照 Exp 3-1 |
| mDeBERTa-v3-base-xnli | ~2GB | NLI 一致性判定 Exp 5-5 |

**🟢 可选下载**：

| 模型 | 大小 | 用途 |
|---|---|---|
| Qwen3-32B-Instruct (int4) | ~18GB | QA 机器标注 |
| BGE-large-zh-v1.5 | ~1.3GB | 旧版嵌入对照基线 |

### 0.5 配置模型路径

编辑 [config/base.yaml](../config/base.yaml)，将 `models.*` 下的占位路径改为实际挂载路径：

```yaml
models:
  llm: "/models/Qwen3-8B-Instruct"
  llm_14b: "/models/Qwen3-14B-Instruct"        # 可选
  llm_32b_int4: "/models/Qwen3-32B-Instruct"    # 可选
  embedding: "/models/Qwen3-Embedding-4B"
  embedding_8b: "/models/Qwen3-Embedding-8B"     # 可选
  reranker: "/models/Qwen3-Reranker-8B"          # 可选
  nli: "/models/mDeBERTa-v3-base-xnli"           # 可选
  ner: null                                       # 使用规则模式
```

### 0.6 安装额外 Python 包

```bash
pip install scikit-learn        # 聚类过滤（当前 1 个测试 skip）
pip install networkx python-louvain  # 证据关系图
pip install jieba               # 中文分词
pip install elasticsearch       # ES 客户端
pip install psycopg2-binary     # PG 客户端
```

---

## 阶段 1：数据构造（2-4 天）

> 目标产出：~31.5 万段知识库 + FAISS 索引 + ES 索引 + 1,276 条 QA + 15 组对抗变体

### 1.1 获取原始语料（策略选择）

**推荐路径**：混合策略 = 公开语料做骨架 + 合成语料填充到目标量。

| 来源 | 预估段数 | 耗时 |
|---|---|---|
| 中文维基百科军事分类 | ~80,000 | 2 小时（下载+筛选） |
| 公开国防白皮书 PDF | ~30,000 | 3 小时（下载+解析） |
| 中国军网公开报道 | ~50,000 | 4 小时（爬取） |
| Qwen3-8B 合成填充 | ~155,000 | 20-30 小时（GPU 生成，可后台） |

详细脚本见 [DATA_BUILD_GUIDE.md](DATA_BUILD_GUIDE.md) §二。

**最小验证路径**（先跑通流水线，确认代码正确）：
1. 生成 200 段合成知识块 + 50 条 QA → 跑通一次完整实验 → 再用完整数据正式跑
2. 最小验证脚本：`python scripts/generate_minimal_data.py`

### 1.2 数据清洗 → NER → 分块

```bash
# 清洗：HTML 去除 / 全半角 / 繁简 / 标点归一化
python scripts/run_clean.py

# NER：军事实体识别（先用规则模式，后续可选模型模式）
python scripts/run_ner.py

# 分块：语义边界优先 + 滑窗 512/64
python scripts/run_chunk.py
```

### 1.3 构建向量索引

```bash
# FAISS HNSW (M=32, efConstruction=200) + ES 倒排 + PG 元信息
python scripts/run_index.py
```

验证检索效果：
```bash
python -c "
from milrag.retrieval.embedding import Embedder
from milrag.data.index_build import load_faiss
from milrag.retrieval.hybrid import dense_retrieve
import json

chunks = json.loads(open('data/kb/chunks.json', encoding='utf-8').read())
index, chunk_ids, _ = load_faiss('data/kb')
embedder = Embedder('/models/Qwen3-Embedding-4B', device='cuda')

query = '歼-20 战斗机的作战半径是多少公里'
results = dense_retrieve(query, embedder, index, chunk_ids, topk=5)
for chunk_id, score in results:
    chunk = next(c for c in chunks if c['chunk_id'] == chunk_id)
    print(f'  [{score:.3f}] {chunk[\"text\"][:100]}...')
"
```

### 1.4 构造 QA 数据集

```bash
# 机器初标：Qwen3-32B int4 标注 1,276 条（或使用已有 QA 数据）
python scripts/run_build_qa.py
```

**数据集拆分**：train 1,020 / val 128 / test 128

**⚠️ 人工校验**：机器初标后抽查 20%（~255 条），记录修正率和 Cohen's κ。论文要求 κ ≥ 0.82。

### 1.5 构造对抗变体

```bash
# 3 策略 × 5 投毒率 = 15 组数据集
python scripts/run_build_adversarial.py
```

三策略：
- **直接矛盾**：将答案替换为与事实相反的内容
- **部分替换**：只篡改数字实体（最难防）
- **提示注入**：在证据文本中嵌入操纵指令

投毒率：`ρ ∈ {1%, 5%, 10%, 20%, 30%}`

---

## 阶段 2：阈值重新标定（0.5-1 天）

> 论文原阈值基于 Qwen2.5 + BGE 空间标定。代码已保留原值，但**必须在 Qwen3 + Qwen3-Embedding 上重扫**。

### 2.1 需要重标的阈值

| 参数 | 原值 | 配置位置 | 标定方法 |
|---|---|---|---|
| 触发阈值 `τ` | 0.62 | `config/dynamic.yaml` | 验证集 PR 曲线，取 F1 最大点 |
| 离群阈值 `θ_out` | 1.85 | `config/defense.yaml` | 网格搜索 [1.0, 3.0]，最大化 F1_Robust |
| Score_trust 权重 `λ1..λ4` | (0.45, 0.22, 0.13, 0.20) | `config/defense.yaml` | 网格搜索，Σ=1 约束 |
| 一致性 `α/β` | 0.4/0.6 | `config/defense.yaml` | 网格搜索 α ∈ [0.2, 0.6]，β=1-α |
| EC 阈值 `τ_EC` | 0.5 | `config/defense.yaml` | 验证集 F1_Robust 最大点 |
| Unc 阈值 `τ_Unc` | 0.5 | `config/defense.yaml` | 验证集拒答精确率/召回率最优 |

### 2.2 标定流程

```bash
# 使用验证集（128 条）做网格搜索
python scripts/calibrate_thresholds.py
```

> 此脚本需要新建。流程：遍历参数组合 → 在验证集上计算 F1_Robust / 触发 F1 → 取最优值 → 写回 `config/dynamic.yaml` 和 `config/defense.yaml`。

---

## 阶段 3：跑通 31 个实验（3-7 天，取决于 GPU 可用性）

> 这是最核心的工作。每个实验跑 5 个种子，自动写入 `experiments/<exp_id>/<timestamp>/`。

### 3.1 实验执行顺序（建议）

| 顺序 | 实验 | 内容 | 预估耗时 | 依赖 |
|---|---|---|---|---|
| 1 | Exp 3-1 | 嵌入模型对比（Zero-shot vs LoRA） | 2-3h | 阶段 1 |
| 2 | Exp 3-2 ~ 3-7 | 检索消融（LoRA vs 全参 / 秩 / 负样本 / 块大小 / 重排 / 混合） | 4-6h | Exp 3-1 |
| 3 | Exp 4-1 | 动态检索总体对比 | 3-4h | Exp 3-1 |
| 4 | Exp 4-2 ~ 4-8 | 动态检索消融（信号 / 阈值 / 重构 / 策略 / 循环 / 规模 / 跨域） | 8-12h | Exp 4-1 |
| 5 | Exp 5-1 | 对抗鲁棒总体 | 4-6h | 阶段 1 对抗数据 |
| 6 | Exp 5-2 ~ 5-10 | 防御消融（模块 / 聚类 / 阈值 / 一致性 / 注入 / 自评估 / 决策 / 协同 / 跨域） | 12-18h | Exp 5-1 |
| 7 | Exp 6-1 ~ 6-6 | 系统集成（总体 / 消融 / 规模 / 延迟 / 前端 / 跨场景） | 8-12h | 全部前面 |

**时间估算**：如果 GPU 24 小时可用，全部跑完约 **3-5 天**。如果按时计费，约 **5-7 天**（分时段跑）。

### 3.2 运行命令

```bash
# 单实验运行
bash scripts/run_exp3_1.sh

# 批量运行（注意 GPU 显存占用）
for exp in 3_1 3_2 3_3 3_4 3_5 3_6 3_7; do
  bash scripts/run_exp${exp}.sh
done

# 或使用总控脚本
bash scripts/run_all.sh
```

### 3.3 运行中需记录的内容

每个实验完成后，在 [PROGRESS.md](PROGRESS.md) 的数值对账表中填入：

- 实测均值 ± 标准差（5 种子）
- 与论文原值的差距
- 差距分析（Qwen3 差异？数据差异？阈值未重标？）

### 3.4 异常处理

| 情况 | 处理方式 |
|---|---|
| 显存不足（OOM） | 减小 batch size，用梯度累积；或换 int4/int8 量化 |
| 某个实验指标显著差于论文 | 先排查实现 bug → 数据质量 → 阈值未标定 → 记录原因 |
| 某个实验指标显著好于论文 | 检查是否过拟合验证集 → 记录为 Qwen3 增益 |
| ES/PG 连接失败 | 检查 Docker 容器状态；允许降级为纯 FAISS 运行 |
| 模型下载中断 | 使用 `HF_HUB_ENABLE_HF_TRANSFER=1` 加速；或从 ModelScope 下载 |

---

## 阶段 4：论文修改（3-5 天）

> 实验跑完后，用实际数据更新论文。

### 4.1 必须修改的部分

| 位置 | 修改内容 | 原因 |
|---|---|---|
| 摘要/引言 | 加入"实验基于 Qwen3 系列模型进行了复现与扩展" | 模型迁移 |
| §3.4.1 | 嵌入模型从"BGE-large-zh-v1.5"改为"Qwen3-Embedding-4B"为主线 | 嵌入升级 |
| §4.6.1 | 生成主干从"Qwen2.5-7B"改为"Qwen3-8B" | LLM 迁移 |
| §6.4 | 软件栈版本：torch 2.4 / transformers 4.51 / vLLM 0.7，硬件从 A100 80GB×2 改为 vGPU-48GB | 环境更新 |

### 4.2 实验表需要新增的列

| 表 | 新增列 | 说明 |
|---|---|---|
| 表 3-4（嵌入对比） | Qwen3-Embedding-4B (Zero/LoRA) | 新主线 |
| 表 4-2（动态检索对比） | Qwen3-8B（本文） | 与 Qwen2.5-7B 并排 |
| 表 4-4（主干规模迁移） | Qwen3-14B | 替代 Qwen2.5-14B 列 |
| 表 5-2（对抗鲁棒对比） | Qwen3-8B ASR | 预期更低 |
| 表 6-3（端到端总体） | Qwen3-8B 全链路 | 最终系统指标 |

### 4.3 数值更新

将 [PROGRESS.md](PROGRESS.md) 中的实测数据填入论文所有实验表，替换原 Qwen2.5 占位数值。

若某些指标的 Qwen3 实测值与论文 Qwen2.5 原值差距 >3 个点，需在论文正文中分析原因（如 Qwen3 的 GQA 注意力对 s_a 信号的影响、Embedding 空间分布差异等）。

### 4.4 可讨论的未来方向（论文 §7）

在论文"总结与展望"章加入：

1. Qwen3 的 GQA 注意力机制对 s_a 信号的影响
2. Qwen3 32K 长上下文对循环检索预算的优化空间
3. Qwen3-Embedding 的 Matryoshka 表示对存储效率的改进
4. 多模态军事情报（卫星图 + 文本）的 RAG 扩展

---

## 阶段 5：答辩准备（2-3 天）

### 5.1 PPT 结构建议

```
1. 研究背景与问题 (3 页)
   - 军事情报问答的特殊需求（离线、可信、动态）
   - 现有 RAG 的两个痛点：静态检索 + 脆弱检索
   
2. 方法总览 (2 页)
   - 三层架构图
   - 第4+5章闭环协同流程图
   
3. 动态检索触发 (5-6 页)
   - 三信号定义（s_p, s_a, s_h）+ 融合公式
   - 熵趋势增强（与 ETC 的关系 / 创新点）
   - 三类查询重构
   - 关键实验结果：与 FLARE/Self-RAG 等对比
   
4. 对抗防御与可信过滤 (5-6 页)
   - 聚类过滤 + Score_trust
   - 证据关系图 + Louvain
   - 四维自评估 + 拒答决策
   - 关键实验结果：ASR 降低 / F1_Robust 提升
   
5. 系统集成与协同 (2 页)
   - 动态检索 ↔ 可信过滤闭环
   - 端到端实验结果
   
6. 总结与展望 (2 页)
   - 主要贡献
   - 局限性与未来方向
   
= 约 20-22 页，讲 25-30 分钟 =
```

### 5.2 可能的答辩提问及准备

| 问题 | 准备 |
|---|---|
| "你和 ETC/TrustRAG 的本质区别是什么？" | 引用 [ETC_TRUSTRAG_DIFF.md](ETC_TRUSTRAG_DIFF.md)，强调三信号融合、查询重构、证据图、四维评估、闭环协同 |
| "为什么选 Qwen3 而不是 Qwen2.5？" | Qwen3 是最新开源模型；GQA 更高效；32K 上下文；复现门槛更低 |
| "τ=0.62 是怎么定的？" | 验证集 PR 曲线 F1 最大点；Qwen3 上重新标定 |
| "拒答会不会太保守？" | 宁可拒答不可错答是军事场景的基本要求；报告拒答精确率/召回率 |
| "数据集怎么构造的？有没有数据合规问题？" | 公开语料 + 合成填充 + 无涉密内容；sensitive/ 永不入 git |
| "实验的 5 种子均值±标准差合理吗？" | 学术规范；所有结果都可复现 |

### 5.3 提交前检查

```
[ ] Git 仓库干净，无涉密文件残留
[ ] experiments/ 下有 31 个实验的完整日志和结果
[ ] PROGRESS.md 数值对账表全部填写
[ ] 论文中所有超参与 config/ 中一致
[ ] 论文中所有实验表中的数值与实验结果 JSON 一致
[ ] README.md 更新了运行指南和依赖
[ ] 单测 47/47 全通过（sklearn 安装后）
[ ] requirements.txt 版本锁定无冲突
[ ] 盲审 / 查重版本已去除个人身份信息
```

---

## 时间线总览

```
第 1 周：
  Day 1:  阶段 0 — 环境搭建 + 模型下载
  Day 2-4: 阶段 1 — 数据构造
  Day 5:  阶段 1 — 数据构造收尾 + 最小验证集跑通

第 2 周：
  Day 1:  阶段 2 — 阈值重标定
  Day 2-7: 阶段 3 — 跑 31 个实验（可后台持续运行）

第 3 周：
  Day 1-2: 阶段 3 — 实验收尾 + 结果整理
  Day 3-5: 阶段 4 — 论文修改（填入实测数据）
  Day 6-7: 阶段 5 — PPT + 答辩准备

总计：约 3 周（含 GPU 运行等待时间）
```

---

## 风险提示

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| Qwen3-8B 生成质量显著低于 Qwen2.5-7B | 低 | 高 | 保留 Qwen2.5 作为对照；论文中如实分析 |
| 模型下载不完整/中断 | 中 | 中 | 用 ModelScope 镜像；用 aria2 多线程 |
| GPU 资源排队/不可用 | 中 | 高 | 提前充值；选择非高峰时段跑长实验 |
| 对抗数据集构造出 bug 导致 ASR 异常 | 中 | 中 | 先用最小集验证流水线正确性 |
| 阈值重标定后与原值差距大 | 中 | 低 | 这是预期内的（不同模型空间），如实记录 |
| 31 个实验中有 1-2 个无法复现 | 中 | 中 | 优先排查代码 bug；若确为模型差异，在论文中标注"Qwen3 上该实验暂未复现" |
| ES/PG 服务不稳定 | 低 | 低 | 允许降级为纯 FAISS 模式运行 |

---

## 快速验收标准

以下条件全部满足 → 达到毕业答辩条件：

1. ✅ 31 个实验均跑出 5 种子均值±标准差
2. ✅ Exp 4-1 核心指标 F1 ≥ 58（Qwen3-8B 预期不低于 Qwen2.5-7B 的 61.5 太多）
3. ✅ Exp 5-1 ASR ≤ 35（无论 Qwen2.5 还是 Qwen3，有显著下降）
4. ✅ Exp 5-8 协同 ASR 低于 Exp 5-1 单体 ASR（证明协同有效）
5. ✅ 论文实验表全部填入实测值，来源可追溯到 `experiments/` 下的 JSON
6. ✅ 所有阈值来自 config，不硬编码，可在验证集上复现标定过程
7. ✅ 代码在另一台机器上可以按 README 步骤跑通
