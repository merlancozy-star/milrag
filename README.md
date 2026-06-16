# milrag — Military Intelligence Dynamic & Robust RAG

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![torch 2.1](https://img.shields.io/badge/torch-2.1-red)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**面向中文军事情报问答的检索增强生成方法研究** — 完整实验代码库。

复现并扩展硕士论文的三条主线：

| 章节 | 方法 | 核心贡献 |
|---|---|---|
| 第 3 章 | 领域知识库 + 嵌入 LoRA 微调 | 1276 条 QA 构建；BGE-LoRA 将 R@10 从 71.4 提升到 78.6 |
| 第 4 章 | 多信号动态检索与查询重构 | token概率+注意力+熵三信号融合触发；三类自适应查询重构 |
| 第 5 章 | 对抗环境两阶段可信过滤与拒答 | 聚类前置过滤+证据关系图+四维自评估；ASR 从 47.5 降到 32.5 |
| 第 6 章 | 三层架构离线部署 | FastAPI + vLLM + Streamlit；完全离线内网运行 |

两个外部参考实现（均 MIT 协议，复用边界见 `CLAUDE.md §4`）：
- [ETC](https://github.com/WisdomShell/ETC) (AAAI'26 Oral) — 熵趋势触发，对应第 4 章
- [TrustRAG](https://github.com/HuichiZhou/TrustRAG) (arXiv 2501.00879) — 聚类过滤+冲突消解，对应第 5 章

---

## 快速开始

### 环境要求

```
Python 3.10
torch 2.4.x  |  transformers 4.51  |  peft 0.14
vllm 0.7     |  faiss 1.8.0       |  elasticsearch 8.13
sentence-transformers (Qwen3-Embedding)  |  postgresql 16
硬件: vGPU-48GB 单卡（Qwen3-14B 需 device_map="auto"）
```

### 安装

```bash
git clone https://github.com/YOUR_USERNAME/milrag.git
cd milrag
pip install -r requirements.txt
```

### 运行实验

```bash
# 单实验复现
bash scripts/run_exp4_1.sh    # 动态检索总体对比 (F1 ~61.5)

# 全部实验
bash scripts/run_all.sh

# 评测入口
python -m eval.run_eval --config config/experiments/exp4_1.yaml
```

每个实验落盘 `experiments/<exp_id>/<timestamp>/`，自动记录 git hash + config 快照 + 种子。

---

## 项目结构

```
milrag/
├── CLAUDE.md                    # 项目最高优先级约定（先读这个！）
├── config/                      # 超参唯一真相（base + retrieval + dynamic + defense + 31个实验配置）
├── docs/                        # 分章设计文档 (ch3_data.md ~ ch6_system.md)
├── data/                        # 原始语料 / 知识库 / QA / 对抗变体
├── src/milrag/
│   ├── data/                    # 清洗 / NER / 分块 / 索引 / QA构造 / 投毒构造
│   ├── retrieval/               # 嵌入封装 / LoRA微调 / 混合RRF / Cross-Encoder重排
│   ├── dynamic/                 # ★三信号 / ETC熵趋势 / 判别器 / 分类器 / 重构 / 循环
│   ├── defense/                 # ★聚类过滤 / 注入检测 / 先验 / 一致性+图 / 自评估 / 决策
│   ├── pipeline/                # RAG基线 / ★端到端编排(4+5章协同) / 部署服务
│   ├── llm/                     # 双后端(vLLM/HF eager) / 白盒信号钩子
│   └── utils/                   # 日志 / 种子 / IO
├── eval/
│   ├── metrics.py               # 全部指标实现（检索/生成/鲁棒/效率）
│   └── run_eval.py              # 统一评测入口 → 31个实验全覆盖
├── scripts/                     # 31个实验复现脚本 (run_exp3_1.sh ~ run_exp6_6.sh)
├── experiments/                 # 产出根目录（logs/results/ckpts）
└── tests/                       # 核心算法单测（46/47 pass）
```

---

## 硬约束

1. **白盒模型** — 触发检测依赖 logits/attention，只用开源模型（Qwen3-8B 主线）
2. **完全离线** — 运行时零外网请求，所有模型/索引本地加载
3. **数据合规** — 不提交涉密语料，示例只用合成/公开样本
4. **复现性** — 固定 5 种子，报均值±标准差，超参只能来自 `config/`
5. **不夸大** — 学术实验，指标如实，不为凑数改阈值

---

## 评测指标

| 类别 | 指标 |
|---|---|
| 检索 | Recall@{1,5,10,20}, MRR, nDCG@10 |
| 生成 | EM, F1, Faithfulness (NLI), Answer Relevance |
| 鲁棒 | ASR↓, PDR↓, RP↑, Recall_clean↑, FPR↓, 拒答 P/R, F1_Robust |
| 效率 | N̄_R, 平均延迟, Token 消耗 |

---

## 论文实验结果（验收基准）

| 实验 | 核心指标 | 论文值 |
|---|---|---|
| Exp 3-1 | BGE-large-zh R@10 (LoRA) | 78.6 |
| Exp 3-6 | +Cross-Encoder R@10 | 81.3 |
| Exp 4-1 | 动态检索 F1 (vs DRAGIN 60.1) | 61.5 |
| Exp 5-1 | 对抗鲁棒 ASR↓ (ρ=10%) | 32.5 |
| Exp 5-8 | 第4+5章协同 ASR↓ | 28.3 |
| Exp 6-1 | 端到端 F1 / ASR | 61.5 / 28.3 |

---

## 引用

若本工作对您的研究有帮助：

```bibtex
@mastersthesis{milrag2025,
  title     = {面向军事情报问答的检索增强生成方法研究},
  school    = {},
  year      = {2025},
  note      = {代码: \url{https://github.com/YOUR_USERNAME/milrag}}
}
```

## License

MIT — 本仓库为学术实验代码。参考实现 ETC 与 TrustRAG 均为 MIT 许可。
