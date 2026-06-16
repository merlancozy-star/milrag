# 第6章 design doc — 系统集成与综合评估

> **2026-06 Qwen3 迁移**：LLM 主线 Qwen3-8B，嵌入 Qwen3-Embedding-4B，重排 Qwen3-Reranker-8B。旧版 Qwen2.5 数值保留为对照基线。

## 三层架构（论文 6.2）
数据层（知识库/嵌入/FAISS+ES/PG元信息）/ 控制层（vLLM/检测器/重构器/循环）/ 安全防御层（聚类/一致性/自评估/决策）。
前端 Streamlit；防御模块 FastAPI+gRPC 微服务；完全离线内网。

## 模块 ↔ 实现
| 论文 | 模块 |
|---|---|
| 6.2 编排+协同 | pipeline/orchestrator.py |
| 6.3.2 主循环 | dynamic/loop.py |
| 6.3.3 信号采集 | llm/hooks.py |
| 6.2/6.3.4/6.3.5 部署 | pipeline/serve.py |

## 部署环境（论文 6.4）
vGPU-48GB；Ubuntu22.04；py3.10/torch2.1/vllm0.4/transformers4.40/faiss1.7.4/ES8.13/PG16。
P95 延迟：基础事实≤3s，复杂推理≤10s。

## 实验与验收（论文表 6-3~6-7）
- Exp6-1：本文 F1 61.5/Faith76.2/ASR28.3/延迟3.98/N_R2.8
- Exp6-2：朴素→完整 F1+~11pt，ASR-~19pt；防御误删致F1此消彼长
- Exp6-3：14B F1 65.2（延迟5.64s）
- Exp6-6：专家盲评(8人×30) 准确性+0.71/可信度+0.88（小样本，不下强结论）
