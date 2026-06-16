# 第4章 design doc — 动态检索与自适应查询重构

## 目标
一次性 Top-K 难覆盖多跳证据；把检索从生成前一次性步骤改为生成中按需触发。

## 参考实现：ETC（复用边界见 CLAUDE.md §4.1）
借鉴熵趋势（一阶/二阶差分+平滑）增强 s_h；不替换 s_a 与查询重构（本论文增量）。

## 论文小节 ↔ 模块
| 论文 | 模块 |
|---|---|
| 4.3.1 三信号 | dynamic/signals.py, entropy_trend.py |
| 4.3.2 综合判别S(t) | dynamic/detector.py |
| 4.3.3 任务权重 | dynamic/detector.py + classifier.py |
| 4.4 查询重构 | dynamic/reformulate.py + selector.py |
| 4.5 循环/终止 | dynamic/loop.py |
| 4.5.2 证据融合 | dynamic/fuse.py |
| 白盒钩子 | llm/hooks.py, backbone.py |

## 关键超参
Δt8；τ0.62；Kmax4(简单1)；Lstop16；Lmax1024；温度0.3；s_a 末L4层；β0.6；上下文100字；回退sim0.55。
权重表：装备0.6/0.2/0.2，态势0.2/0.5/0.3，对抗0.2/0.2/0.6；分类器acc92.8%，差<0.15回退均值。

## 实验与验收（论文表 4-2~4-4）
- Exp4-1：本文 F1 61.5（朴素50.3，DRAGIN60.1）；代价 N_R2.8/延迟3.92s（最高）
- Exp4-2：三信号融合61.5 > 最好单信号~51.6
- Exp4-7：14B→65.2，DeepSeek-R1-7B→64.0
## 局限
对简单事实收益有限；长链问题后续检索偏离原问题（误差累积）；用更高延迟换准确率。

## ★工程坑（CLAUDE.md §5）
白盒信号必须 HF eager attention；vLLM 不暴露逐步 logits/attn，只用于无动态检索基线。
