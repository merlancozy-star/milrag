# 数据集

> 此目录下仅存放小样本数据用于代码验证。完整的 315K 段落数据集需在 GPU 服务器上重新生成。

## 目录说明

```
data/
├── raw/                    # 原始语料（每个源 ≤ 100 条样本）
│   ├── academic_military/  # 军事学术摘要（10 条样本）
│   ├── baidu_baike_military/ # 百度百科军事词条（10 条样本）
│   ├── international_defense/ # 国际防务媒体（66 条样本）
│   ├── partner_data/       # 合作单位脱密数据（5 条合成样本）
│   ├── professional_military/ # 专业军事分析（1 条样本）
│   ├── synthetic_kb/       # 合成知识库（200 条样本：4类×50）
│   ├── xinhua_military/    # 新华网军事（94 条样本）
│   ├── zhwiki_military/    # 中文维基百科军事（100 条样本）
│   └── sensitive/          # 涉密数据目录（.gitignore，永不提交）
├── kb/                     # 知识库索引（空，需构建）
├── qa/                     # QA 数据集（空，需生成）
└── adversarial/            # 对抗变体（空，需生成）
```

## 完整数据集生成步骤

在 GPU 服务器上依次执行：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 语料采集（需要网络）
bash scripts/download/run_acquisition.sh    # Linux/Mac
# 或
.\scripts\download\run_acquisition.ps1       # Windows

# 3. 合成KB填充（不需要GPU — 纯模板生成）
python scripts/download/generate_synthetic_kb_enhanced.py \
    --equipment 75000 --doctrine 51000 --situation 139000 --case 35000

# 4. 清洗 + 分块
python scripts/run_clean.py
python scripts/run_chunk.py

# 5. 构建索引（需要 GPU 嵌入编码 + ES/PG 服务）
python scripts/run_index.py

# 6. QA 数据集构建（需要 Qwen3-32B int4 GPU）
python scripts/run_build_qa.py

# 7. 对抗变体构造
python scripts/run_build_adversarial.py

# 8. 生成语料清单
python scripts/download/corpus_manifest.py
```

## 目标数据规模（论文表 3-1）

| 来源 | 目标段落数 |
|---|---|
| CMNEE 等开源军事情报 | 174,328 |
| 公开条令/解密手册 | 53,472 |
| 军事百科/专业网站 | 31,854 |
| 权威防务媒体 | 47,690 |
| 合作单位脱密语料 | 7,415 |
| **合计** | **314,759** |

## 数据合规

- `data/raw/sensitive/` 永久 .gitignore——不得提交真实涉密语料
- 所有示例和单测仅用合成或公开样本
- 运行时完全离线（`HF_HUB_OFFLINE=1`）
