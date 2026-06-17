# 一次性下载脚本

> 此目录是项目中**唯一允许联网**的位置。运行时全程离线（`HF_HUB_OFFLINE=1`）。
> 涉密/脱密语料不在此处，由合作单位线下交付到 `data/raw/sensitive/`（git 忽略）。

## 脚本清单

### 工具脚本

| 脚本 | 用途 |
|---|---|
| `corpus_manifest.py` | 采集完成后生成语料清单 `data/raw/corpus_manifest.json` |
| `validate_corpus.py` | 采集后质量校验（总量/分布/格式/空段检测） |
| `run_acquisition.sh` | 主控脚本，顺序/并行调度所有下载 |

### Phase 1: CMNEE + 中文官方源

| 脚本 | 来源 | 预估段数 | 语言 |
|---|---|---|---|
| `import_cmnee.py` | CMNEE 数据集导入 | ~100,000 | zh |
| `download_cn_white_papers.py` | 中国国防白皮书 (mod.gov.cn / scio.gov.cn) | ~5,000 | zh |
| `download_cn_doctrine.py` | PLA 公开条令教材 | ~5,472 | zh |
| `download_81cn.py` | 中国军网 (81.cn) | ~20,000 | zh |
| `download_mod_gov.py` | 国防部官网 (mod.gov.cn) | ~10,000 | zh |
| `download_xinhua_military.py` | 新华网/人民网军事频道 | ~8,000 | zh |
| `download_baidu_baike_military.py` | 百度百科军事词条 | ~12,000 | zh |

### Phase 2: 维基百科

| 脚本 | 来源 | 预估段数 | 语言 |
|---|---|---|---|
| `download_zhwiki_military.py` | 中文维基百科军事分类 | ~50,000 | zh |
| `download_enwiki_military.py` | 英文维基百科军事分类 | ~40,000 | en |

### Phase 3: 美军条令 + 国际源

| 脚本 | 来源 | 预估段数 | 语言 |
|---|---|---|---|
| `download_us_army_fm.py` | US Army Field Manuals | ~20,000 | en |
| `download_us_joint_pubs.py` | US Joint Publications | ~15,000 | en |
| `download_declassified.py` | FAS/CREST 解密文档 | ~8,000 | en |
| `download_international_defense.py` | DefenseNews/BreakingDefense/IISS | ~9,690 | en |
| `download_professional_military.py` | 环球网/观察者/Janes/RAND | ~10,000 | zh/en |
| `download_academic_military.py` | CNKI/arXiv 军事论文摘要 | ~9,854 | zh/en |

### Phase 4: 合成生成 + 合作数据

| 脚本 | 来源 | 预估段数 | 语言 |
|---|---|---|---|
| `generate_synthetic_kb.py` | Qwen3 合成填充 | 按缺口量 | zh |
| `import_partner_data.py` | 合作单位脱密数据 | ~7,415 | zh |

## 使用方式

```bash
# 一键运行全部采集（在联网 GPU 服务器上）
bash scripts/download/run_acquisition.sh

# 或分阶段运行
python scripts/download/import_cmnee.py --input /path/to/cmnee
python scripts/download/download_zhwiki_military.py
# ...
python scripts/download/validate_corpus.py
```

## 输出规范

每个脚本输出到 `data/raw/<source_name>/`：
- `*.txt` — 清洗前的原始文本
- `*.meta.json` — 元信息（来源 URL、采集时间、类别、权威等级、语言）
- `errors.log` — 失败记录（如有）
- `checkpoint.json` — 断点续传状态
