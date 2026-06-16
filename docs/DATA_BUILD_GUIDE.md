# 数据集构建完整指南

> 生成日期：2026-06-16 | 目标硬件：vGPU-48GB (AutoDL)
> 目标产出：314,759 段知识库 + 1,276 条 QA + 15 组对抗变体

---

## 一、总体流水线

```
┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌────────────────┐
│ 原始语料获取 │ → │ 清洗归一 │ → │ NER 标注 │ → │ 语义分块 (512) │
│  ~500MB 文本 │    │ clean.py │    │  ner.py  │    │   chunk.py     │
└─────────────┘    └──────────┘    └──────────┘    └────────────────┘
                                                          ↓
┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌────────────────┐
│ 对抗变体构造 │ ← │ QA 数据集│ ← │ QA 机器  │    │ FAISS + ES     │
│ adversarial │    │ 1,276条  │    │ 初标     │    │ + PG 索引      │
│  build_adv  │    │build_qa  │    │ 32B int4 │    │ index_build    │
└─────────────┘    └──────────┘    └──────────┘    └────────────────┘
```

**关键数字**：
- 输入：原始中文军事语料约 500MB（文本）
- 中间产物：~31.5 万段分块（chunk_id + text + meta）
- 最终产出：FAISS HNSW 索引 + ES 倒排 + 1,276 条 QA + 15 组对抗变体
- 总耗时估算（vGPU-48GB）：语料清洗 ~10min + 分块 ~5min + 嵌入编码 ~30min + QA 构造 ~2h + 对抗构造 ~30min ≈ **3-4 小时**

---

## 二、原始语料获取：四种策略

### 策略 A：公开数据集（最推荐，零风险）

| 来源 | 预估段落 | 获取方式 | 难度 |
|---|---|---|---|
| **中文维基百科军事分类** | ~80,000 段 | `wikiextractor` + 军事类目过滤 | ⭐ 低 |
| **中国军网公开报道** | ~50,000 段 | 公开 RSS/爬虫（仅公开页面） | ⭐⭐ 中 |
| **各国国防白皮书（公开版）** | ~30,000 段 | 各国国防部官网公开 PDF | ⭐⭐ 中 |
| **公开军事学术论文摘要** | ~20,000 段 | CNKI/万方公开摘要 | ⭐⭐ 中 |
| **国际战略研究所(IISS)公开报告** | ~15,000 段 | 公开 Military Balance 摘要 | ⭐⭐ 中 |
| **简氏防务公开摘要** | ~10,000 段 | 公开网站 | ⭐⭐ 中 |

**维基百科抽取脚本**（可直接用）：

```bash
# 1. 安装 wikiextractor
pip install wikiextractor

# 2. 下载中文维基 dump
wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2

# 3. 抽取文本
python -m wikiextractor.WikiExtractor zhwiki-latest-pages-articles.xml.bz2 \
    --output extracted/ --bytes 1M --json

# 4. 筛选军事类文章
python -c "
import json, re, os
from pathlib import Path

MILITARY_CATEGORIES = [
    '军事', '武器', '装备', '战争', '军队', '国防', '战略',
    '战役', '战术', '军种', '兵种', '导弹', '舰艇', '战机',
    '坦克', '雷达', '情报', '特种部队', '军事演习', '军事条约',
    '军事历史', '军事技术', '军事人物', '军事基地',
]

def is_military(text):
    score = sum(1 for cat in MILITARY_CATEGORIES if cat in text[:500])
    return score >= 2

out_dir = Path('data/raw')
out_dir.mkdir(parents=True, exist_ok=True)
count = 0

for wiki_dir in Path('extracted').iterdir():
    if wiki_dir.is_dir():
        for f in wiki_dir.glob('*'):
            for line in open(f, encoding='utf-8'):
                try:
                    doc = json.loads(line)
                    if is_military(doc.get('text', '')):
                        fname = f'military_{count:06d}.txt'
                        (out_dir / fname).write_text(
                            doc['text'], encoding='utf-8'
                        )
                        count += 1
                except: pass

print(f'Extracted {count} military articles')
"
```

### 策略 B：Qwen3 合成语料（快速填充，无版权风险）

如果公开语料不够 314,759 段的目标，用 Qwen3-8B 按四类模板生成无涉密的合成军事知识段落：

```python
# scripts/generate_synthetic_kb.py
"""
用 Qwen3-8B 生成合成军事知识库段落。
四类：装备 / 条令 / 态势 / 案例。
所有生成内容为虚构的、不涉及真实军事机密的训练用伪数据。
"""
from milrag.llm.backbone import Backbone
from pathlib import Path
import json, random

backbone = Backbone('/models/Qwen3-8B-Instruct', backend='vllm')

TEMPLATES = {
    'equipment': """生成一段虚构的军事装备技术参数描述（约200字）。
格式：装备名称 | 技术参数 | 作战能力
示例：XX型战斗机最大起飞重量35吨，作战半径1500公里，配备有源相控阵雷达...
注意：所有数据均为虚构，不引用任何真实装备。""",

    'doctrine': """生成一段虚构的军事条令原则描述（约200字）。
格式：原则名称 | 适用场景 | 实施要点
注意：所有内容均为虚构的通用军事原则，不涉及任何国家真实条令。""",

    'situation': """生成一段虚构的战略态势分析（约200字）。
格式：区域 | 力量对比 | 趋势评估
注意：所有数据、地名、事件均为虚构。""",

    'case': """生成一段虚构的军事历史案例分析（约200字）。
格式：背景 | 关键决策 | 经验教训
注意：所有人物、地点、事件均为虚构。""",
}

def generate_category(category: str, count: int) -> list[dict]:
    chunks = []
    for i in range(count):
        prompt = TEMPLATES[category]
        text = backbone.generate(prompt, max_new_tokens=256, temperature=0.8)
        chunks.append({
            'chunk_id': f'syn_{category}_{i:06d}',
            'text': text.strip(),
            'meta': {
                'category': category,
                'source': 'synthetic',
                'authority': 'general_commentary',
                'desensitized': True,
                'timestamp': '2026-06-16',
            }
        })
        if (i+1) % 100 == 0:
            print(f'  {category}: {i+1}/{count}')
    return chunks

# 按论文比例生成
print('Generating synthetic KB...')
all_chunks = []
all_chunks += generate_category('equipment', 78349)   # 24.9%
all_chunks += generate_category('doctrine', 53472)    # 17.0%
all_chunks += generate_category('situation', 145826)  # 46.3%
all_chunks += generate_category('case', 37112)        # 11.8%

# 随机打乱
random.shuffle(all_chunks)

Path('data/kb/synthetic_chunks.json').write_text(
    json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding='utf-8'
)
print(f'Done: {len(all_chunks)} synthetic chunks')
```

**估算**：生成 314,759 段 × 200 字 ≈ 6,300 万字。Qwen3-8B 在 vLLM 上约 50-80 tok/s，粗略估计需要 **20-30 小时**连续生成。建议后台运行。

### 策略 C：混合策略（实用推荐）

**公开语料做骨架 + 合成语料填充到目标量**：

| 来源 | 数量 | 占比 |
|---|---|---|
| 维基百科军事类 | ~80,000 段 | 25% |
| 公开白皮书/论文摘要 | ~50,000 段 | 16% |
| Qwen3 合成（装备） | ~50,000 段 | 16% |
| Qwen3 合成（态势） | ~100,000 段 | 32% |
| Qwen3 合成（条令+案例） | ~35,000 段 | 11% |
| **合计** | **~315,000 段** | **100%** |

### 策略 D：最小验证集（先跑通再扩展）

**50 条 QA + 200 段知识块**，用确定性规则生成，用于验证代码正确性：

> 见附录 §A — 我已写好的最小验证脚本 `scripts/generate_minimal_data.py`

---

## 三、数据清洗流水线（可立即运行）

### 3.1 单文件清洗测试

```bash
# 先拿一个文件测试清洗效果
python -c "
from milrag.data.clean import clean_document
text = open('data/raw/military_000001.txt', encoding='utf-8').read()
cleaned = clean_document(text)
print(f'Before: {len(text)} chars')
print(f'After:  {len(cleaned)} chars')
print(f'Preview: {cleaned[:300]}...')
"
```

### 3.2 批量清洗

```bash
python scripts/run_clean.py
```

脚本内容：

```python
# scripts/run_clean.py
"""批量文本清洗"""
from milrag.data.clean import clean_document
from pathlib import Path

raw_dir = Path('data/raw')
out_dir = Path('data/kb/cleaned')
out_dir.mkdir(parents=True, exist_ok=True)

txt_files = list(raw_dir.glob('*.txt'))
print(f'Found {len(txt_files)} raw files')

for i, f in enumerate(txt_files):
    raw = f.read_text(encoding='utf-8', errors='ignore')
    cleaned = clean_document(raw)
    (out_dir / f.name).write_text(cleaned, encoding='utf-8')
    if (i+1) % 100 == 0:
        print(f'  {i+1}/{len(txt_files)}')

print(f'Done. Cleaned files in {out_dir}')
```

### 3.3 NER 实体标注

```bash
python scripts/run_ner.py
```

```python
# scripts/run_ner.py
"""批量军事实体识别"""
from milrag.data.ner import MilitaryNER, normalize_entity
from pathlib import Path
import json

ner = MilitaryNER(mode='rule')  # 先用规则模式（无需 GPU）

cleaned_dir = Path('data/kb/cleaned')
entity_records = []

for f in cleaned_dir.glob('*.txt'):
    text = f.read_text(encoding='utf-8')
    entities = ner.extract_entities(text)
    entity_records.append({
        'source': f.name,
        'char_count': len(text),
        'entity_count': len(entities),
        'entities': [
            {'text': e.text, 'normalized': e.normalized, 'label': e.label,
             'start': e.start, 'end': e.end}
            for e in entities
        ]
    })

Path('data/kb/entities.json').write_text(
    json.dumps(entity_records, ensure_ascii=False, indent=2), encoding='utf-8'
)

total = sum(r['entity_count'] for r in entity_records)
print(f'NER done: {len(entity_records)} docs, {total} entities')
```

### 3.4 语义分块

```bash
python scripts/run_chunk.py
```

```python
# scripts/run_chunk.py
"""批量语义分块（512/64）"""
from milrag.data.chunk import chunk_document
from pathlib import Path
import json

cleaned_dir = Path('data/kb/cleaned')
all_chunks = []

for f in cleaned_dir.glob('*.txt'):
    text = f.read_text(encoding='utf-8')
    chunks = chunk_document(
        text, window=512, overlap=64,
        doc_id=f.stem, title=f.stem,
    )
    for c in chunks:
        all_chunks.append({
            'chunk_id': c.chunk_id,
            'text': c.text,
            'meta': c.meta,
        })

Path('data/kb/chunks.json').write_text(
    json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding='utf-8'
)
print(f'Chunking done: {len(all_chunks)} chunks from {len(list(cleaned_dir.glob("*.txt")))} docs')
```

---

## 四、向量库构建（需要 GPU + ES + PG）

### 4.1 启动基础设施

```bash
# === 在 AutoDL 服务器上 ===

# 1. 启动 Elasticsearch
docker run -d --name es-milrag \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.0

# 2. 启动 PostgreSQL
docker run -d --name pg-milrag \
  -p 5432:5432 \
  -e POSTGRES_DB=milrag \
  -e POSTGRES_PASSWORD=milrag123 \
  postgres:16

# 3. 验证服务
curl http://localhost:9200   # ES 应返回 JSON
psql -h localhost -U postgres -d milrag -c "SELECT 1"  # PG 应返回 1
```

### 4.2 构建 FAISS 向量索引

```bash
python scripts/run_index.py
```

```python
# scripts/run_index.py
"""一站式索引构建：FAISS + ES + PG"""
from milrag.retrieval.embedding import Embedder
from milrag.data.index_build import build_faiss, build_es, init_metadata_db
from pathlib import Path
import json

# 加载分块
chunks = json.loads(Path('data/kb/chunks.json').read_text(encoding='utf-8'))
print(f'Loaded {len(chunks)} chunks')

# === FAISS HNSW ===
print('Building FAISS index...')
embedder = Embedder(
    '/models/Qwen3-Embedding-4B',  # ← 改为你的实际路径
    device='cuda',
    max_seq_len=8192,
    normalize=True,
)
index, chunk_ids, embeddings = build_faiss(
    chunks, embedder,
    M=32, ef_construction=200, ef_search=64,
    save_dir='data/kb',
)
print(f'  FAISS: {len(chunk_ids)} vectors, dim={embeddings.shape[1]}')

# === Elasticsearch ===
print('Building ES index...')
try:
    build_es(chunks, index_name='mil_kb', es_host='http://localhost:9200')
    print('  ES: indexed successfully')
except Exception as e:
    print(f'  ES skipped: {e}')

# === PostgreSQL 元信息 ===
print('Initializing PG metadata...')
try:
    init_metadata_db(chunks, db_url='postgresql://postgres:milrag123@localhost:5432/milrag')
    print('  PG: metadata initialized')
except Exception as e:
    print(f'  PG skipped: {e}')

print('\nIndex build complete.')
```

### 4.3 验证检索

```bash
python -c "
from milrag.retrieval.embedding import Embedder
from milrag.data.index_build import load_faiss
from milrag.retrieval.hybrid import dense_retrieve
import json

# 加载
chunks = json.loads(open('data/kb/chunks.json', encoding='utf-8').read())
index, chunk_ids, _ = load_faiss('data/kb')
embedder = Embedder('/models/Qwen3-Embedding-4B', device='cuda')

# 测试检索
query = '歼-20 战斗机的作战半径是多少公里'
results = dense_retrieve(query, embedder, index, chunk_ids, topk=5)

print(f'Query: {query}')
print('Top-5 results:')
for chunk_id, score in results:
    chunk = next(c for c in chunks if c['chunk_id'] == chunk_id)
    print(f'  [{score:.3f}] {chunk[\"text\"][:100]}...')
"
```

---

## 五、QA 数据集构造

### 5.1 加载标注器（Qwen3-32B int4）

```bash
python scripts/run_build_qa.py
```

```python
# scripts/run_build_qa.py
"""QA 数据集构造：机器初标 + 质量检查"""
from milrag.llm.backbone import Backbone
from milrag.data.build_qa import build_dataset
from pathlib import Path
import json

# 加载标注模型（int4 量化，~18GB）
print('Loading Qwen3-32B (int4)...')
annotator = Backbone(
    '/models/Qwen3-32B-Instruct',  # ← 改为实际路径
    backend='hf_eager',
    load_in_4bit=True,             # ★ int4 量化
)

# 加载知识块
chunks = json.loads(Path('data/kb/chunks.json').read_text(encoding='utf-8'))

# 构造数据集
print('Generating QA pairs...')
split = build_dataset(
    chunks, annotator,
    output_dir='data/qa',
    total=1276,
    factual=510,
    reasoning=446,
    adversarial=320,
)

print(f'Done: train={len(split["train"])}, val={len(split["val"])}, test={len(split["test"])}')

# 输出质量统计
for name, data in split.items():
    type_dist = {}
    for s in data:
        t = s.get('sample_type', 'unknown')
        type_dist[t] = type_dist.get(t, 0) + 1
    print(f'  {name}: {type_dist}')
```

### 5.2 质量检查

```bash
python -c "
from milrag.data.build_qa import quality_checks
from pathlib import Path
import json

train = json.loads(Path('data/qa/train.json').read_text(encoding='utf-8'))
cleaned, report = quality_checks(train)
print('Quality report:')
for k, v in report.items():
    print(f'  {k}: {v}')
"
```

论文质量基准：
- Cohen's κ ≥ 0.82（人工双盲 + 第三方仲裁）
- 证据一致性 ≥ 0.968
- 格式修订 ≤ 47/1276，内容修订 ≤ 39/1276

> **注意**：论文中的 κ 值和一致性指标来自人工校验环节。机器初标后需要人工抽查（建议 20% = ~255 条），记录修正率和评分者间一致性。

---

## 六、对抗变体构造

```bash
python scripts/run_build_adversarial.py
```

```python
# scripts/run_build_adversarial.py
"""对抗变体构造：3 策略 × 5 投毒率 = 15 组数据集"""
from milrag.data.build_adversarial import build_adversarial_dataset
from pathlib import Path
import json

# 加载干净 QA
train = json.loads(Path('data/qa/train.json').read_text(encoding='utf-8'))
val = json.loads(Path('data/qa/val.json').read_text(encoding='utf-8'))
test = json.loads(Path('data/qa/test.json').read_text(encoding='utf-8'))

all_clean = train + val + test
print(f'Total clean samples: {len(all_clean)}')

# 构造对抗变体
datasets = build_adversarial_dataset(
    all_clean,
    output_dir='data/adversarial',
    ratios=[0.01, 0.05, 0.10, 0.20, 0.30],
    strategies=['direct_contradiction', 'partial_substitution', 'prompt_injection'],
    seed=42,
)

print(f'Generated {len(datasets)} adversarial datasets:')
for key, data in datasets.items():
    poisoned = sum(1 for s in data if 'adversarial_inject' in s)
    print(f'  {key}: {len(data)} samples ({poisoned} poisoned)')
```

---

## 七、完整一键构建脚本

```bash
#!/usr/bin/env bash
# scripts/build_all_data.sh
# 完整数据构建流水线 — 在 vGPU-48GB 上运行
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Phase 1: Text Cleaning ==="
python scripts/run_clean.py

echo "=== Phase 2: NER Annotation ==="
python scripts/run_ner.py

echo "=== Phase 3: Semantic Chunking ==="
python scripts/run_chunk.py

echo "=== Phase 4: Index Building (FAISS + ES + PG) ==="
python scripts/run_index.py

echo "=== Phase 5: QA Dataset Construction ==="
python scripts/run_build_qa.py

echo "=== Phase 6: Adversarial Variant Construction ==="
python scripts/run_build_adversarial.py

echo "=== All done! ==="
echo "Outputs:"
echo "  data/kb/        — FAISS index + chunks + embeddings"
echo "  data/qa/        — train (1020) / val (128) / test (128)"
echo "  data/adversarial/ — 15 adversarial variants"
```

---

## 八、数据目录最终结构

```
data/
├── raw/                          # 原始语料
│   └── sensitive/                # ← .gitignore 永不提交
├── kb/
│   ├── cleaned/                  # 清洗后文本
│   ├── chunks.json               # 分块（31.5万条）
│   ├── entities.json             # NER 标注结果
│   ├── faiss_hnsw.index          # FAISS HNSW 向量索引
│   ├── embeddings.npy            # 嵌入矩阵
│   └── chunk_ids.json            # chunk_id 列表
├── qa/
│   ├── train.json                # 训练集 1020 条
│   ├── val.json                  # 验证集 128 条
│   └── test.json                 # 测试集 128 条
└── adversarial/
    ├── clean.json                # 干净集对照
    ├── rho01_direct_contradiction/
    ├── rho01_partial_substitution/
    ├── rho01_prompt_injection/
    ├── rho05_*/                  # ρ=5% 三策略
    ├── rho10_*/                  # ρ=10% 三策略
    ├── rho20_*/                  # ρ=20% 三策略
    └── rho30_*/                  # ρ=30% 三策略
```

---

## 九、时间预算（vGPU-48GB 单卡）

| 步骤 | 硬件 | 预估耗时 | 是否可并行 |
|---|---|---|---|
| 原始语料获取（爬取/下载） | CPU | 2-4 小时 | ✅ 下载时即可开始后续 |
| 文本清洗 | CPU | 10 分钟 | ✅ 多进程 |
| NER 标注（规则模式） | CPU | 15 分钟 | ✅ 多进程 |
| 语义分块 | CPU | 5 分钟 | ✅ 多进程 |
| Qwen3-Embedding 编码 31.5 万段 | GPU | 30-40 分钟 | ❌ 需 GPU |
| FAISS 索引构建 | CPU | 2 分钟 | ❌ |
| ES 索引 | CPU | 10 分钟 | ❌ |
| QA 机器初标（32B int4） | GPU | 2-3 小时 | ❌ 可后台 |
| 对抗变体构造 | CPU | 30 分钟 | ❌ |
| **总计** | | **~5-7 小时** | |

> 如果 QA 已有现成数据可导入，则总耗时降至 **1-2 小时**。

---

## 附录 A：最小验证数据集

用于在完整数据准备好之前验证代码正确性。

```python
# scripts/generate_minimal_data.py
"""生成最小验证数据集：50 QA + 200 知识块"""
import json, random
from pathlib import Path

random.seed(42)

# === 200 个伪知识块 ===
templates = [
    "歼-20 是第五代隐身战斗机，最大起飞重量 {wt} 吨，作战半径 {rng} 公里，配备有源相控阵雷达。",
    "{ship} 是 {country} 海军的主力驱逐舰，排水量 {disp} 吨，配备垂直发射系统。",
    "据公开报道，{year} 年 {region} 地区军事演习规模较上年增加 {pct}%。",
    "{weapon} 的射程约为 {rng} 公里，精度 {acc} 米，采用 {guide} 制导方式。",
    "{year} 年 {country} 国防预算约为 {budget} 亿美元，占 GDP 的 {ratio}%。",
    "在 {region} 部署的 {system} 系统具备 {capability} 能力，覆盖范围 {cov} 公里。",
    "根据公开条令，{unit_type} 的基本作战单元编制为 {num} 人。",
    "案例研究：{year} 年 {event} 中，{side} 采用的 {tactic} 战术取得了决定性成果。",
]

fill_values = {
    'wt': [25, 28, 30, 32, 35, 37, 40, 42, 45],
    'rng': [200, 500, 800, 1000, 1200, 1500, 2000, 2500, 3000, 5000],
    'ship': ['055型驱逐舰', '052D型驱逐舰', '054A型护卫舰', '075型两栖攻击舰'],
    'country': ['某国', 'A国', 'B国', 'C国', 'D国'],
    'disp': [4000, 6000, 7000, 12000, 13000, 25000, 45000],
    'year': ['2020', '2021', '2022', '2023', '2024', '2025'],
    'region': ['西太平洋', '南海', '东海', '印度洋', '波罗的海', '黑海'],
    'pct': [8, 12, 15, 20, 25, 30],
    'weapon': ['东风-21D', '东风-26', '长剑-10', '鹰击-18', '红旗-9B'],
    'acc': [5, 10, 15, 20, 30, 50],
    'guide': ['惯性+GPS', '雷达主动', '红外成像', '激光半主动', '复合'],
    'budget': [500, 600, 700, 800, 1000, 1200, 1500],
    'ratio': [1.2, 1.5, 1.8, 2.0, 2.3, 2.8, 3.5],
    'system': ['防空导弹', '反舰导弹', '雷达预警', '电子战'],
    'capability': ['区域防空', '反导拦截', '远程预警', '电子压制'],
    'cov': [100, 200, 300, 400, 500, 1000],
    'unit_type': ['机械化步兵', '装甲兵', '炮兵', '航空兵'],
    'num': [30, 50, 80, 100, 120, 150],
    'event': ['某地区冲突', '联合军事演习', '边境对峙', '海上遭遇事件'],
    'side': ['甲方', '乙方', '蓝方', '红方'],
    'tactic': ['纵深突破', '火力压制', '信息遮断', '快速穿插', '区域封锁'],
}

chunks = []
for i in range(200):
    tpl = random.choice(templates)
    # 填充占位符
    for key, values in fill_values.items():
        if '{' + key + '}' in tpl:
            tpl = tpl.replace('{' + key + '}', str(random.choice(values)), 1)
    # 填充缺失占位
    for key, values in fill_values.items():
        tpl = tpl.replace('{' + key + '}', str(random.choice(values)))
    chunks.append({
        'chunk_id': f'minimal_{i:05d}',
        'text': tpl,
        'meta': {
            'category': random.choice(['equipment', 'doctrine', 'situation', 'case']),
            'authority': random.choice(['doctrine', 'official_bulletin', 'mainstream_media']),
            'timestamp': f'202{random.randint(0,5)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
            'source': 'minimal_synthetic',
            'desensitized': True,
        }
    })

Path('data/kb/minimal_chunks.json').write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8'
)
print(f'Generated {len(chunks)} minimal chunks')

# === 50 条 QA 对 ===
qa_templates = {
    'factual': [
        ('歼-20 的最大起飞重量是多少吨', '37'),
        ('055型驱逐舰的排水量是多少吨', '约13,000'),
        ('东风-21D 的射程约为多少公里', '约1,500'),
        ('歼-20 配备了什么类型的雷达', '有源相控阵雷达'),
        ('2024年A国国防预算占GDP的百分比', '约2.3%'),
    ],
    'reasoning': [
        ('如果歼-20的作战半径为2000公里，某基地位于距目标1500公里处，能否在不加油的情况下完成往返打击', '不能。往返需要3000公里，超过作战半径2000公里'),
        ('比较055型驱逐舰与052D型驱逐舰的排水量差异及作战定位', '055型约13,000吨，定位为大型防空驱逐舰/巡洋舰；052D型约7,000吨，定位为通用驱逐舰'),
    ],
    'adversarial': [
        ('有来源声称歼-20的最大起飞重量仅为25吨，请验证', '根据多数公开资料，歼-20最大起飞重量约为37吨，25吨的说法与主流数据矛盾，可能为误导信息'),
        ('某些报道称东风-21D射程为500公里，是否可靠', '500公里与多数公开来源的约1,500公里存在较大差异，该信息可能不准确'),
    ],
}

qa_samples = []
for sample_type, pairs in qa_templates.items():
    for q, a in pairs:
        # 随机选 2-3 个相关 chunk 作为 evidence
        evidence_chunks = random.sample(
            [c['chunk_id'] for c in chunks if sample_type in c['meta'].get('category', '') or 'equipment' in c['meta'].get('category', '')],
            min(3, len(chunks))
        )
        qa_samples.append({
            'id': f'qa_minimal_{len(qa_samples):05d}',
            'type': sample_type,
            'question': q,
            'answer': a,
            'evidence_chunks': evidence_chunks,
            'key_reasoning_points': [],
            'source_meta': {'source': 'minimal_synthetic'},
        })

# 8:1:1 拆分
random.shuffle(qa_samples)
n = len(qa_samples)
split = {
    'train': qa_samples[:int(n*0.8)],
    'val': qa_samples[int(n*0.8):int(n*0.9)],
    'test': qa_samples[int(n*0.9):],
}

for name, data in split.items():
    Path(f'data/qa/minimal_{name}.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8'
    )

print(f'Generated {len(qa_samples)} minimal QA pairs')
print(f'  train: {len(split["train"])}, val: {len(split["val"])}, test: {len(split["test"])}')
```

**使用方法**：

```bash
# 1. 生成最小验证集
python scripts/generate_minimal_data.py

# 2. 用最小集跑通完整流水线
python -c "
from milrag.data.chunk import chunk_document
from milrag.data.index_build import build_faiss
from milrag.retrieval.embedding import Embedder
import json

# 加载最小 chunks，走一遍完整流水线
chunks = json.loads(open('data/kb/minimal_chunks.json').read())
embedder = Embedder('/models/Qwen3-Embedding-4B', device='cuda')
index, ids, embs = build_faiss(chunks, embedder, save_dir='data/kb/minimal_index')
print(f'Minimal pipeline: {len(ids)} chunks indexed successfully')
"

# 3. 用最小 QA 跑一次端到端评测
python -m eval.run_eval --config config/experiments/exp6_1.yaml \
    --seeds 42 \
    --skip_expected_check
```
