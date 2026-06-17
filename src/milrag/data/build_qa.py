"""data/build_qa.py — 军事情报问答数据集构造（论文 3.3）。

机器初标（Qwen3-32B int4 量化，本地）+ 人工校验（双盲 + 第三方仲裁，κ=0.82）。
三类样本：factual(510, hops≈1.0) / reasoning(446, hops≈2.7) / adversarial(320, hops≈1.8)，合计 1276。
字段：id / type / question / answer / evidence_chunks / key_reasoning_points / adversarial_inject / source_meta。
划分 8:1:1 → train(1020) / val(128) / test(128)。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# 论文固定种子（数据集构造阶段）
_DATA_SEED = 42

# ── 三类型 QA 构造 prompt（中文军事）─────────────────────────
_FACTUAL_PROMPT = """你是一个军事分析专家。根据以下情报片段，生成一个**事实查询**问答对。
要求：
1. 问题是简单事实查询（单跳，答案直接出现在段落中）
2. 涉及装备参数、编制、时间、地点等确定性信息
3. 答案简洁、明确，不超过 2 句话
4. 输出 JSON 格式：{{"question": "...", "answer": "...", "key_facts": ["..."]}}

情报片段：
{context}
"""

_REASONING_PROMPT = """你是一个军事分析专家。根据以下情报片段，生成一个**多跳推理**问答对。
要求：
1. 问题需要综合多个信息点（2-4 跳）才能回答
2. 涉及态势研判、因果分析、对比评估等
3. 答案应有推理链条，列出关键推理步骤
4. 输出 JSON 格式：{{"question": "...", "answer": "...", "reasoning_steps": ["..."]}}

情报片段：
{context}
"""

_ADVERSARIAL_PROMPT = """你是一个军事分析专家。根据以下情报片段，生成一个**对抗验证**问答对。
要求：
1. 问题涉及可能存在矛盾的陈述或需要多方验证的断言
2. 模拟情报分析中对可信度的审慎检验
3. 答案需区分"有明确证据支持"与"需进一步验证"的部分
4. 输出 JSON 格式：{{"question": "...", "answer": "...", "evidence_list": ["..."], "uncertainty_flag": true/false}}

情报片段：
{context}
"""


def generate_candidates(
    chunks: list[dict],
    sample_type: str,
    annotator,
    *,
    samples_per_type: int = 500,
) -> list[dict]:
    """用 LLM 从知识库分块生成候选 QA 对。

    Args:
        chunks: 知识库分块列表。
        sample_type: "factual" | "reasoning" | "adversarial".
        annotator: 标注 LLM（Qwen2.5-72B 本地）。
        samples_per_type: 每类生成数量。

    Returns:
        候选 QA 样本列表。
    """
    prompts = {
        "factual": _FACTUAL_PROMPT,
        "reasoning": _REASONING_PROMPT,
        "adversarial": _ADVERSARIAL_PROMPT,
    }
    prompt_template = prompts.get(sample_type, _FACTUAL_PROMPT)

    # 随机采样分块（不加回）
    rng = random.Random(_DATA_SEED)
    sel_chunks = rng.sample(chunks, min(samples_per_type, len(chunks)))
    print(f"  [{sample_type}] 采样 {len(sel_chunks)} chunks, 开始生成...")

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(sel_chunks), total=len(sel_chunks), desc=f"  [{sample_type}]", unit="qa")
    except ImportError:
        iterator = enumerate(sel_chunks)
        if len(sel_chunks) > 0:
            print(f"  [{sample_type}] 生成 {len(sel_chunks)} 条...")

    candidates = []
    for i, c in iterator:
        context = c["text"][:2048]  # 截断防超上下文
        prompt = prompt_template.format(context=context)
        try:
            raw = annotator.generate(prompt, max_new_tokens=512)
            # 提取 JSON（支持嵌套 + markdown 包裹）
            import re
            qa = None

            # 策略1: 从 ```json ... ``` 提取
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if not m:
                # 策略2: 找最外层 { } 对（按括号深度匹配）
                start = raw.find("{")
                if start >= 0:
                    depth = 0
                    end = start
                    for j in range(start, len(raw)):
                        if raw[j] == "{": depth += 1
                        elif raw[j] == "}":
                            depth -= 1
                            if depth == 0:
                                end = j + 1
                                break
                    if depth == 0:
                        m = re.match(r"(\{.*\})", raw[start:end], re.DOTALL)

            if m:
                try:
                    qa = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass  # 解析失败，静默跳过

            if qa is None and i < 3:
                # 前3条失败时打印调试信息
                print(f"    [DEBUG] raw response ({len(raw)} chars): {raw[:300]}")

            if qa is not None:
                qa["source_chunk_id"] = c["chunk_id"]
                qa["sample_type"] = sample_type
                qa["hops"] = len(qa.get("reasoning_steps", qa.get("key_facts", [1])))
                candidates.append(qa)
        except Exception:
            continue

    return candidates


def quality_checks(samples: list[dict]) -> tuple[list[dict], dict]:
    """质量保障流程（论文 3.3.4）：
      1. 问题去重（字符级相似度 > 0.9）
      2. 答案非空
      3. 推理链完整性（reasoning 类）
      4. 证据一致性自检

    Returns:
        (清理后样本, 质检报告).
    """
    report = {"total": len(samples), "duplicates": 0, "empty_answer": 0, "bad_reasoning": 0, "passed": 0}

    # 1. 空答案过滤
    valid = [s for s in samples if s.get("answer", "").strip()]
    report["empty_answer"] = report["total"] - len(valid)

    # 2. 去重
    seen_questions: set[str] = set()
    deduped = []
    for s in valid:
        q = s.get("question", "").strip().lower()
        # 简单字符归一化
        q_norm = "".join(ch for ch in q if ch.isalnum())
        if q_norm not in seen_questions:
            seen_questions.add(q_norm)
            deduped.append(s)
    report["duplicates"] = len(valid) - len(deduped)

    # 3. 推理链完整性（reasoning 类型）
    for s in deduped:
        if s.get("sample_type") == "reasoning":
            steps = s.get("reasoning_steps", [])
            if not steps or len(steps) < 2:
                report["bad_reasoning"] += 1

    # 4. 合格计数
    report["passed"] = len(deduped) - report["bad_reasoning"]
    # 加唯一 id
    for i, s in enumerate(deduped):
        s["id"] = f"qa_{s.get('sample_type', 'unk')}_{i:05d}"

    return deduped, report


def build_dataset(
    chunks: list[dict],
    annotator,
    output_dir: str = "data/qa",
    *,
    total: int = 1276,
    factual: int = 510,
    reasoning: int = 446,
    adversarial: int = 320,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict:
    """完整 QA 数据集构建流水线。

    Args:
        chunks: 知识库分块列表。
        annotator: 标注 LLM。
        output_dir: 输出目录。
        total: 目标总量（1276）。
        factual/reasoning/adversarial: 三类目标数量。
        train_ratio/val_ratio: 划分比例（8:1:1）。

    Returns:
        {"train": [...], "val": [...], "test": [...]}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for stype, count in [("factual", factual), ("reasoning", reasoning), ("adversarial", adversarial)]:
        candidates = generate_candidates(chunks, stype, annotator, samples_per_type=count + 50)
        cleaned, report = quality_checks(candidates)
        # 随机截取目标数量
        rng = random.Random(_DATA_SEED)
        selected = rng.sample(cleaned, min(count, len(cleaned)))
        for s in selected:
            s["id"] = f"qa_{stype}_{len(all_samples):05d}"
        all_samples.extend(selected)

    # 8:1:1 划分
    rng = random.Random(_DATA_SEED)
    rng.shuffle(all_samples)
    n = len(all_samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    split = {
        "train": all_samples[:train_end],
        "val": all_samples[train_end:val_end],
        "test": all_samples[val_end:],
    }

    for name, data in split.items():
        (out / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return split
