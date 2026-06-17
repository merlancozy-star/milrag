#!/usr/bin/env python3
"""对抗变体构造流水线。
读取 data/qa/ 下的干净 QA 样本，构造 3 策略 × 4 投毒率 = 12 组对抗数据集。

用法: python scripts/run_build_adversarial.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from milrag.data.build_adversarial import build_adversarial_dataset

    # 加载 QA 数据
    all_samples = []
    qa_dir = Path("data/qa")
    for split_name in ["train.json", "val.json", "test.json"]:
        split_path = qa_dir / split_name
        if split_path.exists():
            data = json.loads(split_path.read_text(encoding="utf-8"))
            all_samples.extend(data)

    if not all_samples:
        print("错误: data/qa/ 下没有 QA 数据，请先运行 scripts/run_build_qa.py")
        sys.exit(1)

    print(f"对抗变体构造: {len(all_samples)} 条干净 QA")
    print(f"  策略: direct_contradiction, partial_substitution, prompt_injection")
    print(f"  投毒率: 1%, 5%, 10%, 20%")

    datasets = build_adversarial_dataset(
        all_samples,
        output_dir="data/adversarial",
        ratios=[0.01, 0.05, 0.10, 0.20],
        strategies=["direct_contradiction", "partial_substitution", "prompt_injection"],
        seed=42,
    )

    for key, data in datasets.items():
        poisoned = sum(1 for s in data if "adversarial_inject" in s)
        print(f"  {key}: {len(data)} 样本 ({poisoned} 投毒)")


if __name__ == "__main__":
    main()
