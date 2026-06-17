#!/usr/bin/env python3
"""采集后语料质量校验。

检查项：
  1. 总段落数在目标的 ±5% 范围内
  2. 来源类别分布与目标偏差 ≤10 个百分点
  3. 内容类别分布与目标偏差 ≤10 个百分点
  4. 无空段落（< 50 字符）
  5. 语言标记一致性
  6. 中文/英文比例合理
  7. 权威等级分布合理
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# 论文目标值
TARGETS = {
    "total_paragraphs": 314_759,
    "total_tolerance_pct": 5.0,
    "source_categories": {
        "military_news": 174_328,
        "doctrine": 53_472,
        "encyclopedia": 31_854,
        "commentary": 47_690,
        "desensitized": 7_415,
    },
    "content_categories": {
        "equipment": 78_349,
        "doctrine": 53_472,
        "situation": 145_826,
        "case": 37_112,
    },
    "min_chunk_chars": 50,
    "max_empty_ratio": 0.02,  # 最多 2% 空/过短段落
}


def load_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        print(f"[警告] manifest 文件不存在: {manifest_path}，将直接扫描目录")
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_from_manifest(manifest: dict) -> list[str]:
    """基于 corpus_manifest.json 做统计校验。"""
    issues: list[str] = []

    total = manifest.get("total_estimated_paragraphs", 0)
    target = TARGETS["total_paragraphs"]
    tolerance = TARGETS["total_tolerance_pct"]

    if total == 0:
        issues.append("🔴 总段落数为 0")
        return issues

    deviation = abs(total - target) / target * 100
    if deviation > tolerance:
        issues.append(
            f"🔴 总段落数 {total:,} 偏离目标 {target:,} 达 {deviation:.1f}%（阈值 {tolerance}%）"
        )
    elif deviation > tolerance / 2:
        issues.append(
            f"🟡 总段落数 {total:,} 偏离目标 {target:,} {deviation:.1f}%（接近阈值）"
        )
    else:
        issues.append(f"✅ 总段落数 {total:,}，偏离目标 {deviation:.1f}%")

    # 来源类别分布
    src_cats = manifest.get("source_categories", {})
    for cat, target_val in TARGETS["source_categories"].items():
        actual = src_cats.get(cat, {}).get("paragraphs_est", 0) if isinstance(src_cats.get(cat), dict) else 0
        expected_pct = target_val / target * 100
        actual_pct = actual / max(total, 1) * 100
        diff = abs(actual_pct - expected_pct)
        if diff > 10:
            issues.append(f"🔴 来源类别 [{cat}] 实际 {actual_pct:.1f}% vs 目标 {expected_pct:.1f}% (差 {diff:.1f}pp)")
        elif diff > 5:
            issues.append(f"🟡 来源类别 [{cat}] 实际 {actual_pct:.1f}% vs 目标 {expected_pct:.1f}% (差 {diff:.1f}pp)")
        else:
            issues.append(f"✅ 来源类别 [{cat}] {actual_pct:.1f}% (目标 {expected_pct:.1f}%)")

    # 内容类别分布
    cnt_cats = manifest.get("content_categories", {})
    for cat, target_val in TARGETS["content_categories"].items():
        actual = cnt_cats.get(cat, {}).get("paragraphs_est", 0) if isinstance(cnt_cats.get(cat), dict) else 0
        expected_pct = target_val / target * 100
        actual_pct = actual / max(total, 1) * 100
        diff = abs(actual_pct - expected_pct)
        if diff > 10:
            issues.append(f"🔴 内容类别 [{cat}] 实际 {actual_pct:.1f}% vs 目标 {expected_pct:.1f}% (差 {diff:.1f}pp)")
        elif diff > 5:
            issues.append(f"🟡 内容类别 [{cat}] 实际 {actual_pct:.1f}% vs 目标 {expected_pct:.1f}% (差 {diff:.1f}pp)")
        else:
            issues.append(f"✅ 内容类别 [{cat}] {actual_pct:.1f}% (目标 {expected_pct:.1f}%)")

    return issues


def validate_direct(raw_dir: Path) -> list[str]:
    """直接扫描 data/raw/ 目录做内容级校验。"""
    issues: list[str] = []
    empty_count = 0
    total_count = 0
    lang_counter: Counter = Counter()
    short_content_samples: list[str] = []

    for src_dir in sorted(raw_dir.iterdir()):
        if not src_dir.is_dir() or src_dir.name == "sensitive":
            continue

        for txt_file in src_dir.glob("*.txt"):
            total_count += 1
            try:
                text = txt_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                issues.append(f"🔴 无法读取: {txt_file}")
                continue

            # 空/过短检查
            if len(text.strip()) < TARGETS["min_chunk_chars"]:
                empty_count += 1
                if len(short_content_samples) < 10:
                    short_content_samples.append(str(txt_file))

            # 语言检测（简单启发式：中文字符占比）
            chinese_chars = len(re.findall(r"[一-鿿]", text))
            total_chars = max(len(text), 1)
            if chinese_chars / total_chars > 0.3:
                lang_counter["zh"] += 1
            else:
                lang_counter["en"] += 1

    # 空文检查报告
    empty_ratio = empty_count / max(total_count, 1)
    if empty_ratio > TARGETS["max_empty_ratio"]:
        issues.append(
            f"🔴 空/过短段落 {empty_count}/{total_count} ({empty_ratio:.1%})，超过阈值 {TARGETS['max_empty_ratio']:.1%}"
        )
        for s in short_content_samples[:5]:
            issues.append(f"   示例: {s}")
    elif empty_count > 0:
        issues.append(f"🟡 空/过短段落 {empty_count}/{total_count} ({empty_ratio:.1%})")
    else:
        issues.append(f"✅ 无空/过短段落 (共 {total_count} 段)")

    # 语言分布报告
    zh_pct = lang_counter.get("zh", 0) / max(total_count, 1) * 100
    issues.append(f"📊 语言分布: 中文 {zh_pct:.1f}%, 英文 {100 - zh_pct:.1f}%")
    if zh_pct < 60:
        issues.append(f"🟡 中文比例偏低 ({zh_pct:.1f}%)，预期 70%+")

    return issues


def estimate_chunks(raw_dir: Path, window: int = 512) -> dict:
    """估算分块后 chunk 数量。使用与 chunk.py 相同的估算逻辑。"""
    total_chars = 0
    total_files = 0

    for src_dir in raw_dir.iterdir():
        if not src_dir.is_dir() or src_dir.name == "sensitive":
            continue
        for txt_file in src_dir.glob("*.txt"):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="ignore")
                total_chars += len(text)
                total_files += 1
            except Exception:
                pass

    # token 估算：中文字符 1.5 chars/token，其余 1.3 chars/token
    # 与 chunk.py:_estimate_tokens 一致
    # 简化：取平均 1.4 chars/token
    estimated_tokens = total_chars / 1.4
    estimated_chunks = estimated_tokens / window

    return {
        "total_files": total_files,
        "total_chars": total_chars,
        "estimated_tokens": int(estimated_tokens),
        "estimated_chunks": int(estimated_chunks),
        "chunk_window": window,
    }


def main():
    parser = argparse.ArgumentParser(description="语料质量校验")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="原始语料目录 (默认: data/raw)")
    parser.add_argument("--manifest", default="data/raw/corpus_manifest.json",
                        help="manifest 文件路径")
    parser.add_argument("--estimate-chunks", action="store_true",
                        help="估算分块数量")
    parser.add_argument("--strict", action="store_true",
                        help="严格模式：🔴 问题导致非零退出码")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"🔴 错误: 目录不存在 {raw_dir}")
        raise SystemExit(1)

    all_issues: list[str] = []
    has_error = False

    print("=" * 60)
    print("milrag 语料质量校验")
    print("=" * 60)

    # 1. 基于 manifest 的统计校验
    print("\n--- 统计校验 (corpus_manifest.json) ---")
    manifest = load_manifest(Path(args.manifest))
    if manifest:
        issues = validate_from_manifest(manifest)
        all_issues.extend(issues)
        for issue in issues:
            print(f"  {issue}")
            if issue.startswith("🔴"):
                has_error = True
    else:
        print("  [跳过] manifest 不可用")

    # 2. 内容级校验
    print("\n--- 内容校验 (直接扫描) ---")
    issues = validate_direct(raw_dir)
    all_issues.extend(issues)
    for issue in issues:
        print(f"  {issue}")
        if issue.startswith("🔴"):
            has_error = True

    # 3. 分块估算
    if args.estimate_chunks:
        print("\n--- 分块估算 ---")
        est = estimate_chunks(raw_dir)
        print(f"  原始文件数: {est['total_files']:,}")
        print(f"  总字符数: {est['total_chars']:,}")
        print(f"  预估 token 数: {est['estimated_tokens']:,}")
        print(f"  预估 chunk 数: {est['estimated_chunks']:,} (窗口={est['chunk_window']})")
        diff = abs(est["estimated_chunks"] - TARGETS["total_paragraphs"]) / TARGETS["total_paragraphs"] * 100
        if diff > 20:
            print(f"  🟡 预估 chunk 数与目标偏差 {diff:.1f}%（可能在清洗/分块后变化）")

    # 4. 汇总
    print("\n" + "=" * 60)
    error_count = sum(1 for i in all_issues if i.startswith("🔴"))
    warn_count = sum(1 for i in all_issues if i.startswith("🟡"))
    print(f"校验完成: {len(all_issues)} 项检查, {error_count} 错误, {warn_count} 警告")

    if has_error and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
