"""defense/ngram_filter.py — n-gram 重叠去重（论文 5.3，配合聚类过滤）。

# Adapted from TrustRAG group_n_gram_filtering：
  rouge-L f-measure > 0.25 视为高重叠（投毒或冗余），建议移除。

在 cluster_filter 的同簇内部使用：若两条证据 rouge-L > threshold，则标记为可能
的同源投毒（攻击者生成的变体）并去重。
"""
from __future__ import annotations

from collections import Counter


def _lcs_len(a: list[str], b: list[str]) -> int:
    """最长公共子序列长度（DP）。"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l_f1(hyp: str, ref: str) -> float:
    """计算两条文本的 rouge-L F-measure。

    使用字符级 unigram（中文友好）或空格分词。
    """
    # 中文按字符切分
    if any('一' <= ch <= '鿿' for ch in hyp + ref):
        h_tokens = list(hyp.replace(" ", ""))
        r_tokens = list(ref.replace(" ", ""))
    else:
        h_tokens = hyp.split()
        r_tokens = ref.split()

    if not h_tokens or not r_tokens:
        return 0.0

    lcs = _lcs_len(h_tokens, r_tokens)
    if lcs == 0:
        return 0.0

    p = lcs / len(h_tokens)
    r = lcs / len(r_tokens)
    return 2 * p * r / (p + r)


def ngram_overlap_filter(
    contents: list[str],
    rouge_threshold: float = 0.25,
) -> list[int]:
    """成对检查 n-gram 重叠，标记应删除的索引。

    策略（对齐 TrustRAG group_n_gram_filtering）：
      对同一簇内所有文档对，若 rouge-L > threshold，标记其中"较短/权威性低"者删除。

    Args:
        contents: 文档内容列表（同一簇内）。
        rouge_threshold: rouge-L F-measure 阈值（默认 0.25）。

    Returns:
        应删除的文档索引列表。
    """
    n = len(contents)
    if n <= 1:
        return []

    # 计算成对 rouge-L
    to_remove: set[int] = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            score = rouge_l_f1(contents[i], contents[j])
            if score > rouge_threshold:
                # 保留较长的（信息更完整）
                if len(contents[i]) >= len(contents[j]):
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break  # i 被删，不再用它比后续

    return sorted(to_remove)


def filter_by_rouge(
    ids: list[str],
    contents: list[str],
    threshold: float = 0.25,
) -> tuple[list[str], list[str]]:
    """便捷函数：按 rouge-L 去重，返回 (保留ids, 移除ids)。"""
    remove_idx = set(ngram_overlap_filter(contents, threshold))
    kept = [did for i, did in enumerate(ids) if i not in remove_idx]
    removed = [did for i, did in enumerate(ids) if i in remove_idx]
    return kept, removed
