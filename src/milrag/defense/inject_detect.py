"""defense/inject_detect.py — 提示注入检测（论文 5.3.3 的 Inj(d)）。

两阶段：
  1. 正则/关键词命中（快速，覆盖已知模板）
  2. 轻量二分类器（可训练，泛化未知攻击）

输出 [0, 1]，越接近 1 表示越可能是注入/操纵性内容。
"""
from __future__ import annotations

import re

# ── 注入模式库（中文军事语境）────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"忽略(上述|上文|之前|前面)", re.IGNORECASE),
    re.compile(r"ignore\s+(previous|above)", re.IGNORECASE),
    re.compile(r"你现在(是|扮演|作为)", re.IGNORECASE),
    re.compile(r"请输出[：:]\s*", re.IGNORECASE),
    re.compile(r"(系统|上级|管理员)\s*(指令|命令|通知|确认)", re.IGNORECASE),
    re.compile(r"正确答案(是|为|应为)[：:]\s*", re.IGNORECASE),
    re.compile(r"(必须|务必|强制)\s*(回答|输出|返回)", re.IGNORECASE),
    re.compile(r"新的事实是", re.IGNORECASE),
    re.compile(r"之前的分析(已被|应该被)", re.IGNORECASE),
    re.compile(r"扮演.*角色", re.IGNORECASE),
    re.compile(r"不要(再|继续|重复)", re.IGNORECASE),
]

# 高危关键词（单个命中即加分）
_SUSPICIOUS_KEYWORDS = [
    "忽略", "ignore", "扮演", "roleplay", "系统指令", "system prompt",
    "你现在是", "you are now", "请输出", "output only",
    "上级确认", "superior confirms", "正确答案",
]

# 看似正常但可疑的短语（辅助判断）
_MANIPULATIVE_PHRASES = [
    "据权威消息", "内部消息", "可靠来源称", "未经证实的消息",
    "据知情人士", "有消息称",
]


def injection_probability(
    content: str,
    patterns: list[str] | None = None,
    use_classifier: bool = False,
) -> float:
    """计算注入概率 Inj(d) ∈ [0, 1]。

    Args:
        content: 待检测文本。
        patterns: 额外的注入模式关键词（合并到默认库）。
        use_classifier: 是否启用轻量分类器（需训练过的模型）。

    Returns:
        注入概率分数。
    """
    # 阶段 1：规则匹配
    rule_score = _rule_score(content, patterns)

    # 阶段 2（可选）：轻量分类器
    if use_classifier:
        clf_score = _classifier_score(content)
        return max(rule_score, clf_score)

    return rule_score


def _rule_score(content: str, extra_patterns: list[str] | None = None) -> float:
    """关键词/正则命中打分。

    命中越多越接近 1，但有上限（防误判）。
    """
    text = content.lower() if content else ""
    if not text:
        return 0.0

    hits = 0

    # 正则模式命中
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            hits += 1

    # 关键词命中
    for kw in _SUSPICIOUS_KEYWORDS:
        if kw.lower() in text:
            hits += 0.5

    # 操纵性短语（权重较低 — 可能只是正常引述）
    for phrase in _MANIPULATIVE_PHRASES:
        if phrase in text:
            hits += 0.2

    # 额外模式
    if extra_patterns:
        for ep in extra_patterns:
            if re.search(ep, text, re.IGNORECASE):
                hits += 0.5

    # 归一化到 [0, 1]
    # 经验性地，3 个命中 = 高可疑，封顶在 ~0.95
    score = min(0.95, hits / 3.0)
    return score


def _classifier_score(content: str) -> float:
    """轻量二分类器（占位，可替换为实际模型）。"""
    # TODO: 接入实际训练好的轻量分类器
    # 使用类似 textcnn / fasttext 的模型
    return 0.0
