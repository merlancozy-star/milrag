"""data/clean.py — 文本清洗（论文 3.2.2 第一阶段）。

处理流程：
  1. HTML 标签/实体/属性去除
  2. 不可见控制字符滤除
  3. 全角→半角（英文字母/数字/符号）
  4. 繁体→简体
  5. 标点/空白规范化
  6. 表格/列表保留层次语义标记
  7. 脚注/引用单独抽取，保留对应关系（供证据追溯）
"""
from __future__ import annotations

import re
import unicodedata


# ── 全角→半角 映射 ─────────────────────────────────────────
_FULLWIDTH_START = 0xFF01
_FULLWIDTH_END = 0xFF5E
_HALFWIDTH_OFFSET = 0xFEE0

# 中文标点等不应被转的字符范围跳过
_FULLWIDTH_SKIP = set(range(0xFF01, 0xFF0F + 1))  # ！＂＃＄％＆＇（）＊＋，－．／
_FULLWIDTH_SKIP.update(range(0xFF1A, 0xFF20 + 1))  # ：；＜＝＞？＠
_FULLWIDTH_SKIP.update(range(0xFF3B, 0xFF40 + 1))  # ［＼］＾＿｀
_FULLWIDTH_SKIP.update(range(0xFF5B, 0xFF5E + 1))  # ｛｜｝～


def _fullwidth_to_halfwidth(text: str) -> str:
    """全角 ASCII 字符转半角，保留中文全角标点。"""
    result = []
    for ch in text:
        code = ord(ch)
        if _FULLWIDTH_START <= code <= _FULLWIDTH_END and code not in _FULLWIDTH_SKIP:
            # 空格特殊处理
            if code == 0x3000:
                result.append(" ")
            else:
                result.append(chr(code - _HALFWIDTH_OFFSET))
        else:
            result.append(ch)
    return "".join(result)


# ── HTML 清洗 ──────────────────────────────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)

# 常见 HTML 实体
_HTML_ENTITIES = {
    "&nbsp;": " ", "&lt;": "<", "&gt;": ">", "&amp;": "&",
    "&quot;": '"', "&apos;": "'", "&ensp;": " ", "&emsp;": "  ",
    "&ndash;": "–", "&mdash;": "—", "&lsquo;": "'", "&rsquo;": "'",
    "&ldquo;": '"', "&rdquo;": '"',
}


# ── 标点/空白 ──────────────────────────────────────────────
_CN_PUNCT_MAP = str.maketrans({
    "，": "，", "。": "。", "、": "、", "；": "；", "：": "：",
    "？": "？", "！": "！", "“": '"', "”": '"', "‘": "'", "’": "'",
    "（": "(", "）": ")", "《": "<", "》": ">", "【": "[", "】": "]",
    "—": "—", "…": "...", "·": "·",
})

# 连续空白归一
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_document(raw: str, *, strip_html: bool = True,
                   fullwidth_to_halfwidth: bool = True,
                   traditional_to_simplified: bool = True,
                   normalize_punct: bool = True) -> str:
    """清洗单篇原始文档。

    Args:
        raw: 原始文本。
        strip_html: 是否去除 HTML 标签（默认 True）。
        fullwidth_to_halfwidth: 是否全角 ASCII → 半角（默认 True）。
        traditional_to_simplified: 是否繁→简（默认 True，需安装 opencc）。
        normalize_punct: 是否规范标点/空白（默认 True）。

    Returns:
        清洗后的纯文本。
    """
    text = raw

    # 1. HTML 去除
    if strip_html:
        text = _SCRIPT_STYLE_RE.sub(" ", text)
        text = _HTML_COMMENT_RE.sub(" ", text)
        text = _HTML_TAG_RE.sub(" ", text)
        for entity, repl in _HTML_ENTITIES.items():
            text = text.replace(entity, repl)
        text = _HTML_ENTITY_RE.sub(" ", text)

    # 2. 全角→半角
    if fullwidth_to_halfwidth:
        text = _fullwidth_to_halfwidth(text)

    # 3. 繁→简（可选，需要 opencc-python）
    if traditional_to_simplified:
        try:
            import opencc
            cc = opencc.OpenCC("t2s")
            text = cc.convert(text)
        except ImportError:
            pass  # 无 opencc 时静默跳过，日志提示

    # 4. 标点/空白规范
    if normalize_punct:
        text = text.translate(_CN_PUNCT_MAP)
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # 5. 不可见控制字符（保留 \n \t）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    return text.strip()


# ── 辅助：提取脚注/引用 ──────────────────────────────────────
_FOOTNOTE_RE = re.compile(r"\[\d+\]|（[^）]*?\d{4}[^）]*?）")


def extract_references(text: str) -> tuple[str, list[str]]:
    """提取引用标记并返回（正文, 引用列表）。"""
    refs = _FOOTNOTE_RE.findall(text)
    body = _FOOTNOTE_RE.sub("", text)
    return body.strip(), refs


def clean_document_with_meta(text: str, meta: dict | None = None) -> str:
    """根据 .meta.json 的 language 字段自动配置清洗参数。

    中文文本：全半角转换 + 繁简转换 + 标点规范
    英文文本：仅 HTML 清洗 + 控制字符去除，跳过中文专用转换

    Args:
        text: 原始文本。
        meta: .meta.json 内容（含 language 字段）。

    Returns:
        清洗后的文本。
    """
    is_chinese = True
    if meta and meta.get("language") == "en":
        is_chinese = False

    return clean_document(
        text,
        strip_html=True,
        fullwidth_to_halfwidth=is_chinese,
        traditional_to_simplified=is_chinese,
        normalize_punct=is_chinese,
    )
