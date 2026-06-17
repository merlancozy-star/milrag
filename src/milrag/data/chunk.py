"""data/chunk.py — 情报分块（论文 3.2.2）。

最终方案：语义边界优先 + 滑窗 512/64。
  1. 先按自然段落/章节边界切分，尽量保持语义完整
  2. 超长段落在语义边界的约束下使用滑窗（512 token / 重叠 64 token）
  3. 每块保留元信息：标题、章节编号、来源

对照 Exp 3-5：固定 256/512/1024 vs 语义+滑窗。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: dict = field(default_factory=dict)


# ── 章节/段落边界检测 ─────────────────────────────────────────
_SECTION_RE = re.compile(
    r"^(?:第[一二三四五六七八九十\d]+[章节条款]|[①②③④⑤⑥⑦⑧⑨⑩\d]+[\.、．)）]|\d+[\.、．)）])",
    re.MULTILINE,
)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")


def _estimate_tokens(text: str, language: str = "zh") -> int:
    """粗略估算 token 数（双语支持）。

    中文: ~1.5 字符/token（汉字+标点）
    英文: ~1.3 字符/token（空格分词+字母+标点）
    混合文本按中文字符占比自动加权。
    """
    cn_chars = len(re.findall(r"[一-鿿]", text))
    other = len(text) - cn_chars
    # 英文纯文本按空格分词，token 边界通常在词级；
    # 中文无空格，token 边界在子词级。两者 chars/token 比不同。
    return int(cn_chars / 1.5 + other / 1.3)


def _split_by_semantic_boundaries(text: str) -> list[str]:
    """按章节/段落等语义边界切分。"""
    # 先按双换行分段落
    paragraphs = _PARAGRAPH_SPLIT_RE.split(text.strip())
    # 若某段以章节标记开始，视为新语义单元起点
    segments: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        if not para.strip():
            continue
        if _SECTION_RE.match(para.strip()) and current:
            segments.append("\n\n".join(current))
            current = [para]
        else:
            current.append(para)
    if current:
        segments.append("\n\n".join(current))
    return segments if segments else [text]


def chunk_document(
    text: str,
    window: int = 512,
    overlap: int = 64,
    doc_id: str = "",
    title: str = "",
    section_id: str = "",
) -> list[Chunk]:
    """语义边界优先 + 滑窗分块。

    Args:
        text: 输入文档文本。
        window: 滑窗大小（token 数，默认 512）。
        overlap: 重叠 token 数（默认 64）。
        doc_id: 文档 ID（用于生成 chunk_id）。
        title: 文档标题（附带元信息）。
        section_id: 章节编号（附带元信息）。

    Returns:
        Chunk 列表。
    """
    segments = _split_by_semantic_boundaries(text)
    chunks: list[Chunk] = []
    idx = 0

    for seg in segments:
        seg_tokens = _estimate_tokens(seg)
        if seg_tokens <= window:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk{idx:05d}",
                text=seg,
                meta={"title": title, "section_id": section_id, "source_doc": doc_id},
            ))
            idx += 1
        else:
            # 滑窗切分超长段落，保持语义边界约束
            # 按句子切分（中英文句号）
            sentences = re.split(r"(?<=[。！？.!?\n])\s*", seg)
            sentences = [s for s in sentences if s.strip()]

            buffer: list[str] = []
            buf_len = 0

            def flush_buf(keep_last: int = 0):
                nonlocal buffer, buf_len
                if not buffer:
                    return
                # 保留末尾 keep_last 句作为下一窗开头
                overlap_sents = buffer[-keep_last:] if keep_last > 0 and len(buffer) > keep_last else []
                chunk_text = "".join(buffer)
                nonlocal idx
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_chunk{idx:05d}",
                    text=chunk_text,
                    meta={"title": title, "section_id": section_id, "source_doc": doc_id},
                ))
                idx += 1
                buffer = overlap_sents.copy()
                buf_len = _estimate_tokens("".join(buffer))

            for sent in sentences:
                st = _estimate_tokens(sent)
                if buf_len + st > window and buffer:
                    # 计算保留多少句做重叠
                    overlap_sents_count = 0
                    acc = 0
                    for s in reversed(buffer):
                        acc += _estimate_tokens(s)
                        overlap_sents_count += 1
                        if acc >= overlap:
                            break
                    flush_buf(keep_last=overlap_sents_count)
                buffer.append(sent)
                buf_len += st

            flush_buf()  # 尾段

    return chunks


def chunk_document_with_meta(
    text: str,
    meta: dict | None = None,
    window: int = 512,
    overlap: int = 64,
    doc_id: str = "",
) -> list[Chunk]:
    """带 meta 信息的分块（双语 KB 入口）。

    从 .meta.json 的 language 字段判断语言，将语言标记写入 chunk meta，
    供后续 NER（英文跳过中文 NER）和嵌入（多语言模型）使用。

    Args:
        text: 输入文档文本。
        meta: .meta.json 内容（含 language 等字段）。
        window: 滑窗大小。
        overlap: 重叠量。
        doc_id: 文档 ID。

    Returns:
        Chunk 列表，每个 chunk 的 meta 中包含 language 标记。
    """
    lang = meta.get("language", "zh") if meta else "zh"
    title = meta.get("title", "") if meta else ""
    chunks = chunk_document(
        text, window=window, overlap=overlap,
        doc_id=doc_id, title=title,
    )
    # 注入语言标记
    for c in chunks:
        c.meta["language"] = lang
        if meta:
            c.meta["authority"] = meta.get("authority", "general_commentary")
            c.meta["source_category"] = meta.get("source_category", "")
            c.meta["content_category"] = meta.get("content_category", "")
            c.meta["desensitized"] = meta.get("desensitized", False)
    return chunks
