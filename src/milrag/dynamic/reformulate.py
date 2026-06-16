"""dynamic/reformulate.py — 自适应查询重构（论文 4.4）。

三策略：
  1. 实体增强 (4.4.1)：α_i = β·att(e_i) + (1-β)·IDF(e_i)，β=0.6，取 Top-3~5 实体。
     → 装备参数类。
  2. 实词聚焦 (4.4.2)：中文词性标注 + 军事词典保留实词，按注意力排序取 Top-K。
     → 对抗环境类（过滤噪声）。★坑：短问题勿删限定词导致语义漂移。
  3. 上下文融合 (4.4.3)：q' = q + "当前推理：" + 触发点前~100字已生成片段。
     → 战略态势类。

超参从 config/dynamic.yaml: reformulate 读。
"""
from __future__ import annotations

# ── 中文军事术语词典（合成/公开语料）───────────────────────
_MILITARY_TERMS: set[str] = {
    "作战半径", "最大起飞重量", "巡航速度", "隐身性能", "有源相控阵",
    "垂直起降", "超音速巡航", "态势感知", "电子战", "信息优势",
    "火力投射", "精确制导", "饱和攻击", "区域拒止", "反介入",
    "联合作战", "兵力投送", "海空一体", "网络中心战", "多域作战",
    "弹道导弹", "巡航导弹", "高超音速", "核威慑", "常规威慑",
    "航母战斗群", "两栖戒备大队", "远征打击大队", "防空识别区",
    "歼击机", "强击机", "预警机", "加油机", "电子侦察机",
}

# 中文词性标签映射
_POS_KEEP = {"n", "v", "nr", "ns", "nt", "nz", "vn", "an", "m", "q"}


class QueryReformulator:
    """三策略查询重构器。"""

    def __init__(self, cfg: dict, ner, military_dict: set[str] | None = None,
                 idf_table: dict[str, float] | None = None):
        """
        Args:
            cfg: config/dynamic.yaml → reformulate 段。
            ner: MilitaryNER 实例（实体抽取用）。
            military_dict: 军事术语词典（增强实词聚焦）。
            idf_table: IDF 查询表（用于实体增强的 IDF 项）。
        """
        r = cfg["reformulate"]
        self.beta: float = r["entity_enhance"]["beta"]                      # 0.6
        self.topk_entities: list[int] = r["entity_enhance"]["topk_entities"] # [3,5]
        self.keep_pos: list[str] = r.get("content_word_focus", {}).get("keep_pos", list(_POS_KEEP))
        self.prefix_chars: int = r["context_fusion"]["prefix_chars"]         # 100
        self.ctx_template: str = r["context_fusion"]["template"]             # "{q}当前推理：{ctx}"
        self.ner = ner
        self.military_dict = military_dict or _MILITARY_TERMS
        self.idf_table = idf_table or {}

    # ── 策略 1：实体增强（装备参数类）────────────────────────
    def entity_enhance(self, query: str, generated: str,
                       attn_to_entities: "np.ndarray | None" = None) -> str:
        """α_i = β·att(e_i) + (1-β)·IDF(e_i)，取 Top-3~5 实体附加到查询。

        Args:
            query: 原始查询。
            generated: 当前已生成文本（用于抽取新实体）。
            attn_to_entities: shape [|E|] — 来自 detector，对实体的注意力依赖。

        Returns:
            增强后的查询。
        """
        import numpy as np
        entities = self.ner.extract_entities(query + " " + generated[:200])
        if not entities:
            return query

        # 计算每个实体的综合权重 α_i
        scores = []
        for i, e in enumerate(entities):
            att_val = float(attn_to_entities[i]) if (
                attn_to_entities is not None and i < len(attn_to_entities)
            ) else 0.0
            idf_val = float(self.idf_table.get(e.normalized, 0.0))
            alpha = self.beta * att_val + (1 - self.beta) * idf_val / max(1.0, max(self.idf_table.values(), default=1))
            scores.append((alpha, e.normalized))

        # 按权重排序取 Top-K
        scores.sort(key=lambda x: x[0], reverse=True)
        top_n = self.topk_entities[0]  # 默认 3
        # 若生成较长，多用一些实体
        if len(generated) > 200:
            top_n = self.topk_entities[1]  # 5

        enhanced_entities = [ent for _, ent in scores[:top_n]]
        if enhanced_entities:
            return query + " 相关实体：" + "、".join(enhanced_entities)
        return query

    # ── 策略 2：实词聚焦（对抗环境类）───────────────────────
    def content_word_focus(self, query: str, generated: str,
                           attn_to_entities: "np.ndarray | None" = None) -> str:
        """保留中文名词/动词/专名/数词 + 军事术语，按注意力排序取 Top-K。

        ★坑（CLAUDE.md §6）：短问题（<8 字）勿删限定词，避免语义漂移。
        """
        import re
        import numpy as np

        # CRITICAL: 短问题不删词，直接返回增强版
        if len(query) < 8:
            # 仅附加军事术语中存在于 query 的部分
            matched_terms = [t for t in self.military_dict if t in query or t in generated[:100]]
            if matched_terms:
                return query + " 关键术语：" + "、".join(matched_terms[:5])
            return query

        # 分词（jieba 若可用，否则用字符 unigram fallback）
        try:
            import jieba
            import jieba.posseg as pseg
            words = [(w.word, w.flag) for w in pseg.cut(query)]
        except ImportError:
            # fallback: 按字符 + 简单词性启发
            words = _simple_segment(query)

        # 保留实词：词性在 keep_pos 中，或在军事词典中
        kept_words = []
        for word, pos in words:
            if pos in self.keep_pos or word in self.military_dict:
                kept_words.append(word)

        if not kept_words:
            return query

        # 按注意力排序（若提供了 attn_to_entities）
        if attn_to_entities is not None and len(attn_to_entities) > 0:
            # 实体位置→词映射（简化：按字符位置）
            word_attn = []
            offset = 0
            for w in kept_words:
                w_len = len(w)
                idx_start = query.index(w) if w in query else offset
                # 取词范围内实体的平均注意力
                rel_attn = np.mean([
                    attn_to_entities[i] for i in range(idx_start, min(idx_start + w_len, len(attn_to_entities)))
                ])
                word_attn.append((rel_attn, w))
                offset += w_len
            word_attn.sort(key=lambda x: x[0], reverse=True)
            kept_words = [w for _, w in word_attn]

        # Top-K = min(len(kept_words), 取原词的 60%~80%)
        keep_n = max(3, int(len(query) * 0.15))  # ~15% 的词
        focused = kept_words[:max(keep_n, len(kept_words) // 2)]
        return "".join(focused) if focused else query

    # ── 策略 3：上下文融合（战略态势类）──────────────────────
    def context_fusion(self, query: str, generated: str, trigger_pos: int = -1) -> str:
        """q' = q + "当前推理：" + 触发点前~100字已生成片段。

        Args:
            query: 原始查询。
            generated: 当前已生成文本。
            trigger_pos: 触发时的生成位置（字符）。
        Returns:
            融合后查询。
        """
        if not generated:
            return query

        # 取触发点前的 ~100 字
        if trigger_pos > 0:
            ctx = generated[max(0, trigger_pos - self.prefix_chars):trigger_pos]
        else:
            ctx = generated[-self.prefix_chars:]

        if not ctx.strip():
            return query

        return self.ctx_template.format(q=query, ctx=ctx.strip())
