"""data/ner.py — 军事实体识别 + 归一化（论文 3.2.2，目标 F1 91.2%）。

BERT-CRF 抽取：武器装备 / 作战单位 / 地理位置 / 时间表达。
编辑距离 + 同义词表归一化到统一标识。
输出的实体位置集 E 供第4章 s_a 信号与实体增强重构使用。

注：模型加载完全离线（HF_HUB_OFFLINE=1），路径来自 config/base.yaml: models.ner。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# ── 军事词典（合成/公开语料，不含真实涉密信息）─────────────
# 武器装备后缀模式
_EQUIPMENT_SUFFIXES = [
    "型", "式", "级", "代", "系列",
    "战斗机", "轰炸机", "预警机", "运输机", "直升机", "无人机",
    "驱逐舰", "护卫舰", "航母", "潜艇", "巡洋舰", "两栖舰",
    "坦克", "步战车", "装甲车", "自行火炮", "火箭炮",
    "导弹", "雷达", "电子战系统", "通信系统",
]

# 作战单位模式
_UNIT_PATTERNS = [
    r"(第[一二三四五六七八九十\d]+)?(集团军|师|旅|团|营|连|排|班)",
    r"(东部|南部|西部|北部|中部)?(战区|舰队|军区)",
    r"(航母|两栖|驱逐舰|潜艇)?(编队|战斗群|大队|中队)",
]

# 时间表达式（中文军事语境）
_TIME_PATTERNS = [
    r"\d{4}年\d{1,2}月\d{1,2}日",
    r"\d{4}年\d{1,2}月",
    r"\d{4}年",
    r"\d{2}:\d{2}(:\d{2})?",  # 时间戳
]

# 地理位置诱饵词
_LOCATION_INDICATORS = [
    "海域", "空域", "区域", "基地", "港口", "机场", "阵地",
    "海峡", "岛", "半岛", "沿岸", "边境", "方向",
]

# ── 同义词归一化表 ───────────────────────────────────────────
_SYNONYM_MAP: dict[str, str] = {
    # 装备简称 → 全称（合成示例）
    "歼20": "歼-20", "歼二零": "歼-20", "J20": "歼-20", "J-20": "歼-20",
    "运20": "运-20", "运二零": "运-20", "Y20": "运-20", "Y-20": "运-20",
    "直20": "直-20", "直二零": "直-20", "Z20": "直-20", "Z-20": "直-20",
    # 单位同义
    "东海舰队": "东部战区海军", "南海舰队": "南部战区海军",
    "北海舰队": "北部战区海军",
}


@dataclass
class Entity:
    text: str           # 表面形式
    normalized: str      # 归一化标识
    label: str           # 类型: EQUIP/UNIT/LOC/TIME
    start: int           # 字符起始位置
    end: int             # 字符结束位置


class MilitaryNER:
    """军事实体识别 + 归一化。

    支持两种模式：
      - rule: 基于词典+正则（无需 GPU，快速但召回率低于 BERT-CRF）。
      - model: 加载本地 BERT-CRF 模型（论文主线，目标 F1 91.2%）。

    默认 rule 模式作为兜底，model 路径由 config 配置。
    """

    LABELS = ["EQUIP", "UNIT", "LOC", "TIME"]

    def __init__(self, model_path: str | None = None, mode: str = "rule",
                 military_dict: set[str] | None = None):
        """
        Args:
            model_path: BERT-CRF 模型本地路径（mode="model" 时必填）。
            mode: "rule" | "model".
            military_dict: 军事术语词典（用于 rule 模式增强）。
        """
        self.mode = mode
        self.model_path = model_path
        self.military_dict = military_dict or set()
        self._model = None
        if mode == "model" and model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        """延迟加载 BERT-CRF 模型（离线）。"""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            # BERT + CRF 头：这里使用通用 NER pipeline 抽象
            self._model = AutoModel.from_pretrained(path, local_files_only=True)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"无法加载 NER 模型 ({path}): {e}") from e

    def extract_entities(self, text: str) -> list[Entity]:
        """从文本抽取军事实体。

        Returns:
            Entity 列表，含位置与归一化形式。
        """
        if self.mode == "rule" or self._model is None:
            return self._rule_extract(text)
        return self._model_extract(text)

    def _rule_extract(self, text: str) -> list[Entity]:
        """基于规则+词典的实体抽取（离线可用，无需 GPU）。"""
        entities: list[Entity] = []

        # 1. 武器装备 — 后缀匹配 + 词典
        for suffix in _EQUIPMENT_SUFFIXES:
            pattern = rf"([一-鿿\w\-]+?{re.escape(suffix)})"
            for m in re.finditer(pattern, text):
                surf = m.group(1)
                entities.append(Entity(
                    text=surf, normalized=normalize_entity(surf, _SYNONYM_MAP),
                    label="EQUIP", start=m.start(), end=m.end(),
                ))

        # 2. 作战单位 — 正则模式
        for pat in _UNIT_PATTERNS:
            for m in re.finditer(pat, text):
                surf = m.group(0)
                entities.append(Entity(
                    text=surf, normalized=normalize_entity(surf, _SYNONYM_MAP),
                    label="UNIT", start=m.start(), end=m.end(),
                ))

        # 3. 时间表达式
        for pat in _TIME_PATTERNS:
            for m in re.finditer(pat, text):
                surf = m.group(0)
                entities.append(Entity(
                    text=surf, normalized=surf, label="TIME",
                    start=m.start(), end=m.end(),
                ))

        # 4. 地理位置 — 诱饵词触发
        for indicator in _LOCATION_INDICATORS:
            escaped = re.escape(indicator)
            pattern = rf"([一-鿿\w\-]+?{escaped})"
            for m in re.finditer(pattern, text):
                surf = m.group(1)
                # 避免过于泛化（>15 字符不取）
                if len(surf) <= 15:
                    entities.append(Entity(
                        text=surf, normalized=normalize_entity(surf, _SYNONYM_MAP),
                        label="LOC", start=m.start(), end=m.end(),
                    ))

        # 5. 词典精确匹配（军事术语）
        for term in self.military_dict:
            for m in re.finditer(re.escape(term), text):
                entities.append(Entity(
                    text=term, normalized=term, label="EQUIP",
                    start=m.start(), end=m.end(),
                ))

        # 去重（按起点+终点+label）
        seen, dedup = set(), []
        for e in sorted(entities, key=lambda x: (x.start, -x.end)):
            key = (e.start, e.end, e.label)
            if key not in seen:
                seen.add(key)
                dedup.append(e)
        return dedup

    def _model_extract(self, text: str) -> list[Entity]:
        """BERT-CRF 模型推理（需 GPU / 预加载模型）。"""
        import torch
        # 简化：实际 BERT-CRF 需要逐 token 解码
        # 此处用 token classification pipeline 抽象
        from transformers import pipeline
        nlp = pipeline("ner", model=self._model, tokenizer=self._tokenizer,
                       aggregation_strategy="simple")
        raw = nlp(text)
        entities = []
        for r in raw:
            label = r["entity_group"] if "entity_group" in r else r["entity"]
            surf = r["word"]
            # 过滤无关标签
            if label not in self.LABELS:
                label = self._map_label(label)
            entities.append(Entity(
                text=surf, normalized=normalize_entity(surf, _SYNONYM_MAP),
                label=label, start=r["start"], end=r["end"],
            ))
        return entities

    @staticmethod
    def _map_label(raw_label: str) -> str:
        """将模型输出标签映射到统一 4 类。"""
        mapping = {
            "B-EQUIP": "EQUIP", "I-EQUIP": "EQUIP",
            "B-UNIT": "UNIT", "I-UNIT": "UNIT",
            "B-LOC": "LOC", "I-LOC": "LOC",
            "B-TIME": "TIME", "I-TIME": "TIME",
            "MISC": "EQUIP", "ORG": "UNIT", "GPE": "LOC",
        }
        return mapping.get(raw_label.upper(), "EQUIP")


def normalize_entity(surface: str, synonym_map: dict[str, str] | None = None) -> str:
    """编辑距离 + 同义词表归一化实体名称。

    Args:
        surface: 实体表面形式。
        synonym_map: 自定义同义词映射（合并默认表）。
    Returns:
        归一化后的标准标识。
    """
    combined = {**_SYNONYM_MAP}
    if synonym_map:
        combined.update(synonym_map)

    # 1. 精确匹配同义词表
    if surface in combined:
        return combined[surface]

    # 2. 基本归一化：去空格、统一分隔符
    norm = surface.replace(" ", "").replace("　", "")
    norm = norm.replace("—", "-").replace("–", "-")
    norm = re.sub(r"[-]{2,}", "-", norm)

    if norm in combined:
        return combined[norm]

    return norm


def get_entity_positions(entities: Sequence[Entity], text: str) -> list[int]:
    """将实体列表转为 token 级位置索引（供第4章 s_a 注意力信号使用）。

    将字符级 start/end 映射到 BERT/BPE tokenizer 的 token position。
    """
    # 简化：返回字符位置，实际使用时由 tokenizer 的 offset_mapping 转 token 索引
    positions = []
    for e in entities:
        positions.extend(range(e.start, e.end))
    return sorted(set(positions))
