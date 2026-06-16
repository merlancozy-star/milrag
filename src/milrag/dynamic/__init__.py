"""milrag.dynamic — 第4章 动态检索与自适应查询重构。"""
from milrag.dynamic.signals import attention_to_entities, extract_step_logprob, compute_s_a
from milrag.dynamic.entropy_trend import token_entropy, EntropyTrendTracker
from milrag.dynamic.detector import NeedDetector, SlidingZScore, TriggerDecision
from milrag.dynamic.classifier import QuestionTypeClassifier
from milrag.dynamic.reformulate import QueryReformulator
from milrag.dynamic.selector import StrategySelector
from milrag.dynamic.fuse import fuse_evidence, EvidenceItem
from milrag.dynamic.loop import DynamicRetrievalLoop, LoopState

__all__ = [
    "attention_to_entities", "extract_step_logprob", "compute_s_a",
    "token_entropy", "EntropyTrendTracker",
    "NeedDetector", "SlidingZScore", "TriggerDecision",
    "QuestionTypeClassifier",
    "QueryReformulator",
    "StrategySelector",
    "fuse_evidence", "EvidenceItem",
    "DynamicRetrievalLoop", "LoopState",
]
