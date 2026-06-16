"""milrag.defense — 第5章 对抗环境 两阶段可信增强与噪声过滤。"""
from milrag.defense.cluster_filter import ClusterFilter, FilterResult, outlier_scores
from milrag.defense.ngram_filter import ngram_overlap_filter, rouge_l_f1, filter_by_rouge
from milrag.defense.inject_detect import injection_probability
from milrag.defense.prior import PriorExtractor, InternalPrior, extract_claims_from_response
from milrag.defense.consistency import ConsistencyChecker, ConsistencyResult, pairwise_consistency, nli_score
from milrag.defense.self_assess import SelfAssessor, ReliabilityVector
from milrag.defense.decision import Decider, Action

__all__ = [
    "ClusterFilter", "FilterResult", "outlier_scores",
    "ngram_overlap_filter", "rouge_l_f1", "filter_by_rouge",
    "injection_probability",
    "PriorExtractor", "InternalPrior", "extract_claims_from_response",
    "ConsistencyChecker", "ConsistencyResult", "pairwise_consistency", "nli_score",
    "SelfAssessor", "ReliabilityVector",
    "Decider", "Action",
]
