"""测试：一致性检查 ConsistencyChecker。"""
import numpy as np
from milrag.defense.consistency import pairwise_consistency, nli_score


class MockNLIModel:
    """Mock NLI 模型用于测试。"""
    def predict(self, premise: str, hypothesis: str) -> dict:
        # 简单启发式：若 premise 与 hypothesis 共享关键词 → entail
        p_words = set(premise)
        h_words = set(hypothesis)
        overlap = len(p_words & h_words) / max(len(h_words), 1)
        if overlap > 0.5:
            return {"entail": 0.8, "neutral": 0.15, "contradict": 0.05}
        elif overlap > 0.2:
            return {"entail": 0.3, "neutral": 0.5, "contradict": 0.2}
        else:
            return {"entail": 0.05, "neutral": 0.05, "contradict": 0.9}


def test_nli_score_entail():
    nli = MockNLIModel()
    score = nli_score(nli, "歼-20 作战半径 2000 公里", "歼-20 的作战半径为 2000 公里")
    assert score > 0  # entail > contradict


def test_nli_score_contradict():
    nli = MockNLIModel()
    score = nli_score(nli, "歼-20 作战半径 2000 公里", "月球绕地球运行")
    assert score < 0  # contradict > entail


def test_pairwise_consistency():
    nli = MockNLIModel()
    c = pairwise_consistency(
        nli,
        evidence="歼-20 作战半径 2000 公里",
        prior_claim="歼-20 的作战半径为 2000 公里",
        sim_sem=0.9,
        alpha=0.4,
        beta=0.6,
    )
    expected = 0.4 * 0.9 + 0.6 * (0.8 - 0.05)  # 0.36 + 0.45 = 0.81
    assert abs(c - 0.81) < 0.05
