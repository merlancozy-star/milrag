"""核心指标实现正确性（断言实现，不是断言论文数值）。"""
from eval.metrics import (token_f1, exact_match, recall_at_k, mrr, ndcg_at_k, f1_robust, refusal_pr)

def test_em_f1():
    assert exact_match("歼-20", "歼-20") == 1.0
    assert 0 < token_f1("歼20最大起飞重量37吨", "歼20起飞重量约37吨") < 1.0

def test_retrieval():
    r = ["a","b","c","d"]; gold = {"c"}
    assert recall_at_k(r, gold, 1) == 0.0
    assert recall_at_k(r, gold, 3) == 1.0
    assert abs(mrr(r, gold) - 1/3) < 1e-9
    assert 0 < ndcg_at_k(r, gold, 4) <= 1.0

def test_f1_robust():
    # R_clean=0.91, FPR=0 -> 接近 0.91+ 区间
    assert 0 < f1_robust(0.91, 0.0) <= 1.0

def test_refusal():
    pr = refusal_pr([True, False, True], [True, True, False])
    assert pr["refusal_precision"] == 0.5 and pr["refusal_recall"] == 0.5
