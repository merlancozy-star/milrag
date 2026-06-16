"""测试：n-gram 去重。"""
from milrag.defense.ngram_filter import rouge_l_f1, ngram_overlap_filter, filter_by_rouge


def test_rouge_identical():
    assert rouge_l_f1("歼-20 作战半径 2000 公里", "歼-20 作战半径 2000 公里") > 0.9


def test_rouge_different():
    score = rouge_l_f1("歼-20 作战半径 2000 公里", "运-20 最大载重 66 吨")
    assert score < 0.5


def test_rouge_empty():
    assert rouge_l_f1("", "test") == 0.0
    assert rouge_l_f1("test", "") == 0.0


def test_overlap_filter_no_overlap():
    contents = [
        "歼-20 隐身战斗机 作战半径 2000 公里",
        "东风-21D 反舰导弹 射程 1500",
        "辽宁舰 航母 排水量 6万吨 滑跃",
    ]
    to_remove = ngram_overlap_filter(contents, rouge_threshold=0.25)
    assert len(to_remove) == 0


def test_overlap_filter_duplicate():
    contents = [
        "歼-20 作战半径 2000 公里",
        "歼-20 作战半径 2000 公里（精确值为 1852 公里）",
        "运-20 最大载重 66 吨",
    ]
    to_remove = ngram_overlap_filter(contents, rouge_threshold=0.25)
    # 前两条高度重叠 → 应删除较短的一条
    assert len(to_remove) >= 1


def test_filter_by_rouge():
    ids = ["a", "b", "c"]
    contents = ["相同内容", "相同内容", "不同内容"]
    kept, removed = filter_by_rouge(ids, contents, threshold=0.25)
    # 两个相同内容应只保留一个
    assert len(kept) <= 2
