"""熵趋势：阶跃上升时趋势信号显著为正，平稳时 ~0。"""
from milrag.dynamic.entropy_trend import EntropyTrendTracker

def test_trend_rising():
    t = EntropyTrendTracker(smoothing_alpha=1.0)  # 无平滑便于断言
    for h in [1.0, 1.0, 1.0]: t.update(h)
    flat = t.update(1.0)
    for h in [1.0, 2.0, 4.0]: rising = t.update(h)
    assert rising > flat
