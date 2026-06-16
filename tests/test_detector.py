"""测试：综合判别器 NeedDetector + SlidingZScore。"""
import numpy as np
from milrag.dynamic.detector import SlidingZScore, NeedDetector, TriggerDecision


def test_sliding_zscore_initial():
    z = SlidingZScore(window=16)
    # 第一个值：窗口不足，返回 0
    assert z.normalize(5.0) == 0.0
    # 第二个值开始有标准差
    second = z.normalize(3.0)
    assert isinstance(second, float)


def test_sliding_zscore_convergence():
    z = SlidingZScore(window=8)
    for x in [1.0, 1.0, 1.0, 1.0, 1.0]:
        z.normalize(x)
    # 全是相同值，标准差接近 0 → z-score 接近 0
    val = z.normalize(1.0)
    assert abs(val) < 1e-6


def test_detector_initialization():
    cfg = {
        "detector": {
            "threshold": 0.62,
            "task_weights": {
                "equipment_param": {"w_p": 0.6, "w_a": 0.2, "w_h": 0.2},
                "strategic_status": {"w_p": 0.2, "w_a": 0.5, "w_h": 0.3},
                "adversarial_check": {"w_p": 0.2, "w_a": 0.2, "w_h": 0.6},
            },
            "fallback_weights": {"w_p": 0.3334, "w_a": 0.3333, "w_h": 0.3333},
            "zscore_window": 16,
        },
        "classifier": {"flatten_fallback_gap": 0.15},
        "signals": {
            "entropy": {"topk_truncate": 50},
            "attention": {"last_layers": 4},
            "entropy_trend": {
                "smoothing_alpha": 0.3,
                "first_order": True,
                "second_order": True,
            },
        },
    }
    detector = NeedDetector(cfg)
    assert detector.threshold == 0.62
    assert detector.flatten_gap == 0.15


def test_s_p():
    assert NeedDetector.s_p(-0.5) == 0.5
    assert NeedDetector.s_p(-2.0) == 2.0


def test_s_a_no_entities():
    d = _make_detector()
    val = d.s_a(np.array([]), 0.8)
    assert val == 0.0


def test_s_a_with_entities():
    d = _make_detector()
    # 注意力集中在实体位置
    attn = np.array([0.3, 0.5, 0.2])
    val = d.s_a(attn, 0.6)
    expected = attn.mean() * (1 - 0.6)
    assert abs(val - expected) < 1e-8


def test_trigger_decision():
    d = _make_detector()
    # 高 token 概率 低不确定性 → 不应触发
    type_probs = {"equipment_param": 0.8, "strategic_status": 0.1, "adversarial_check": 0.1}
    logits = np.ones(1000) * 0.001
    logits[0] = 10.0  # 高置信度
    decision = d.step(
        token_logprob=-0.1,
        token_prob=0.9,
        logits=logits,
        attn_to_entities=np.array([0.1, 0.1]),
        type_probs=type_probs,
    )
    assert not decision.triggered  # 高置信度 → 不需要检索


def test_select_weights_fallback():
    d = _make_detector()
    # top1-top2 差距小 → 退回均值
    type_probs = {"equipment_param": 0.4, "strategic_status": 0.35, "adversarial_check": 0.25}
    wp, wa, wh = d.select_weights(type_probs)
    assert abs(wp - 0.3334) < 0.01
    assert abs(wa - 0.3333) < 0.01
    assert abs(wh - 0.3333) < 0.01


def test_select_weights_equipment():
    d = _make_detector()
    type_probs = {"equipment_param": 0.85, "strategic_status": 0.1, "adversarial_check": 0.05}
    wp, wa, wh = d.select_weights(type_probs)
    assert wp == 0.6
    assert wa == 0.2
    assert wh == 0.2


def _make_detector() -> NeedDetector:
    cfg = {
        "detector": {
            "threshold": 0.62,
            "task_weights": {
                "equipment_param": {"w_p": 0.6, "w_a": 0.2, "w_h": 0.2},
                "strategic_status": {"w_p": 0.2, "w_a": 0.5, "w_h": 0.3},
                "adversarial_check": {"w_p": 0.2, "w_a": 0.2, "w_h": 0.6},
            },
            "fallback_weights": {"w_p": 0.3334, "w_a": 0.3333, "w_h": 0.3333},
            "zscore_window": 16,
        },
        "classifier": {"flatten_fallback_gap": 0.15},
        "signals": {
            "entropy": {"topk_truncate": 50},
            "attention": {"last_layers": 4},
            "entropy_trend": {
                "smoothing_alpha": 1.0,
                "first_order": True,
                "second_order": False,
            },
        },
    }
    return NeedDetector(cfg)
