"""测试：决策器 Decider。"""
from milrag.defense.decision import Decider, Action
from milrag.defense.self_assess import ReliabilityVector


def test_threshold_decide_use_ext():
    cfg = {
        "decision": {
            "threshold_policy": {"ec_min": 0.5, "unc_max": 0.5},
            "learned_policy": {"type": "mlp", "in_dim": 12},
            "principle": "宁可拒答不可错答",
            "refuse_with_reason": True,
        },
    }
    d = Decider(cfg)
    v_ext = ReliabilityVector(ec=0.8, ea=0.9, sc=0.85, unc=0.2)
    v_prior = ReliabilityVector(ec=0.3, ea=0.3, sc=0.5, unc=0.6)

    action, reason = d.threshold_decide(v_ext, v_prior, has_conflict=False)
    assert action == Action.USE_EXT
    assert reason == ""


def test_threshold_decide_refuse():
    cfg = {
        "decision": {
            "threshold_policy": {"ec_min": 0.5, "unc_max": 0.5},
            "learned_policy": {"type": "mlp", "in_dim": 12},
            "principle": "宁可拒答不可错答",
            "refuse_with_reason": True,
        },
    }
    d = Decider(cfg)
    v_ext = ReliabilityVector(ec=0.2, ea=0.1, sc=0.3, unc=0.9)
    v_prior = ReliabilityVector(ec=0.1, ea=0.1, sc=0.2, unc=0.8)

    action, reason = d.threshold_decide(v_ext, v_prior, has_conflict=False)
    assert action == Action.REFUSE
    assert len(reason) > 0


def test_threshold_decide_conflict_retriggers():
    cfg = {
        "decision": {
            "threshold_policy": {"ec_min": 0.5, "unc_max": 0.5},
            "learned_policy": {"type": "mlp", "in_dim": 12},
            "principle": "宁可拒答不可错答",
            "refuse_with_reason": True,
        },
    }
    d = Decider(cfg)
    v_ext = ReliabilityVector(ec=0.3, ea=0.4, sc=0.5, unc=0.6)
    v_prior = ReliabilityVector(ec=0.2, ea=0.2, sc=0.4, unc=0.7)

    action, reason = d.threshold_decide(v_ext, v_prior, has_conflict=True)
    assert action == Action.RE_RETRIEVE


def test_learned_decide_fallback():
    """无 MLP 模型时回退到阈值式。"""
    cfg = {
        "decision": {
            "threshold_policy": {"ec_min": 0.5, "unc_max": 0.5},
            "learned_policy": {"type": "mlp", "in_dim": 12},
            "principle": "宁可拒答不可错答",
            "refuse_with_reason": True,
        },
    }
    d = Decider(cfg)
    # features: [ec_ext, ea_ext, sc_ext, unc_ext, ec_prior, ea_prior, sc_prior, unc_prior, has_conflict, n_evidence]
    features = [0.8, 0.9, 0.85, 0.2, 0.3, 0.3, 0.5, 0.6, 0.0, 5.0]
    action, confidence = d.learned_decide(features)
    assert action in Action
    assert 0.0 <= confidence <= 1.0
