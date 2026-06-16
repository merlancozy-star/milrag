"""defense/decision.py — 可靠来源选择与拒答决策（论文 5.5.2 / 5.5.3）。

双路候选：y_prior（内部知识）vs y_ext（外部过滤证据）。
对两路各算四维可靠性向量 v_prior / v_ext。

两类决策策略：
  1. 阈值式（5.5.2）：EC(y_ext) > τ_EC 且 Unc(y_ext) < τ_Unc → 用 y_ext；
     否则进一步判断（内部 vs 补检索 vs 拒答）。
  2. 学习式（5.5.3）：小型 MLP（10~12 维特征）→ 5 类：
     use_ext / use_prior / re_retrieve / ask_user / refuse。

核心原则：宁可拒答，不可错答。拒答必须附带原因。
"""
from __future__ import annotations

from enum import Enum

import numpy as np


class Action(str, Enum):
    USE_EXT = "use_ext"
    USE_PRIOR = "use_prior"
    RE_RETRIEVE = "re_retrieve"
    ASK_USER = "ask_user"
    REFUSE = "refuse"


# ── 拒答模板（中文军事语境）─────────────────────────────────
_REFUSAL_TEMPLATES = {
    "low_ec": "无法验证该回答的证据支持度，建议提供更具体的信息或重新提问。",
    "high_unc": "当前证据存在较大不确定性，暂时无法给出可靠回答。",
    "conflict": "检索到的证据间存在严重冲突，需要进一步验证。",
    "no_evidence": "未检索到相关权威证据，建议补充查询条件。",
    "general": "出于信息可靠性考虑，暂时无法回答该问题。",
}


class Decider:
    """双路决策器：对外部证据回答 vs 内部知识先验 vs 拒答。"""

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: config/defense.yaml → decision 段。
        """
        self.policy = cfg["decision"]
        self.threshold_policy = self.policy.get("threshold_policy", {})
        self.learned_policy = self.policy.get("learned_policy", {})
        self.principle = self.policy.get("principle", "宁可拒答不可错答")
        self.refuse_with_reason = self.policy.get("refuse_with_reason", True)

        # 阈值（从 config 读，验证集网格搜索结果）
        self.tau_ec = self.threshold_policy.get("ec_min", 0.5)
        self.tau_unc = self.threshold_policy.get("unc_max", 0.5)

        # MLP 模型（惰性加载）
        self._mlp_model = None

    def threshold_decide(
        self,
        v_ext: "ReliabilityVector",
        v_prior: "ReliabilityVector",
        has_conflict: bool = False,
    ) -> tuple[Action, str]:
        """阈值式决策（论文 5.5.2）。

        决策树：
          1. EC(y_ext) > τ_EC 且 Unc(y_ext) < τ_Unc → USE_EXT
          2. 否则若 EC(y_prior) > EC(y_ext) 且 Unc(y_prior) < τ_Unc → USE_PRIOR
          3. 否则若 has_conflict → RE_RETRIEVE
          4. 否则 → REFUSE

        Args:
            v_ext: 外部证据回答的四维向量。
            v_prior: 内部先验回答的四维向量。
            has_conflict: 是否从图检测到冲突簇。

        Returns:
            (Action, reason_string).
        """
        # 规则 1：外部证据足够好 → 直接用
        if v_ext.ec > self.tau_ec and v_ext.unc < self.tau_unc:
            return Action.USE_EXT, ""

        # 规则 2：外部不可靠但内部更可靠
        if v_prior.ec > v_ext.ec and v_prior.unc < self.tau_unc:
            return Action.USE_PRIOR, "外部证据覆盖度不足，回退至内部知识"

        # 规则 3：存在冲突 → 建议补检索
        if has_conflict:
            return Action.RE_RETRIEVE, "证据间存在冲突，建议重新检索"

        # 规则 4：都不行 → 拒答
        reason = _refusal_reason(v_ext, has_conflict)
        return Action.REFUSE, reason

    def learned_decide(self, features: list[float]) -> tuple[Action, float]:
        """学习式决策（论文 5.5.3）。

        特征向量 (10~12 维)：
          v_ext.ec, v_ext.ea, v_ext.sc, v_ext.unc,
          v_prior.ec, v_prior.ea, v_prior.sc, v_prior.unc,
          has_conflict(0/1), num_evidence, evidence_authority_mean

        Args:
            features: 特征向量。

        Returns:
            (Action, confidence).
        """
        if self._mlp_model is not None:
            x = np.asarray(features, dtype=np.float32).reshape(1, -1)
            logits = self._mlp_model.predict(x)[0]  # type: ignore[union-attr]
            idx = int(np.argmax(logits))
            actions = list(Action)
            return actions[idx], float(np.max(logits) / logits.sum())

        # 无 MLP 模型时回退到阈值式（用 features 前 8 维构造 v_ext/v_prior）
        from milrag.defense.self_assess import ReliabilityVector
        v_ext = ReliabilityVector(features[0], features[1], features[2], features[3])
        v_prior = ReliabilityVector(features[4], features[5], features[6], features[7])
        has_conflict = len(features) > 8 and features[8] > 0.5
        action, reason = self.threshold_decide(v_ext, v_prior, has_conflict)
        return action, 1.0

    def decide(
        self,
        v_ext: "ReliabilityVector",
        v_prior: "ReliabilityVector",
        features: list[float] | None = None,
        has_conflict: bool = False,
    ) -> dict:
        """统一决策入口（论文 5.5.2 / 5.5.3 自动路由）。

        Returns:
            {
                "action": str,
                "reason": str,
                "confidence": float,
                "refused": bool,
            }
        """
        use_learned = self.learned_policy.get("type") == "mlp" and features is not None

        if use_learned and features:
            action, confidence = self.learned_decide(features)
            reason = ""
        else:
            action, reason = self.threshold_decide(v_ext, v_prior, has_conflict)
            confidence = _confidence_from_vectors(v_ext, action)

        refused = action in (Action.REFUSE, Action.ASK_USER)

        if refused and not reason:
            reason = _refusal_reason(v_ext, has_conflict)

        return {
            "action": action.value,
            "reason": reason,
            "confidence": confidence,
            "refused": refused,
        }

    def load_mlp(self, checkpoint_path: str):
        """加载训练好的 MLP 模型。"""
        import joblib
        self._mlp_model = joblib.load(checkpoint_path)


def _refusal_reason(v_ext: "ReliabilityVector", has_conflict: bool) -> str:
    """生成拒答原因。"""
    if v_ext.ec < 0.3:
        return _REFUSAL_TEMPLATES["low_ec"]
    if has_conflict:
        return _REFUSAL_TEMPLATES["conflict"]
    if v_ext.unc > 0.7:
        return _REFUSAL_TEMPLATES["high_unc"]
    if v_ext.ea < 0.3:
        return _REFUSAL_TEMPLATES["no_evidence"]
    return _REFUSAL_TEMPLATES["general"]


def _confidence_from_vectors(v_ext: "ReliabilityVector", action: Action) -> float:
    """基于四维向量的决策置信度启发式估计。"""
    if action == Action.USE_EXT:
        return float((v_ext.ec + v_ext.ea + v_ext.sc + (1 - v_ext.unc)) / 4)
    elif action == Action.USE_PRIOR:
        return float((v_ext.ec + v_ext.sc + (1 - v_ext.unc)) / 3)
    elif action == Action.REFUSE:
        return 0.9  # 拒答本身是高置信度决定
    return 0.5
