"""Plan §35 — bootstrap multiplicity and exact gate thresholds."""
from __future__ import annotations

import math

from ..evaluation.bootstrap import paired_event_bootstrap
from ..evaluation.heterogeneity import gate_p1
from ..evaluation.structural_interaction import gate_p2
from ..evaluation.unseen_reader import gate_p4
from ..evaluation.utility_prediction import gate_p3


def test_bootstrap_preserves_multiplicity():
    draws = []

    def statistic(payloads):
        ids = [id(p) for p in payloads]
        draws.append(ids)
        return 1.0

    per_event = {f"e{i}": {"v": i} for i in range(5)}
    out = paired_event_bootstrap(per_event, statistic, iterations=200,
                                 seed=7319)
    assert out["n_events"] == 5
    # every draw has exactly n_events payloads, repeats included
    assert all(len(d) == 5 for d in draws)
    # at least one draw repeats an event payload -> multiplicity is kept
    assert any(len(set(d)) < len(d) for d in draws)
    assert out["observed"] == 1.0


def test_bootstrap_is_seed_deterministic():
    def statistic(payloads):
        return sum(p["v"] for p in payloads)

    per_event = {f"e{i}": {"v": i} for i in range(6)}
    a = paired_event_bootstrap(per_event, statistic, iterations=100, seed=7319)
    b = paired_event_bootstrap(per_event, statistic, iterations=100, seed=7319)
    assert a["samples"] == b["samples"]


def test_gate_p1_exact_thresholds():
    report = {"macro_mean_disagreement": 0.10,
              "pairs": [{"disagreement": 0.05}, {"disagreement": 0.05},
                        {"disagreement": 0.0499}]}
    assert gate_p1(report)["pass"] is True
    report["macro_mean_disagreement"] = 0.0999
    assert gate_p1(report)["pass"] is False
    # only one pair above threshold -> fails the 2/3 rule
    report = {"macro_mean_disagreement": 0.20,
              "pairs": [{"disagreement": 0.20}, {"disagreement": 0.04},
                        {"disagreement": 0.04}]}
    assert gate_p1(report)["pass"] is False


def test_gate_p2_exact_thresholds():
    assert gate_p2({"edge": {"delta": 0.02, "ci_low": 1e-9}})["pass"] is True
    assert gate_p2({"edge": {"delta": 0.02, "ci_low": 0.0}})["pass"] is False
    assert gate_p2({"edge": {"delta": 0.0199, "ci_low": 0.01}})["pass"] is False


def test_gate_p3_exact_thresholds():
    ci = {"ci_low": 0.005, "ci_high": 0.03, "n_events": 10}
    b3 = {"macro_f1": 0.52, "spearman": 0.60}
    base = {"B0": {"macro_f1": 0.50, "spearman": 0.60}}
    assert gate_p3(b3, base, delta_ci=ci)["pass"] is True
    # a point estimate without the event-level bootstrap fails closed
    assert gate_p3(b3, base)["pass"] is False
    # CI lower bound must be strictly positive
    assert gate_p3(b3, base, delta_ci={"ci_low": 0.0, "ci_high": 0.1})["pass"]         is False
    lower = {"B0": {"macro_f1": 0.50, "spearman": 0.60},
             "B1": {"macro_f1": 0.55, "spearman": 0.60}}
    assert gate_p3(b3, lower, delta_ci=ci)["pass"] is False
    worse_spearman = {"B0": {"macro_f1": 0.50, "spearman": 0.70}}
    assert gate_p3(b3, worse_spearman, delta_ci=ci)["pass"] is False


def _rot(held, delta, ok=True):
    return {"held_out_reader": held, "delta": delta, "token_target_ok": ok}


def test_gate_p4_exact_thresholds():
    ok = [_rot("internlm", 0.02), _rot("glm", 0.015), _rot("qwen", -0.004)]
    assert gate_p4(ok)["pass"] is True
    # worst rotation below -0.005 fails
    bad_worst = [_rot("internlm", 0.02), _rot("glm", -0.006),
                 _rot("qwen", 0.02)]
    assert gate_p4(bad_worst)["pass"] is False
    # only one positive rotation fails the 2/3 rule
    one_positive = [_rot("internlm", 0.02), _rot("glm", -0.001),
                    _rot("qwen", -0.001)]
    assert gate_p4(one_positive)["pass"] is False
    # token target breach fails
    tokens_bad = [_rot("internlm", 0.02, False), _rot("glm", 0.02),
                  _rot("qwen", 0.02)]
    assert gate_p4(tokens_bad)["pass"] is False


def test_gate_p4_requires_exactly_three_distinct_rotations():
    # only two rotations -> fail closed even when both deltas are excellent
    two = [_rot("internlm", 0.05), _rot("glm", 0.05)]
    assert gate_p4(two)["pass"] is False
    assert gate_p4(two)["rotations_complete"] is False
    # duplicated held-out reader -> fail closed
    dup = [_rot("internlm", 0.05), _rot("internlm", 0.05),
           _rot("glm", 0.05)]
    assert gate_p4(dup)["pass"] is False
    # extra rotation -> fail closed
    extra = [_rot("internlm", 0.05), _rot("glm", 0.05), _rot("qwen", 0.05),
             _rot("qwen", 0.05)]
    assert gate_p4(extra)["pass"] is False


def test_final_decision_never_auto_upgrades():
    from ..evaluation.unseen_reader import final_decision
    gates = {"P0": {"pass": True}, "P1": {"pass": True}, "P2": {"pass": True},
             "P3": {"pass": True}, "P4": {"pass": False}}
    decision = final_decision(gates, pheme={"pass": True, "complete": True})
    assert decision["recommendation"] in (
        "STOP_FOR_RESEARCH_REVIEW", "STOP_CR_TSER")
    all_pass = {"P0": {"pass": True}, "P1": {"pass": True},
                "P2": {"pass": True}, "P3": {"pass": True}, "P4": {"pass": True}}
    assert final_decision(all_pass,
                          pheme={"pass": True, "complete": True})[
        "recommendation"] == "START_FULL_CR_TSER_METHOD_DEVELOPMENT"
    # PHEME alone can never produce FULL_GO
    assert final_decision(all_pass, pheme={"pass": False,
                                           "complete": True})["decision"] != \
        "FULL_GO"


def test_final_decision_requires_complete_pheme_evidence():
    from ..evaluation.unseen_reader import final_decision
    all_pass = {"P0": {"pass": True}, "P1": {"pass": True},
                "P2": {"pass": True}, "P3": {"pass": True},
                "P4": {"pass": True}}
    # missing PHEME secondary evidence must never produce FULL_GO
    assert final_decision(all_pass)["decision"] != "FULL_GO"
    assert final_decision(all_pass)["pheme_evidence_complete"] is False
    # incomplete PHEME rotations must never produce FULL_GO either
    assert final_decision(all_pass, pheme={"pass": True,
                                           "complete": False})["decision"] != \
        "FULL_GO"
    # complete PHEME evidence + non-catastrophic -> FULL_GO
    assert final_decision(all_pass, pheme={"pass": True, "complete": True})[
        "decision"] == "FULL_GO"


def test_pheme_secondary_requires_complete_rotations():
    from ..evaluation.unseen_reader import pheme_secondary
    two = [_rot("internlm", 0.01), _rot("glm", 0.01)]
    assert pheme_secondary(two)["pass"] is False
    assert pheme_secondary(two)["complete"] is False
    full = [_rot("internlm", 0.01), _rot("glm", 0.01), _rot("qwen", 0.01)]
    assert pheme_secondary(full)["pass"] is True


def test_gate_logic_exact():
    """All four gates combine exactly as the plan freezes them (plan §25)."""
    from ..config import pilot_config as C
    from ..evaluation.unseen_reader import final_decision

    # thresholds come from the frozen module, never from literals in the gates
    assert C.P1_MEAN_DISAGREEMENT_MIN == 0.10
    assert C.P1_PAIR_DISAGREEMENT_MIN == 0.05
    assert C.P2_EDGE_DELTA_MIN == 0.02
    assert C.P3_MACRO_F1_DELTA_MIN == 0.02
    assert C.P4_MEAN_DELTA_MIN == 0.01
    assert C.P4_WORST_ROTATION_MIN == -0.005
    assert C.PHEME_MEAN_DELTA_MIN == -0.005

    p1 = gate_p1({"macro_mean_disagreement": 0.10,
                  "pairs": [{"disagreement": 0.05}, {"disagreement": 0.05},
                            {"disagreement": 0.0}]})
    p2 = gate_p2({"edge": {"delta": 0.02, "ci_low": 1e-12}})
    p3 = gate_p3({"macro_f1": 0.52, "spearman": 0.5},
                 {"B0": {"macro_f1": 0.50, "spearman": 0.5}},
                 delta_ci={"ci_low": 0.004, "ci_high": 0.03, "n_events": 12})
    p4 = gate_p4([_rot("internlm", 0.02), _rot("glm", 0.015),
                  _rot("qwen", -0.004)])
    assert all(g["pass"] for g in (p1, p2, p3, p4))

    gates = {"P0": {"pass": True}, "P1": p1, "P2": p2, "P3": p3, "P4": p4}
    decision = final_decision(gates, pheme={"pass": True, "complete": True})
    assert decision["decision"] == "FULL_GO"
    assert decision["recommendation"] == \
        "START_FULL_CR_TSER_METHOD_DEVELOPMENT"

    # one core failure that is not P0 -> research review, never FULL_GO
    gates_fail = dict(gates)
    gates_fail["P3"] = {"pass": False}
    assert final_decision(gates_fail, pheme={"pass": True,
                                             "complete": True})["decision"] ==         "PARTIAL_GO"
    # P0 failure is a hard stop
    gates_p0 = dict(gates)
    gates_p0["P0"] = {"pass": False}
    assert final_decision(gates_p0, pheme={"pass": True,
                                           "complete": True})["decision"] ==         "NO_GO"


def test_auroc_and_macro_f1_sane():
    from ..evaluation.bootstrap import auroc, macro_f1_from_counts
    counts = {(1, 1): 5, (1, 0): 1, (0, 0): 4, (0, 1): 0}
    value = macro_f1_from_counts(counts, labels=(0, 1))
    assert 0.0 < value <= 1.0
    perfect = auroc([0.9, 0.8, 0.1], [True, True, False])
    assert math.isclose(perfect, 1.0)
    reversed_auc = auroc([0.1, 0.2, 0.9], [True, True, False])
    assert math.isclose(reversed_auc, 0.0)
