"""M1-F analysis tests: alignment, the primary comparison, Task E and the decision."""
from __future__ import annotations

import pytest

from ..config import protocol as P
from ..evaluation import utility_metrics as um
from ..m1f import analysis

SIGNS = ("HELPFUL", "NEUTRAL", "HARMFUL")
UTILITIES = (0.4, 0.0, -0.4)


def _rows(held):
    rows = []
    for event in ("u0", "u1", "u2"):
        for k, sign in enumerate(SIGNS):
            rows.append({"key": f"{event}|15|n{k}", "event_id": event,
                         "cutoff": 15, "reader": held, "sign": sign,
                         "utility": UTILITIES[k]})
    return rows


def _pred(rows, correct=True):
    probs, utilities = [], []
    for row in rows:
        sign = row["sign"] if correct else \
            ("HARMFUL" if row["sign"] == "HELPFUL" else
             "HELPFUL" if row["sign"] == "HARMFUL" else "NEUTRAL")
        one_hot = [0.0] * len(P.SIGN_CLASSES)
        one_hot[um.SIGN_TO_INDEX[sign]] = 1.0
        probs.append(one_hot)
        utilities.append(float(row["utility"]) if correct
                         else -float(row["utility"]))
    return {"utility": utilities, "sign_probs": probs}


def _results(reversed_variants=()):
    results = {}
    for variant in P.M1F_VARIANTS:
        rotations = {}
        for rotation in P.LORO_ROTATIONS:
            held = rotation[2]
            rows = _rows(held)
            if variant in reversed_variants:
                rows = list(reversed(rows))
            rotations[held] = {
                "held_out": held, "train_readers": [rotation[0], rotation[1]],
                "eval_rows": rows,
                "prediction": _pred(rows, correct=variant != P.M1F_S0),
            }
        results[variant] = {"variant": variant, "rotations": rotations}
    return results


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------
def test_align_rotations_restores_a_shared_row_order():
    aligned = analysis.align_rotations(_results(reversed_variants=(P.M1F_S4,)))
    for held, entry in aligned.items():
        base_keys = [r["key"] for r in entry[P.M1F_S0]["rows"]]
        assert [r["key"] for r in entry[P.M1F_S4]["rows"]] == base_keys
        # the reordered prediction must follow the reference rows
        for row, utility in zip(entry[P.M1F_S4]["rows"],
                                entry[P.M1F_S4]["prediction"]["utility"]):
            assert (utility > 0) == (row["sign"] == "HELPFUL")


def test_align_rotations_detects_gold_drift():
    results = _results()
    held = P.LORO_ROTATIONS[0][2]
    results[P.M1F_S3]["rotations"][held]["eval_rows"][0]["utility"] = 0.99
    with pytest.raises(analysis.AnalysisRefused):
        analysis.align_rotations(results)


def test_align_rotations_detects_a_different_evidence_set():
    results = _results()
    held = P.LORO_ROTATIONS[0][2]
    results[P.M1F_S3]["rotations"][held]["eval_rows"].pop()
    with pytest.raises(analysis.AnalysisRefused):
        analysis.align_rotations(results)


def test_pool_concatenates_every_rotation():
    aligned = analysis.align_rotations(_results())
    rows, pred = analysis.pool(aligned, P.M1F_S4)
    assert len(rows) == 3 * 9
    assert len(pred["utility"]) == 3 * 9
    assert len(pred["sign_probs"]) == 3 * 9


# --------------------------------------------------------------------------
# Task C
# --------------------------------------------------------------------------
def test_primary_comparison_uses_the_frozen_bootstrap():
    aligned = analysis.align_rotations(_results())
    out = analysis.primary_comparison(aligned, P.M1F_S4, P.M1F_S0,
                                      iterations=40, seed=P.BOOTSTRAP_SEED)
    assert out["primary"] == P.M1F_S4 and out["comparator"] == P.M1F_S0
    assert out["aggregate"]["mean_delta_macro_f1"] > 0.0
    assert out["aggregate"]["iterations"] == 40
    assert out["aggregate"]["positive_readers"] == 3
    assert set(out["per_reader_macro_f1_delta"]) == set(P.READER_KEYS)
    for name in ("balanced_accuracy", "active_macro_f1", "harmful_auprc",
                 "centered_spearman"):
        assert name in out["secondary_metric_deltas"]


# --------------------------------------------------------------------------
# Task E
# --------------------------------------------------------------------------
def test_within_snapshot_audit_reports_all_variants_and_comparisons():
    aligned = analysis.align_rotations(_results())
    out = analysis.within_snapshot_audit(aligned, "maweibo", iterations=25,
                                         seed=P.BOOTSTRAP_SEED)
    assert out["min_rows"] == P.M1F_WITHIN_MIN_ROWS
    assert out["reference_variant"] == P.M1F_S2
    assert set(out["variants"]) == set(P.M1F_TRAINED_VARIANTS + (P.M1F_S4,))
    for variant in P.M1F_TRAINED_VARIANTS + (P.M1F_S4,):
        entry = out["variants"][variant]
        assert entry["n_snapshots_used"] == 9
        assert "pairwise_concordance" in entry
    assert set(out["comparisons_vs_S2"]) == {P.M1F_S3, P.M1F_S4}


def test_within_snapshot_variant_is_positive_for_a_correct_prediction():
    aligned = analysis.align_rotations(_results())
    entry = analysis.within_snapshot_variant(aligned, P.M1F_S4, iterations=25,
                                             seed=P.BOOTSTRAP_SEED)
    assert entry["centered_pooled_spearman"] == pytest.approx(1.0)
    assert entry["pairwise_concordance"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# decision
# --------------------------------------------------------------------------
def _comparison(mean=0.05, ci_low=0.01, positive=3, active=0.05,
                harmful_f1=0.02, helpful_f1=0.01, harmful_auprc=0.01):
    return {
        "aggregate": {"mean_delta_macro_f1": mean, "ci_low": ci_low,
                      "ci_high": 0.1, "positive_readers": positive,
                      "worst_reader_delta": 0.01, "iterations": 10000,
                      "seed": 7319, "n_events_per_reader": 25},
        "secondary_metric_deltas": {
            "active_macro_f1": {"observed": active},
            "harmful_f1": {"observed": harmful_f1},
            "helpful_f1": {"observed": helpful_f1},
            "harmful_auprc": {"observed": harmful_auprc},
        },
    }


def _within(delta=0.05):
    return {"comparisons_vs_S2": {
        P.M1F_S4: {"observed": delta, "ci_low": 0.0, "ci_high": 0.1}}}


def test_decide_continue_requires_every_condition():
    out = analysis.decide(_comparison(), _within(), validity_blocked=False,
                          active_improves=True,
                          harmful_auprc_not_decreased=True)
    assert out["verdict"] == "M1F_CONTINUE_TO_CONFIRMATION_REVIEW"
    assert all(out["checks"].values())
    assert out["authorizes_m2"] is False


def test_decide_closes_when_the_strong_comparator_wins():
    out = analysis.decide(_comparison(mean=0.001, ci_low=-0.02, positive=1),
                          _within(), False, True, True)
    assert out["verdict"] == "M1F_CLOSE_BCR_UTILITY_METHOD"
    assert out["checks"]["mean_delta_gte_0.02"] is False


def test_decide_closes_without_within_snapshot_improvement():
    out = analysis.decide(_comparison(), _within(delta=-0.01), False, True,
                          True)
    assert out["verdict"] == "M1F_CLOSE_BCR_UTILITY_METHOD"
    assert out["checks"][
        "within_snapshot_centered_spearman_improves_over_S2"] is False


def test_decide_closes_when_the_gain_is_neutral_only():
    out = analysis.decide(_comparison(active=-0.01, harmful_f1=0.0,
                                      helpful_f1=0.0),
                          _within(), False, False, True)
    assert out["verdict"] == "M1F_CLOSE_BCR_UTILITY_METHOD"
    assert out["checks"]["gain_not_only_neutral"] is False


def test_decide_closes_when_harmful_auprc_decreases():
    out = analysis.decide(_comparison(harmful_auprc=-0.03), _within(),
                          validity_blocked=False, active_improves=True,
                          harmful_auprc_not_decreased=False)
    assert out["verdict"] == "M1F_CLOSE_BCR_UTILITY_METHOD"


def test_validity_blocked_takes_precedence():
    out = analysis.decide(_comparison(), _within(), validity_blocked=True,
                          active_improves=True,
                          harmful_auprc_not_decreased=True)
    assert out["verdict"] == "M1F_VALIDITY_BLOCKED"


def test_gain_only_neutral_rule():
    assert analysis.gain_only_neutral(
        _comparison(active=-0.01, harmful_f1=0.0,
                    helpful_f1=0.0))["gain_only_neutral"] is True
    assert analysis.gain_only_neutral(
        _comparison(active=0.02, harmful_f1=0.01,
                    helpful_f1=0.01))["gain_only_neutral"] is False
