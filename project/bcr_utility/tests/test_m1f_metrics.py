"""M1-F metric tests (pure functions; no torch, no data, no model)."""
from __future__ import annotations

import pytest

from ..config import protocol as P
from ..evaluation import utility_metrics as um
from ..m1f import metrics as M


def _row(key, sign, utility, pred, event="u0", cutoff=15, reader="qwen"):
    return {"key": key, "event_id": event, "cutoff": cutoff, "reader": reader,
            "sign": sign, "utility": utility, "utility_pred": pred}


def _pred(signs, utilities=None):
    index = um.SIGN_TO_INDEX
    probs = []
    for sign in signs:
        row = [0.0] * len(P.SIGN_CLASSES)
        row[index[sign]] = 1.0
        probs.append(row)
    return {"utility": list(utilities or [0.0] * len(signs)),
            "sign_probs": probs}


# --------------------------------------------------------------------------
# Task C
# --------------------------------------------------------------------------
def test_balanced_accuracy_perfect_and_constant():
    gold = ["HELPFUL", "HARMFUL", "NEUTRAL"]
    assert M.balanced_accuracy(gold, gold) == 1.0
    assert M.balanced_accuracy(["HELPFUL", "HARMFUL"],
                               ["NEUTRAL", "NEUTRAL"]) == 0.0


def test_overall_metrics_shape():
    rows = [_row("a", "HELPFUL", 0.4, 0.5), _row("b", "HARMFUL", -0.4, -0.5),
            _row("c", "NEUTRAL", 0.0, 0.0)]
    out = M.overall_metrics(rows, _pred(["HELPFUL", "HARMFUL", "NEUTRAL"],
                                        [0.5, -0.5, 0.0]))
    assert out["n"] == 3
    assert out["macro_f1"] == pytest.approx(1.0)
    assert set(out["per_class_f1"]) == set(P.SIGN_CLASSES)
    assert out["harmful_auprc"] == pytest.approx(1.0)
    assert out["utility_spearman"] == pytest.approx(1.0)
    assert out["mae"] == pytest.approx(0.2 / 3)


# --------------------------------------------------------------------------
# Task D
# --------------------------------------------------------------------------
def test_active_metrics_keeps_neutral_as_error():
    rows = [_row("a", "HELPFUL", 0.4, 0.4), _row("b", "HARMFUL", -0.4, -0.4),
            _row("c", "NEUTRAL", 0.0, 0.0)]
    out = M.active_metrics(rows, _pred(["HELPFUL", "HARMFUL", "NEUTRAL"]))
    assert out["defined"] is True
    assert out["n_active"] == 2
    assert out["n_gold_helpful"] == 1 and out["n_gold_harmful"] == 1
    assert out["helpful_vs_harmful_macro_f1"] == 1.0
    assert out["neutral_prediction_rate"] == 0.0
    assert out["active_prevalence"] == 2 / 3


def test_active_metrics_neutral_prediction_is_an_error():
    rows = [_row("a", "HELPFUL", 0.4, 0.0), _row("b", "HARMFUL", -0.4, 0.0)]
    out = M.active_metrics(rows, _pred(["NEUTRAL", "NEUTRAL"]))
    assert out["helpful_vs_harmful_macro_f1"] == 0.0
    assert out["neutral_prediction_rate"] == 1.0
    assert out["balanced_accuracy"] == 0.0


def test_active_metrics_undefined_when_no_active_rows():
    rows = [_row("a", "NEUTRAL", 0.0, 0.0)]
    assert M.active_metrics(rows, _pred(["NEUTRAL"]))["defined"] is False


# --------------------------------------------------------------------------
# Task F
# --------------------------------------------------------------------------
def test_threshold_signs_boundaries_are_inclusive():
    assert M.threshold_signs([0.05, -0.05, 0.0499, -0.0499], 0.05) == \
        ["HELPFUL", "HARMFUL", "NEUTRAL", "NEUTRAL"]


def test_threshold_diagnostics_reports_every_fixed_threshold():
    rows = [_row("a", "HELPFUL", 0.4, 0.4), _row("b", "HARMFUL", -0.4, -0.04)]
    out = M.threshold_diagnostics(rows, _pred(["HELPFUL", "NEUTRAL"],
                                              [0.4, -0.04]))
    assert sorted(out) == sorted(f"{t:.2f}" for t in P.M1F_THRESHOLDS)
    assert out["0.05"]["predicted_activity_rate"] == 0.5
    assert out["0.03"]["predicted_activity_rate"] == 1.0


def test_near_boundary_and_activity_rate():
    assert M.activity_rate(["HELPFUL", "NEUTRAL"]) == pytest.approx(0.5)
    assert M.near_boundary_rate([0.05, 0.0499, 0.2], 0.05, 0.01) == \
        pytest.approx(2 / 3)


def test_conditional_sign_matrix():
    rows = [_row("a", "HELPFUL", 0.4, 0.4), _row("b", "HELPFUL", 0.4, 0.0),
            _row("c", "HARMFUL", -0.4, 0.0)]
    matrix = M.conditional_sign_matrix(rows, _pred(["HELPFUL", "NEUTRAL",
                                                    "NEUTRAL"]))
    assert matrix["HELPFUL"]["n"] == 2
    assert matrix["HELPFUL"]["p_pred_HELPFUL"] == 0.5
    assert matrix["HARMFUL"]["p_pred_NEUTRAL"] == 1.0
    assert matrix["NEUTRAL"]["n"] == 0


# --------------------------------------------------------------------------
# Task E
# --------------------------------------------------------------------------
def _within_rows():
    return [
        # snapshot A: 3 rows, strictly monotone -> rho 1, concordance 1
        _row("a1", "HARMFUL", -0.4, -0.5),
        _row("a2", "NEUTRAL", 0.0, 0.1),
        _row("a3", "HELPFUL", 0.4, 0.6),
        # snapshot B: 2 rows -> too few
        _row("b1", "HELPFUL", 0.4, 0.4, event="u1"),
        _row("b2", "HARMFUL", -0.4, -0.4, event="u1"),
        # snapshot C: 3 rows, zero gold variance -> excluded
        _row("c1", "NEUTRAL", 0.0, 0.2, event="u2"),
        _row("c2", "NEUTRAL", 0.0, 0.1, event="u2"),
        _row("c3", "NEUTRAL", 0.0, -0.1, event="u2"),
    ]


def test_within_snapshot_stats_filters_and_pools():
    out = M.within_snapshot_stats(_within_rows(), pairwise=True)
    assert out["n_snapshots_total"] == 3
    assert out["n_snapshots_used"] == 1
    assert out["skipped"] == {"too_few_rows": 1, "zero_variance": 1}
    assert out["mean_within_snapshot_spearman"] == pytest.approx(1.0)
    assert out["centered_pooled_spearman"] == pytest.approx(1.0)
    assert out["pairwise_concordance"] == pytest.approx(1.0)
    assert len(out["per_snapshot"]) == 1


def test_within_snapshot_reversed_prediction_is_negative():
    rows = [
        _row("a1", "HARMFUL", -0.4, 0.5),
        _row("a2", "NEUTRAL", 0.0, -0.1),
        _row("a3", "HELPFUL", 0.4, -0.6),
    ]
    out = M.within_snapshot_stats(rows, pairwise=True)
    assert out["centered_pooled_spearman"] == pytest.approx(-1.0)
    assert out["pairwise_concordance"] == pytest.approx(0.0)


def test_snapshot_key_is_event_cutoff_reader():
    row = _row("a", "NEUTRAL", 0.0, 0.0, event="u9", cutoff=60,
               reader="mistral")
    assert M.snapshot_key(row) == ("u9", 60, "mistral")


# --------------------------------------------------------------------------
# event-clustered bootstrap
# --------------------------------------------------------------------------
def _bootstrap_rows():
    rows = []
    for event in ("u0", "u1", "u2", "u3"):
        rows.append(_row(f"{event}h", "HELPFUL", 0.4, 0.4, event=event))
        rows.append(_row(f"{event}a", "HARMFUL", -0.4, -0.4, event=event))
    return rows


def test_clustered_event_bootstrap_is_paired_and_event_level():
    rows = _bootstrap_rows()
    good = _pred(["HELPFUL", "HARMFUL"] * 4)
    bad = _pred(["HARMFUL", "HELPFUL"] * 4)
    out = M.clustered_event_bootstrap(
        rows, good, bad, lambda r, p: um.macro_f1([x["sign"] for x in r],
                                                  um.predicted_signs(p)),
        iterations=25, seed=P.BOOTSTRAP_SEED)
    assert out["n_events"] == 4
    # three-class Macro-F1: NEUTRAL is never predicted or gold here, so the
    # perfect active classes give 2/3
    assert out["observed"] == pytest.approx(2 / 3)
    assert out["ci_low"] <= out["observed"] <= out["ci_high"]
    assert "reps" not in out or isinstance(out["reps"], list)


def test_clustered_event_ci_single_prediction():
    rows = _bootstrap_rows()
    good = _pred(["HELPFUL", "HARMFUL"] * 4)
    out = M.clustered_event_ci(
        rows, good, lambda r, p: um.macro_f1([x["sign"] for x in r],
                                            um.predicted_signs(p)),
        iterations=25, seed=P.BOOTSTRAP_SEED)
    assert out["observed"] == pytest.approx(2 / 3)
    assert out["n_events"] == 4
    assert out["n_finite_replicates"] > 0


def test_take_prediction_preserves_order():
    pred = {"utility": [1.0, 2.0, 3.0],
            "sign_probs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    out = M.take_prediction(pred, [2, 0])
    assert out["utility"] == [3.0, 1.0]
    assert out["sign_probs"] == [[0, 0, 1], [1, 0, 0]]
