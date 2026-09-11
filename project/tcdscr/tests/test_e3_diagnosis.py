"""E3 failure-diagnosis tests (E3 diagnosis order §1, §20).

Pins the corrected bootstrap (multiplicity preserved, sample size == n
events, static/dynamic paired alignment) and the diagnosis metric helpers
(exact-match, ranking change, budget masking, prediction transitions,
fixed pressure bins), plus the protocol guard that the diagnosis pipeline
only ever reads the frozen predictions.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ..evaluation.e3_diagnosis import (  # noqa: E402
    budget_masking_classification, paired_bootstrap, pressure_bin, resample,
    selection_change_metrics, selection_pressure_summary,
    spearman_rank_correlation, topk_set_changed)


class _StubRng:
    """randrange always returns 0 -> every draw picks the first key."""

    def randrange(self, n):  # noqa: A003
        return 0


def _unit(eid, triples_by_cutoff, seqs):
    return {"pairs": triples_by_cutoff,
            "static_seq": [s[0] for s in seqs],
            "dynamic_seq": [s[1] for s in seqs]}


def _two_event_units():
    """Event A: static wrong->dynamic correct everywhere; B: reverse."""
    a_pairs = {"5": [(1, 0, 1)] * 3, "15": [(1, 0, 1)] * 3}
    b_pairs = {"5": [(1, 1, 0)] * 3, "15": [(1, 1, 0)] * 3}
    a = _unit("A", a_pairs, [([0, 0], [1, 1]), ([0, 0], [1, 1]),
                             ([0, 0], [1, 1])])
    b = _unit("B", b_pairs, [([1, 1], [0, 0]), ([1, 1], [0, 0]),
                             ([1, 1], [0, 0])])
    return {"A": a, "B": b}


def test_bootstrap_preserves_duplicate_events():
    keys = ["A", "B"]
    sample = resample(keys, _StubRng(), 5)
    assert len(sample) == 5
    # every draw hit the first key and every occurrence was kept
    assert sample == ["A"] * 5
    # paired_bootstrap itself must not fold repeats: with a single event the
    # resample is constant, so the CI collapses onto the point estimate
    units = {"X": _unit("X", {"5": [(1, 0, 1)] * 4},
                        [([0, 0], [1, 1]), ([0, 0], [1, 1])])}
    b = paired_bootstrap(units, n_iter=200, seed=3090)
    lo = b["delta_macro_f1"]["ci_low"]
    hi = b["delta_macro_f1"]["ci_high"]
    assert abs(lo - hi) < 1e-12
    assert abs(lo - b["delta_macro_f1"]["point"]) < 1e-12


def test_bootstrap_sample_size_equals_n_events():
    rng = random.Random(7)
    keys = list("ABCDE")
    for n in (1, 3, 10):
        assert len(resample(keys, rng, n)) == n
    # every iteration samples exactly n_events occurrences (n=1 collapse
    # above already pins the size implicitly; here check the draw size)
    sample = resample(keys, rng, len(keys))
    assert len(sample) == len(keys)


def test_paired_bootstrap_static_dynamic_alignment():
    units = _two_event_units()
    b = paired_bootstrap(units, n_iter=500, seed=3090)
    # point estimate over the full event set, hand-computed:
    # event A: 3 seeds x 2 cutoffs of (gold=1, static=0, dynamic=1)
    # event B: 3 seeds x 2 cutoffs of (gold=1, static=1, dynamic=0)
    # pooled static: 6 correct (B) + 6 wrong (A) -> acc 0.5 on rumor-heavy
    # pooled dynamic: mirrored -> the delta must be exactly 0.0
    assert abs(b["delta_macro_f1"]["point"]) < 1e-12
    assert b["delta_macro_f1"]["ci_low"] <= \
        b["delta_macro_f1"]["point"] <= b["delta_macro_f1"]["ci_high"]
    assert b["multiplicity"] == "preserved"
    assert b["n_iterations"] == 500


def test_selection_exact_match_metric():
    rows = [
        {"static_selected_node_ids": ["a", "b"],
         "dynamic_selected_node_ids": ["a", "b"]},
        {"static_selected_node_ids": ["a", "b"],
         "dynamic_selected_node_ids": ["a", "b"]},
        {"static_selected_node_ids": ["a", "b"],
         "dynamic_selected_node_ids": ["a", "c"]},
    ]
    m = selection_change_metrics(rows)
    assert m["exact_match_rate"] == 2 / 3
    assert m["replacement_rate"] == 1 / 3
    assert abs(m["jaccard_mean"] - (1.0 + 1.0 + 1 / 3) / 3) < 1e-12
    assert m["removed_from_static"] == 1
    assert m["added_by_dynamic"] == 1
    assert m["symmetric_difference_size"] == 2


def test_ranking_change_metric():
    a = [1.0, 2.0, 3.0, 4.0]
    assert abs(spearman_rank_correlation(a, a) - 1.0) < 1e-12
    assert abs(spearman_rank_correlation(a, list(reversed(a))) + 1.0) \
        < 1e-12
    ids_a = ["n1", "n2", "n3", "n4"]
    assert not topk_set_changed(ids_a, ids_a, 3)
    assert topk_set_changed(ids_a, ["n9", "n2", "n3", "n4"], 1)
    assert topk_set_changed(ids_a, ["n9", "n2", "n3", "n4"], 3)


def test_budget_masking_classification():
    rows = [
        {"static_selected_node_ids": ["a"], "dynamic_selected_node_ids": ["a"]},
        {"static_selected_node_ids": ["a"], "dynamic_selected_node_ids": ["a"]},
        {"static_selected_node_ids": ["a"], "dynamic_selected_node_ids": ["b"]},
    ]

    def ranking_changed(r):
        return r["static_selected_node_ids"] == ["a"] and \
            r["dynamic_selected_node_ids"] == ["b"]

    out = budget_masking_classification(rows, ranking_changed)
    # row0/1: ranking unchanged + selection unchanged -> A; row2: C
    assert out["A_unchanged"] == 2
    assert out["C_ranking_changed_selection_changed"] == 1
    assert out["B_ranking_changed_selection_same"] == 0


def test_prediction_transition_categories():
    rows = [
        {"gold": 1, "static_prediction": 0, "dynamic_prediction": 1,
         "static_selected_node_ids": ["a"],
         "dynamic_selected_node_ids": ["b"]},
        {"gold": 1, "static_prediction": 1, "dynamic_prediction": 0,
         "static_selected_node_ids": ["a"],
         "dynamic_selected_node_ids": ["b"]},
        {"gold": 1, "static_prediction": 1, "dynamic_prediction": 1,
         "static_selected_node_ids": ["a", "c"],
         "dynamic_selected_node_ids": ["b"]},
    ]
    from ..evaluation.e3_diagnosis import prediction_transition_categories
    t = prediction_transition_categories(rows)
    assert t["wrong_to_correct"] == 1
    assert t["correct_to_wrong"] == 1
    assert t["net_correction_gain"] == 0
    assert t["both_correct"] == 1


def test_selection_pressure_bins_fixed():
    assert pressure_bin(0.9) == "low"
    assert pressure_bin(0.8) == "low"
    assert pressure_bin(0.79) == "medium"
    assert pressure_bin(0.4) == "medium"
    assert pressure_bin(0.39) == "high"
    rows = [
        {"gold": 1, "static_prediction": 1, "dynamic_prediction": 1,
         "static_selected_node_ids": ["a"],
         "dynamic_selected_node_ids": ["a"],
         "n_candidates": 2},
        {"gold": 1, "static_prediction": 0, "dynamic_prediction": 1,
         "static_selected_node_ids": ["a"],
         "dynamic_selected_node_ids": ["b"],
         "n_candidates": 10},
    ]
    out = selection_pressure_summary(rows)
    assert out["medium"]["n"] == 1   # saturation 0.5
    assert out["high"]["n"] == 1     # saturation 0.1
    assert out["low"]["n"] == 0


def test_diagnosis_uses_existing_predictions_only():
    src = (SCRIPTS_DIR / "tcdscr_e3_failure_diagnosis.py").read_text(
        encoding="utf-8")
    for forbidden in ("backward(", ".step()", "AutoModel", "AutoTokenizer",
                      "optimizer", "out_root = results/tcdscr/formal_e3_test"):
        assert forbidden not in src, forbidden
    assert "e3_failure_diagnosis" in src
    assert "test_predictions.jsonl" in src  # reads frozen predictions only
