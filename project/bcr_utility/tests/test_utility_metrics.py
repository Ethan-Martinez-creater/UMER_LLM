"""M1 metric tests."""
from __future__ import annotations

import pytest

from ..config import protocol as P
from ..evaluation import utility_metrics as um


def test_macro_f1_perfect_and_partial():
    gold = ["HELPFUL", "NEUTRAL", "HARMFUL", "NEUTRAL"]
    assert um.macro_f1(gold, gold) == pytest.approx(1.0)
    pred = ["HELPFUL", "HARMFUL", "HARMFUL", "NEUTRAL"]
    # HELPFUL: tp=1 fp=0 fn=0 -> 1; NEUTRAL: tp=1 fp=0 fn=1 -> 2/3;
    # HARMFUL: tp=1 fp=1 fn=0 -> 2/3
    assert um.macro_f1(gold, pred) == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)


def test_macro_f1_missing_classes_are_zero_not_nan():
    gold = ["NEUTRAL", "NEUTRAL"]
    pred = ["NEUTRAL", "NEUTRAL"]
    assert um.macro_f1(gold, pred) == pytest.approx(1 / 3)


def test_auprc_known_example():
    scores = [0.9, 0.8, 0.7, 0.6]
    positives = [True, False, True, False]
    # recall steps: 1/2 at precision 1, then 1 at precision 2/3
    assert um.auprc(scores, positives) == pytest.approx(0.5 * 1.0
                                                        + 0.5 * (2 / 3))
    assert um.auprc([], []) != um.auprc([], [])  # nan
    assert um.auprc([0.1], [False]) != um.auprc([0.1], [False])


def test_evaluate_predictions_full_set():
    gold_rows = [
        {"sign": "HELPFUL", "utility": 0.2},
        {"sign": "NEUTRAL", "utility": 0.0},
        {"sign": "HARMFUL", "utility": -0.3},
    ]
    pred = {"utility": [0.25, 0.05, -0.2],
            "sign_probs": [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]}
    metrics = um.evaluate_predictions(gold_rows, pred)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx((0.05 + 0.05 + 0.1) / 3)
    assert metrics["utility_spearman"] == pytest.approx(1.0)
    assert metrics["harmful_auprc"] == pytest.approx(1.0)
    assert metrics["helpful_auprc"] == pytest.approx(1.0)


def test_disagreement_mask_uses_training_signs_only():
    eval_rows = [{"key": "k1"}, {"key": "k2"}, {"key": "k3"}]
    training = {"k1": {"qwen": "HELPFUL", "mistral": "HARMFUL"},
                "k2": {"qwen": "NEUTRAL", "mistral": "NEUTRAL"}}
    mask = um.disagreement_mask(eval_rows, training)
    assert mask == [True, False, False]
    subset = um.subset(eval_rows, mask)
    assert [r["key"] for r in subset] == ["k1"]


def test_subset_prediction():
    pred = {"utility": [1.0, 2.0, 3.0],
            "sign_probs": [[0.7, 0.2, 0.1]] * 3}
    sub = um.subset_prediction(pred, [True, False, True])
    assert sub["utility"] == [1.0, 3.0]
    assert len(sub["sign_probs"]) == 2
