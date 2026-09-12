"""Plan §35 — sequence scoring and utility sign definition."""
from __future__ import annotations

import math

import pytest

from ..models.utility_heads import tri_class_label, utility_of, utility_record
from ..readers.sequence_scorer import ab_scores, gold_probability, is_correct


def test_sequence_score_A_B():
    out = ab_scores({"A": -0.5, "B": -1.5})
    assert out["prediction"] == "A"
    assert math.isclose(out["p_rumor"] + out["p_nonrumor"], 1.0, rel_tol=1e-9)
    ea, eb = math.exp(-0.5), math.exp(-1.5)
    assert math.isclose(out["p_rumor"], ea / (ea + eb), rel_tol=1e-9)
    # deterministic tie handling
    assert ab_scores({"A": 0.0, "B": 0.0})["prediction"] == "A"
    # extremes stay finite (numerically stable softmax)
    huge = ab_scores({"A": -1000.0, "B": 0.0})
    assert huge["p_rumor"] == pytest.approx(0.0, abs=1e-12)


def test_utility_sign_definition():
    assert utility_of(0.8, 0.3) == pytest.approx(0.5)
    assert utility_of(0.2, 0.7) == pytest.approx(-0.5)
    out = ab_scores({"A": 0.0, "B": 0.0})
    assert gold_probability(out, 1) == pytest.approx(0.5)
    assert gold_probability(out, 0) == pytest.approx(0.5)
    assert is_correct(ab_scores({"A": 1.0, "B": -1.0}), 1)
    assert not is_correct(ab_scores({"A": 1.0, "B": -1.0}), 0)
    # a full record recomputes utility from the cached probabilities
    before = ab_scores({"A": 1.0, "B": 0.0})
    after = ab_scores({"A": 0.0, "B": 1.0})
    rec = utility_record(1, before, after)
    assert rec["utility"] == pytest.approx(
        rec["gold_probability_before"] - rec["gold_probability_after"])
    assert rec["prediction_before"] == "A"
    assert rec["prediction_after"] == "B"
    assert rec["label_flip"] is True


def test_correct_to_wrong_is_helpful():
    assert tri_class_label(0.0, True, False) == "HELPFUL"
    # the transition overrides a negative measured delta
    assert tri_class_label(-0.4, True, False) == "HELPFUL"


def test_wrong_to_correct_is_harmful():
    assert tri_class_label(0.0, False, True) == "HARMFUL"
    assert tri_class_label(0.4, False, True) == "HARMFUL"


def test_neutral_threshold_is_exactly_five_percent():
    assert tri_class_label(0.05, True, True) == "HELPFUL"
    assert tri_class_label(0.049999, True, True) == "NEUTRAL"
    assert tri_class_label(-0.05, True, True) == "HARMFUL"
    assert tri_class_label(-0.049999, True, True) == "NEUTRAL"
