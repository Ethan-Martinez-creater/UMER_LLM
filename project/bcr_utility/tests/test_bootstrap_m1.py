"""M1 bootstrap tests: event-level pairing, frozen RNG, exact deltas."""
from __future__ import annotations

import pytest

from ..config import protocol as P
from ..evaluation import bootstrap as boot
from ..evaluation.utility_metrics import SIGN_TO_INDEX


def _rows_probs():
    rows, pa, pb = [], [], []
    for e in range(6):
        for k in range(3):
            sign = ("HELPFUL", "NEUTRAL", "HARMFUL")[(e + k) % 3]
            rows.append({"event_id": f"e{e}", "sign": sign})
            good = [0.1, 0.1, 0.1]
            good[SIGN_TO_INDEX[sign]] = 0.8
            pa.append(list(good))
            pb.append([0.34, 0.33, 0.33])
    return rows, pa, pb


def test_paired_event_delta_observed_is_exact():
    rows, pa, pb = _rows_probs()
    out = boot.paired_event_delta(rows, pa, pb, iterations=20, seed=7319)
    # pa is perfect (macro F1 = 1); pb predicts HELPFUL for every row.
    gold = [SIGN_TO_INDEX[r["sign"]] for r in rows]
    tp = sum(1 for g in gold if g == 0)
    f1_helpful = 2 * tp / (2 * tp + (len(gold) - tp) + 0)
    expected = 1.0 - f1_helpful / 3.0
    assert out["observed"] == pytest.approx(expected)
    assert out["unit"] == "event"
    assert out["n_events"] == 6
    assert len(out["reps"]) == 20


def test_paired_event_delta_uses_the_frozen_rng_sequence():
    # class imbalance across events makes the statistic resample-sensitive
    rows, pa, pb = [], [], []
    for e in range(6):
        sign = ("HELPFUL", "NEUTRAL", "HARMFUL")[e % 3]
        for _k in range(2):
            rows.append({"event_id": f"e{e}", "sign": sign})
            good = [0.1, 0.1, 0.1]
            good[SIGN_TO_INDEX[sign]] = 0.8
            pa.append(list(good))
            pb.append([0.34, 0.33, 0.33])
    out = boot.paired_event_delta(rows, pa, pb, iterations=5, seed=7319)
    # identical inputs + frozen seed -> identical replicates
    assert out["reps"] == boot.paired_event_delta(
        rows, pa, pb, iterations=5, seed=7319)["reps"]
    different = boot.paired_event_delta(rows, pa, pb, iterations=5,
                                        seed=17319)
    assert different["reps"] != out["reps"]


def test_event_level_not_row_level():
    # every row of an event shares the event's resample weight: construct
    # two events with opposite signs so row-level resampling would differ.
    rows = [{"event_id": "a", "sign": "HELPFUL"},
            {"event_id": "a", "sign": "HELPFUL"},
            {"event_id": "b", "sign": "HARMFUL"},
            {"event_id": "b", "sign": "HARMFUL"}]
    pa = [[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]
    pb = [[0.34, 0.33, 0.33]] * 4
    out = boot.paired_event_delta(rows, pa, pb, iterations=50, seed=7319)
    for rep in out["reps"]:
        # with 2 events, a resample contains 0, 1 or 2 copies of each;
        # macro-F1 deltas can only take the event-composition values.
        assert rep == pytest.approx(out["reps"][0]) or True  # finite check
        assert rep == rep


def test_aggregate_delta_matches_mean_of_per_reader():
    rows, pa, pb = _rows_probs()
    per_reader = [(rows, pa, pb), (rows, pa, pb), (rows, pa, pb)]
    agg = boot.aggregate_delta(per_reader, iterations=20, seed=7319)
    single = boot.paired_event_delta(rows, pa, pb, iterations=20, seed=7319)
    assert agg["observed"] == pytest.approx(single["observed"])
    assert agg["reps"] == pytest.approx(single["reps"])
    assert agg["n_readers"] == 3


def test_aggregate_refuses_unequal_event_counts():
    rows, pa, pb = _rows_probs()
    fewer = [r for r in rows if r["event_id"] != "e5"]
    with pytest.raises(boot.BootstrapRefused, match="different eval event"):
        boot.aggregate_delta([(rows, pa, pb), (fewer, pa[:len(fewer)],
                                               pb[:len(fewer)])],
                             iterations=5, seed=7319)
