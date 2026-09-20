"""M1-E dataset-shift audit tests."""
from __future__ import annotations

import pytest

from ..attribution import dataset_shift as ds
from ..config import protocol as P
from . import m1_synth as S


def test_summarise_known_values():
    out = ds.summarise([1.0, 3.0, 5.0])
    assert out["n"] == 3
    assert out["mean"] == pytest.approx(3.0)
    assert out["std"] == pytest.approx(2.0)
    assert out["median"] == pytest.approx(3.0)
    assert out["iqr"] == pytest.approx(2.0)   # q1=2, q3=4
    assert out["min"] == 1.0 and out["max"] == 5.0
    assert ds.summarise([])["n"] == 0


def test_wasserstein_is_exact_on_known_cases():
    assert ds.wasserstein_1d([0.0, 1.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert ds.wasserstein_1d([0.0], [2.0]) == pytest.approx(2.0)
    assert ds.wasserstein_1d([1.0, 2.0, 3.0],
                             [1.0, 2.0, 3.0]) == pytest.approx(0.0)
    # unequal sample sizes, same distribution support
    assert ds.wasserstein_1d([0.0, 0.0, 1.0], [0.0, 1.0, 1.0]) == \
        pytest.approx(1 / 3)
    with pytest.raises(ds.ShiftAuditRefused):
        ds.wasserstein_1d([], [1.0])


def test_e3_by_reader_cutoff_and_reports():
    rows = []
    for dataset in P.DATASETS:
        for e in S.synth_atomic_entries(dataset)[:6]:
            for r in S.synth_e3_rows([e]):
                rows.append(r)
    by_cell = ds.e3_by_reader_cutoff(rows)
    assert sorted(by_cell) == sorted(P.READER_KEYS)
    dist = ds.distribution_report({"maweibo": by_cell})
    cell = dist["maweibo"]["qwen"]["15"]
    assert sorted(cell) == sorted(P.E3_FEATURE_NAMES)
    assert cell["nll_gap"]["n"] > 0

    # a two-dataset shift report needs both sides
    maweibo = ds.e3_by_reader_cutoff(
        [r for r in rows if r["key"].startswith("maweibo")])
    pheme = ds.e3_by_reader_cutoff(
        [r for r in rows if r["key"].startswith("pheme")])
    shift = ds.shift_report({"maweibo": maweibo, "pheme": pheme})
    assert shift["metric"] == "wasserstein_1d"
    assert shift["datasets"] == list(P.DATASETS)
    entry = shift["per_reader_cutoff"]["qwen"]["15"]["nll_gap"]
    assert entry["wasserstein"] >= 0.0
    assert entry["mean_shift"] == pytest.approx(
        entry["mean_left"] - entry["mean_right"])


def test_association_report_per_split_and_post_hoc_flag():
    dataset = "maweibo"
    entries = S.synth_atomic_entries(dataset)
    rows = []
    for e in entries:
        rows += S.synth_e3_rows([e])
    split = S.synth_split()
    report = ds.association_report(rows, entries, split)
    assert sorted(report) == sorted(P.READER_KEYS)
    for fold in ("utility_train", "utility_dev", "utility_eval"):
        cell = report["qwen"][fold]
        assert cell["post_hoc"] is (fold == "utility_eval")
        assert cell["n"] > 0
        for name in P.E3_FEATURE_NAMES:
            assert name in cell
        assert -1.0 <= cell["nll_gap"]["spearman_with_utility"] <= 1.0
        assert 0.0 <= cell["nll_gap"]["auroc_helpful"] <= 1.0
