"""M1-F strong-baseline variant tests over the synthetic M1 fixtures."""
from __future__ import annotations

import pytest

from ..config import protocol as P
from ..evaluation.unseen_reader import build_feature_table
from ..m1f import variants
from . import m1_synth


def _table(dataset="maweibo"):
    entries = m1_synth.synth_atomic_entries(dataset)
    table = build_feature_table(dataset, entries,
                                m1_synth.synth_e0_rows(entries),
                                m1_synth.synth_e1_rows(entries),
                                m1_synth.synth_e2_rows(entries),
                                e3_rows=m1_synth.synth_e3_rows(entries))
    return entries, table


def test_variant_dims_follow_the_frozen_blocks():
    assert variants.variant_dim(P.M1F_S2) == \
        len(P.E3_SOURCE_STATE_NAMES) + 1
    assert variants.variant_dim(P.M1F_S3) == \
        (len(P.E0_FEATURE_NAMES) + len(P.E1_FEATURE_NAMES)
         + len(P.E2_FEATURE_NAMES) + len(P.E3_SOURCE_STATE_NAMES) + 1)
    with pytest.raises(variants.VariantRefused):
        variants.variant_dim(P.M1F_S0)
    with pytest.raises(variants.VariantRefused):
        variants.variant_dim(P.M1F_S1)


def test_s2_vector_is_source_state_plus_cutoff_only():
    entries, table = _table()
    key = entries[0]["key"]
    row = table[key]
    vector = variants.variant_vector(row, "qwen", P.M1F_S2)
    assert len(vector) == variants.variant_dim(P.M1F_S2)
    assert vector[-1] == float(row["cutoff"])
    e3 = row["e3"]["qwen"]
    expected = [e3[P.E3_FEATURE_NAMES.index(name)]
                for name in P.E3_SOURCE_STATE_NAMES]
    assert vector[:-1] == expected
    # the evidence-familiarity block must not leak into S2
    for name in P.E3_EVIDENCE_FAMILIARITY_NAMES:
        assert e3[P.E3_FEATURE_NAMES.index(name)] not in vector[:-1]


def test_s3_vector_is_z_plus_source_state_and_cutoff():
    entries, table = _table()
    row = table[entries[0]["key"]]
    vector = variants.variant_vector(row, "internlm", P.M1F_S3)
    assert len(vector) == variants.variant_dim(P.M1F_S3)
    z = (len(P.E0_FEATURE_NAMES) + len(P.E1_FEATURE_NAMES)
         + len(P.E2_FEATURE_NAMES))
    assert vector[:z] == list(row["e0e1"]) + list(row["e2"]["internlm"])
    assert vector[-1] == float(row["cutoff"])
    e3 = row["e3"]["internlm"]
    for name in P.E3_EVIDENCE_FAMILIARITY_NAMES:
        assert e3[P.E3_FEATURE_NAMES.index(name)] not in vector


def test_build_rows_respects_reader_and_event_filters():
    entries, table = _table()
    rows = variants.build_rows(P.M1F_S2, entries, table, ["qwen", "mistral"],
                               {"u000", "u001"})
    assert {r["reader"] for r in rows} == {"qwen", "mistral"}
    assert {r["event_id"] for r in rows} == {"u000", "u001"}
    assert all(len(r["x"]) == variants.variant_dim(P.M1F_S2) for r in rows)
    assert all(r["sign"] in P.SIGN_CLASSES for r in rows)


def test_trained_versus_constant_classification():
    for variant in P.M1F_TRAINED_VARIANTS:
        assert variants.is_trained(variant) is True
    for variant in P.M1F_CONSTANT_VARIANTS + (P.M1F_S4,):
        assert variants.is_trained(variant) is False


def test_prior_and_constant_predictions():
    rows = [{"sign": "HARMFUL", "utility": -0.3, "reader": "qwen"},
            {"sign": "HARMFUL", "utility": -0.5, "reader": "mistral"},
            {"sign": "NEUTRAL", "utility": 0.0, "reader": "qwen"}]
    assert variants.prior_sign(rows) == "HARMFUL"
    assert variants.prior_utility(rows) == pytest.approx((-0.3 - 0.5) / 3)
    pred = variants.s1_prediction(rows, rows)
    assert pred["utility"] == [pytest.approx((-0.3 - 0.5) / 3)] * 3
    assert all(p[0] == 0.0 and p[1] == 0.0 and p[2] == 1.0
               for p in pred["sign_probs"])
    s0 = variants.s0_prediction(rows)
    assert s0["utility"] == [0.0, 0.0, 0.0]
    assert all(p[1] == 1.0 for p in s0["sign_probs"])


def test_prior_requires_rows():
    with pytest.raises(variants.VariantRefused):
        variants.prior_sign([])
