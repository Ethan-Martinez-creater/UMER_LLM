"""M1-E attribution-variant tests (synthetic, tiny end-to-end)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ..attribution import feature_ablation as fa      # noqa: E402
from ..config import protocol as P                    # noqa: E402
from ..evaluation import unseen_reader as lor         # noqa: E402
from . import m1_synth as S                           # noqa: E402

FAST = {"max_epochs": 3, "patience": 2, "batch_size": 256}


def _dataset(name="maweibo"):
    entries = S.synth_atomic_entries(name)
    table = lor.build_feature_table(
        name, entries, S.synth_e0_rows(entries), S.synth_e1_rows(entries),
        S.synth_e2_rows(entries), e3_rows=S.synth_e3_rows(entries))
    return entries, table, S.synth_split(), S.synth_fingerprints()


def test_variant_evidence_dims_are_frozen():
    dims = {v: fa.variant_evidence_dim(v) for v in P.ATTRIBUTION_VARIANTS}
    assert dims == {"D0_Z": 26, "D1_Z_F": 26, "D2_Z_F_S": 29,
                    "D3_Z_F_C": 29, "D4_Z_F_S_C": 32, "D5_Z_S_C": 32}


def test_variant_fingerprint_flags():
    for v in P.ATTRIBUTION_FINGERPRINT_VARIANTS:
        assert fa.variant_uses_fingerprint(v) is True
    for v in P.ATTRIBUTION_NO_FINGERPRINT_VARIANTS:
        assert fa.variant_uses_fingerprint(v) is False
    with pytest.raises(fa.AttributionRefused):
        fa.variant_uses_fingerprint("D9")


def test_variant_evidence_vector_groups_are_exact():
    entries, table, _split, _fp = _dataset()
    row = table[entries[0]["key"]]
    z = fa.variant_evidence_vector(row, "qwen", "D0_Z")
    assert z == list(row["e0e1"]) + list(row["e2"]["qwen"])
    d2 = fa.variant_evidence_vector(row, "qwen", "D2_Z_F_S")
    assert d2[:26] == z
    assert d2[26:] == [row["e3"]["qwen"][i] for i in (0, 1, 2)]
    d3 = fa.variant_evidence_vector(row, "qwen", "D3_Z_F_C")
    assert d3[26:] == [row["e3"]["qwen"][i] for i in (3, 4, 5)]
    d4 = fa.variant_evidence_vector(row, "qwen", "D4_Z_F_S_C")
    assert d4[26:] == list(row["e3"]["qwen"])
    d5 = fa.variant_evidence_vector(row, "qwen", "D5_Z_S_C")
    assert d5 == z + list(row["e3"]["qwen"])


def test_variant_requires_e3_rows():
    entries = S.synth_atomic_entries("maweibo")
    table = lor.build_feature_table("maweibo", entries,
                                    S.synth_e0_rows(entries),
                                    S.synth_e1_rows(entries),
                                    S.synth_e2_rows(entries))
    with pytest.raises(fa.AttributionRefused, match="needs E3"):
        fa.variant_evidence_vector(table[entries[0]["key"]], "qwen", "D2_Z_F_S")


def test_build_rows_reader_and_event_scope():
    entries, table, split, _fp = _dataset()
    rows = fa.build_rows("D0_Z", entries, table, ["qwen", "mistral"],
                         split["utility_train"][:2])
    assert {r["reader"] for r in rows} == {"qwen", "mistral"}
    assert {r["event_id"] for r in rows} == set(split["utility_train"][:2])
    assert len(rows[0]["x"]) == 26


def test_run_variant_end_to_end_and_comparison():
    entries, table, split, fingerprints = _dataset()
    d0 = fa.run_variant("maweibo", "D0_Z", entries, table, split,
                        fingerprints, train_kwargs=FAST)
    assert d0["uses_fingerprint"] is False
    assert d0["model_kind"] == P.MODEL_B0
    d5 = fa.run_variant("maweibo", "D5_Z_S_C", entries, table, split,
                        fingerprints, train_kwargs=FAST)
    assert d5["uses_fingerprint"] is False
    assert d5["rotations"]["qwen"]["evidence_dim"] == 32
    d1 = fa.run_variant("maweibo", "D1_Z_F", entries, table, split,
                        fingerprints, train_kwargs=FAST)
    assert d1["uses_fingerprint"] is True
    assert d1["model_kind"] == P.MODEL_B4
    summary = fa.summarise_variant(d0)
    assert "eval_rows" not in str(summary)
    comparison = fa.compare_to_baseline(d5, d0, iterations=50, seed=7319)
    assert comparison["baseline"] == "D0_Z"
    assert sorted(comparison["per_reader_delta"]) == sorted(P.READER_KEYS)
    assert comparison["aggregate"]["iterations"] == 50
    assert comparison["aggregate"]["worst_reader_delta"] == min(
        comparison["per_reader_delta"].values())
