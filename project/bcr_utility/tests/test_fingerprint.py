"""M1 fingerprint tests (synthetic, model-free)."""
from __future__ import annotations

import math

import pytest

from ..config import protocol as P
from ..probes import fingerprint as fp
from . import m1_synth as S


def test_binary_entropy_bounds():
    assert fp.binary_entropy(0.5) == pytest.approx(1.0)
    assert fp.binary_entropy(1.0) == pytest.approx(0.0, abs=1e-6)
    assert fp.binary_entropy(0.0) == pytest.approx(0.0, abs=1e-6)


def test_response_metrics():
    m = fp.response_metrics(2.0, -1.0, 0.9)
    assert m["signed_margin"] == pytest.approx(3.0)
    assert m["abs_margin"] == pytest.approx(3.0)
    assert 0.0 < m["entropy"] < 1.0


def test_audit_accepts_full_synthetic_responses():
    audit = fp.audit_responses(S.synth_probe_rows(), S.synth_manifest())
    assert audit["ok"] is True
    assert audit["n_rows"] == P.PROBE_TOTAL_EVALS == 864
    assert audit["per_reader_counts"] == {
        r: P.PROBE_CONTEXT_EVALS_PER_READER for r in P.READER_KEYS}


def test_audit_refuses_missing_rows():
    rows = S.synth_probe_rows()[:-1]
    with pytest.raises(fp.FingerprintRefused, match="missing"):
        fp.audit_responses(rows, S.synth_manifest())


def test_audit_refuses_duplicates():
    rows = S.synth_probe_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(fp.FingerprintRefused, match="duplicate"):
        fp.audit_responses(rows, S.synth_manifest())


def test_audit_refuses_extra_items_outside_manifest():
    rows = S.synth_probe_rows()
    extra = dict(rows[0])
    extra["event_id"] = "outside_event"
    rows.append(extra)
    with pytest.raises(fp.FingerprintRefused, match="outside"):
        fp.audit_responses(rows, S.synth_manifest())


@pytest.mark.parametrize("field", ["utility", "sign", "gold_label",
                                   "is_helpful", "harmful_flag",
                                   "correct_before"])
def test_audit_refuses_forbidden_fields(field):
    rows = S.synth_probe_rows()
    rows[0][field] = 0.0
    with pytest.raises(fp.FingerprintRefused, match="forbidden"):
        fp.audit_responses(rows, S.synth_manifest())


def test_audit_refuses_model_id_drift():
    rows = S.synth_probe_rows()
    rows[0]["model_id"] = "google/gemma-3-4b-it"
    with pytest.raises(fp.FingerprintRefused, match="model id"):
        fp.audit_responses(rows, S.synth_manifest())


def test_audit_refuses_non_finite():
    rows = S.synth_probe_rows()
    rows[0]["signed_margin"] = float("nan")
    with pytest.raises(fp.FingerprintRefused, match="not finite"):
        fp.audit_responses(rows, S.synth_manifest())


def test_compact_fingerprint_layout_and_dim():
    built = fp.build_fingerprints(S.synth_probe_rows(), S.synth_manifest())
    for reader in P.READER_KEYS:
        compact = built["readers"][reader]["compact"]
        assert len(compact["vector"]) == P.FINGERPRINT_DIM == 216
        assert len(compact["groups"]) == 24  # 2 datasets x 3 cutoffs x 4 ctx
        assert all(math.isfinite(v) for v in compact["vector"])
        p0 = compact["groups"]["maweibo|15|P0"]
        assert tuple(p0) == P.FINGERPRINT_BASE_METRICS
        p1 = compact["groups"]["maweibo|15|P1"]
        assert tuple(p1) == (P.FINGERPRINT_BASE_METRICS
                             + P.FINGERPRINT_DELTA_METRICS)


def test_compact_fingerprint_delta_block_is_exact():
    rows = S.synth_probe_rows()
    built = fp.build_fingerprints(rows, S.synth_manifest())
    # recompute one group by hand
    group = [r for r in rows if r["reader"] == "qwen"
             and r["dataset"] == "maweibo" and r["cutoff"] == 15
             and r["context"] == "P1"]
    p0 = {(r["dataset"], r["event_id"], r["cutoff"]): r
          for r in rows if r["reader"] == "qwen" and r["context"] == "P0"}
    expected_mean_delta = sum(
        r["signed_margin"]
        - p0[(r["dataset"], r["event_id"], r["cutoff"])]["signed_margin"]
        for r in group) / len(group)
    got = built["readers"]["qwen"]["compact"]["groups"]["maweibo|15|P1"][
        "mean_delta_margin"]
    assert got == pytest.approx(expected_mean_delta)


def test_scaler_uses_training_readers_only_and_handles_zero_variance():
    fingerprints = S.synth_fingerprints()
    scaler = fp.fit_fingerprint_scaler(fingerprints,
                                       ("qwen", "mistral"))
    assert scaler["fit_readers"] == ["qwen", "mistral"]
    assert all(s >= 1.0 or s > 1e-12 for s in scaler["std"])
    # a constant dimension across the two training readers gets std 1.0
    vec_q = fp.fingerprint_vector(fingerprints, "qwen")
    vec_m = fp.fingerprint_vector(fingerprints, "mistral")
    for j, (a, b) in enumerate(zip(vec_q, vec_m)):
        if a == b:
            assert scaler["std"][j] == 1.0
    # transforming the held-out reader never changes the scaler
    before = dict(scaler)
    fp.transform_fingerprint(
        fp.fingerprint_vector(fingerprints, "internlm"), scaler)
    assert scaler == before


def test_fingerprint_distances_symmetric_and_zero_for_identical():
    fingerprints = S.synth_fingerprints()
    scaler = fp.fit_fingerprint_scaler(fingerprints, P.READER_KEYS)
    distances = fp.fingerprint_distances(fingerprints, scaler, P.READER_KEYS)
    assert set(distances) == {"qwen|mistral", "qwen|internlm",
                              "mistral|internlm"}
    assert all(d > 0 for d in distances.values())


def test_fingerprints_digest_is_reader_order_independent():
    fingerprints = S.synth_fingerprints()
    again = fp.build_fingerprints(S.synth_probe_rows(), S.synth_manifest())
    assert fp.fingerprints_digest(fingerprints) == \
        fp.fingerprints_digest(again)
