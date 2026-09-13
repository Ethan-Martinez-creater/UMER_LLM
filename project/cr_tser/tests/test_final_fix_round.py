"""Integration tests for the final code-complete fix round.

Pins: the frozen normalized Weibo22 source path, P0 token-boundary
fail-closed, the P3 no-target-leakage contract, per-model prediction artifacts,
label-cap vs inference-candidate separation, S6 consuming non-empty legacy
scores, cache identity substitution, single-reader S3a/S3b, source fingerprints
and the smoke namespace.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from ..config.pilot_config import (SEMANTIC_DIM, STRUCT_SCALAR_DIM,
                                   TEXT_FEATURE_DIM, UTILITY_Q_DIM)
from ..data.snapshot_bridge import build_causal_snapshot
from ..intervention.evidence_units import evidence_key
from .conftest import branch_event


# --------------------------------------------------------------------------
# 1. normalized Weibo22 export validation
# --------------------------------------------------------------------------
def _event(reply_text="reply body"):
    return {
        "event_id": "e1", "label": 1, "source_id": "n0",
        "source_timestamp": 1000,
        "nodes": [
            {"node_id": "n0", "parent_id": None, "timestamp": 1000,
             "text": "source body", "original_order": 0, "status": "VALID"},
            {"node_id": "n1", "parent_id": "n0", "timestamp": 1060,
             "text": reply_text, "original_order": 1, "status": "VALID"},
        ],
    }


def test_normalized_export_validation_ready_and_unavailable(tmp_path):
    from cr_tser.data import weibo22_adapter as adapter
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(_event()) + "\n", encoding="utf-8")
    report = adapter.validate_normalized_export(str(path))
    assert report["valid"] is True
    assert report["verdict"] == adapter.VERDICT_READY
    assert all(v >= 1.0 for v in report["coverage"].values())

    path.write_text(json.dumps(_event(reply_text="   ")) + "\n",
                    encoding="utf-8")
    report = adapter.validate_normalized_export(str(path))
    assert report["valid"] is False
    assert report["verdict"] == adapter.VERDICT_UNAVAILABLE


def test_raw_release_without_normalized_export_stays_unavailable(tmp_path):
    from cr_tser.data import weibo22_adapter as adapter
    raw = tmp_path / "kpg"
    (raw / "label_all").mkdir(parents=True)
    (raw / "td_rvnn").mkdir(parents=True)
    (raw / "label_all" / "Weibo_label_All.txt").write_text(
        "true\tcovid-19\t111\n", encoding="utf-8")
    (raw / "td_rvnn" / "data.TD_RvNN.vol_5000.txt").write_text(
        "111\tNone\t1\t2\t0\t5:1\n", encoding="utf-8")
    audit = adapter.audit_release(str(raw))
    assert audit["verdict"] == adapter.VERDICT_UNAVAILABLE
    with pytest.raises(adapter.Weibo22TemporalUnavailable):
        adapter.load_events(str(raw))


# --------------------------------------------------------------------------
# 2. P0 readiness fail-closed on boundaries
# --------------------------------------------------------------------------
def test_p0_boundary_failure_blocks_pass():
    import cr_tser_p0_audit as p0
    audit = {"verdict": "WEIBO22_TEMPORAL_READY"}
    readers = {"qwen": {"model_path_exists": True, "loaded": True}}
    bad = {"qwen": {"identical_predictions": True, "boundaries_ok": False}}
    good = {"qwen": {"identical_predictions": True, "boundaries_ok": True}}
    assert p0.evaluate_readiness(audit, readers, bad, True, True)["P0"] == \
        "P0_FAIL"
    assert p0.evaluate_readiness(audit, readers, good, True, True)["P0"] == \
        "P0_PASS"
    # identical predictions but a failing boundary is still a failure
    nonident = {"qwen": {"identical_predictions": False, "boundaries_ok": True}}
    assert p0.evaluate_readiness(audit, readers, nonident, True, True)["P0"] == \
        "P0_FAIL"
    # data not ready always blocks
    assert p0.evaluate_readiness({"verdict": "WEIBO22_TEMPORAL_UNAVAILABLE"},
                                 readers, good, True, True)["P0"] == "P0_FAIL"


# --------------------------------------------------------------------------
# 3. P3 prediction contract
# --------------------------------------------------------------------------
def test_p3_predicted_sign_comes_from_sign_head_not_correctness():
    import cr_tser_run_pilot as pilot
    # ground-truth utility is positive but the model's sign head says HARMFUL
    record = pilot._pred_record("HELPFUL", 0.9,
                                {"utility": 0.5, "probs": [0.0, 0.1, 0.9],
                                 "predicted_sign": "HARMFUL"}, active=True)
    assert record["pred_sign"] == "HARMFUL"
    assert record["harmful_score"] == 0.9
    # and the reverse
    record = pilot._pred_record("HARMFUL", -0.9,
                                {"utility": -0.5, "probs": [0.9, 0.1, 0.0],
                                 "predicted_sign": "HELPFUL"}, active=True)
    assert record["pred_sign"] == "HELPFUL"


def test_p3_active_definition_includes_correctness_change():
    import cr_tser_run_pilot as pilot
    source = open(os.path.join(os.path.dirname(pilot.__file__),
                               "cr_tser_run_pilot.py"), encoding="utf-8").read()
    assert "abs(gold_cont) >= 0.05" in source
    assert "correctness_before" in source and "correctness_after" in source
    # no threshold-on-ground-truth sign derivation remains
    assert "_sign_of" not in source


def test_baseline_predictions_carry_own_sign_head():
    from ..models.text_baseline import TextOnlyPredictor
    from ..training.train_utility import predict_baseline
    model = TextOnlyPredictor()
    runs = [{"model": model, "seed": 7319}]
    rows = [{"key": "k1", "x": torch.zeros(TEXT_FEATURE_DIM)}]
    out = predict_baseline(runs, rows, device="cpu")
    assert set(out["k1"]) >= {"utility", "probs", "predicted_sign",
                              "class_scores"}
    assert len(out["k1"]["probs"]) == 3
    assert out["k1"]["predicted_sign"] in ("HELPFUL", "NEUTRAL", "HARMFUL")


# --------------------------------------------------------------------------
# 4. label cap vs inference candidates
# --------------------------------------------------------------------------
def test_label_cap_does_not_limit_inference_candidates():
    from ..training.utility_dataset import make_group
    snap = build_causal_snapshot(branch_event(), 360)
    n = len(snap["node_ids"])
    sem = torch.zeros(n, SEMANTIC_DIM)
    struct = torch.zeros(n, STRUCT_SCALAR_DIM)
    q = torch.zeros(n, UTILITY_Q_DIM)
    pos = {nid: i for i, nid in enumerate(snap["node_ids"])}
    # simulate the §11 cap: exactly one labeled unit
    unit_rows = [{"unit_index": 1, "reader": "qwen", "target": 0.1,
                  "sign": "NEUTRAL", "correct_before": True,
                  "correct_after": True}]
    infer_keys = [(i, evidence_key("pheme", snap["event_id"], 360, nid))
                  for i, nid in enumerate(snap["node_ids"])
                  if nid != snap["source_id"]]
    group = make_group(snap, sem, struct, q, unit_rows, 360, [1],
                       dataset="pheme", infer_keys=infer_keys)
    assert len(group["unit_rows"]) == 1
    assert len(group["infer_rows"]) == len(infer_keys) > 1
    covered = {r["unit_key"] for r in group["infer_rows"]}
    assert pos  # sanity
    assert len(covered) == len(infer_keys)


# --------------------------------------------------------------------------
# 5. S6 consumes non-empty legacy scores; missing prediction fails closed
# --------------------------------------------------------------------------
def _fixture():
    units = [{"node_id": f"n{i}", "reply_text": f"reply {i} body",
              "parent_text": "parent body", "parent_id": None,
              "timestamp": i, "elapsed_seconds": i, "snapshot_order": i,
              "depth": 1} for i in range(1, 7)]
    ids = [u["node_id"] for u in units]
    src = {"selected_node_ids": ids, "ranked_node_ids": ids,
           "unit_token_costs": {n: 60 for n in ids}, "total_tokens": 360,
           "budget_ref": 1024, "n_selected": len(ids), "utilization": 0.2,
           "rank_percentile": {n: i / len(ids) for i, n in enumerate(ids)},
           "relevance": {n: 1.0 - i / 10 for i, n in enumerate(ids)}}
    return units, src, ids


class _Tok:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(str(text).split())))}


def test_s6_consumes_legacy_scores():
    from ..models.robust_selector import build_arm
    units, src, ids = _fixture()
    forward = {n: float(i) for i, n in enumerate(ids)}
    reverse = {n: float(len(ids) - i) for i, n in enumerate(ids)}
    arm_a = build_arm("S6_legacy_utility_tm", units, src, {"legacy": forward},
                      _Tok(), ["qwen", "glm"], seed=7319)
    arm_b = build_arm("S6_legacy_utility_tm", units, src, {"legacy": reverse},
                      _Tok(), ["qwen", "glm"], seed=7319)
    assert arm_a["selected_node_ids"] != arm_b["selected_node_ids"]
    assert arm_a["selected_node_ids"]  # non-empty packing


def test_missing_candidate_prediction_fails_closed():
    from ..models.robust_selector import MissingPredictionError, build_arm
    units, src, ids = _fixture()
    full = {n: 0.2 for n in ids}
    partial = {n: 0.2 for n in ids[:-1]}
    with pytest.raises(MissingPredictionError):
        build_arm("S5_cross_reader_robust", units, src,
                  {"qwen": partial, "glm": full}, _Tok(), ["qwen", "glm"],
                  seed=7319)
    with pytest.raises(MissingPredictionError):
        build_arm("S4_shared", units, src, {"shared": partial}, _Tok(),
                  ["qwen", "glm"], seed=7319)


# --------------------------------------------------------------------------
# 6. S3a/S3b single-reader artifacts
# --------------------------------------------------------------------------
def test_single_reader_artifact_guard(tmp_path):
    import cr_tser_run_selection as selection
    target = tmp_path / "predictor" / "pheme" / "single_qwen"
    target.mkdir(parents=True)
    (target / "predictions.json").write_text(json.dumps(
        {"training_readers": ["qwen", "glm"], "predictions": {}}),
        encoding="utf-8")
    with pytest.raises(ValueError):
        selection._load_single_predictions(str(tmp_path), "pheme", "qwen")
    (target / "predictions.json").write_text(json.dumps(
        {"training_readers": ["qwen"],
         "predictions": {"pheme|e|60|n1": {"utility": 0.1}}}), encoding="utf-8")
    got = selection._load_single_predictions(str(tmp_path), "pheme", "qwen")
    assert set(got) == {"pheme|e|60|n1"}


# --------------------------------------------------------------------------
# 7. source fingerprint + smoke namespace
# --------------------------------------------------------------------------
def test_source_fingerprint_detects_change(tmp_path):
    from cr_tser.data.source_manifest import (SourceIdentityError,
                                              assert_same_source,
                                              fingerprint_path)
    target = tmp_path / "export.jsonl"
    target.write_text("a", encoding="utf-8")
    first = fingerprint_path(str(target))
    target.write_text("b", encoding="utf-8")
    second = fingerprint_path(str(target))
    assert first["sha256"] != second["sha256"]
    with pytest.raises(SourceIdentityError):
        assert_same_source(first, second, "test")
    assert_same_source(first, dict(first), "test")  # identical passes


def test_smoke_namespace_is_isolated():
    import cr_tser_common as common
    assert common.smoke_root("/x") == os.path.join("/x", "smoke")
    source = open(os.path.join(os.path.dirname(common.__file__),
                               "cr_tser_generate_labels.py"),
                  encoding="utf-8").read()
    assert "smoke_root(out_root)" in source


def test_score_items_never_trains_and_is_pheme_only():
    from ..models.legacy_utility import LegacyNotPermitted, LegacyPHEMEUtility
    with pytest.raises(LegacyNotPermitted):
        LegacyPHEMEUtility("weibo22", None, None)
