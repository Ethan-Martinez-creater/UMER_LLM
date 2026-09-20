"""M1 verifier tests over a synthetic results tree."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from .. import verifier_m1
from ..config import protocol as P
from ..probes import fingerprint as fp
from . import m1_synth as S


def _write(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if isinstance(payload, str):
            fh.write(payload)
        else:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
    return path


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _rotation(held, train):
    models = {}
    for kind in (P.MODEL_B0, P.MODEL_B1, P.MODEL_B4):
        models[kind] = {
            "selected_config": {"hidden": 32, "dropout": 0.0, "lr": 1e-3,
                                "weight_decay": 0.0},
            "dev_macro_f1": 0.4, "dev_loss": 1.0, "n_parameters": 10000,
            "grid_summary": [{"config": {"hidden": 32}}] * 8,
            "scaler_rows": 100,
            "fold_sizes": {"train": 100, "dev": 30, "eval": 50},
            "eval_metrics": {"macro_f1": 0.4},
        }
    models[P.MODEL_B3] = {"nearest_reader": train[0],
                          "distances": {train[0]: 1.0, train[1]: 2.0},
                          "eval_metrics": {"macro_f1": 0.3}}
    return {"dataset": "maweibo", "held_out": held,
            "train_readers": list(train), "models": models,
            "disagreement_subset": {"n": 5, "of": 50}}


def _synthetic_tree(tmp_path, mutate=None):
    root = str(tmp_path)
    # probe manifest (bootstrap namespace) --------------------------------
    manifest = S.synth_manifest()
    manifest_path = os.path.join(
        root, P.RESULTS_ROOT, P.BOOTSTRAP_DIRNAME,
        P.PROBE_MANIFEST_FILENAME)
    _write(manifest_path, manifest)
    manifest_sha = _sha(manifest_path)

    # probe shards ---------------------------------------------------------
    rows = S.synth_probe_rows()
    for dataset in P.DATASETS:
        for reader in P.READER_KEYS:
            shard = [r for r in rows
                     if r["dataset"] == dataset and r["reader"] == reader]
            _write(os.path.join(
                root, P.RESULTS_ROOT, P.M1_DIRNAME, "probe_responses",
                f"probe_responses_{dataset}_{reader}.jsonl"),
                "".join(json.dumps(r) + "\n" for r in shard))

    # fingerprints ---------------------------------------------------------
    built = fp.build_fingerprints(rows, manifest)
    fingerprints = {
        "probe_manifest_sha256": manifest_sha,
        "dim": P.FINGERPRINT_DIM,
        "readers": {r: {"model_id": P.READER_MODEL_IDS[r],
                        "compact": built["readers"][r]["compact"],
                        "raw_sha256": built["readers"][r]["raw_sha256"]}
                    for r in P.READER_KEYS},
    }
    fingerprints["sha256"] = fp.fingerprints_digest(
        {"readers": fingerprints["readers"]})
    _write(os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME, "fingerprints",
                        "fingerprints.json"), fingerprints)
    _write(os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME, "fingerprints",
                        "fingerprint_audit.json"),
           {"probe_manifest_sha256": manifest_sha,
            "response_audit": built["audit"], "ok": True})

    # features -------------------------------------------------------------
    feature_entries = {}
    for dataset in P.DATASETS:
        entry = {}
        for stage, n_rows in (("e0", P.FROZEN_ATOMIC_KEYS[dataset]),
                              ("e1", P.FROZEN_ATOMIC_KEYS[dataset]),
                              ("e2", P.FROZEN_ATOMIC_KEYS[dataset] * 3)):
            path = _write(os.path.join(
                root, P.RESULTS_ROOT, P.M1_DIRNAME, "features",
                f"{stage}_{dataset}.jsonl"), "x" * 100)
            entry[stage] = {"rows": n_rows, "path": path,
                            "sha256": _sha(path)}
        entry["e1"]["extractor"] = {"model_id": P.NLI_MODEL_ID}
        feature_entries[dataset] = entry
    _write(os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME, "features",
                        "feature_audit.json"),
           {"features": feature_entries, "ok": True})

    # evaluation -----------------------------------------------------------
    aggregate = {"mean_delta_macro_f1": 0.04, "ci_low": 0.01, "ci_high": 0.07,
                 "per_reader_delta": {"qwen": 0.05, "mistral": 0.04,
                                      "internlm": 0.03},
                 "positive_readers": 3, "worst_reader_delta": 0.03}
    rotations = {rot[2]: _rotation(rot[2], rot[:2])
                 for rot in P.LORO_ROTATIONS}
    datasets = {
        "maweibo": {"decides_m1_gate": True, "diagnostic_only": False,
                    "rotations": rotations, "aggregate": aggregate,
                    "b2_in_domain_diagnostic": {"in_domain": True}},
        "pheme": {"decides_m1_gate": False, "diagnostic_only": True,
                  "rotations": rotations, "aggregate": aggregate,
                  "b2_in_domain_diagnostic": {"in_domain": True}},
    }
    _write(os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME, "evaluation",
                        "evaluation.json"), {"datasets": datasets})
    gate = {"dataset": "maweibo", "decides_m1_gate": True,
            "thresholds": {"mean_delta_min": P.GATE_MEAN_DELTA_MIN,
                           "ci_alpha": P.GATE_CI_ALPHA,
                           "positive_readers_min": P.GATE_POSITIVE_READERS_MIN,
                           "worst_reader_min": P.GATE_WORST_READER_MIN},
            "aggregate": aggregate, "passed": True}
    _write(os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME, "evaluation",
                        P.M1_GATE_FILENAME), gate)
    _write(os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME,
                        P.M1_VERDICT_FILENAME),
           {"zero_touch_passed": True,
            "pheme_diagnostic": {"decides_m1_gate": False}})
    if mutate:
        mutate(root)
    return root


def test_verifier_m1_clean_tree_has_no_issues(tmp_path):
    root = _synthetic_tree(tmp_path)
    out = verifier_m1.verify(root)
    assert out["issues"] == [], out["issues"]
    names = {c["check"] for c in out["checks"]}
    assert "m1_probe_rows_exact_864" in names
    assert "m1_feature_coverage_and_nli_identity" in names
    assert "m1_loro_rotations_splits_grid_exact" in names
    assert "m1_gate_thresholds_frozen" in names


def test_verifier_m1_detects_probe_row_count_drift(tmp_path):
    def mutate(root):
        path = os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME,
                            "probe_responses",
                            "probe_responses_maweibo_qwen.jsonl")
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines[:-1])
    out = verifier_m1.verify(_synthetic_tree(tmp_path, mutate))
    assert "m1_probe_rows_exact_864" in out["issues"]


def test_verifier_m1_detects_forbidden_probe_field(tmp_path):
    def mutate(root):
        path = os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME,
                            "probe_responses",
                            "probe_responses_pheme_internlm.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            row = S.synth_probe_rows()[0]
            row["dataset"] = "pheme"
            row["reader"] = "internlm"
            row["event_id"] = "pheme_p000"
            row["utility"] = 0.5
            fh.write(json.dumps(row) + "\n")
    out = verifier_m1.verify(_synthetic_tree(tmp_path, mutate))
    assert "m1_probe_rows_have_no_utility_or_gold_fields" in out["issues"]


def test_verifier_m1_detects_nli_model_drift(tmp_path):
    def mutate(root):
        path = os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME, "features",
                            "feature_audit.json")
        data = json.load(open(path, encoding="utf-8"))
        data["features"]["maweibo"]["e1"]["extractor"]["model_id"] = \
            "some/other-nli"
        _write(path, data)
    out = verifier_m1.verify(_synthetic_tree(tmp_path, mutate))
    assert "m1_feature_coverage_and_nli_identity" in out["issues"]


def test_verifier_m1_detects_pheme_deciding_the_gate(tmp_path):
    def mutate(root):
        path = os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME, "evaluation",
                            "evaluation.json")
        data = json.load(open(path, encoding="utf-8"))
        data["datasets"]["pheme"]["decides_m1_gate"] = True
        data["datasets"]["pheme"]["diagnostic_only"] = False
        _write(path, data)
    out = verifier_m1.verify(_synthetic_tree(tmp_path, mutate))
    assert "m1_loro_rotations_splits_grid_exact" in out["issues"]


def test_verifier_m1_detects_e3_before_zero_touch_failure(tmp_path):
    def mutate(root):
        _write(os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME, "features",
                            "e3_maweibo.jsonl"), "x")
    out = verifier_m1.verify(_synthetic_tree(tmp_path, mutate))
    assert "m1_no_e3_before_zero_touch_failure" in out["issues"]


def test_verifier_m1_detects_fingerprint_digest_drift(tmp_path):
    def mutate(root):
        path = os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME,
                            "fingerprints", "fingerprints.json")
        data = json.load(open(path, encoding="utf-8"))
        data["readers"]["qwen"]["compact"]["vector"][0] += 1.0
        _write(path, data)
    out = verifier_m1.verify(_synthetic_tree(tmp_path, mutate))
    assert "m1_fingerprints_exact_and_pinned" in out["issues"]


def test_verifier_m1_detects_gate_threshold_drift(tmp_path):
    def mutate(root):
        path = os.path.join(root, P.RESULTS_ROOT, P.M1_DIRNAME, "evaluation",
                            P.M1_GATE_FILENAME)
        data = json.load(open(path, encoding="utf-8"))
        data["thresholds"]["mean_delta_min"] = 0.01
        _write(path, data)
    out = verifier_m1.verify(_synthetic_tree(tmp_path, mutate))
    assert "m1_gate_thresholds_frozen" in out["issues"]


def test_verifier_m1_detects_m2_artifacts(tmp_path):
    def mutate(root):
        os.makedirs(os.path.join(root, P.RESULTS_ROOT, "m2_pilot"))
    out = verifier_m1.verify(_synthetic_tree(tmp_path, mutate))
    assert "m1_no_m2_or_reader_panel_expansion" in out["issues"]
