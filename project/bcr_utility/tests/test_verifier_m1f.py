"""M1-F verifier tests over a synthetic M0/M1/M1-E/M1-F tree."""
from __future__ import annotations

import json
import os
import subprocess

from .. import verifier_m1f
from ..config import protocol as P


def _write(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)


def _write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


# --------------------------------------------------------------------------
# frozen upstream tree
# --------------------------------------------------------------------------
def _frozen_tree(root):
    _write(P.m1_path(root, P.M1_VERDICT_FILENAME),
           {"final_outcome": P.M1E_EXPECTED_M1_VERDICT,
            "zero_touch_passed": False, "light_touch_passed": True})
    ev = P.m1_path(root, "evaluation")
    _write_jsonl(os.path.join(ev, P.M1_PREDICTIONS_FILENAME),
                 [{"dataset": "maweibo"}])
    _write_jsonl(os.path.join(ev, "predictions_light_touch.jsonl"),
                 [{"dataset": "maweibo"}])
    for name in (P.M1_EVALUATION_FILENAME, "evaluation_light_touch.json",
                 "gate.json", "gate_light_touch.json"):
        _write(os.path.join(ev, name), {"ok": True})
    for dataset in P.DATASETS:
        for prefix in ("e0", "e1", "e2", "e3"):
            _write_jsonl(P.m1_path(root, "features",
                                   f"{prefix}_{dataset}.jsonl"),
                         [{"key": "k"}])
        _write_jsonl(P.m1_path(root, "probe_responses",
                               f"probe_responses_{dataset}_qwen.jsonl"),
                     [{"dataset": dataset}])
    _write(P.m1_path(root, "fingerprints", "fingerprints.json"), {"dim": 216})
    _write(P.result_path(root, P.PROBE_MANIFEST_FILENAME), {"items": 72})
    _write(P.result_path(root, P.ATOMIC_INDEX_FILENAME), {"datasets": {}})
    _write(P.m1e_path(root, P.M1E_VERDICT_FILENAME),
           {"status": "COMPLETED", "defines_new_gate": False,
            "m1_verdict_unchanged": True})
    for dataset in P.DATASETS:
        _write_jsonl(P.m1e_path(root, P.M1E_ATTRIBUTION_DIRNAME,
                                f"predictions_{dataset}_D5_Z_S_C.jsonl"),
                     [{"dataset": dataset, "model": "D5_Z_S_C"}])
        _write(os.path.join(P.historical_root(root), "manifests", dataset,
                            "event_split.json"), {"dataset": dataset})


# --------------------------------------------------------------------------
# M1-F artifacts
# --------------------------------------------------------------------------
def _rotation(rotation, kind):
    return {
        "held_out": rotation[2], "train_readers": [rotation[0], rotation[1]],
        "selected_config": {"hidden": 32, "dropout": 0.0, "lr": 1e-3,
                            "weight_decay": 0.0} if kind == "trained" else None,
        "n_parameters": 0, "scaler_rows": 100 if kind == "trained" else 0,
        "fold_sizes": {"train": 100, "dev": 30, "eval": 50},
        "evidence_dim": 4, "dev_macro_f1": 0.3,
    }


def _variant_block(variant):
    kind = ("reused" if variant == P.M1F_S4 else
            "constant" if variant in P.M1F_CONSTANT_VARIANTS else "trained")
    rotations = {r[2]: _rotation(r, kind) for r in P.LORO_ROTATIONS}
    return {
        "dataset": "maweibo", "variant": variant, "kind": kind,
        "frozen_variant": "D5_Z_S_C" if variant == P.M1F_S4 else None,
        "uses_fingerprint": False, "model_kind": P.MODEL_B0,
        "rotations": rotations,
        "per_reader_macro_f1": {h: 0.3 for h in rotations},
    }


def _baseline_payload(dataset):
    readers = {r[2]: 0.3 for r in P.LORO_ROTATIONS}
    return {
        "protocol": P.PROTOCOL_VERSION, "stage": "m1f_baselines",
        "dataset": dataset, "scope": "falsification_audit",
        "defines_gate": False,
        "variants": {v: _variant_block(v) for v in P.M1F_VARIANTS},
        "dev_macro_f1": {v: dict(readers) for v in P.M1F_VARIANTS},
        "comparator": {
            "selected": P.M1F_S2, "pool": list(P.M1F_COMPARATOR_POOL),
            "dev_macro_f1_mean": {v: 0.3 for v in P.M1F_COMPARATOR_POOL},
            "selected_by": "mean utility_dev Macro-F1, training readers only"},
        "predictions": {},
    }


def _active_bundle(variant):
    return {
        "variant": variant, "overall": {"macro_f1": 0.3},
        "active": {"defined": True, "n_active": 10, "n_gold_helpful": 5,
                   "n_gold_harmful": 5, "neutral_prediction_rate": 0.2,
                   "helpful_vs_harmful_macro_f1": 0.3,
                   "balanced_accuracy": 0.3, "harmful_auprc": 0.2,
                   "helpful_auprc": 0.2},
        "per_reader_macro_f1": {},
    }


def _primary_comparison():
    return {
        "primary": P.M1F_S4, "comparator": P.M1F_S2,
        "aggregate": {"mean_delta_macro_f1": 0.01, "ci_low": -0.01,
                      "ci_high": 0.03, "positive_readers": 2,
                      "worst_reader_delta": -0.02, "iterations": 10000,
                      "seed": 7319, "n_events_per_reader": 25},
        "per_reader_macro_f1_delta": {
            held: {"observed": 0.01, "ci_low": -0.02, "ci_high": 0.03}
            for held in P.READER_KEYS},
        "secondary_metric_deltas": {
            name: {"observed": 0.01, "ci_low": -0.01, "ci_high": 0.02}
            for name in ("balanced_accuracy", "active_macro_f1",
                         "harmful_auprc", "helpful_auprc",
                         "utility_spearman", "centered_spearman",
                         "harmful_f1", "helpful_f1", "neutral_f1")},
    }


def _active_payload():
    return {
        "protocol": P.PROTOCOL_VERSION, "stage": "m1f_active_signed_audit",
        "primary_dataset": P.PRIMARY_DATASET,
        "official_threshold": P.M1F_OFFICIAL_THRESHOLD,
        "primary": _primary_comparison(),
        "variants": {v: _active_bundle(v) for v in P.M1F_VARIANTS},
    }


def _within_payload():
    variants = {v: {"variant": v, "centered_pooled_spearman": 0.1,
                    "centered_spearman_ci": {
                        "observed": 0.1, "ci_low": -0.01, "ci_high": 0.2,
                        "iterations": 10000, "seed": 7319, "n_events": 25},
                    "pairwise_concordance": 0.5, "n_snapshots_used": 10,
                    "n_snapshots_total": 12, "n_rows_used": 100,
                    "mean_within_snapshot_spearman": 0.05,
                    "raw_pooled_spearman": 0.08, "skipped":
                        {"too_few_rows": 1, "zero_variance": 1}}
                for v in P.M1F_TRAINED_VARIANTS + (P.M1F_S4,)}
    return {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1f_within_snapshot_audit",
        "dataset": P.PRIMARY_DATASET, "min_rows": P.M1F_WITHIN_MIN_ROWS,
        "variance_eps": P.M1F_WITHIN_VARIANCE_EPS,
        "variants": variants,
        "comparisons_vs_S2": {
            v: {"observed": 0.01, "ci_low": -0.02, "ci_high": 0.04}
            for v in (P.M1F_S3, P.M1F_S4)},
    }


def _robustness_payload():
    return {
        "protocol": P.PROTOCOL_VERSION, "stage": "m1f_robustness_diagnostic",
        "dataset": P.PRIMARY_DATASET,
        "official_threshold": P.M1F_OFFICIAL_THRESHOLD,
        "thresholds": list(P.M1F_THRESHOLDS),
        "threshold_selected": False,
        "diagnostics": {f"{t:.2f}": {P.M1F_S4: {"macro_f1": 0.3}}
                        for t in P.M1F_THRESHOLDS},
        "activity": {"near_boundary_rate": {P.M1F_S4: 0.1},
                     "gold_near_boundary_rate": {P.M1F_S4: 0.1},
                     "gold_activity_rate": {P.M1F_S4: 0.4},
                     "predicted_activity_rate": {P.M1F_S4: 0.4},
                     "near_boundary_threshold": P.M1F_OFFICIAL_THRESHOLD,
                     "band": P.M1F_NEAR_BOUNDARY_BAND},
        "ab_label_swap": {"status": P.M1F_AB_SWAP_STATUS},
    }


def _task_validity_payload():
    return {
        "protocol": P.PROTOCOL_VERSION, "stage": "m1f",
        "relabelled_historical_data": False,
        "changed_threshold_or_split": False,
        "utility_meaning": {"is_verification_utility": False,
                            "claim_allowed":
                                "reader-specific rumour-label evidence utility",
                            "claim_forbidden":
                                "fact-checking / truth-verification utility"},
        "datasets": {
            "maweibo": {"is_truth_verification": False,
                        "label_semantics":
                            "event-level rumour / non-rumour annotation",
                        "boundary": "a rumour annotation, not a truth verdict"},
            "pheme": {"is_truth_verification": False,
                      "label_semantics":
                          "event-level rumour / non-rumour annotation",
                      "boundary": "rumour != false and non-rumour != true"},
        },
        "cross_dataset_boundary": {"is_language_only_shift": False},
        "conclusion": {"validity_blocked": False,
                       "supports_intended_claim": True,
                       "reason": "utility is rumour-label agreement utility"},
    }


def _verdict_payload():
    return {
        "protocol": P.PROTOCOL_VERSION, "stage": "m1f", "status": "COMPLETED",
        "verdict": "M1F_CLOSE_BCR_UTILITY_METHOD",
        "primary_dataset": P.PRIMARY_DATASET,
        "defines_new_gate": False, "m1_verdict_unchanged": True,
        "m1e_verdict_unchanged": True, "m2_entered": False,
        "checks": {"mean_delta_gte_0.02": False, "ci_low_gt_0": False,
                   "positive_readers_gte_2": True,
                   "active_macro_f1_improves": False,
                   "harmful_auprc_not_decreased": True,
                   "within_snapshot_centered_spearman_improves_over_S2": False,
                   "gain_not_only_neutral": False},
        "note": "test fixture",
        "pheme_diagnostic": {"diagnostic_only": True, "decides_gate": False},
    }


def _m1f_tree(root):
    _write(P.m1f_path(root, P.M1F_EVIDENCE_PINS_FILENAME), {"ok": True})
    _write(P.m1f_path(root, P.M1F_TASK_VALIDITY_FILENAME),
           _task_validity_payload())
    _write_text(P.m1f_path(root, P.M1F_TASK_VALIDITY_REPORT_FILENAME),
                "# TASK_VALIDITY\n")
    base_dir = P.m1f_path(root, P.M1F_BASELINES_DIRNAME)
    for dataset in P.DATASETS:
        _write(os.path.join(base_dir, f"{dataset}.json"),
               _baseline_payload(dataset))
    _write(P.m1f_path(root, P.M1F_ACTIVE_AUDIT_FILENAME), _active_payload())
    _write(P.m1f_path(root, P.M1F_WITHIN_SNAPSHOT_FILENAME),
           _within_payload())
    _write(P.m1f_path(root, P.M1F_ROBUSTNESS_FILENAME),
           _robustness_payload())
    _write(P.m1f_path(root, P.M1F_PHEME_FILENAME), {"diagnostic_only": True})
    _write(P.m1f_path(root, P.M1F_VERDICT_FILENAME), _verdict_payload())
    _write_text(P.m1f_path(root, P.M1F_REPORT_FILENAME), "# M1F_REPORT\n")


def _tree(tmp_path, mutate=None):
    root = str(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _frozen_tree(root)
    _m1f_tree(root)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "tree")
    if mutate:
        mutate(root)
    return root


def _rewrite(path, mutate_payload):
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    mutate_payload(payload)
    _write(path, payload)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_verifier_m1f_clean_tree(tmp_path):
    out = verifier_m1f.verify(_tree(tmp_path))
    assert out["issues"] == [], out["issues"]
    names = {c["check"] for c in out["checks"]}
    assert "m1f_frozen_inputs_pinned" in names
    assert "m1f_task_validity_boundaries" in names
    assert "m1f_baselines_fixed_and_eval_never_used_for_selection" in names
    assert "m1f_active_signed_audit_complete" in names
    assert "m1f_within_snapshot_discrimination_reported" in names
    assert "m1f_thresholds_fixed_and_diagnostic_only" in names
    assert "m1f_pheme_remains_diagnostic" in names
    assert "m1f_no_m2_artifacts" in names
    # the frozen label caches are server-only and must be pending, not passed
    assert out["pending"] == ["m1f_historical_caches_recomputed"]


def test_verifier_m1f_detects_frozen_input_drift(tmp_path):
    def mutate(root):
        _write(P.m1_path(root, "evaluation", "gate.json"), {"passed": True})
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_frozen_inputs_pinned" in out["issues"]


def test_verifier_m1f_detects_comparator_drift(tmp_path):
    def mutate(root):
        _rewrite(os.path.join(P.m1f_path(root, P.M1F_BASELINES_DIRNAME),
                              "maweibo.json"),
                 lambda p: p["comparator"].update({"selected": P.M1F_S4}))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_baselines_fixed_and_eval_never_used_for_selection" in \
        out["issues"]


def test_verifier_m1f_detects_held_out_reader_in_training(tmp_path):
    def mutate(root):
        def patch(payload):
            payload["variants"][P.M1F_S2]["rotations"]["qwen"][
                "train_readers"] = ["qwen", "mistral"]
        _rewrite(os.path.join(P.m1f_path(root, P.M1F_BASELINES_DIRNAME),
                              "pheme.json"), patch)
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_baselines_fixed_and_eval_never_used_for_selection" in \
        out["issues"]


def test_verifier_m1f_detects_scaler_fitted_on_eval_rows(tmp_path):
    def mutate(root):
        def patch(payload):
            payload["variants"][P.M1F_S3]["rotations"]["mistral"][
                "scaler_rows"] = 50
        _rewrite(os.path.join(P.m1f_path(root, P.M1F_BASELINES_DIRNAME),
                              "maweibo.json"), patch)
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_baselines_fixed_and_eval_never_used_for_selection" in \
        out["issues"]


def test_verifier_m1f_detects_s4_no_longer_reusing_d5(tmp_path):
    def mutate(root):
        def patch(payload):
            payload["variants"][P.M1F_S4]["frozen_variant"] = "D0_Z"
        _rewrite(os.path.join(P.m1f_path(root, P.M1F_BASELINES_DIRNAME),
                              "maweibo.json"), patch)
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_baselines_fixed_and_eval_never_used_for_selection" in \
        out["issues"]


def test_verifier_m1f_detects_utility_called_verification(tmp_path):
    def mutate(root):
        _rewrite(P.m1f_path(root, P.M1F_TASK_VALIDITY_FILENAME),
                 lambda p: p["utility_meaning"].update(
                     {"is_verification_utility": True}))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_task_validity_boundaries" in out["issues"]


def test_verifier_m1f_detects_threshold_drift(tmp_path):
    def mutate(root):
        _rewrite(P.m1f_path(root, P.M1F_ROBUSTNESS_FILENAME),
                 lambda p: p.update({"official_threshold": 0.03}))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_thresholds_fixed_and_diagnostic_only" in out["issues"]


def test_verifier_m1f_detects_threshold_selection(tmp_path):
    def mutate(root):
        _rewrite(P.m1f_path(root, P.M1F_ROBUSTNESS_FILENAME),
                 lambda p: p.update({"threshold_selected": True}))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_thresholds_fixed_and_diagnostic_only" in out["issues"]


def test_verifier_m1f_detects_missing_within_comparison(tmp_path):
    def mutate(root):
        _rewrite(P.m1f_path(root, P.M1F_WITHIN_SNAPSHOT_FILENAME),
                 lambda p: p["comparisons_vs_S2"].pop(P.M1F_S4))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_within_snapshot_discrimination_reported" in out["issues"]


def test_verifier_m1f_detects_undefined_active_audit(tmp_path):
    def mutate(root):
        _rewrite(P.m1f_path(root, P.M1F_ACTIVE_AUDIT_FILENAME),
                 lambda p: p["variants"][P.M1F_S4]["active"].update(
                     {"defined": False}))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_active_signed_audit_complete" in out["issues"]


def test_verifier_m1f_detects_pheme_deciding(tmp_path):
    def mutate(root):
        _rewrite(P.m1f_path(root, P.M1F_VERDICT_FILENAME),
                 lambda p: p.update({"pheme_diagnostic": {
                     "diagnostic_only": False, "decides_gate": True}}))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_pheme_remains_diagnostic" in out["issues"]


def test_verifier_m1f_detects_an_invalid_verdict(tmp_path):
    def mutate(root):
        _rewrite(P.m1f_path(root, P.M1F_VERDICT_FILENAME),
                 lambda p: p.update({"verdict": "M1F_LOOKS_GOOD"}))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_verdict_is_valid_outcome" in out["issues"]


def test_verifier_m1f_detects_m2_artifacts(tmp_path):
    def mutate(root):
        os.makedirs(os.path.join(root, P.RESULTS_ROOT, "m2_pilot"))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_no_m2_artifacts" in out["issues"]


def test_verifier_m1f_detects_reader_inference_artifacts(tmp_path):
    def mutate(root):
        os.makedirs(P.m1f_path(root, "probe_responses"))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_no_reader_inference_or_label_artifacts" in out["issues"]


def test_verifier_m1f_detects_a_missing_artifact(tmp_path):
    def mutate(root):
        os.remove(P.m1f_path(root, P.M1F_ACTIVE_AUDIT_FILENAME))
    out = verifier_m1f.verify(_tree(tmp_path, mutate))
    assert "m1f_required_artifacts_present" in out["issues"]
