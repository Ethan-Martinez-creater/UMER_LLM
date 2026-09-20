"""M1-E verifier tests over a synthetic M1+M1-E tree."""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from .. import verifier_m1e
from ..config import protocol as P


def _write(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def _rotation(rotation, variant, dim):
    return {
        "held_out": rotation[2],
        "train_readers": [rotation[0], rotation[1]],
        "selected_config": {"hidden": 32, "dropout": 0.0, "lr": 1e-3,
                            "weight_decay": 0.0},
        "dev_macro_f1": 0.3, "n_parameters": 12000,
        "fold_sizes": {"train": 100, "dev": 30, "eval": 50},
        "scaler_rows": 100, "evidence_dim": dim,
        "eval_metrics": {"macro_f1": 0.3},
    }


def _variant_block(variant, dim):
    rotations = {r[2]: _rotation(r, variant, dim)
                 for r in P.LORO_ROTATIONS}
    f1 = {h: 0.3 for h in rotations}
    return {"dataset": "maweibo", "variant": variant,
            "uses_fingerprint": variant in P.ATTRIBUTION_FINGERPRINT_VARIANTS,
            "evidence_dim": dim, "rotations": rotations,
            "per_reader_macro_f1": f1, "mean_macro_f1": 0.3,
            "worst_reader_macro_f1": 0.3}


def _attribution_payload(dataset):
    dims = {"D0_Z": 26, "D2_Z_F_S": 29, "D3_Z_F_C": 29, "D5_Z_S_C": 32}
    variants = {v: _variant_block(v, d) for v, d in dims.items()}
    reused = {}
    for variant, model_kind in P.ATTRIBUTION_REUSED.items():
        block = _variant_block(variant, 26 if variant == "D1_Z_F" else 32)
        block.update({"frozen_model": model_kind, "reused": True})
        reused[variant] = block
    comparisons = {v: {"baseline": "D0_Z",
                       "aggregate": {"mean_delta_macro_f1": 0.0,
                                     "ci_low": -0.01, "ci_high": 0.01,
                                     "iterations": 10000, "seed": 7319,
                                     "positive_readers": 1,
                                     "worst_reader_delta": 0.0}}
                   for v in P.ATTRIBUTION_VARIANTS if v != "D0_Z"}
    return {
        "protocol": P.PROTOCOL_VERSION, "stage": "m1e_attribution",
        "dataset": dataset, "scope": "post_hoc_diagnostic_only",
        "defines_gate": False, "variants": variants,
        "reused_variants": reused, "comparisons_vs_D0": comparisons,
        "fingerprint_increment_D4_vs_D5": {
            "variant": "D4_Z_F_S_C", "baseline": "D5_Z_S_C",
            "aggregate": {"mean_delta_macro_f1": 0.0}},
        "secondary_metrics": {},
    }


def _tree(tmp_path, mutate=None):
    root = str(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _write(P.m1_path(root, P.M1_VERDICT_FILENAME),
           {"final_outcome": P.M1E_EXPECTED_M1_VERDICT,
            "zero_touch_passed": False, "light_touch_passed": True})
    for dataset in P.DATASETS:
        _write(P.m1e_path(root, P.M1E_ATTRIBUTION_DIRNAME,
                          f"{dataset}.json"), _attribution_payload(dataset))
    _write(P.m1e_path(root, P.M1E_VERDICT_FILENAME),
           {"m1_verdict_unchanged": True, "defines_new_gate": False,
            "pheme_diagnostic": {"diagnostic_only": True,
                                 "decides_gate": False}})
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "tree")
    if mutate:
        # mutations happen *after* the commit: the pinned M1 tree must then
        # show up as drift, and the m1e artifacts are read from the worktree
        mutate(root)
    return root


def test_verifier_m1e_clean_tree(tmp_path):
    out = verifier_m1e.verify(_tree(tmp_path))
    assert out["issues"] == [], out["issues"]
    names = {c["check"] for c in out["checks"]}
    assert "m1e_m1_evidence_unchanged" in names
    assert "m1e_m1_verdict_unchanged" in names
    assert "m1e_variants_fixed_and_eval_never_trained_on" in names
    assert "m1e_pheme_remains_diagnostic" in names


def test_verifier_m1e_detects_variant_drift(tmp_path):
    def mutate(root):
        path = P.m1e_path(root, P.M1E_ATTRIBUTION_DIRNAME, "maweibo.json")
        data = json.load(open(path, encoding="utf-8"))
        data["variants"].pop("D3_Z_F_C")
        _write(path, data)
    out = verifier_m1e.verify(_tree(tmp_path, mutate))
    assert "m1e_variants_fixed_and_eval_never_trained_on" in out["issues"]


def test_verifier_m1e_detects_held_out_reader_in_training(tmp_path):
    def mutate(root):
        path = P.m1e_path(root, P.M1E_ATTRIBUTION_DIRNAME, "pheme.json")
        data = json.load(open(path, encoding="utf-8"))
        rot = data["variants"]["D0_Z"]["rotations"]["qwen"]
        rot["train_readers"] = ["qwen", "mistral"]
        _write(path, data)
    out = verifier_m1e.verify(_tree(tmp_path, mutate))
    assert "m1e_variants_fixed_and_eval_never_trained_on" in out["issues"]


def test_verifier_m1e_detects_scaler_trained_on_eval_rows(tmp_path):
    def mutate(root):
        path = P.m1e_path(root, P.M1E_ATTRIBUTION_DIRNAME, "maweibo.json")
        data = json.load(open(path, encoding="utf-8"))
        rot = data["variants"]["D5_Z_S_C"]["rotations"]["mistral"]
        rot["scaler_rows"] = 50          # equals the eval fold, not train
        _write(path, data)
    out = verifier_m1e.verify(_tree(tmp_path, mutate))
    assert "m1e_variants_fixed_and_eval_never_trained_on" in out["issues"]


def test_verifier_m1e_detects_m1_mutation(tmp_path):
    def mutate(root):
        _write(P.m1_path(root, "evaluation", "gate.json"), {"passed": True})
    out = verifier_m1e.verify(_tree(tmp_path, mutate))
    # a new tracked M1 file is a tracked change under the M1 root
    assert "m1e_m1_evidence_unchanged" in out["issues"]


def test_verifier_m1e_detects_m2_artifacts(tmp_path):
    def mutate(root):
        os.makedirs(os.path.join(root, P.RESULTS_ROOT, "m2_pilot"))
    out = verifier_m1e.verify(_tree(tmp_path, mutate))
    assert "m1e_no_m2_artifacts" in out["issues"]


def test_verifier_m1e_detects_pheme_deciding(tmp_path):
    def mutate(root):
        _write(P.m1e_path(root, P.M1E_VERDICT_FILENAME),
               {"m1_verdict_unchanged": True, "defines_new_gate": False,
                "pheme_diagnostic": {"diagnostic_only": False,
                                     "decides_gate": True}})
    out = verifier_m1e.verify(_tree(tmp_path, mutate))
    assert "m1e_pheme_remains_diagnostic" in out["issues"]


def test_verifier_m1e_detects_new_gate(tmp_path):
    def mutate(root):
        _write(P.m1e_path(root, P.M1E_VERDICT_FILENAME),
               {"m1_verdict_unchanged": True, "defines_new_gate": True,
                "pheme_diagnostic": {"diagnostic_only": True,
                                     "decides_gate": False}})
    out = verifier_m1e.verify(_tree(tmp_path, mutate))
    assert "m1e_no_new_gate_defined" in out["issues"]
