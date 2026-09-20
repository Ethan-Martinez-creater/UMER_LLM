"""M1-E concentration / B3-contract / evidence-pin tests."""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from ..attribution import b3_contract, concentration, evidence_pins
from ..config import protocol as P
from . import m1_synth as S


def _write(path, payload, jsonl=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if jsonl:
            for row in payload:
                fh.write(json.dumps(row) + "\n")
        else:
            json.dump(payload, fh, indent=1)


def _prediction_rows(dataset, model, primary: bool):
    rows = []
    for entry in S.synth_atomic_entries(dataset):
        held = "qwen"
        good = ["HELPFUL", "NEUTRAL", "HARMFUL"].index(entry["sign"]["qwen"])
        probs = [0.1, 0.1, 0.1]
        probs[good] = 0.8
        rows.append({
            "dataset": dataset, "held_out": held, "model": model,
            "key": entry["key"], "event_id": entry["event_id"],
            "reader": "qwen", "gold_sign": entry["sign"]["qwen"],
            "gold_utility": entry["utility"]["qwen"],
            "pred_utility": entry["utility"]["qwen"] if primary else 0.0,
            "pred_sign": entry["sign"]["qwen"] if primary else "NEUTRAL",
            "prob_helpful": probs[0], "prob_neutral": probs[1],
            "prob_harmful": probs[2],
        })
    return rows


def test_load_paired_predictions_and_concentration(tmp_path):
    root = str(tmp_path)
    rows_all = _prediction_rows("maweibo", P.MODEL_B0, False) \
        + _prediction_rows("maweibo", P.MODEL_B5, True)
    _write(P.m1_path(root, "evaluation", "predictions_light_touch.jsonl"),
           rows_all, jsonl=True)
    rows = concentration.load_paired_predictions(root, "maweibo",
                                                P.MODEL_B5, P.MODEL_B0)
    assert rows
    assert all(r["held_out"] == "qwen" for r in rows)
    report = concentration.concentration_report(rows, P.MODEL_B5, P.MODEL_B0)
    assert report["scope"] == "post_hoc_diagnostic_only"
    assert report["uses_existing_predictions_only"] is True
    assert report["reader_deltas"]["qwen"] > 0        # primary is perfect
    assert sorted(report["cutoff_deltas"], key=int) == \
        [str(c) for c in P.CUTOFFS_MIN]
    assert sorted(report["class_deltas"]) == sorted(P.SIGN_CLASSES)
    assert report["dominance"]["reader"]["largest"] == "qwen"


def test_b3_contract_detects_same_key_transfer(tmp_path):
    root = str(tmp_path)
    entries = S.synth_atomic_entries("maweibo")
    _write(os.path.join(P.bootstrap_dir(root), P.ATOMIC_INDEX_FILENAME),
           {"datasets": {"maweibo": {"entries": entries}}})
    rows = []
    for rotation in P.LORO_ROTATIONS:
        held = rotation[2]
        nearest = rotation[0]
        for entry in entries[:20]:
            rows.append({
                "dataset": "maweibo", "held_out": held, "model": P.MODEL_B3,
                "key": entry["key"], "event_id": entry["event_id"],
                "reader": held,
                "gold_sign": entry["sign"][held],
                "gold_utility": entry["utility"][held],
                "pred_utility": entry["utility"][nearest],
                "pred_sign": entry["sign"][nearest],
                "prob_helpful": 0.1, "prob_neutral": 0.1, "prob_harmful": 0.1,
            })
    _write(P.m1_path(root, "evaluation", "predictions.jsonl"), rows, jsonl=True)
    _write(P.m1_path(root, "evaluation", "evaluation.json"),
           {"datasets": {"maweibo": {"rotations": {
               rot[2]: {"models": {P.MODEL_B3: {
                   "nearest_reader": rot[0],
                   "distances": {rot[0]: 1.0, rot[1]: 2.0}}}}
               for rot in P.LORO_ROTATIONS}}}})
    audit = b3_contract.audit(root, "maweibo")
    assert audit["conclusion"]["is_same_evidence_cross_reader_transfer"] is True
    assert audit["conclusion"][
        "prediction_is_the_nearest_training_reader_label"] is True
    assert audit["conclusion"][
        "requires_utility_label_on_at_least_one_existing_reader"] is True
    assert audit["conclusion"][
        "held_out_utility_labels_used_for_selection"] is False
    assert "same-evidence cross-reader transfer baseline" in \
        audit["conclusion"]["statement"]
    for held, cell in audit["per_rotation"].items():
        assert cell["match_rate"] == 1.0
        assert cell["nearest_reader_is_a_training_reader"] is True


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def test_evidence_pins_accepts_matching_tree(tmp_path):
    root = str(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _write(P.m1_path(root, P.M1_VERDICT_FILENAME),
           {"final_outcome": P.M1E_EXPECTED_M1_VERDICT,
            "zero_touch_passed": False, "light_touch_passed": True})
    _write(P.m1_path(root, "evaluation", "gate.json"), {"passed": True})
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "m1")
    pin = evidence_pins.pin_m1_evidence(root)
    assert pin["ok"] is True
    assert pin["m1_verdict"] == P.M1E_EXPECTED_M1_VERDICT
    assert pin["n_artifacts"] == 2
    # mutating a pinned artifact must be refused
    _write(P.m1_path(root, "evaluation", "gate.json"), {"passed": False})
    with pytest.raises(evidence_pins.EvidencePinRefused, match="blob"):
        evidence_pins.pin_m1_evidence(root)


def test_evidence_pins_refuses_wrong_verdict(tmp_path):
    root = str(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _write(P.m1_path(root, P.M1_VERDICT_FILENAME),
           {"final_outcome": "M1_NO_GO", "zero_touch_passed": False,
            "light_touch_passed": False})
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "m1")
    with pytest.raises(evidence_pins.EvidencePinRefused, match="verdict"):
        evidence_pins.pin_m1_evidence(root)
