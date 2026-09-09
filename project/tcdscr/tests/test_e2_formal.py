"""Formal E2 unit tests (plan §38 ordering; E2 §27–§28).

Pins the E2 invariants that cannot be left to runtime luck:
- the encoder source is always the Random-init E1 checkpoint (UMER-init is
  refused before any training);
- freezing is real (parameter checksum stable, requires_grad False);
- the E1/E2 split audit flags mismatches (identical builder + counts +
  independent E1 test-ids);
- the static selector carries no dynamic memory / novelty / persistence;
- the readiness delta is 0.005 (0.5 percentage point), not 0.5 absolute;
- the training pass never evaluates the test split (that only happens in
  the explicit E2-B pass gated by readiness).
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tcdscr_run_e2 as e2  # noqa: E402

from ..models.selector import StaticUtilitySelector  # noqa: E402


def test_e2_uses_random_e1_encoder_only(tmp_path):
    dataset, fold, seed = "pheme", 0, 2000
    run_dir = tmp_path / "e1" / dataset / f"fold{fold}_random_seed{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"init": "umer", "split": {"train": 1, "validation": 1,
                                              "test": 1}}),
        encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Random-init"):
        e2.load_random_e1_encoder(dataset, fold, seed,
                                  str(tmp_path / "e1"), "cpu")


def test_e2_encoder_frozen_checksum():
    import torch
    from ..models.causal_social_encoder import CausalSocialEncoder
    torch.manual_seed(0)
    encoder = CausalSocialEncoder()
    before = e2.state_sha256(encoder.state_dict())
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    after = e2.state_sha256(encoder.state_dict())
    assert after == before  # freezing must not touch parameter values
    assert all(not p.requires_grad for p in encoder.parameters())
    # checksum is content-sensitive: perturbing one parameter changes it
    with torch.no_grad():
        next(encoder.parameters()).add_(1.0)
    assert e2.state_sha256(encoder.state_dict()) != after


def test_e2_split_matches_e1(tmp_path):
    registry = {f"e{i:02d}": i % 2 for i in range(15)}
    test_ids = ["e00", "e01", "e02"]
    run_dir = tmp_path / "e1" / "pheme" / "fold0_random_seed2000"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "init": "random",
        "split": {"train": 10, "validation": 2, "test": 3}}),
        encoding="utf-8")
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for eid in test_ids:
            for cutoff in ("5", "360"):
                fh.write(json.dumps({"event_id": eid, "cutoff": cutoff})
                         + "\n")
    split = {"train": [f"e{i:02d}" for i in range(3, 13)],
             "validation": ["e13", "e14"],
             "test": test_ids}
    events = {"train": [{"event_id": e} for e in split["train"]],
              "validation": [{"event_id": e} for e in split["validation"]],
              "test": [{"event_id": e} for e in split["test"]]}
    parity = e2.check_split_parity("pheme", 0, 2000, split, events,
                                   str(tmp_path / "e1"), registry)
    assert parity["train_exact_match"]
    assert parity["validation_exact_match"]
    assert parity["test_exact_match"]
    # introduce a mismatched test id: the audit must detect it
    split_bad = dict(split)
    split_bad["test"] = ["e00", "e01", "e00"]
    parity_bad = e2.check_split_parity("pheme", 0, 2000, split_bad, events,
                                       str(tmp_path / "e1"), registry)
    assert not parity_bad["test_exact_match"]
    assert not parity_bad["train_exact_match"]


def test_static_selector_has_no_dynamic_memory():
    sel = StaticUtilitySelector()
    assert not hasattr(sel, "memory")
    sig = inspect.signature(sel.forward)
    for forbidden in ("novelty", "persistence", "memory", "dynamic"):
        assert forbidden not in sig.parameters
    assert all("memory" not in k for k in sel.state_dict())


def test_readiness_delta_is_0_005_not_0_5():
    # 0.805 vs 0.800 = +0.005 = the exact gate; a misread "+0.5" threshold
    # would reject this, so this test pins the percentage-point meaning.
    assert e2.readiness_pass(0.805, 0.800)
    assert not e2.readiness_pass(0.800, 0.800)
    assert not e2.readiness_pass(0.800, 0.805)
    assert e2.readiness_pass(0.85, 0.80)
    # far below the gate stays below
    assert not e2.readiness_pass(0.8001, 0.8000)


def test_test_not_evaluated_before_readiness():
    # the training pass writes validation artifacts only; test scoring lives
    # exclusively in the explicit E2-B pass, which marks the caller gate.
    src_train = inspect.getsource(e2.run_one)
    assert "test_metrics" not in src_train
    assert "test_predictions" not in src_train
    src_test = inspect.getsource(e2.evaluate_test_only)
    assert "test_metrics" in src_test
    assert "readiness_gate_passed_by_caller" in src_test