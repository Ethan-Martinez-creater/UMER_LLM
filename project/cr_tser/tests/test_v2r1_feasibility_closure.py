"""V2R1 feasibility closure after the P2 failure.

The closure is a document about frozen evidence, so the tests check that it
cannot disagree with that evidence:

* on the frozen artifacts it produces exactly P1 PASS / P2 FAIL / P3 NOT RUN /
  P4 NOT RUN / NO_GO with the expected reason code;
* it refuses when a verdict, a threshold, a reader set, the dataset role or the
  frozen input digests do not match;
* it refuses as soon as a P3/P4 artifact exists;
* the frozen numbers in the closure are the ones in the gate artifacts.

The frozen gate artifacts are read from the repository; mutations are applied
to temporary copies, so the real artifacts are never written to.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "scripts", REPO / "project"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GATES = ("p1_maweibo.json", "p2_maweibo.json", "p1_pheme_diagnostic.json",
         "p2_pheme_diagnostic.json")


def _copy_gates(tmp_path):
    root = tmp_path / "v2r1"
    (root / "gates").mkdir(parents=True)
    for name in GATES:
        shutil.copy(REPO / "results" / "cr_tser_v2r1" / "gates" / name,
                    root / "gates" / name)
    return root


def _mutate(root, name, mutate):
    path = root / "gates" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def test_closure_reports_the_frozen_verdicts(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    payload = closure.build_closure(str(root), "deadbeef")

    assert payload["P1"]["verdict"] == "P1_PASS"
    assert payload["P2"]["verdict"] == "P2_FAIL"
    assert payload["P3"] == "NOT_RUN"
    assert payload["P4"] == "NOT_RUN"
    assert payload["final_feasibility_verdict"] == "NO_GO"
    assert payload["reason_code"] == "STRUCTURED_INTERACTION_GATE_NOT_SUPPORTED"
    assert payload["protocol"] == "v2r1"
    assert payload["reader_keys"] == ["qwen", "mistral", "internlm"]
    assert payload["dataset_roles"]["primary_decision_dataset"] == "maweibo"
    assert payload["dataset_roles"]["pheme_decides_any_gate"] is False
    assert payload["baseline_commit"] == "deadbeef"
    assert payload["p3_p4_artifacts_absent"] == {"predictor": True,
                                                "unseen_reader": True}


def test_closure_numbers_are_the_gate_artifact_numbers(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    payload = closure.build_closure(str(root), "deadbeef")
    p1_gate = json.loads((root / "gates" / "p1_maweibo.json").read_text(
        encoding="utf-8"))
    p2_gate = json.loads((root / "gates" / "p2_maweibo.json").read_text(
        encoding="utf-8"))

    assert payload["P1"]["mean_disagreement"] == \
        p1_gate["report"]["macro_mean_disagreement"]
    assert payload["P2"]["delta_edge"] == p2_gate["report"]["edge"]["delta"]
    assert payload["P2"]["edge_ci_low"] == p2_gate["report"]["edge"]["ci_low"]
    assert payload["P2"]["edge_ci_high"] == p2_gate["report"]["edge"]["ci_high"]
    assert payload["P2"]["delta_subtree"] == \
        p2_gate["report"]["subtree"]["delta"]
    assert payload["P2"]["edge_n_events"] == \
        p2_gate["report"]["edge"]["n_events"]
    # the reported failure is exactly the threshold/CI failure
    assert payload["P2"]["delta_edge"] < payload["frozen_thresholds"][
        "p2_edge_delta_min"]
    assert payload["P2"]["edge_ci_low"] <= 0.0


def test_closure_is_a_no_go_without_claiming_no_effect(tmp_path):
    """The wording must not overstate the finding."""
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    payload = closure.build_closure(str(root), "deadbeef")
    reason = payload["reason"]
    assert "did not establish" in reason
    assert "positive" in reason
    assert "below the pre-registered threshold" in reason
    assert "crossed zero" in reason
    for overstatement in ("no effect", "no structural effect",
                          "structure does not matter"):
        assert overstatement not in reason.lower()

    md = closure._write_markdown(payload, str(tmp_path / "closure.md"))
    text = Path(md).read_text(encoding="utf-8")
    for expected in ("P1 = PASS", "P2 = FAIL", "P3 = NOT RUN", "P4 = NOT RUN",
                     "CR_TSER_V2R1_FEASIBILITY = NO_GO",
                     "reason = STRUCTURED_INTERACTION_GATE_NOT_SUPPORTED"):
        assert expected in text


# --------------------------------------------------------------------------
# fail-closed behaviour
# --------------------------------------------------------------------------
def test_wrong_p1_verdict_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p1_maweibo.json", lambda p: p.update(verdict="P1_FAIL"))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_wrong_p2_verdict_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p2_maweibo.json", lambda p: p.update(verdict="P2_PASS"))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_changed_threshold_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p2_maweibo.json",
            lambda p: p["gate"].update(edge_threshold=0.005))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_changed_reader_set_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p1_maweibo.json",
            lambda p: p.update(readers_present=["qwen", "internlm"]))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_retired_reader_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p1_maweibo.json",
            lambda p: p.update(retired_readers_present=["glm"]))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_pheme_deciding_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p1_pheme_diagnostic.json",
            lambda p: p.update(diagnostic_only=False,
                               decides_primary_gate=True))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_pheme_dataset_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p2_pheme_diagnostic.json",
            lambda p: p.update(dataset="maweibo"))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_drifted_cache_digest_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p1_maweibo.json",
            lambda p: p["frozen_inputs"]["labels_sha256"].update(
                maweibo="0" * 64))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_drifted_manifest_digest_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p2_maweibo.json",
            lambda p: p["frozen_inputs"]["manifest_sha256"].update(
                {"maweibo/event_split.json": "0" * 64}))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_missing_gate_artifact_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    (root / "gates" / "p2_maweibo.json").unlink()
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_p3_artifact_presence_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    (root / "gates" / "p3_maweibo.json").write_text("{}", encoding="utf-8")
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_predictor_namespace_presence_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    (root / "predictor" / "maweibo").mkdir(parents=True)
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_split_and_cutoff_drift_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p1_maweibo.json",
            lambda p: p["coverage"].update(cutoffs=[15, 60]))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")


def test_bootstrap_protocol_drift_is_refused(tmp_path):
    import cr_tser_feasibility_closure as closure

    root = _copy_gates(tmp_path)
    _mutate(root, "p2_maweibo.json",
            lambda p: p["report"].update(bootstrap_unit="snapshot"))
    with pytest.raises(closure.ClosureRefused):
        closure.build_closure(str(root), "deadbeef")
