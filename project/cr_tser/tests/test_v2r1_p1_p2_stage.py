"""V2R1 P1/P2 stage — thin entrypoint, fail-closed inputs, frozen statistics.

The stage script is only allowed to load frozen artifacts, call the frozen
evaluation functions and serialise results, so these tests check exactly that:

* a drifted manifest or a missing/mismatched label cache is refused;
* a reader set that is not exactly qwen/mistral/internlm is refused;
* PHEME is diagnostic-only and never decides a primary gate;
* the artifacts carry the values the frozen functions produce (verified against
  a hand-computed synthetic case);
* the script does not reimplement a gate or a statistic.

No model, no GPU and no real label cache is needed: the synthetic root below is
built in a temporary directory.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "scripts", REPO / "project"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

READERS = ("qwen", "mistral", "internlm")
EVENTS = ("e1", "e2", "e3", "e4")

#: Per-reader synthetic utilities. The two groups disagree on sign (so P1 has a
#: disagreement) while their ``|I2| - |I3|`` and ``|I4| - |I5|`` gaps are the
#: same positive constant (so P2 has a clean, exactly reproducible delta).
SYNTH = {
    "qwen": {"I1": 0.5, "I2": 1.0, "I3": 0.55, "I4": 1.2, "I5": 0.6},
    "internlm": {"I1": 0.5, "I2": 1.0, "I3": 0.55, "I4": 1.2, "I5": 0.6},
    "mistral": {"I1": -0.5, "I2": 0.0, "I3": -0.45, "I4": -1.2, "I5": -0.6},
}
EXPECTED_EDGE_DELTA = 0.45   # |1.0 - 0.5| - |0.55 - 0.5| = 0.5 - 0.05
EXPECTED_SUBTREE_DELTA = 0.6  # |1.2 - 0.5| - |0.6  - 0.5| = 0.7 - 0.1


def _row(dataset, event, reader, intervention_type, utility, affected):
    return {
        "dataset": dataset, "event_id": event, "cutoff": 15, "reader": reader,
        "intervention_id": intervention_type, "intervention_type":
            intervention_type,
        "utility": utility, "affected_reply_ids": list(affected),
        "correctness_before": True, "correctness_after": True,
        "gold": 1, "score_A": 0.0, "score_B": -1.0, "p_rumor": 0.7,
        "p_nonrumor": 0.3,
    }


def _write_root(tmp_path, dataset="maweibo", readers=READERS,
                events=EVENTS, label_override=None):
    root = tmp_path / dataset
    (root / "manifests" / dataset).mkdir(parents=True)
    (root / "utility_labels" / dataset).mkdir(parents=True)
    split = {
        "dataset": dataset, "split_seed": 7319,
        "sizes": {"foundation_train": 80, "utility_train": 50,
                  "utility_dev": 15, "utility_eval": 25},
        "utility_train": list(events), "utility_dev": [], "utility_eval": [],
        "foundation_train": [], "unused": [],
        "source": {"kind": "synthetic", "path": str(root),
                   "sha256": "0" * 64, "n_files": 1, "exists": True},
    }
    (root / "manifests" / dataset / "event_split.json").write_text(
        json.dumps(split), encoding="utf-8")

    lines = []
    for event in events:
        for reader in readers:
            # an unexpected reader still needs rows, so the refusal under test
            # is the reader set itself and not a missing fixture
            values = SYNTH.get(reader) or SYNTH["qwen"]
            node = f"{event}_n1"
            lines.append(_row(dataset, event, reader, "I1_atomic",
                              values["I1"], [node]))
            for key, itype in (("I2", "I2_parent_child"),
                               ("I3", "I3_matched_nonadjacent"),
                               ("I4", "I4_subtree"),
                               ("I5", "I5_matched_disconnected")):
                lines.append(_row(dataset, event, reader, itype,
                                  values[key], [node]))
    if label_override is not None:
        lines = label_override(lines)
    (root / "utility_labels" / dataset / "labels.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return root


def _frozen_for(root):
    """The frozen-input dict shape the stage expects (hashes are not checked)."""
    return {"manifest_sha256": {}, "labels_sha256": {}, "labels_rows": {}}


def _load_rows(root, dataset="maweibo"):
    import cr_tser_run_pilot as pilot
    return pilot._read_jsonl(pilot.labels_path(str(root), dataset))


# --------------------------------------------------------------------------
# 1. the frozen statistics are produced, not reimplemented
# --------------------------------------------------------------------------
def test_p1_report_and_gate_come_from_the_frozen_functions(tmp_path):
    import cr_tser_eval_p1_p2 as stage
    from cr_tser.evaluation.heterogeneity import (gate_p1,
                                                  heterogeneity_report)
    import cr_tser_run_pilot as pilot

    root = _write_root(tmp_path)
    rows = _load_rows(root)
    payload = stage.evaluate_p1(str(root), "maweibo", rows, _frozen_for(root))

    expected_report = heterogeneity_report(
        pilot.load_unit_table(str(root), "maweibo"), READERS)
    assert payload["report"]["macro_mean_disagreement"] == \
        expected_report["macro_mean_disagreement"]
    assert payload["gate"] == gate_p1(expected_report)

    # two of three pairs disagree on every key, one never does
    by_pair = {(p["reader_a"], p["reader_b"]): p["disagreement"]
               for p in payload["report"]["pairs"]}
    assert by_pair[("qwen", "internlm")] == 0.0
    assert by_pair[("mistral", "internlm")] == 1.0
    assert by_pair[("qwen", "mistral")] == 1.0
    assert payload["report"]["macro_mean_disagreement"] == pytest.approx(2 / 3)
    assert payload["verdict"] == "P1_PASS"
    assert payload["decides_primary_gate"] is True
    assert payload["diagnostic_only"] is False
    assert payload["reader_set_exact"] is True
    assert payload["retired_readers_present"] == []
    # every atomic key is active for every reader at |u| = 0.5
    assert payload["active_counts"]["qwen"]["active"] == len(EVENTS)
    assert payload["active_counts"]["utility_threshold"] == 0.05


def test_p2_report_and_gate_come_from_the_frozen_functions(tmp_path):
    import cr_tser_eval_p1_p2 as stage
    from cr_tser.evaluation.structural_interaction import (
        gate_p2, structural_interaction_report)
    import cr_tser_run_pilot as pilot

    root = _write_root(tmp_path)
    rows = _load_rows(root)
    payload = stage.evaluate_p2(str(root), "maweibo", rows, _frozen_for(root))

    expected = structural_interaction_report(
        pilot.load_interaction_records(str(root), "maweibo"))
    assert payload["gate"] == gate_p2(expected)
    assert payload["report"]["edge"]["delta"] == pytest.approx(
        EXPECTED_EDGE_DELTA)
    assert payload["report"]["subtree"]["delta"] == pytest.approx(
        EXPECTED_SUBTREE_DELTA)
    assert payload["report"]["edge"]["ci_low"] > 0
    assert payload["verdict"] == "P2_PASS"
    assert payload["record_counts"]["structured_records"] == \
        len(EVENTS) * len(READERS) * 4
    assert payload["slot_means"]["pc"]["mean_abs_interaction"] == \
        pytest.approx(0.5)   # qwen/internlm |1.0-0.5|, mistral |0.0+0.5|
    assert payload["slot_means"]["na"]["mean_abs_interaction"] == \
        pytest.approx(0.05)
    assert payload["frozen_statistics"]["bootstrap_iterations"] == 10000
    assert payload["frozen_statistics"]["bootstrap_seed"] == 7319
    assert payload["frozen_statistics"]["matching_unit"] == \
        "reader_x_snapshot"
    assert payload["frozen_statistics"]["bootstrap_unit"] == "event"


# --------------------------------------------------------------------------
# 2. PHEME is descriptive only
# --------------------------------------------------------------------------
def test_pheme_is_diagnostic_only_and_never_decides_a_gate(tmp_path):
    import cr_tser_eval_p1_p2 as stage

    root = _write_root(tmp_path, dataset="pheme")
    rows = _load_rows(root, "pheme")
    for payload in (stage.evaluate_p1(str(root), "pheme", rows,
                                      _frozen_for(root)),
                    stage.evaluate_p2(str(root), "pheme", rows,
                                      _frozen_for(root))):
        assert payload["diagnostic_only"] is True
        assert payload["decides_primary_gate"] is False
        assert payload["role"] == "diagnostic_only"
        assert payload["verdict"] == "DIAGNOSTIC_ONLY"
        assert "diagnostic_gate" in payload
        assert "verdict" not in payload["diagnostic_gate"]
    # and the primary dataset does the opposite
    maw = _write_root(tmp_path / "m", dataset="maweibo")
    primary = stage.evaluate_p1(str(maw), "maweibo", _load_rows(maw),
                                _frozen_for(maw))
    assert primary["decides_primary_gate"] is True
    assert primary["verdict"].startswith("P1_")


# --------------------------------------------------------------------------
# 3. fail-closed inputs
# --------------------------------------------------------------------------
def test_reader_set_mismatch_is_refused(tmp_path):
    import cr_tser_eval_p1_p2 as stage

    for p1 in (True, False):
        root = _write_root(tmp_path / ("p1" if p1 else "p2"),
                           readers=("qwen", "internlm"))
        rows = _load_rows(root)
        fn = stage.evaluate_p1 if p1 else stage.evaluate_p2
        with pytest.raises(stage.StageRefused):
            fn(str(root), "maweibo", rows, _frozen_for(root))


def test_retired_reader_in_the_cache_is_refused(tmp_path):
    import cr_tser_eval_p1_p2 as stage

    root = _write_root(tmp_path, readers=READERS + ("glm",))
    rows = _load_rows(root)
    with pytest.raises(stage.StageRefused):
        stage.evaluate_p1(str(root), "maweibo", rows, _frozen_for(root))


def test_drifted_manifest_is_refused(tmp_path):
    import cr_tser_eval_p1_p2 as stage

    root = _write_root(tmp_path)
    (root / "manifests" / "maweibo" / "source.json").write_text(
        json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(stage.StageRefused):
        stage._assert_frozen_inputs(str(root))


def test_missing_label_cache_is_refused(tmp_path):
    import cr_tser_eval_p1_p2 as stage

    root = _write_root(tmp_path)
    (root / "utility_labels" / "maweibo" / "labels.jsonl").unlink()
    with pytest.raises(stage.StageRefused):
        stage._assert_frozen_inputs(str(root))


def test_wrong_label_cache_digest_is_refused(tmp_path):
    import cr_tser_eval_p1_p2 as stage

    root = _write_root(tmp_path)
    with pytest.raises(stage.StageRefused) as exc:
        stage._assert_frozen_inputs(str(root))
    assert "labels" in str(exc.value)


def test_the_real_repository_manifests_are_the_pinned_ones():
    """The pins in the stage match the pins in the verifier."""
    import cr_tser_eval_p1_p2 as stage
    import cr_tser_verify_pilot as verifier

    verifier_pins = {key.split("manifests/", 1)[1]: value for key, value
                     in verifier.V2R1_FROZEN_MANIFEST_SHA256.items()}
    assert stage.FROZEN_MANIFEST_SHA256 == verifier_pins
    for dataset in ("maweibo", "pheme"):
        path = REPO / "results" / "cr_tser_v2r1" / "manifests" / dataset / \
            "event_split.json"
        assert stage._canonical_sha256(str(path)) == \
            stage.FROZEN_MANIFEST_SHA256[f"{dataset}/event_split.json"]


# --------------------------------------------------------------------------
# 4. the stage must not reimplement the statistics
# --------------------------------------------------------------------------
def test_stage_reuses_the_frozen_evaluation_functions():
    source = (REPO / "scripts" / "cr_tser_eval_p1_p2.py").read_text(
        encoding="utf-8")
    for required in ("from cr_tser.evaluation.heterogeneity import",
                     "gate_p1", "heterogeneity_report",
                     "from cr_tser.evaluation.structural_interaction import",
                     "gate_p2", "structural_interaction_report",
                     "pilot.load_unit_table", "pilot.load_interaction_records"):
        assert required in source, f"{required} is not reused by the stage"
    # and it never contains a gate threshold literal of its own
    for literal in ("0.10", "0.05)", "0.02"):
        assert literal not in source, \
            f"the stage must not carry its own threshold literal {literal!r}"
    assert hashlib.sha256(source.encode()).hexdigest()
