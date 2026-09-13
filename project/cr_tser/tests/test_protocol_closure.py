"""Protocol-closure tests (final fix round).

Pins: S6 uses the frozen StaticUtilitySelector (not the Proxy), P3 keeps every
LORO rotation's predictions, P3/P4 fail closed on incomplete rotations, P0
normalized coverage cannot produce a false READY, frozen subsets are immutable,
and ``CRTSER_SMOKE`` resolves to the smoke namespace.
"""
from __future__ import annotations

import hashlib
import json

import pytest
import torch
import torch.nn as nn

from ..config.pilot_config import paths_from_env


# --------------------------------------------------------------------------
# 1. legacy S6 scores come from StaticUtilitySelector
# --------------------------------------------------------------------------
class _Selector(nn.Module):
    """Records its calls and returns a fixed, easily-ordered utility."""

    def __init__(self, values):
        super().__init__()
        self.values = values
        self.calls = 0

    def forward(self, h_cand, event_repr, sem_c, sem_src, struct_c):
        self.calls += 1
        return torch.tensor(self.values[:h_cand.shape[0]], dtype=torch.float32)


class _Proxy(nn.Module):
    """Opposite-direction classifier; must never be used for S6."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def classify(self, h_source, z_sel):
        self.calls += 1
        # deliberately reversed relative to the selector ordering
        return torch.tensor([[9.0, 0.0]])


def test_legacy_s6_uses_static_utility_selector():
    from ..models.legacy_utility import LegacyPHEMEUtility
    selector = _Selector([3.0, 2.0, 1.0])
    proxy = _Proxy()
    util = LegacyPHEMEUtility("pheme", encoder=None, selector=selector,
                              proxy=proxy)
    node_repr = torch.zeros(4, 768)
    sem = torch.zeros(4, 384)
    struct3 = torch.zeros(4, 3)
    scores = util.per_unit_scores(node_repr, torch.zeros(768), sem, struct3,
                                  source_pos=0, candidate_indices=[1, 2, 3])
    # u_i is exactly the selector's output
    assert scores == {1: 3.0, 2: 2.0, 3: 1.0}
    assert selector.calls == 1
    assert proxy.calls == 0, "S6 must not use the Proxy classifier"


def test_legacy_selector_is_required():
    from ..models.legacy_utility import LegacyPHEMEUtility, LegacyUnavailable
    with pytest.raises(LegacyUnavailable):
        LegacyPHEMEUtility("pheme", encoder=None, selector=None, proxy=_Proxy())


def test_legacy_ranking_follows_selector_not_proxy():
    """A reversing Proxy must not change the S6 ranking."""
    from ..models.robust_selector import build_arm
    from ..models.legacy_utility import LegacyPHEMEUtility

    units = [{"node_id": f"n{i}", "reply_text": f"r{i}", "parent_text": "p",
              "parent_id": None, "timestamp": i, "elapsed_seconds": i,
              "snapshot_order": i, "depth": 1} for i in range(1, 7)]
    ids = [u["node_id"] for u in units]
    src = {"selected_node_ids": ids, "ranked_node_ids": ids,
           "unit_token_costs": {n: 60 for n in ids}, "total_tokens": 360,
           "budget_ref": 1024, "n_selected": len(ids), "utilization": 0.2,
           "rank_percentile": {}, "relevance": {}}
    util = LegacyPHEMEUtility("pheme", encoder=None,
                              selector=_Selector([1.0, 2.0, 3.0, 4.0, 5.0,
                                                  6.0]), proxy=_Proxy())
    scores = util.per_unit_scores(torch.zeros(7, 768), torch.zeros(768),
                                  torch.zeros(7, 384), torch.zeros(7, 3),
                                  source_pos=0,
                                  candidate_indices=list(range(1, 7)))
    legacy = {f"n{i}": scores[i] for i in range(1, 7)}

    class _Tok:
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": list(range(len(str(text).split())))}

    arm = build_arm("S6_legacy_utility_tm", units, src, {"legacy": legacy},
                    _Tok(), ["qwen", "glm"], seed=7319)
    # the highest selector utilities win the packing
    assert set(arm["selected_node_ids"]) == {"n6", "n5", "n4"}


# --------------------------------------------------------------------------
# 2. P3 keeps every rotation's predictions (no overwrite)
# --------------------------------------------------------------------------
def _p3_fixture(tmp_path, rotations):
    root = tmp_path
    (root / "pheshell").mkdir(exist_ok=True)
    (root / "manifests" / "weibo22").mkdir(parents=True)
    (root / "manifests" / "weibo22" / "event_split.json").write_text(
        json.dumps({"dataset": "weibo22", "utility_eval": ["e1"],
                    "utility_train": [], "utility_dev": [],
                    "foundation_train": [], "unused": []}), encoding="utf-8")
    (root / "utility_labels" / "weibo22").mkdir(parents=True)
    labels = [{"dataset": "weibo22", "event_id": "e1", "cutoff": 60,
               "reader": "qwen", "intervention_type": "I1_atomic",
               "intervention_id": "I1:n1", "affected_reply_ids": ["n1"],
               "utility": 0.4, "sign": "HELPFUL", "correctness_before": True,
               "correctness_after": True},
              {"dataset": "weibo22", "event_id": "e1", "cutoff": 60,
               "reader": "glm", "intervention_type": "I1_atomic",
               "intervention_id": "I1:n1", "affected_reply_ids": ["n1"],
               "utility": -0.3, "sign": "HARMFUL", "correctness_before": True,
               "correctness_after": True}]
    (root / "utility_labels" / "weibo22" / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    key = "weibo22|e1|60|n1"
    for rid, qwen_utility in rotations.items():
        d = root / "predictor" / "weibo22" / f"rotation_{rid}"
        d.mkdir(parents=True)
        payload = {
            "dataset": "weibo22", "train_readers": rid.split("_"),
            "held_out_reader": "x",
            "predictions": {
                rid.split("_")[0]: {key: {"utility": qwen_utility,
                                          "probs": [0.8, 0.1, 0.1],
                                          "predicted_sign": "HELPFUL"}},
                rid.split("_")[1]: {key: {"utility": 0.1,
                                          "probs": [0.1, 0.8, 0.1],
                                          "predicted_sign": "NEUTRAL"}},
            },
            "baselines": {"B0_text": {"predictions": {key: {
                "utility": 0.2, "probs": [0.4, 0.3, 0.3],
                "predicted_sign": "HELPFUL"}}}},
        }
        (d / "predictions.json").write_text(json.dumps(payload),
                                            encoding="utf-8")
    return str(root)


def test_p3_keeps_predictions_from_every_rotation(tmp_path):
    import cr_tser_run_pilot as pilot
    root = _p3_fixture(tmp_path, {
        "qwen_glm": 0.9, "qwen_internlm": -0.9, "glm_internlm": 0.3})
    split = {"utility_eval": ["e1"]}
    gate = pilot.utility_prediction_gate(root, "weibo22", split)
    assert gate is not None
    assert gate["rotations_complete"] is True
    # qwen appears in two rotations; both observations must survive
    assert gate["n_observations"] == 4      # 2 readers x 2 rotations with qwen
    metrics = gate["per_model_metrics"]["B3_text_graph"]
    assert metrics["n"] == 4


def test_p3_missing_rotation_fails_closed(tmp_path):
    import cr_tser_run_pilot as pilot
    root = _p3_fixture(tmp_path, {"qwen_glm": 0.9})
    gate = pilot.utility_prediction_gate(root, "weibo22",
                                         {"utility_eval": ["e1"]})
    assert gate["pass"] is False
    assert gate["reason"] == "incomplete LORO predictor artifacts"
    assert gate["rotations_required"] == 3
    assert len(gate["rotations_missing"]) == 2


# --------------------------------------------------------------------------
# 3. frozen subsets are immutable; Stage A never builds a held-out reader
# --------------------------------------------------------------------------
def test_freeze_stage_never_references_held_out_reader_builder():
    import cr_tser_run_selection as selection
    freeze_names = set(selection.freeze_subsets.__code__.co_names)
    score_names = set(selection.score_heldout.__code__.co_names)
    assert "build_reader" not in freeze_names
    assert "build_reader" in score_names


def test_modified_frozen_subset_is_rejected(tmp_path):
    import cr_tser_run_selection as selection
    root = str(tmp_path)
    frozen = selection.frozen_dir(root, "pheme")
    import os
    os.makedirs(frozen, exist_ok=True)
    record = {"stage": "A_freeze_subsets", "dataset": "pheme",
              "train_readers": ["qwen", "glm"], "held_out_reader": "internlm",
              "subsets": [{"arm": "S0_src_full", "event_id": "e1", "cutoff": 60,
                           "gold": 1, "selected_node_ids": ["n1"],
                           "total_tokens": 1, "target_tokens": 1,
                           "content_hash": "h"}]}
    record["sha256"] = selection._canonical_sha(record)
    path = selection._frozen_path(root, "pheme", "internlm")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    with open(path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(record["sha256"])
    # intact artifact verifies
    assert selection.load_frozen_subsets(root, "pheme", "internlm")["sha256"]
    # tamper with the subset -> refuse to score
    record["subsets"][0]["selected_node_ids"] = ["n1", "n2"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    with pytest.raises(selection.FrozenSubsetChanged):
        selection.load_frozen_subsets(root, "pheme", "internlm")


def test_missing_frozen_subset_fails_closed(tmp_path):
    import cr_tser_run_selection as selection
    with pytest.raises(selection.FrozenSubsetMissing):
        selection.load_frozen_subsets(str(tmp_path), "pheme", "qwen")


# --------------------------------------------------------------------------
# 4. normalized coverage cannot produce a false READY
# --------------------------------------------------------------------------
def _node(nid, parent, ts, text, order=0):
    return {"node_id": nid, "parent_id": parent, "timestamp": ts, "text": text,
            "original_order": order, "status": "VALID"}


def _event(nodes, label=1):
    source = next(n for n in nodes if n["parent_id"] is None)
    return {"event_id": f"e{source['timestamp']}", "label": label,
            "source_id": source["node_id"],
            "source_timestamp": source["timestamp"], "nodes": nodes}


def _write(tmp_path, events):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n",
                    encoding="utf-8")
    return str(path)


def test_missing_parent_blocks_ready(tmp_path):
    from cr_tser.data import weibo22_adapter as adapter
    path = _write(tmp_path, [_event([
        _node("n0", None, 1000, "source body"),
        _node("n1", "n0", 1060, "reply one"),
        _node("n2", None, 1120, "reply two"),   # no parent
    ])])
    report = adapter.validate_normalized_export(path)
    assert report["valid"] is False
    assert report["verdict"] == adapter.VERDICT_UNAVAILABLE
    assert report["parent_coverage"] < 1.0
    assert report["missing_parent_count"] == 1


def test_reply_text_coverage_counts_every_reply(tmp_path):
    from cr_tser.data import weibo22_adapter as adapter
    path = _write(tmp_path, [_event([
        _node("n0", None, 1000, "source body"),
        _node("n1", "n0", 1060, "reply one"),
        _node("n2", "n0", 1120, "   "),         # empty reply text
    ])])
    report = adapter.validate_normalized_export(path)
    assert report["valid"] is False
    assert report["reply_text_coverage"] == pytest.approx(0.5)


def test_unresolvable_parent_and_child_earlier_are_reported(tmp_path):
    from cr_tser.data import weibo22_adapter as adapter
    path = _write(tmp_path, [_event([
        _node("n0", None, 1000, "source body"),
        _node("n1", "ghost", 1060, "reply one"),      # unresolvable parent
        _node("n2", "n1", 1000, "child earlier"),     # child earlier than n1
        _node("n3", "n0", 1120, "reply three"),
    ])])
    report = adapter.validate_normalized_export(path)
    assert report["valid"] is False
    assert report["unresolvable_parent_count"] == 1
    assert report["child_earlier_than_parent_count"] == 1
    assert report["unresolvable_parent_rate"] > 0.0


def test_valid_export_is_ready_with_full_30_fields(tmp_path):
    from cr_tser.data import weibo22_adapter as adapter
    path = _write(tmp_path, [
        _event([_node("n0", None, 1000, "source body"),
                _node("n1", "n0", 1060, "reply one"),
                _node("n2", "n1", 1120, "reply two")]),
        _event([_node("n0", None, 2000, "another source"),
                _node("n1", "n0", 2060, "another reply")], label=0),
    ])
    report = adapter.validate_normalized_export(path)
    assert report["valid"] is True
    assert report["verdict"] == adapter.VERDICT_READY
    for field in ("event_count", "label_distribution", "source_text_coverage",
                  "reply_text_coverage", "timestamp_coverage",
                  "parent_coverage", "duplicate_ids", "negative_timestamps",
                  "child_earlier_than_parent_count", "unresolvable_parent_rate",
                  "events_with_ge1_valid_reply", "events_viable_15m",
                  "events_viable_1h", "events_viable_6h"):
        assert field in report, field
    assert report["events_viable_15m"] == 2
    assert report["original_order_used_as_time"] is False


def test_duplicate_node_ids_block_ready(tmp_path):
    from cr_tser.data import weibo22_adapter as adapter
    path = _write(tmp_path, [_event([
        _node("n0", None, 1000, "source body"),
        _node("n1", "n0", 1060, "reply one"),
        _node("n1", "n0", 1120, "duplicate id"),
    ])])
    report = adapter.validate_normalized_export(path)
    assert report["valid"] is False


# --------------------------------------------------------------------------
# 5. CRTSER_SMOKE resolves without TypeError
# --------------------------------------------------------------------------
def test_crtser_smoke_env_builds_smoke_root():
    paths = paths_from_env({"CRTSER_SMOKE": "1",
                            "CRTSER_OUT_ROOT": "/tmp/out"})
    assert paths.out_root.endswith("smoke")
    plain = paths_from_env({"CRTSER_OUT_ROOT": "/tmp/out"})
    assert plain.out_root == "/tmp/out"
