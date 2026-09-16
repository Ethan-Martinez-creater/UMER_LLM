"""Protocol-closure tests (final fix round + code-freeze hotfix).

Pins: S6 uses the frozen StaticUtilitySelector (not the Proxy), P3 keeps every
LORO rotation's predictions, P3/P4 fail closed on incomplete rotations, P0
normalized coverage cannot produce a false READY, frozen subsets are immutable,
and ``CRTSER_SMOKE`` resolves to the smoke namespace.

The hotfix round adds the *aggregator* contract: Stage-B rotation identity must
survive ``compute_gates`` (three rotations complete, two fail closed, PHEME
complete three), Stage A must be write-once, and the PHEME B2 legacy diagnostic
must reach the report through a real ``b2_surface`` without touching a gate.
"""
from __future__ import annotations

import hashlib
import json

import pytest
import torch
import torch.nn as nn

from ..config.pilot_config import paths_from_env
from ..models.legacy_utility import LegacyPHEMEUtility


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
                    _Tok(), ["qwen", "mistral"], seed=7319)
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
               "reader": "mistral", "intervention_type": "I1_atomic",
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
        "qwen_mistral": 0.9, "qwen_internlm": -0.9, "mistral_internlm": 0.3})
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
    root = _p3_fixture(tmp_path, {"qwen_mistral": 0.9})
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
              "train_readers": ["qwen", "mistral"], "held_out_reader": "internlm",
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


# --------------------------------------------------------------------------
# 6. aggregator contract — Stage-B rotation identity must survive
#    ``compute_gates``, because a lost identity makes every three-rotation
#    condition fail closed on complete evidence.
# --------------------------------------------------------------------------
THREE_HELD_OUT = ("internlm", "mistral", "qwen")


def _write_aggregator_tree(root, dataset, held_readers, delta=0.05,
                           token_ok=True, p0="P0_PASS"):
    """Formal-shape artifacts: identity at the top level, Δ nested."""
    import os as _os
    _os.makedirs(_os.path.join(root, "manifests", dataset), exist_ok=True)
    with open(_os.path.join(root, "manifests", dataset, "event_split.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"dataset": dataset, "utility_eval": [], "utility_train": [],
                   "utility_dev": [], "foundation_train": [], "unused": []}, fh)
    _os.makedirs(_os.path.join(root, "unseen_reader", dataset), exist_ok=True)
    for held in held_readers:
        with open(_os.path.join(root, "unseen_reader", dataset,
                                f"rotation_{held}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"stage": "B_score_heldout", "dataset": dataset,
                       "held_out_reader": held,
                       "delta": {"delta": delta, "primary_macro_f1": 0.5,
                                 "best_simple_macro_f1": 0.5 - delta,
                                 "token_target_ok": token_ok}}, fh)
    _os.makedirs(_os.path.join(root, "p0"), exist_ok=True)
    with open(_os.path.join(root, "p0", "p0_readiness.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"P0": p0}, fh)
    return str(root)


def test_aggregator_preserves_three_rotation_identity(tmp_path):
    import cr_tser_run_pilot as pilot
    from ..evaluation.unseen_reader import EXPECTED_HELDOUT_READERS
    root = _write_aggregator_tree(str(tmp_path), "maweibo", THREE_HELD_OUT)
    gates, _reports, decision = pilot.compute_gates(root)
    p4 = gates["P4"]
    assert p4["rotation_completeness"]["complete"] is True
    # the identity itself must survive, not merely the row count
    assert p4["rotation_completeness"]["distinct_held_out"] == \
        sorted(EXPECTED_HELDOUT_READERS)
    assert p4["pass"] is True
    # PHEME evidence is genuinely absent, not read as a pass
    assert decision["pheme_evidence_complete"] is False
    assert decision["decision"] != "FULL_GO"


def test_aggregator_two_rotations_fail_closed(tmp_path):
    import cr_tser_run_pilot as pilot
    root = _write_aggregator_tree(str(tmp_path), "maweibo",
                                  THREE_HELD_OUT[:2])
    gates, _reports, _decision = pilot.compute_gates(root)
    assert gates["P4"]["rotation_completeness"]["complete"] is False
    assert gates["P4"]["pass"] is False


def test_aggregator_recognizes_complete_pheme_secondary(tmp_path):
    import cr_tser_run_pilot as pilot
    root = _write_aggregator_tree(str(tmp_path), "maweibo", THREE_HELD_OUT)
    _write_aggregator_tree(root, "pheme", THREE_HELD_OUT)
    gates, _reports, decision = pilot.compute_gates(root)
    assert gates["P4"]["rotation_completeness"]["complete"] is True
    assert gates["P4"]["pass"] is True
    assert decision["pheme_evidence_complete"] is True
    assert decision["pheme_secondary"] is True


def test_aggregator_incomplete_pheme_never_full_go(tmp_path):
    import cr_tser_run_pilot as pilot
    root = _write_aggregator_tree(str(tmp_path), "maweibo", THREE_HELD_OUT)
    _write_aggregator_tree(root, "pheme", THREE_HELD_OUT[:2])
    gates, _reports, decision = pilot.compute_gates(root)
    assert gates["P4"]["pass"] is True          # primary evidence complete
    assert decision["pheme_evidence_complete"] is False
    assert decision["decision"] != "FULL_GO"    # secondary cannot be a pass


# --------------------------------------------------------------------------
# 7. Stage A is write-once — a second freeze must fail and change nothing
# --------------------------------------------------------------------------
class _Tok:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(str(text).split())))}


def _freeze_paths():
    import types
    return types.SimpleNamespace(semantic_model="m", canonical_tokenizer="t")


def _freeze_environment(monkeypatch, root):
    """A minimal synthetic Stage-A environment (no real data or encoders)."""
    import os as _os
    import cr_tser_common as common
    import cr_tser_run_selection as selection

    _os.makedirs(_os.path.join(root, "manifests", "pheme"), exist_ok=True)
    with open(_os.path.join(root, "manifests", "pheme", "event_split.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"dataset": "pheme", "utility_eval": ["e1"],
                   "utility_train": [], "utility_dev": [],
                   "foundation_train": [], "unused": []}, fh)
    event = {"event_id": "e1", "label": 1, "source_id": "n0",
             "source_timestamp": 1000,
             "nodes": [{"node_id": "n0", "parent_id": None, "timestamp": 1000,
                        "text": "src", "original_order": 0,
                        "status": "VALID"},
                       {"node_id": "n1", "parent_id": "n0", "timestamp": 1060,
                        "text": "reply", "original_order": 1,
                        "status": "VALID"}]}
    item = {"event_id": "e1", "cutoff_minutes": 60, "label": 1,
            "num_nodes": 2, "node_ids": ["n0", "n1"], "source_pos": 0}
    art = {"zero_reply": False, "units": [{"node_id": "n1"}],
           "src": {"selected_node_ids": ["n1"], "total_tokens": 10},
           "snapshot": {"node_ids": ["n0", "n1"]},
           "semantic": torch.zeros(2, 384)}
    fake_arms = {name: {"selected_node_ids": ["n1"], "total_tokens": 5,
                        "target_tokens": 10, "content_hash": "h"}
                 for name in ("S0_src_full", "S1_random_tm", "S2_semantic_tm",
                              "S3a_source", "S3b_source", "S4_shared",
                              "S5_cross_reader_robust",
                              "S6_legacy_utility_tm")}
    monkeypatch.setattr(common, "load_dataset_events",
                        lambda dataset, paths: [event])
    monkeypatch.setattr(common, "CrSemanticEncoder",
                        lambda model, dataset, device="cpu": nn.Module())
    monkeypatch.setattr(common, "canonical_tokenizer", lambda path: _Tok())
    monkeypatch.setattr(common, "snapshot_artifacts",
                        lambda e, c, enc, tok: art)
    monkeypatch.setattr(common, "source_fingerprint_for",
                        lambda dataset, paths: {})
    monkeypatch.setattr(selection, "_load_predictor",
                        lambda out_root, dataset, readers: {
                            "predictions": {}, "prediction_coverage": {}})
    monkeypatch.setattr(selection, "_load_single_predictions",
                        lambda out_root, dataset, reader: {})
    monkeypatch.setattr(selection, "_legacy_item",
                        lambda e, snapshot, sem_rows: item)
    monkeypatch.setattr(selection, "all_arms", lambda *a, **k: dict(fake_arms))
    return item


def test_stage_a_is_write_once(tmp_path, monkeypatch):
    import cr_tser_run_selection as selection
    root = str(tmp_path)
    _freeze_environment(monkeypatch, root)
    paths = _freeze_paths()
    first = selection.freeze_subsets("pheme", paths, root, max_snapshots=1)
    assert len(first) == 3

    path = selection._frozen_path(root, "pheme", "internlm")
    with open(path, encoding="utf-8") as fh:
        before = fh.read()
    with open(path + ".sha256", encoding="utf-8") as fh:
        before_sha = fh.read()

    # a second freeze — even with a different predictor — must not overwrite
    monkeypatch.setattr(selection, "_load_predictor",
                        lambda out_root, dataset, readers: {
                            "predictions": {"changed": True},
                            "prediction_coverage": {"changed": True}})
    with pytest.raises(selection.FrozenSubsetAlreadyExists):
        selection.freeze_subsets("pheme", paths, root, max_snapshots=1)
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == before
    with open(path + ".sha256", encoding="utf-8") as fh:
        assert fh.read() == before_sha
    # the frozen artifact still verifies after the refused second freeze
    assert selection.load_frozen_subsets(root, "pheme", "internlm")


def test_stage_a_refuses_when_a_single_rotation_exists(tmp_path, monkeypatch):
    import os as _os
    import cr_tser_run_selection as selection
    root = str(tmp_path)
    _freeze_environment(monkeypatch, root)
    _os.makedirs(selection.frozen_dir(root, "pheme"), exist_ok=True)
    record = {"stage": "A_freeze_subsets", "dataset": "pheme",
              "held_out_reader": "mistral", "subsets": []}
    record["sha256"] = selection._canonical_sha(record)
    path = selection._frozen_path(root, "pheme", "mistral")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    with pytest.raises(selection.FrozenSubsetAlreadyExists):
        selection.freeze_subsets("pheme", _freeze_paths(), root)


# --------------------------------------------------------------------------
# 8. PHEME B2 legacy diagnostic — real surface into the artifact, never a gate
# --------------------------------------------------------------------------
class _B2Selector(nn.Module):
    def forward(self, *args):
        return torch.zeros(1)


class _B2Proxy(nn.Module):
    """Deterministic classification; index 1 is *rumor* (TC-DSCR convention)."""

    def __init__(self, table=None):
        super().__init__()
        self.table = table or [[0.0, 9.0]]
        self.calls = 0

    def classify(self, h_source, z_sel):
        index = min(self.calls, len(self.table) - 1)
        self.calls += 1
        return torch.tensor(self.table[index])


class _B2Scorer(LegacyPHEMEUtility):
    """Real ``b2_items``/``b2_surface`` path with a stubbed S6 selector."""

    def score_items(self, items, device=None):
        return [{nid: 0.1 for nid in item["node_ids"]
                 if nid != item["node_ids"][item["source_pos"]]}
                for item in items]


def _b2_scorer(proxy):
    return _B2Scorer("pheme", encoder=nn.Module(), selector=_B2Selector(),
                     proxy=proxy)


def test_b2_surface_outputs_enter_pheme_artifact(tmp_path, monkeypatch):
    import os as _os
    import cr_tser_run_selection as selection
    import tcdscr_run_e2
    monkeypatch.setattr(
        tcdscr_run_e2, "encoder_forward_batch",
        lambda encoder, items, device, batch_size=32: [
            (torch.zeros(it["num_nodes"], 768), torch.zeros(768),
             torch.zeros(2)) for it in items])
    scorer = _b2_scorer(_B2Proxy())
    item = {"event_id": "e1", "cutoff_minutes": 60, "label": 1,
            "num_nodes": 2, "node_ids": ["n0", "n1"], "source_pos": 0}
    frozen = [{"held_out_reader": "internlm", "sha256": "sha-a"},
              {"held_out_reader": "mistral", "sha256": "sha-b"},
              {"held_out_reader": "qwen", "sha256": "sha-c"}]
    payload = selection._write_b2_diagnostic(
        "pheme", str(tmp_path), scorer, [item], [["n1"]], frozen)
    path = _os.path.join(str(tmp_path), "unseen_reader", "pheme",
                         "b2_legacy_diagnostic.json")
    assert _os.path.exists(path)
    on_disk = json.loads(open(path, encoding="utf-8").read())
    assert on_disk["diagnostic"] == "B2_legacy_tcdscr"
    assert on_disk["diagnostic_only"] is True
    assert on_disk["participates_in_primary_gate"] is False
    assert on_disk["never_in_p3_baseline_competition"] is True
    assert on_disk["never_in_p4_primary_comparison"] is True
    assert on_disk["frozen_fingerprint"]["s6_score_source"] == \
        "StaticUtilitySelector"
    assert on_disk["frozen_subset_sha256"] == {
        "internlm": "sha-a", "mistral": "sha-b", "qwen": "sha-c"}
    # the metric is the real Proxy surface (gold=1, p[1] maximal -> pred 1)
    assert on_disk["metrics"]["n"] == 1
    assert on_disk["metrics"]["accuracy"] == 1.0
    row = on_disk["rows"][0]
    assert row["pred"] == 1 and row["gold"] == 1
    assert row["p_rumor"] > row["p_nonrumor"]
    assert payload["n_snapshots"] == 1


def test_pheme_legacy_freeze_writes_b2_artifact(tmp_path, monkeypatch):
    """The PHEME legacy execution stage itself produces the B2 artifact."""
    import os as _os
    import cr_tser_run_selection as selection
    import tcdscr_run_e2
    root = str(tmp_path)
    _freeze_environment(monkeypatch, root)
    monkeypatch.setattr(
        tcdscr_run_e2, "encoder_forward_batch",
        lambda encoder, items, device, batch_size=32: [
            (torch.zeros(it["num_nodes"], 768), torch.zeros(768),
             torch.zeros(2)) for it in items])
    records = selection.freeze_subsets("pheme", _freeze_paths(), root,
                                       max_snapshots=1, legacy=True,
                                       legacy_scorer=_b2_scorer(_B2Proxy()))
    assert all(record["legacy_s6_enabled"] for record in records)
    path = _os.path.join(root, "unseen_reader", "pheme",
                         "b2_legacy_diagnostic.json")
    assert _os.path.exists(path)
    artifact = json.loads(open(path, encoding="utf-8").read())
    assert artifact["stage"] == "C_b2_legacy_diagnostic"
    assert artifact["dataset"] == "pheme"
    assert artifact["rows"][0]["selected_node_ids"] == ["n1"]
    assert artifact["frozen_subset_sha256"] == {
        record["held_out_reader"]: record["sha256"] for record in records}


def test_b2_artifact_does_not_change_weibo22_gates(tmp_path):
    import os as _os
    import cr_tser_run_pilot as pilot
    root = _write_aggregator_tree(str(tmp_path), "maweibo", THREE_HELD_OUT)
    _write_aggregator_tree(root, "pheme", THREE_HELD_OUT)
    gates_before, _r, decision_before = pilot.compute_gates(root)
    assert decision_before["legacy_diagnostics"]["b2_legacy_tcdscr"] is None
    with open(_os.path.join(root, "unseen_reader", "pheme",
                            "b2_legacy_diagnostic.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"dataset": "pheme", "diagnostic": "B2_legacy_tcdscr",
                   "diagnostic_only": True,
                   "participates_in_primary_gate": False,
                   "metrics": {"accuracy": 0.9, "macro_f1": 0.9, "n": 4},
                   "rows": [{"event_id": "e1"}, {"event_id": "e2"},
                            {"event_id": "e3"}]}, fh)
    gates_after, _r2, decision_after = pilot.compute_gates(root)
    assert gates_after == gates_before
    b2 = decision_after["legacy_diagnostics"]["b2_legacy_tcdscr"]
    assert b2["diagnostic"] == "B2_legacy_tcdscr"
    assert b2["n_rows"] == 3          # rows are summarised, never re-scored
    assert "rows" not in b2


def test_b2_is_never_part_of_p3_or_p4_comparison(tmp_path):
    import cr_tser_run_pilot as pilot
    from ..config.pilot_config import SIMPLE_BASELINE_ARMS
    from ..evaluation.unseen_reader import (best_simple_baseline,
                                            rotation_delta)
    root = _p3_fixture(tmp_path, {"qwen_mistral": 0.9, "qwen_internlm": -0.9,
                                  "mistral_internlm": 0.3})
    gate = pilot.utility_prediction_gate(root, "weibo22",
                                         {"utility_eval": ["e1"]})
    assert set(gate["per_model_metrics"]) <= {
        "B3_text_graph", "B0_text", "B1_scalar_structure"}
    assert "S6_legacy_utility_tm" not in gate["per_model_metrics"]

    metrics = {"S1_random_tm": {"macro_f1": 0.10, "mean_social_tokens": 1},
               "S2_semantic_tm": {"macro_f1": 0.20, "mean_social_tokens": 1},
               "S5_cross_reader_robust": {"macro_f1": 0.50,
                                          "mean_social_tokens": 1},
               "S6_legacy_utility_tm": {"macro_f1": 0.99,
                                        "mean_social_tokens": 1}}
    name, _ = best_simple_baseline(metrics)
    assert name in SIMPLE_BASELINE_ARMS
    assert name != "S6_legacy_utility_tm"
    delta = rotation_delta(metrics)
    assert delta["best_simple_arm"] == name
    assert delta["delta"] == pytest.approx(0.50 - 0.20)

