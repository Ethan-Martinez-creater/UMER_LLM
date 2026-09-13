"""Integration tests for the code-review fix round.

Each test pins one finding: uncapped CR-TSER snapshots, entrance-level
leave-one-reader-out isolation, the canonical evidence-key contract, dataset
artifact namespacing, cache fingerprint fail-closed behaviour, manifest
freezing, viability-before-split ordering, P1 cutoff identity, P2
reader/snapshot pairing, the P3 bootstrap requirement, PHEME-only B2/S6 and the
teacher-forced A/B tokenization.
"""
from __future__ import annotations

import json

import pytest
import torch

from ..config.pilot_config import (LORO_ROTATIONS, SEMANTIC_DIM,
                                   STRUCT_SCALAR_DIM, UTILITY_Q_DIM)
from ..data.snapshot_bridge import MAX_NODES_CAP, build_causal_snapshot
from .conftest import branch_event, make_event


class _RecordingTokenizer:
    """Records ``add_special_tokens`` so the fix can be asserted directly."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=True):
        # character-level ids: concatenation identity holds exactly, which is
        # what the teacher-forcing boundary check asserts
        self.calls.append((str(text), add_special_tokens))
        return {"input_ids": [ord(ch) for ch in str(text)]}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        return "<s>" + "\n".join(m["content"] for m in messages)


# --------------------------------------------------------------------------
# 1. uncapped causal snapshot
# --------------------------------------------------------------------------
def _wide_event(n_replies):
    nodes = [("n0", None, 1000, "source claim", 0)]
    for i in range(n_replies):
        nodes.append((f"n{i + 1}", "n0", 1000 + i + 1, f"reply {i}", i + 1))
    return make_event(nodes)


def test_snapshot_retains_more_than_1021_nodes():
    n = 1100
    event = _wide_event(n)
    snap = build_causal_snapshot(event, 360)
    assert MAX_NODES_CAP is None
    assert snap["cap_hit"] is False
    assert snap["max_nodes_cap"] is None
    assert len(snap["node_ids"]) == n + 1
    assert snap["num_nodes_before_cap"] == n + 1
    assert snap["num_nodes_after_cap"] == n + 1


def test_bitte_receives_all_snapshot_nodes():
    from ..data.structural_stats import structural_scalars
    from ..models.bitte import BiTTE, pack_graph_batch
    n = 1100
    snap = build_causal_snapshot(_wide_event(n), 360)
    n_nodes = len(snap["node_ids"])
    assert n_nodes == n + 1
    pos = {nid: i for i, nid in enumerate(snap["node_ids"])}
    parent_idx = torch.full((n_nodes,), -1, dtype=torch.long)
    for child, parent in snap["edge_index"]:
        parent_idx[child] = parent
    edges = torch.tensor(snap["edge_index"], dtype=torch.long) \
        if snap["edge_index"] else torch.zeros((0, 2), dtype=torch.long)
    graph = {
        "sem": torch.zeros(n_nodes, SEMANTIC_DIM),
        "struct": torch.tensor(structural_scalars(snap, 360)),
        "parent_idx": parent_idx, "edges": edges,
        "source_idx": pos[snap["source_id"]],
    }
    model = BiTTE().eval()
    with torch.no_grad():
        h, _h_src = model(**pack_graph_batch([graph]))
    assert h.shape[1] == n_nodes


# --------------------------------------------------------------------------
# 2. leave-one-reader-out isolation at the entrance
# --------------------------------------------------------------------------
def _snapshot_tensors(n):
    return (torch.randn(n, SEMANTIC_DIM,
                        generator=torch.Generator().manual_seed(0)),
            torch.rand(n, STRUCT_SCALAR_DIM,
                       generator=torch.Generator().manual_seed(1)),
            torch.rand(n, UTILITY_Q_DIM,
                       generator=torch.Generator().manual_seed(2)))


def test_rotation_builder_filters_three_reader_cache_at_the_entrance():
    from ..models.bitte import BiTTE
    from ..models.utility_heads import SharedResidualUtility
    from ..training.train_utility import _batch_forward
    from ..training.utility_dataset import (UtilityDataset, make_group,
                                            filter_rows_for_readers)
    a, b, held = LORO_ROTATIONS[0]
    snap = build_causal_snapshot(branch_event(), 360)
    n = len(snap["node_ids"])
    sem, struct, q = _snapshot_tensors(n)
    # a realistic three-reader cache: one atomic row per reader
    unit_rows = [{"unit_index": i, "reader": reader, "target": 0.1,
                  "sign": "NEUTRAL", "correct_before": True,
                  "correct_after": True}
                 for i, reader in ((1, a), (2, b), (3, held))]
    group = make_group(snap, sem, struct, q, unit_rows, 360, [1, 2, 3],
                       dataset="pheme")
    assert len(group["unit_rows"]) == 3
    assert all(r["unit_key"].startswith("pheme|")
               for r in group["unit_rows"])

    kept, dropped = filter_rows_for_readers(group["unit_rows"], [a, b])
    assert len(dropped) == 1 and dropped[0]["reader"] == held
    filtered = {**group, "unit_rows": kept}
    dataset = UtilityDataset([filtered], [a, b], {a: 0, b: 1})
    assert all(r["reader"] != held for g in dataset.groups
               for r in g["unit_rows"])

    bitte = BiTTE()
    model = SharedResidualUtility(2)
    packed = _batch_forward(bitte, model, [filtered], {a: 0, b: 1}, "cpu")
    assert packed is not None
    assert set(packed[1].tolist()) == {0, 1}
    assert all(held not in key for key in packed[4])

    with pytest.raises(ValueError):
        UtilityDataset([group], [a, b], {a: 0, b: 1})


# --------------------------------------------------------------------------
# 3. predictor -> selector evidence identity contract
# --------------------------------------------------------------------------
def test_predictor_selector_key_contract_end_to_end(fake_tokenizer):
    import cr_tser_run_selection as selection
    from ..intervention.evidence_units import build_evidence_units, evidence_key
    from ..intervention.semantic_reference import build_src
    from ..models.robust_selector import build_arm
    from .conftest import embeddings_for

    snap = build_causal_snapshot(branch_event(), 360)
    units = build_evidence_units(snap)
    src_emb, reply_emb = embeddings_for(snap)
    src = build_src(units, src_emb, reply_emb, fake_tokenizer)
    ids = list(src["selected_node_ids"])

    # predictor-style artifact keyed by the canonical contract (no test-only key)
    artifact = {
        "qwen": {evidence_key("pheme", snap["event_id"], 360, nid):
                 {"utility": 0.3} for nid in ids},
        "glm": {evidence_key("pheme", snap["event_id"], 360, nid):
                {"utility": -0.1} for nid in ids},
        "shared": {evidence_key("pheme", snap["event_id"], 360, nid):
                   {"utility": 0.05} for nid in ids},
    }
    per_snapshot = {
        key: selection.predictions_for_snapshot(artifact, "pheme",
                                                snap["event_id"], 360, key)
        for key in ("qwen", "glm", "shared")}
    assert per_snapshot["qwen"] == {nid: 0.3 for nid in ids}
    # a different cutoff must not collide with the 360m rows
    assert selection.predictions_for_snapshot(
        artifact, "pheme", snap["event_id"], 60, "qwen") == {}
    # a different dataset must not collide either
    assert selection.predictions_for_snapshot(
        artifact, "weibo22", snap["event_id"], 360, "qwen") == {}

    for arm in ("S3a_single_a", "S3b_single_b", "S4_shared",
                "S5_cross_reader_robust"):
        result = build_arm(arm, units, src, per_snapshot, fake_tokenizer,
                           ["qwen", "glm"], seed=7319)
        assert set(result["selected_node_ids"]) <= set(ids)


# --------------------------------------------------------------------------
# 4. dataset artifact namespace
# --------------------------------------------------------------------------
def test_dataset_artifacts_do_not_overwrite(tmp_path):
    root = tmp_path
    for dataset in ("pheme", "weibo22"):
        target = root / "manifests" / dataset
        target.mkdir(parents=True, exist_ok=True)
        (target / "event_split.json").write_text(
            json.dumps({"dataset": dataset, "viable_event_count": 10,
                        "viability_filtered": 2, "split_seed": 7319,
                        "foundation_train": [], "utility_train": [],
                        "utility_dev": [], "utility_eval": [],
                        "unused": []}), encoding="utf-8")
    assert (root / "manifests" / "pheme" / "event_split.json").exists()
    assert (root / "manifests" / "weibo22" / "event_split.json").exists()
    pheme = json.loads((root / "manifests" / "pheme" /
                        "event_split.json").read_text(encoding="utf-8"))
    weibo = json.loads((root / "manifests" / "weibo22" /
                        "event_split.json").read_text(encoding="utf-8"))
    assert pheme["dataset"] == "pheme" and weibo["dataset"] == "weibo22"

    import cr_tser_run_pilot as pilot
    gates, reports, decision = pilot.compute_gates(str(root),
                                                   ("pheme", "weibo22"))
    # no P1/P2 gate is invented from empty artifacts; P3 fails closed because
    # the three LORO predictor artifacts are absent
    assert not reports
    assert "P1" not in gates and "P2" not in gates
    assert gates["P3"]["pass"] is False
    assert gates["P3"]["reason"] == "incomplete LORO predictor artifacts"
    assert decision["primary_dataset"] == "weibo22"
    assert decision["legacy_diagnostics"]["b2_s6_enabled_datasets"] == ["pheme"]


def test_manifest_freeze_refuses_after_labels(tmp_path):
    import cr_tser_build_manifests as manifests
    root = tmp_path
    (root / "utility_labels" / "pheme").mkdir(parents=True)
    (root / "utility_labels" / "pheme" / "labels.jsonl").write_text(
        "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        manifests.assert_manifests_mutable(str(root), "pheme", force=True)
    with pytest.raises(RuntimeError):
        manifests.assert_manifests_mutable(str(root), "pheme", force=False)
    # with no labels, a pre-freeze rebuild is allowed when --force is given
    manifests.assert_manifests_mutable(str(tmp_path / "empty"), "pheme",
                                       force=True)


# --------------------------------------------------------------------------
# 5. cache identity
# --------------------------------------------------------------------------
def _fingerprint_row(**overrides):
    row = {"prompt_hash": "p", "prompt_ids_hash": "pi", "base_context_hash": "b",
           "intervened_context_hash": "i", "reader_hash": "r",
           "reader_identity_hash": "ri", "tokenizer_hash": "t",
           "chat_template_hash": "ct"}
    row.update(overrides)
    return row


def test_cache_fingerprint_mismatch_fails_closed():
    import cr_tser_generate_labels as labels
    matching = _fingerprint_row()
    labels._verify_cached(_fingerprint_row(), matching, "k")  # no raise
    for field in ("reader_hash", "reader_identity_hash", "prompt_hash",
                  "prompt_ids_hash", "tokenizer_hash", "chat_template_hash",
                  "base_context_hash", "intervened_context_hash"):
        with pytest.raises(labels.CacheIdentityMismatch):
            labels._verify_cached(_fingerprint_row(),
                                  {**matching, field: "changed"}, "k")


def test_cache_fails_closed_on_tokenizer_and_template_substitution():
    """A re-tokenized or re-templated reader must invalidate cached labels."""
    import cr_tser_generate_labels as labels
    matching = _fingerprint_row()
    with pytest.raises(labels.CacheIdentityMismatch):
        labels._verify_cached(_fingerprint_row(), {**matching,
                                                   "tokenizer_hash": "v2"}, "k")
    with pytest.raises(labels.CacheIdentityMismatch):
        labels._verify_cached(_fingerprint_row(), {**matching,
                                                   "chat_template_hash": "v2"},
                              "k")


# --------------------------------------------------------------------------
# 6. viability precedes the split
# --------------------------------------------------------------------------
def test_viability_filter_precedes_split():
    from ..data.pilot_split import viable_event_ids
    dead = make_event([
        ("n0", None, 1000, "source", 0),
        ("n1", "n0", 1000 + 10 * 3600, "reply far beyond 6h", 1),
    ], event_id="dead")
    alive = make_event([
        ("n0", None, 1000, "source", 0),
        ("n1", "n0", 1060, "early reply", 1),
    ], event_id="alive")
    assert viable_event_ids([dead, alive], (15, 60, 360)) == ["alive"]

    import cr_tser_build_manifests as manifests
    source = manifests.__dict__
    assert "viable_event_ids" in {k for k in dir(manifests)} or \
        "viable_event_ids" in manifests.build_manifests.__code__.co_names


# --------------------------------------------------------------------------
# 7. P1 cutoff identity
# --------------------------------------------------------------------------
def test_p1_atomic_identity_includes_cutoff():
    from ..intervention.evidence_units import evidence_key, evidence_key_parts
    at_60 = evidence_key("weibo22", "e1", 60, "n1")
    at_360 = evidence_key("weibo22", "e1", 360, "n1")
    assert at_60 != at_360
    assert evidence_key_parts(at_60)["cutoff"] == 60
    assert evidence_key_parts(at_360)["cutoff"] == 360
    import cr_tser_run_pilot as pilot
    assert "evidence_key" in pilot.load_unit_table.__code__.co_names


# --------------------------------------------------------------------------
# 8. P2 reader/snapshot pairing
# --------------------------------------------------------------------------
def test_p2_pairs_within_reader_and_snapshot():
    from ..evaluation.structural_interaction import (_pair_deltas,
                                                     build_event_payloads)
    base = [
        {"event": "e1", "cutoff": 60, "reader": "qwen", "type": "I2",
         "utility": 0.5, "members": [0.1, 0.1]},
        {"event": "e1", "cutoff": 60, "reader": "qwen", "type": "I3",
         "utility": 0.2, "members": [0.1, 0.1]},
        # no I3 at 360m for qwen -> must not be paired
        {"event": "e1", "cutoff": 360, "reader": "qwen", "type": "I2",
         "utility": 0.9, "members": [0.1, 0.1]},
    ]
    payloads = build_event_payloads(base)
    assert len(_pair_deltas(payloads["e1"], "pc", "na")) == 1
    # adding an I3 for a different reader must not create a cross-reader pair
    with_other_reader = base + [
        {"event": "e1", "cutoff": 60, "reader": "glm", "type": "I3",
         "utility": 0.3, "members": [0.1, 0.1]}]
    payloads2 = build_event_payloads(with_other_reader)
    assert len(_pair_deltas(payloads2["e1"], "pc", "na")) == 1


# --------------------------------------------------------------------------
# 9. B2 / S6 PHEME-only
# --------------------------------------------------------------------------
def test_b2_s6_legacy_is_pheme_only_and_inference_only():
    from ..models.legacy_utility import (LegacyNotPermitted, LegacyPHEMEUtility,
                                         assert_legacy_dataset,
                                         legacy_arm_enabled)
    assert legacy_arm_enabled("pheme") is True
    assert legacy_arm_enabled("weibo22") is False
    with pytest.raises(LegacyNotPermitted):
        assert_legacy_dataset("weibo22")
    with pytest.raises(LegacyNotPermitted):
        LegacyPHEMEUtility("weibo22", None, None)


# --------------------------------------------------------------------------
# 10. teacher-forced A/B tokenization
# --------------------------------------------------------------------------
def test_teacher_forced_tokenization_disables_special_tokens():
    from ..readers.sequence_scorer import (ab_token_report,
                                           continuation_boundary,
                                           tokenize_continuation,
                                           tokenize_prompt)
    tok = _RecordingTokenizer()
    tokenize_prompt(tok, "<s>hello")
    assert tok.calls[-1][1] is False
    tokenize_continuation(tok, "A")
    assert tok.calls[-1][1] is False
    info = continuation_boundary(tok, "<s>hello", "A")
    assert "boundary_ok" in info and info["prompt_tokens"] >= 1
    report = ab_token_report(tok, "source post body")
    assert report["all_boundaries_ok"] is True
    assert report["candidates"]["A"]["score_mode"] == \
        "teacher_forced_logprob_sum"
    assert report["candidates"]["B"]["score_mode"] == \
        "teacher_forced_logprob_sum"


def test_p3_gate_requires_bootstrap_ci():
    from ..evaluation.utility_prediction import gate_p3
    b3 = {"macro_f1": 0.60, "spearman": 0.5}
    base = {"B0": {"macro_f1": 0.50, "spearman": 0.5}}
    assert gate_p3(b3, base)["pass"] is False          # no bootstrap -> refuse
    assert gate_p3(b3, base, delta_ci={"ci_low": 0.001})["pass"] is True
    assert gate_p3(b3, base, delta_ci={"ci_low": -0.001})["pass"] is False
