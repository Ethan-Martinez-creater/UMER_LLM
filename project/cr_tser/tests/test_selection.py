"""Plan §35 — robust selection and budget tests (plan §21, §22)."""
from __future__ import annotations

from ..data.snapshot_bridge import build_causal_snapshot
from ..intervention.evidence_units import build_evidence_units
from ..intervention.semantic_reference import build_src
from ..models.robust_selector import (build_arm, pilot_target_tokens,
                                      robust_score)
from .conftest import chain_event, embeddings_for


def _setup(fake_tokenizer, n_replies=6, words=40, seed=0):
    event = chain_event(n_replies=n_replies, words=words)
    snap = build_causal_snapshot(event, 360)
    units = build_evidence_units(snap)
    src_emb, reply_emb = embeddings_for(snap, seed=seed)
    src = build_src(units, src_emb, reply_emb, fake_tokenizer)
    return snap, units, src


def test_robust_score_is_min_train_readers():
    assert robust_score({"x": 0.3}, {"x": -0.2}, "x") == -0.2
    assert robust_score({"x": -0.2}, {"x": 0.3}, "x") == -0.2
    assert robust_score({"x": 0.1}, {"x": 0.1}, "x") == 0.1


def test_selection_never_exceeds_50pct_target(fake_tokenizer):
    _snap, units, src = _setup(fake_tokenizer)
    target = pilot_target_tokens(src)
    assert target == int(0.5 * src["total_tokens"])
    ids = src["selected_node_ids"]
    predictions = {"a": {n: 0.4 for n in ids}, "b": {n: -0.1 for n in ids},
                   "shared": {n: 0.05 for n in ids}}
    for arm_name in ("S1_random_tm", "S2_semantic_tm", "S4_shared",
                     "S5_cross_reader_robust"):
        arm = build_arm(arm_name, units, src, predictions, fake_tokenizer,
                        ["a", "b"], seed=7319)
        assert arm["total_tokens"] <= arm["target_tokens"] == target


def test_selection_uses_only_reference_candidates(fake_tokenizer):
    _snap, units, src = _setup(fake_tokenizer)
    ids = set(src["selected_node_ids"])
    predictions = {"a": {n: 0.4 for n in ids}, "b": {n: 0.2 for n in ids},
                   "shared": {n: 0.1 for n in ids}}
    for arm_name in ("S0_src_full", "S1_random_tm", "S2_semantic_tm",
                     "S4_shared", "S5_cross_reader_robust"):
        arm = build_arm(arm_name, units, src, predictions, fake_tokenizer,
                        ["a", "b"], seed=7319)
        assert set(arm["selected_node_ids"]) <= ids


def test_source_full_arm_is_the_whole_reference(fake_tokenizer):
    _snap, units, src = _setup(fake_tokenizer)
    predictions = {"a": {}, "b": {}, "shared": {}}
    arm = build_arm("S0_src_full", units, src, predictions, fake_tokenizer,
                    ["a", "b"], seed=7319)
    assert set(arm["selected_node_ids"]) == set(src["selected_node_ids"])


def test_ranked_ids_in_snapshot_order(fake_tokenizer):
    """Prompt presentation is chronological, not score order (plan §21)."""
    _snap, units, src = _setup(fake_tokenizer)
    ids = src["selected_node_ids"]
    predictions = {"a": {n: float(i) for i, n in enumerate(ids)},
                   "b": {n: float(len(ids) - i) for i, n in enumerate(ids)},
                   "shared": {n: 0.0 for n in ids}}
    arm = build_arm("S5_cross_reader_robust", units, src, predictions,
                    fake_tokenizer, ["a", "b"], seed=7319)
    positions = {nid: i for i, nid in enumerate(ids)}
    order = [positions[n] for n in arm["selected_node_ids"]]
    assert order == sorted(order)
