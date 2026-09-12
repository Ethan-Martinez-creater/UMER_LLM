"""Plan §35 — intervention structure tests."""
from __future__ import annotations

from ..data.snapshot_bridge import build_causal_snapshot
from ..intervention.evidence_units import build_evidence_units
from ..intervention.interaction_controls import (in_one_ancestor_subtree,
                                                 is_ancestor, is_parent_child)
from ..intervention.intervention_generator import (descendants_of,
                                                   generate_interventions)
from ..intervention.semantic_reference import build_src
from .conftest import branch_event, chain_event, embeddings_for


def _setup(event, tokenizer, seed=0):
    snap = build_causal_snapshot(event, 360)
    units = build_evidence_units(snap)
    src_emb, reply_emb = embeddings_for(snap, seed=seed)
    src = build_src(units, src_emb, reply_emb, tokenizer)
    ivs = generate_interventions(snap, units, src, tokenizer)
    return snap, units, src, ivs


def _by_id(ivs, key):
    return next(v for v in ivs if v["intervention_id"] == key)


def test_parent_child_pair_is_real_edge(fake_tokenizer):
    snap, units, src, ivs = _setup(branch_event(), fake_tokenizer)
    i2 = _by_id(ivs, "I2")
    assert i2["status"] == "OK"
    a, b = i2["remove_node_ids"]
    assert is_parent_child(snap, a, b)
    assert set(i2["remove_node_ids"]) <= set(src["selected_node_ids"])


def test_nonadjacent_control_not_ancestor(fake_tokenizer):
    snap, units, src, ivs = _setup(branch_event(), fake_tokenizer)
    i3 = _by_id(ivs, "I3")
    assert i3["status"] == "OK"
    a, b = i3["remove_node_ids"]
    # plan §11 I3: not parent-child and neither is an ancestor of the other
    assert not is_parent_child(snap, a, b)
    assert not is_ancestor(snap, a, b)
    assert not is_ancestor(snap, b, a)
    assert set(i3["remove_node_ids"]) <= set(src["selected_node_ids"])


def test_subtree_group_is_valid_descendant_set(fake_tokenizer):
    snap, units, src, ivs = _setup(branch_event(), fake_tokenizer)
    i4 = _by_id(ivs, "I4")
    assert i4["status"] == "OK"
    root = i4["meta"]["subtree_root"]
    descendants = descendants_of(snap, root)
    assert set(i4["remove_node_ids"]) <= descendants
    assert set(i4["remove_node_ids"]) <= set(src["selected_node_ids"])
    assert len(i4["remove_node_ids"]) <= 5
    assert i4["meta"]["selected_in_subtree"] >= 2


def test_no_parent_child_pair_is_explicit(fake_tokenizer):
    """A star snapshot has no reply–reply edge: I2 must say so, not vanish."""
    from .conftest import star_event
    snap, units, src, ivs = _setup(star_event(), fake_tokenizer)
    i2 = _by_id(ivs, "I2")
    assert i2["status"] == "NO_PARENT_CHILD_PAIR"
    assert i2["remove_node_ids"] == []
    i3 = _by_id(ivs, "I3")
    assert i3["status"] == "NO_MATCHED_CONTROL"


def test_atomic_cap_and_base_present(fake_tokenizer):
    event = chain_event(n_replies=25, words=4)
    snap, units, src, ivs = _setup(event, fake_tokenizer)
    atomic = [v for v in ivs if v["type"] == "I1_atomic"]
    assert len(atomic) <= 20
    assert any(v["intervention_id"] == "I0" for v in ivs)
    assert _by_id(ivs, "I0")["remove_node_ids"] == []
