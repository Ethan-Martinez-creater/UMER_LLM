"""Plan §35 — data, snapshot and SRC tests."""
from __future__ import annotations

import os

import pytest

from ..config.pilot_config import CUTOFFS_MIN, VERDICT_UNAVAILABLE
from ..data import weibo22_adapter as wa
from ..data.pilot_split import assert_event_disjoint, build_pilot_split
from ..data.snapshot_bridge import build_causal_snapshot, cutoffs
from ..data.structural_stats import structural_scalars
from .conftest import embeddings_for, make_event


def _write_kpg_release(root):
    os.makedirs(os.path.join(root, "label_all"), exist_ok=True)
    os.makedirs(os.path.join(root, "td_rvnn"), exist_ok=True)
    with open(os.path.join(root, "label_all", "Weibo_label_All.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("true\tcovid-19\t111\nfalse\tother\t222\n")
    # rows carry a parent index chain and word indices only — no time column
    with open(os.path.join(root, "td_rvnn", "data.TD_RvNN.vol_5000.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("111\tNone\t1\t2\t0\t5:1 7:1\n")
        fh.write("111\t1\t2\t2\t0\t9:1\n")
        fh.write("222\tNone\t1\t1\t0\t3:1\n")
    return root


def test_weibo22_timestamp_not_original_order(tmp_path):
    """The release has no timestamps; nothing may be inferred from row order."""
    root = _write_kpg_release(str(tmp_path / "kpg"))
    audit = wa.audit_release(root)
    assert audit["verdict"] == VERDICT_UNAVAILABLE
    assert audit["timestamp_coverage"] == 0.0
    assert audit["release_defects"]["absolute_node_timestamp"] is False
    with pytest.raises(wa.Weibo22TemporalUnavailable):
        wa.load_events(root)


def test_snapshot_uses_timestamp_not_row_order():
    """A later row with an earlier timestamp must enter an early snapshot."""
    event = make_event([
        ("n0", None, 1000, "source", 0),
        ("n1", "n0", 1000 + 10 * 60, "early but listed second", 2),
        ("n2", "n0", 1000 + 5 * 3600, "late but listed first", 1),
    ])
    snap = build_causal_snapshot(event, 15)
    assert "n1" in snap["node_ids"]
    assert "n2" not in snap["node_ids"]


def test_snapshot_excludes_future_node():
    event = make_event([
        ("n0", None, 1000, "source", 0),
        ("n1", "n0", 1000 + 60 * 60, "at 1h", 1),
        ("n2", "n0", 1000 + 400 * 60, "future", 2),
    ])
    snap = build_causal_snapshot(event, 60)
    assert "n2" not in snap["node_ids"]
    limit = 1000 + 60 * 60
    assert all(ts <= limit for ts in snap["timestamps"])


def test_parent_precedes_child_or_flags_anomaly():
    """A child earlier than its parent must not form an edge in G_t."""
    event = make_event([
        ("n0", None, 1000, "source", 0),
        ("n1", "n2", 1000 + 60, "child earlier than parent", 1),
        ("n2", "n0", 1000 + 120, "parent later", 2),
    ])
    snap = build_causal_snapshot(event, 360)
    pos = {nid: i for i, nid in enumerate(snap["node_ids"])}
    assert snap["parent_ids"][pos["n1"]] is None
    assert snap["parent_ids"][pos["n2"]] == "n0"
    assert all(child != pos["n1"] for child, _parent in snap["edge_index"])


def test_src_budget_never_exceeded(fake_tokenizer):
    from ..intervention.evidence_units import build_evidence_units
    from ..intervention.semantic_reference import build_src, src_token_cost
    from .conftest import chain_event
    event = chain_event(n_replies=6, words=120)
    snap = build_causal_snapshot(event, 360)
    units = build_evidence_units(snap)
    src_emb, reply_emb = embeddings_for(snap)
    src = build_src(units, src_emb, reply_emb, fake_tokenizer)
    assert src["budget_ref"] == 1024
    assert src["total_tokens"] <= 1024
    assert src_token_cost(fake_tokenizer, src, units) <= 1024
    # the budget must actually bind, otherwise the test proves nothing
    assert src["n_selected"] < src["n_visible_replies"]


def test_pilot_split_sizes_disjoint_and_deterministic():
    labels = {f"e{i}": i % 2 for i in range(400)}
    split = build_pilot_split(labels)
    assert [len(split["foundation_train"]), len(split["utility_train"]),
            len(split["utility_dev"]), len(split["utility_eval"])] == \
        [80, 50, 15, 25]
    assert_event_disjoint(split)
    assert split == build_pilot_split(labels)
    # label balance is kept approximately
    for name in ("foundation_train", "utility_train", "utility_dev",
                 "utility_eval"):
        counts = split["label_counts"][name]
        assert abs(counts[0] - counts[1]) <= 2


def test_frozen_cutoffs_and_structural_scalar_shape(simple_chain):
    _event, snap = simple_chain
    assert cutoffs() == CUTOFFS_MIN
    rows = structural_scalars(snap, 360)
    assert len(rows) == len(snap["node_ids"])
    assert all(len(r) == 10 for r in rows)
    assert all(0.0 <= v <= 1.0 for r in rows for v in r)
