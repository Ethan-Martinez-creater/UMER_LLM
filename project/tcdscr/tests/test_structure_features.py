"""Unit tests: structural features (plan §28 —
test_norm_degree_snapshot_internal, test_adj_signature_snapshot_only)."""
import torch

from ..data.snapshot_builder import build_snapshot
from ..data.structural_features import build_snapshot_features, structural_summary
from .conftest import make_event


def _star_event(children):
    nodes = [("n0", None, 1000, "source", 0)]
    for i in range(1, children + 1):
        nodes.append((f"c{i}", "n0", 1000 + i, f"reply {i}", i))
    return make_event(nodes)


def test_norm_degree_snapshot_internal():
    # root with 2 replies: snapshot max raw degree is 2 -> root norm = 1.0
    snap = build_snapshot(_star_event(2), 60)
    summary = structural_summary(snap)
    assert abs(summary[0, 0].item() - 1.0) < 1e-9
    assert abs(summary[1, 0].item()) < 1e-9

    # a branch node with one child scores 0.5 when the snapshot max is 2
    nodes = [("n0", None, 1000, "source", 0),
             ("c1", "n0", 1001, "branch reply", 1),
             ("c2", "n0", 1002, "flat reply", 2),
             ("g1", "c1", 1003, "grandchild", 3)]
    snap_b = build_snapshot(make_event(nodes), 60)
    summary_b = structural_summary(snap_b)
    idx = {nid: i for i, nid in enumerate(snap_b["node_ids"])}
    assert abs(summary_b[idx["c1"], 0].item() - 0.5) < 1e-9

    # inside a smaller snapshot (max raw degree 1) the same node scores 1.0:
    # normalization is snapshot-internal, never dataset/event-wide (§9.1)
    nodes_small = [("n0", None, 1000, "source", 0),
                   ("c1", "n0", 1001, "branch reply", 1),
                   ("g1", "c1", 1003, "grandchild", 2)]
    snap_s = build_snapshot(make_event(nodes_small), 60)
    summary_s = structural_summary(snap_s)
    idx_s = {nid: i for i, nid in enumerate(snap_s["node_ids"])}
    assert abs(summary_s[idx_s["c1"], 0].item() - 1.0) < 1e-9


def test_adj_signature_snapshot_only():
    # the signature is rebuilt from G_t edges only: a parent's row normalizer
    # counts just its currently visible children (§11)
    snap = build_snapshot(_star_event(2), 60)
    feats = build_snapshot_features(snap, torch.zeros(3, 384))
    sig = feats["struct_feat"][:, :1021]
    row = sig[0]                       # source row: 2 children + self = 3
    assert abs(row[0].item() - 1 / 3) < 1e-6
    assert abs(row[1].item() - 1 / 3) < 1e-6
    assert abs(row[2].item() - 1 / 3) < 1e-6
    assert abs(row.sum().item() - 1.0) < 1e-5

    # a future reply must not change the earlier snapshot's source row
    snap_later = build_snapshot(_star_event(3), 60)
    feats_later = build_snapshot_features(
        snap_later, torch.zeros(4, 384))
    row_later = feats_later["struct_feat"][0, :1021]
    assert abs(row_later[0].item() - 1 / 4) < 1e-6


def test_struct_feat_shape():
    snap = build_snapshot(_star_event(2), 60)
    feats = build_snapshot_features(snap, torch.zeros(3, 384))
    assert feats["node_feat"].shape == (3, 384)
    assert feats["struct_feat"].shape == (3, 1024)
    assert feats["source_id"] == "n0"
