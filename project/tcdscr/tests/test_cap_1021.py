"""Unit tests: 1021-node cap determinism (plan §28 —
test_1021_cap_deterministic)."""
from ..data.snapshot_builder import build_snapshot
from .conftest import make_event


def _wide_event(n_replies):
    nodes = [("n0", None, 1000, "source", 0)]
    for i in range(1, n_replies + 1):
        nodes.append((f"n{i}", "n0", 1000 + i, f"reply {i}", i))
    return make_event(nodes)


def test_1021_cap_deterministic():
    event = _wide_event(1500)
    snap_a = build_snapshot(event, 60)
    snap_b = build_snapshot(event, 60)
    assert snap_a["cap_hit"] is True
    assert snap_a["num_nodes_before_cap"] == 1501
    assert snap_a["num_nodes_after_cap"] == 1021
    # repeated construction is bit-identical
    assert snap_a["node_ids"] == snap_b["node_ids"]
    assert snap_a["edge_index"] == snap_b["edge_index"]
    assert snap_a["depths"] == snap_b["depths"]
    # the kept nodes are exactly the earliest in deterministic order
    assert snap_a["node_ids"] == [f"n{i}" for i in range(1021)]
    # no randomness, no degree or label influence: reversing the raw node
    # order must not change the kept sequence (original_order pins it)
    shuffled = {"event_id": event["event_id"], "label": event["label"],
                "source_id": event["source_id"],
                "source_timestamp": event["source_timestamp"],
                "nodes": list(reversed(event["nodes"]))}
    snap_c = build_snapshot(shuffled, 60)
    assert snap_c["node_ids"] == snap_a["node_ids"]
