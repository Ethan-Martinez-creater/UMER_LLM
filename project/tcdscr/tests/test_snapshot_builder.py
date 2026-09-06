"""Unit tests: causal snapshot builder (plan §28 Causality block)."""
from ..data.snapshot_builder import build_snapshot, build_source_only
from .conftest import make_event, simple_event


def test_source_only_exactly_one_node():
    event = simple_event()
    snap = build_source_only(event)
    assert len(snap["node_ids"]) == 1
    assert snap["node_ids"] == [event["source_id"]]
    assert snap["edge_index"] == []
    assert snap["cutoff_minutes"] == "SOURCE_ONLY"


def test_source_only_ignores_same_timestamp_nodes():
    # a node sharing the source's exact timestamp must NOT appear in
    # SOURCE_ONLY (§4.1: never emulate SOURCE_ONLY by timestamp filtering)
    event = make_event([
        ("n0", None, 1000, "source claim", 0),
        ("n1", "n0", 1000, "same-timestamp reply", 1),
    ])
    snap = build_source_only(event)
    assert snap["node_ids"] == ["n0"]


def test_snapshot_node_monotonicity():
    event = simple_event()
    # elapsed = ts - 1000: n1=100s, n3=500s, n4=1000s, n2=7200s
    small = build_snapshot(event, 10)   # 600s:  n0, n1, n3
    mid = build_snapshot(event, 60)     # 3600s: + n4
    big = build_snapshot(event, 360)    # 21600s: + n2
    s, m, b = set(small["node_ids"]), set(mid["node_ids"]), set(big["node_ids"])
    assert s == {"n0", "n1", "n3"}
    assert s <= m <= b
    assert m == {"n0", "n1", "n3", "n4"}
    assert "n2" not in m and "n2" in b


def test_snapshot_edge_monotonicity():
    event = simple_event()
    edges = {}
    for cut in (10, 60, 360):
        edges[cut] = set(map(tuple, build_snapshot(event, cut)["edge_index"]))
    assert edges[10] <= edges[60] <= edges[360]
    # n1 (idx 1) -> parent n0 (idx 0) is visible from 10 minutes on
    assert edges[10] == {(1, 0)}
    assert edges[60] == {(1, 0)}
    # n2 (idx 4 in the 6h order) -> n1 (idx 1) joins at 6h
    assert edges[360] == {(1, 0), (4, 1)}


def test_no_future_text():
    event = simple_event()
    snap = build_snapshot(event, 10)
    future_texts = [n["text"] for n in event["nodes"]
                    if n["timestamp"] > 1000 + 600]
    joined = " ".join(snap["texts"])
    for text in future_texts:
        assert text not in joined


def test_no_future_topology():
    event = simple_event()
    snap = build_snapshot(event, 10)
    inside = set(snap["node_ids"])
    for child, parent in snap["edge_index"]:
        assert snap["node_ids"][child] in inside
        assert snap["node_ids"][parent] in inside
    # only the n1->n0 edge is visible at 10 minutes; n3 has no resolved
    # parent and n2/n4 are future or externally-parented
    assert set(map(tuple, snap["edge_index"])) == {(1, 0)}
    # and at one hour still exactly the source->n1 edge exists
    snap60 = build_snapshot(event, 60)
    assert set(map(tuple, snap60["edge_index"])) == {(1, 0)}


def test_source_timestamp_as_t0():
    # an event containing an earlier-than-source (temporal invalid) node must
    # still compute elapsed time from the source timestamp (§8), not from the
    # event-wide minimum.
    event = make_event([
        ("n0", None, 1000, "source claim", 0),
        ("n1", "n0", 1100, "reply one", 1),
        ("n_bad", None, 500, "earlier than source", 2),
    ])
    # mark the early node explicitly like the adapter would
    event["nodes"][2]["status"] = "TEMPORAL_INVALID_NODE"
    snap = build_snapshot(event, 10)
    assert snap["elapsed_seconds"][snap["node_ids"].index("n1")] == 100
    assert "n_bad" not in snap["node_ids"]


def test_temporal_invalid_node_excluded():
    event = simple_event()
    event["nodes"][3]["status"] = "TEMPORAL_INVALID_NODE"  # n3
    for cut in (10, 60, 1440):
        snap = build_snapshot(event, cut)
        assert "n3" not in snap["node_ids"]
    # SOURCE_ONLY always still contains exactly the source
    assert build_source_only(event)["node_ids"] == [event["source_id"]]


def test_child_before_parent_edge_removed():
    # child at t=1100 whose parent appears at t=1200: the child stays (its own
    # text is causally visible) but the edge must not exist (§10.2/§14.2)
    event = make_event([
        ("n0", None, 1000, "source claim", 0),
        ("p1", "n0", 1200, "late parent", 2),
        ("c1", "p1", 1100, "early child", 1),
    ])
    snap = build_snapshot(event, 60)
    assert set(snap["node_ids"]) == {"n0", "p1", "c1"}
    idx = {nid: i for i, nid in enumerate(snap["node_ids"])}
    assert [idx["c1"], idx["p1"]] not in [list(e) for e in snap["edge_index"]]
    # the child's parent is unavailable downstream
    assert snap["parent_ids"][idx["c1"]] is None
    assert snap["parent_ids"][idx["p1"]] == "n0"


def test_cap_keeps_earliest_nodes():
    nodes = [("n0", None, 1000, "source", 0)]
    for i in range(1, 1100):
        nodes.append((f"n{i}", "n0", 1000 + i, f"reply {i}", i))
    event = make_event(nodes)
    snap = build_snapshot(event, 60)
    assert snap["cap_hit"] is True
    assert snap["num_nodes_before_cap"] == 1100
    assert snap["num_nodes_after_cap"] == 1021
    # earliest 1021 nodes: source + n1..n1020
    assert snap["node_ids"][:3] == ["n0", "n1", "n2"]
    assert snap["node_ids"][-1] == "n1020"
