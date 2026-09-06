"""Unit tests: time features (plan §28 — test_norm_depth_div19,
test_norm_time_30min)."""
from ..data.snapshot_builder import build_snapshot
from ..data.structural_features import structural_summary
from .conftest import make_event


def _chain_event(depth):
    """A linear chain of the given depth below the source, one node/minute."""
    nodes = [("n0", None, 1000, "source", 0)]
    prev = "n0"
    for i in range(1, depth + 1):
        nid = f"d{i}"
        nodes.append((nid, prev, 1000 + i * 60, f"chain {i}", i))
        prev = nid
    return make_event(nodes)


def test_norm_depth_div19():
    event = _chain_event(3)
    snap = build_snapshot(event, 60)
    summary = structural_summary(snap)
    idx = {nid: i for i, nid in enumerate(snap["node_ids"])}
    assert abs(summary[idx["n0"], 1].item() - 0.0) < 1e-6
    assert abs(summary[idx["d1"], 1].item() - 1 / 19) < 1e-6
    assert abs(summary[idx["d3"], 1].item() - 3 / 19) < 1e-6


def test_norm_depth_overflow_not_clipped():
    event = _chain_event(21)
    snap = build_snapshot(event, 60)
    assert snap["depth_overflow_count"] == 2   # depths 20 and 21 exceed 19
    summary = structural_summary(snap)
    idx = {nid: i for i, nid in enumerate(snap["node_ids"])}
    assert abs(summary[idx["d21"], 1].item() - 21 / 19) < 1e-6


def test_norm_time_30min():
    nodes = [
        ("n0", None, 1000, "source", 0),
        ("a", "n0", 1000 + 1799, "just before 30m bin", 1),
        ("b", "n0", 1000 + 1800, "exactly 30m", 2),
        ("c", "n0", 1000 + 1800 * 479, "last bin", 3),
        ("d", "n0", 1000 + 1800 * 479 + 1799, "inside last bin", 4),
        ("e", "n0", 1000 + 400 * 3600, "way beyond horizon", 5),
    ]
    event = make_event(nodes)
    snap = build_snapshot(event, 400 * 60)  # large enough to include node e
    summary = structural_summary(snap)
    idx = {nid: i for i, nid in enumerate(snap["node_ids"])}
    assert abs(summary[idx["n0"], 2].item()) < 1e-6
    assert abs(summary[idx["a"], 2].item() - 0 / 480) < 1e-6
    assert abs(summary[idx["b"], 2].item() - 1 / 480) < 1e-6
    assert abs(summary[idx["c"], 2].item() - 479 / 480) < 1e-6
    assert abs(summary[idx["d"], 2].item() - 479 / 480) < 1e-6
    # 400h exceeds the 480-bin (240h) horizon: clipped to 479, never negative
    assert abs(summary[idx["e"], 2].item() - 479 / 480) < 1e-6
