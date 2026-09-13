"""CR-TSER causal snapshots (plan §3.1.B, §6).

CR-TSER keeps the TC-DSCR causal rule

    V_t = {v_i | timestamp_i <= t}

but owns its own assembly so the retired ``MAX_NODES=1021`` cap can never
truncate a snapshot. The cap existed for the 1021D positional adjacency
signature (plan §3.2), which CR-TSER does not use; a Weibo22 snapshot may
legitimately hold tens of thousands of nodes and every one of them must reach
BiTTE.

Only the *causal filtering* logic is shared with TC-DSCR (the
``TEMPORAL_INVALID_NODE`` exclusion and the parent-before-child edge rule);
the assembly itself is explicit and cap-free here.
"""
from __future__ import annotations

from collections import deque

from ..config.pilot_config import CUTOFFS_MIN
from .structural_stats import assert_frozen_cutoffs, assert_valid_cutoff

__all__ = ["build_causal_snapshot", "assert_causal", "reply_node_ids",
           "snapshot_has_replies", "audit_snapshot", "cutoffs",
           "MAX_NODES_CAP", "snapshot_node_count", "count_parent_cycles",
           "valid_reply_parent_units"]

#: CR-TSER never caps a snapshot. Kept as an explicit, testable value.
MAX_NODES_CAP = None

_DEPTH_OVERFLOW_CONST = 19  # mirrors TC-DSCR's depth-norm denominator (9.2)


def cutoffs():
    """The three frozen pilot cutoffs, in plan order (plan §6)."""
    assert_frozen_cutoffs(CUTOFFS_MIN)
    return tuple(CUTOFFS_MIN)


def _deterministic_order(event: dict):
    """Non-temporally-invalid nodes sorted by (timestamp, original_order).

    ``original_order`` is a **tie-break only** — never a substitute for time
    (plan §4.1).
    """
    nodes = [n for n in event["nodes"]
             if n["status"] != "TEMPORAL_INVALID_NODE"]
    nodes.sort(key=lambda n: (n["timestamp"], n["original_order"]))
    return nodes


def _bfs_depth(num_nodes, edges, source_pos):
    depth = [-1] * num_nodes
    depth[source_pos] = 0
    children = [[] for _ in range(num_nodes)]
    for child, parent in edges:
        children[parent].append(child)
    queue = deque([source_pos])
    while queue:
        node = queue.popleft()
        for child in children[node]:
            if depth[child] == -1:
                depth[child] = depth[node] + 1
                queue.append(child)
    return depth


def _assemble(event, cutoff_label, nodes, t0):
    """Assemble one snapshot from an already-filtered, ordered node list.

    There is **no** node cap: ``num_nodes_before_cap == num_nodes_after_cap``
    and ``cap_hit`` is always ``False``.
    """
    num_nodes = len(nodes)
    pos = {n["node_id"]: i for i, n in enumerate(nodes)}
    source_id = event["source_id"]
    if source_id not in pos:
        raise ValueError(
            f"event {event['event_id']}: source missing from snapshot; "
            "causal snapshots always contain the source")

    edges = []
    parent_ids = []
    for i, node in enumerate(nodes):
        parent = None
        if node["node_id"] != source_id and node["status"] == "VALID" \
                and node["parent_id"] is not None \
                and node["parent_id"] in pos:
            parent_node = nodes[pos[node["parent_id"]]]
            # §14.2: an edge may not exist before its parent does in causal time
            if parent_node["timestamp"] <= node["timestamp"]:
                edges.append([i, pos[node["parent_id"]]])
                parent = node["parent_id"]
        parent_ids.append(parent)

    source_pos = pos[source_id]
    depths = _bfs_depth(num_nodes, edges, source_pos)
    return {
        "event_id": event["event_id"],
        "label": event["label"],
        "cutoff_minutes": cutoff_label,
        "source_id": source_id,
        "node_ids": [n["node_id"] for n in nodes],
        "texts": [n["text"] for n in nodes],
        "timestamps": [n["timestamp"] for n in nodes],
        "elapsed_seconds": [n["timestamp"] - t0 for n in nodes],
        "parent_ids": parent_ids,
        "edge_index": edges,
        "depths": depths,
        "statuses": [n["status"] for n in nodes],
        "num_nodes_before_cap": num_nodes,
        "num_nodes_after_cap": num_nodes,
        "cap_hit": False,
        "max_nodes_cap": MAX_NODES_CAP,
        "max_depth": max(depths) if depths else 0,
        "unreachable_count": sum(1 for i, d in enumerate(depths)
                                 if d == -1 and i != source_pos),
        "depth_overflow_count": sum(1 for d in depths
                                    if d > _DEPTH_OVERFLOW_CONST),
    }


def build_causal_snapshot(event: dict, cutoff_minutes: int) -> dict:
    """All causally visible nodes at ``source_timestamp + cutoff`` (no cap)."""
    assert_valid_cutoff(cutoff_minutes)
    t0 = event["source_timestamp"]
    cutoff_seconds = int(cutoff_minutes) * 60
    nodes = [n for n in _deterministic_order(event)
             if n["timestamp"] <= t0 + cutoff_seconds]
    snap = _assemble(event, int(cutoff_minutes), nodes, t0)
    assert_causal(event, snap, cutoff_minutes)
    return snap


def snapshot_node_count(event: dict, cutoff_minutes: int) -> int:
    """Number of nodes a snapshot would contain — cheap, cap-free counter."""
    t0 = event["source_timestamp"]
    limit = t0 + int(cutoff_minutes) * 60
    return sum(1 for n in event["nodes"]
               if n["status"] != "TEMPORAL_INVALID_NODE"
               and n["timestamp"] <= limit)


def assert_causal(event: dict, snapshot: dict, cutoff_minutes: int) -> None:
    """Fail if any snapshot node lies beyond the cutoff (plan §3.1.B, §34)."""
    t0 = event["source_timestamp"]
    limit = t0 + int(cutoff_minutes) * 60
    for node_id, ts in zip(snapshot["node_ids"], snapshot["timestamps"]):
        if ts > limit:
            raise AssertionError(
                f"event {event['event_id']!r} cutoff {cutoff_minutes}m: node "
                f"{node_id!r} has timestamp {ts} > {limit} (future leak)")
    if event["source_id"] not in snapshot["node_ids"]:
        raise AssertionError("snapshot lost the source node")
    if snapshot.get("cap_hit"):
        raise AssertionError("CR-TSER snapshots must never be capped")


def reply_node_ids(snapshot: dict) -> list:
    """Visible non-source node ids in deterministic snapshot order."""
    return [nid for nid in snapshot["node_ids"]
            if nid != snapshot["source_id"]]


def snapshot_has_replies(snapshot: dict) -> bool:
    return len(reply_node_ids(snapshot)) > 0


def valid_reply_parent_units(event: dict) -> list:
    """``[(reply_ts, parent_ts), ...]`` for every real Reply–Parent unit.

    Amendment V2 §7: a unit requires a ``VALID`` reply with a resolved,
    in-event parent whose timestamp does not follow the reply's. Nodes with
    ``EMPTY_TEXT`` / ``MISSING_PARENT`` / ``EXTERNAL_PARENT`` /
    ``TEMPORAL_INVALID_NODE`` never produce one.
    """
    source_id = event["source_id"]
    by_id = {node["node_id"]: node for node in event["nodes"]}
    units = []
    for node in event["nodes"]:
        if node["node_id"] == source_id or node["status"] != "VALID":
            continue
        parent_id = node["parent_id"]
        if parent_id is None or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        if parent["timestamp"] <= node["timestamp"]:
            units.append((node["timestamp"], parent["timestamp"]))
    return units


def count_parent_cycles(event: dict) -> int:
    """Number of parent-pointer cycles among the valid nodes (V2 §8, §9).

    Only real, in-event parent links of ``VALID`` nodes are followed; an event
    with any cycle is not viable.
    """
    ids = {node["node_id"] for node in event["nodes"]}
    parent = {}
    for node in event["nodes"]:
        if node["status"] != "VALID" or node["node_id"] == event["source_id"]:
            continue
        pid = node["parent_id"]
        if pid is not None and pid in ids:
            parent[node["node_id"]] = pid
    colored = set()
    cycles = 0
    for start in sorted(parent):
        if start in colored:
            continue
        path, seen = [], set()
        current = start
        while current is not None and current not in colored \
                and current not in seen:
            seen.add(current)
            path.append(current)
            current = parent.get(current)
        if current in seen:
            cycles += 1
        colored.update(path)
    return cycles


def audit_snapshot(event: dict, snapshot: dict) -> dict:
    """Snapshot-level audit row (plan §6: zero-reply snapshots are audited)."""
    replies = reply_node_ids(snapshot)
    return {
        "event_id": event["event_id"],
        "cutoff": snapshot["cutoff_minutes"],
        "num_nodes": len(snapshot["node_ids"]),
        "num_replies": len(replies),
        "zero_reply": len(replies) == 0,
        "cap_hit": False,
        "max_nodes_cap": MAX_NODES_CAP,
        "max_depth": snapshot.get("max_depth"),
        "unreachable_count": snapshot.get("unreachable_count"),
        "edge_count": len(snapshot["edge_index"]),
    }
