"""Causal snapshot builder (plan §14, §8, §10, §11).

Time base: t0 = source/root timestamp (§8). A node enters a snapshot iff
``status != TEMPORAL_INVALID_NODE`` and ``timestamp <= t0 + cutoff`` (§14.1).
An edge is created iff child and parent are both included, the parent relation
is resolved inside the event, and ``parent_timestamp <= child_timestamp``
(§14.2) — otherwise the parent stays [UNAVAILABLE] downstream (§10.2).

Node order is deterministic: timestamp ascending, ties broken by the
adapter-provided ``original_order`` (§11). The 1021-node cap keeps the earliest
nodes in that order and never looks at degree, selector, or labels.
"""
from __future__ import annotations

from collections import deque

from ..config.schema import DEPTH_NORM_CONST, MAX_NODES


def _deterministic_order(event: dict):
    """All non-temporally-invalid nodes sorted by (timestamp, original_order)."""
    nodes = [n for n in event["nodes"]
             if n["status"] != "TEMPORAL_INVALID_NODE"]
    nodes.sort(key=lambda n: (n["timestamp"], n["original_order"]))
    return nodes


def _bfs_depth(num_nodes, edges, source_pos):
    """BFS depth from the source within snapshot edges; unreachable = -1."""
    depth = [-1] * num_nodes
    depth[source_pos] = 0
    children = [[] for _ in range(num_nodes)]
    for child, parent in edges:
        children[parent].append(child)
    q = deque([source_pos])
    while q:
        u = q.popleft()
        for v in children[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


def _assemble(event, cutoff_label, nodes, t0, max_nodes=MAX_NODES):
    """Build the snapshot dict for an already-ordered included node list."""
    num_before = len(nodes)
    if num_before > max_nodes:
        nodes = nodes[:max_nodes]
    num_after = len(nodes)

    pos = {n["node_id"]: i for i, n in enumerate(nodes)}
    source_id = event["source_id"]
    if source_id not in pos:
        raise ValueError(
            f"event {event['event_id']}: source missing from snapshot; "
            "SOURCE_ONLY and causal snapshots always contain the source")

    edges = []
    parent_ids = []
    for i, n in enumerate(nodes):
        p = None
        if n["node_id"] != source_id and n["status"] == "VALID" \
                and n["parent_id"] is not None \
                and n["parent_id"] in pos:
            parent = nodes[pos[n["parent_id"]]]
            # §14.2: the parent edge may not appear before its parent exists
            # in causal time either.
            if parent["timestamp"] <= n["timestamp"]:
                edges.append([i, pos[n["parent_id"]]])
                p = n["parent_id"]
        parent_ids.append(p)

    source_pos = pos[source_id]
    depths = _bfs_depth(num_after, edges, source_pos)

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
        "num_nodes_before_cap": num_before,
        "num_nodes_after_cap": num_after,
        "cap_hit": num_before > max_nodes,
        "max_depth": max(depths) if depths else 0,
        "unreachable_count": sum(1 for i, d in enumerate(depths)
                                 if d == -1 and i != source_pos),
        "depth_overflow_count": sum(1 for d in depths if d > DEPTH_NORM_CONST),
    }


def build_snapshot(event: dict, cutoff_minutes: int) -> dict:
    """All causally-visible nodes at ``t0 + cutoff_minutes``."""
    t0 = event["source_timestamp"]
    cutoff_seconds = int(cutoff_minutes) * 60
    nodes = [n for n in _deterministic_order(event)
             if n["timestamp"] <= t0 + cutoff_seconds]
    return _assemble(event, int(cutoff_minutes), nodes, t0)


def build_source_only(event: dict) -> dict:
    """Strictly the source node — never ``timestamp <= t0`` filtering (§4.1)."""
    t0 = event["source_timestamp"]
    source = next(n for n in event["nodes"]
                  if n["node_id"] == event["source_id"])
    return _assemble(event, "SOURCE_ONLY", [source], t0)
