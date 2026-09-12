"""Structural / time scalars for BiTTE node inputs (plan §13.1).

The ten frozen scalars are computed **strictly inside the current snapshot**
``G_t``. No future node, edge, subtree or intervention label may influence a
value at time ``t`` (plan §3.1.B). The returned values are plain Python floats
in ``[0, 1]`` so this module stays independent of torch.
"""
from __future__ import annotations

import math

from ..config.pilot_config import (CUTOFFS_MIN, DEPTH_NORM_CAP,
                                   STRUCT_SCALAR_DIM)

SCALAR_NAMES = (
    "depth_norm",
    "child_count_norm",
    "degree_norm",
    "subtree_size_norm",
    "sibling_count_norm",
    "is_source_child",
    "is_leaf",
    "elapsed_norm",
    "parent_lag_norm",
    "arrival_rank",
)


def _norm(count: int, n_nodes: int) -> float:
    return math.log1p(count) / math.log1p(max(n_nodes, 1))


def children_map(snapshot: dict):
    """``children[i]`` = indices whose edge parent is ``i`` (within G_t)."""
    n = len(snapshot["node_ids"])
    children = [[] for _ in range(n)]
    for child, parent in snapshot["edge_index"]:
        children[parent].append(child)
    return children


def parent_index_map(snapshot: dict):
    """``parent_of[i]`` = edge parent index or -1 (plan §3.1.B edges only)."""
    n = len(snapshot["node_ids"])
    parent_of = [-1] * n
    for child, parent in snapshot["edge_index"]:
        parent_of[child] = parent
    return parent_of


def _descendant_counts(children, n: int) -> list:
    """Subtree size (including self) per node — iterative post-order, O(N)."""
    counts = [1] * n
    done = [False] * n
    for root in range(n):
        if done[root]:
            continue
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                for c in children[node]:
                    counts[node] += counts[c]
                done[node] = True
                continue
            if done[node]:
                continue
            stack.append((node, True))
            for c in children[node]:
                if not done[c]:
                    stack.append((c, False))
    return counts


def structural_scalars(snapshot: dict, cutoff_minutes) -> list:
    """``[[s0..s9], ...]`` in snapshot node order (plan §13.1)."""
    n = len(snapshot["node_ids"])
    if n == 0:
        return []
    children = children_map(snapshot)
    parent_of = parent_index_map(snapshot)
    subs = _descendant_counts(children, n)
    depths = snapshot["depths"]
    elapsed = snapshot["elapsed_seconds"]
    timestamps = snapshot["timestamps"]
    source_id = snapshot["source_id"]
    node_ids = snapshot["node_ids"]
    cutoff_seconds = int(cutoff_minutes) * 60
    log_cutoff = math.log1p(max(cutoff_seconds, 1))

    rows = []
    for i in range(n):
        depth = depths[i] if depths[i] >= 0 else 0
        n_children = len(children[i])
        degree = n_children + (1 if parent_of[i] >= 0 else 0)
        siblings = 0
        if parent_of[i] >= 0:
            siblings = max(len(children[parent_of[i]]) - 1, 0)
        is_source_child = 1.0 if (parent_of[i] >= 0
                                  and node_ids[parent_of[i]] == source_id) else 0.0
        parent_lag = 0
        if parent_of[i] >= 0:
            parent_lag = max(timestamps[i] - timestamps[parent_of[i]], 0)
        arrival_rank = (i / (n - 1)) if n > 1 else 0.0
        rows.append([
            min(depth, DEPTH_NORM_CAP) / DEPTH_NORM_CAP,
            _norm(n_children, n),
            _norm(degree, n),
            _norm(subs[i], n),
            _norm(siblings, n),
            is_source_child,
            1.0 if n_children == 0 else 0.0,
            math.log1p(max(elapsed[i], 0)) / log_cutoff,
            math.log1p(parent_lag) / log_cutoff,
            arrival_rank,
        ])
    return rows


def scalars_dim() -> int:
    return STRUCT_SCALAR_DIM


def assert_valid_cutoff(cutoff) -> None:
    """A single cutoff must be one of the three frozen pilot cutoffs."""
    if int(cutoff) not in CUTOFFS_MIN:
        raise ValueError(
            f"cutoff {cutoff!r} is not one of the frozen cutoffs {CUTOFFS_MIN}")


def assert_frozen_cutoffs(cutoffs) -> None:
    """The full cutoff *set* must equal the frozen tuple (plan §6)."""
    if tuple(int(c) for c in cutoffs) != CUTOFFS_MIN:
        raise ValueError(
            f"CR-TSER cutoffs are frozen to {CUTOFFS_MIN}, got {tuple(cutoffs)}")
