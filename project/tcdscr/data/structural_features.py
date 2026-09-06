"""3D causal structural summary + snapshot feature assembly (plan §9, §15).

Fixed definitions:
  norm_degree — rawDegree = max(outDegree - 1, 0) computed inside the current
      snapshot and normalized by the snapshot-internal max (V2 §9.1; identical
      formula to the historical per-event rule, restricted to G_t).
  norm_depth  — BFS depth from the source divided by the frozen constant 19.
      depth > 19 is recorded as overflow and never clipped; unreachable nodes
      keep the historical -1 sentinel (norm = -1/19), counted in the manifest.
  norm_time   — timeBin = clip(floor(elapsed/1800), 0, 479), norm = bin/480.
"""
from __future__ import annotations

import torch

from ..config.schema import (BIN_SECONDS, DEPTH_NORM_CONST, MAX_BINS,
                             MAX_NODES, STRUCT_ADJ_DIM, STRUCT_SUMMARY_DIM)
from .adjacency_signature import build_adjacency_signature


def structural_summary(snapshot: dict) -> torch.Tensor:
    """Return the (N, 3) [norm_degree, norm_depth, norm_time] tensor."""
    n = len(snapshot["node_ids"])
    edges = snapshot["edge_index"]

    # count children per parent (out-degree without the self-loop)
    child_count = [0] * n
    for _child, parent in edges:
        child_count[parent] += 1

    raw = [float(max(c, 0)) for c in child_count]  # max(outDegree-1, 0)
    max_raw = max(raw) if raw else 0.0
    if max_raw > 0:
        norm_degree = [r / max_raw for r in raw]
    else:
        norm_degree = [0.0] * n

    norm_depth = [d / DEPTH_NORM_CONST for d in snapshot["depths"]]

    norm_time = []
    for elapsed in snapshot["elapsed_seconds"]:
        time_bin = int(min(max(elapsed // BIN_SECONDS, 0), MAX_BINS - 1))
        norm_time.append(time_bin / MAX_BINS)

    return torch.tensor([norm_degree, norm_depth, norm_time],
                        dtype=torch.float32).t()


def build_snapshot_features(snapshot: dict, semantic: torch.Tensor,
                            max_nodes: int = MAX_NODES) -> dict:
    """Assemble the Module-3 feature dict for one snapshot (plan §15)."""
    n = len(snapshot["node_ids"])
    if semantic.shape != (n, 384):
        raise ValueError(
            f"semantic matrix shape {tuple(semantic.shape)} does not match "
            f"snapshot node count {n}")
    adj = build_adjacency_signature(snapshot, max_nodes=max_nodes)
    summary = structural_summary(snapshot)
    struct_feat = torch.cat([adj, summary], dim=1)
    if struct_feat.shape[1] != STRUCT_ADJ_DIM + STRUCT_SUMMARY_DIM:
        raise ValueError("struct_feat must be (N, 1024)")
    edge_index = torch.tensor(snapshot["edge_index"], dtype=torch.long).t() \
        if snapshot["edge_index"] else torch.empty((2, 0), dtype=torch.long)
    return {
        "event_id": snapshot["event_id"],
        "node_feat": semantic.to(torch.float32),
        "struct_feat": struct_feat,
        "edge_index": edge_index,
        "node_ids": list(snapshot["node_ids"]),
        "source_id": snapshot["source_id"],
    }
