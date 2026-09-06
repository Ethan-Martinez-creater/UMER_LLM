"""1021D adjacency signature, rebuilt from the current snapshot only (§11).

Verbatim historical construction verified by the P4 parity audit:
``adj[parent, child] = 1`` plus a self-loop on the diagonal, each row divided
by its own out-degree (+1e-8), columns padded to 1021. Row order is the
deterministic snapshot order (timestamp ascending, original_order ties).
A full-graph signature masked down to G_t is forbidden — the matrix is built
from snapshot edges only, so a parent row's normalizer counts just its
currently visible children.
"""
from __future__ import annotations

import torch

from ..config.schema import MAX_NODES, STRUCT_ADJ_DIM


def build_adjacency_signature(snapshot: dict,
                              max_nodes: int = MAX_NODES) -> torch.Tensor:
    """Return the (N, 1021) float32 row-normalized adjacency signature."""
    n = len(snapshot["node_ids"])
    if n > max_nodes:
        raise ValueError(
            f"snapshot has {n} nodes, cap is {max_nodes}; snapshot_builder "
            "must truncate before feature construction")
    adj = torch.zeros(n, n, dtype=torch.float32)
    for child, parent in snapshot["edge_index"]:
        adj[parent, child] = 1.0
    adj.fill_diagonal_(1.0)
    out_degree = adj.sum(dim=1) + 1e-8
    adj_row = adj / out_degree.unsqueeze(1)
    if n < STRUCT_ADJ_DIM:
        pad = torch.zeros(n, STRUCT_ADJ_DIM - n, dtype=torch.float32)
        adj_row = torch.cat([adj_row, pad], dim=1)
    return adj_row
