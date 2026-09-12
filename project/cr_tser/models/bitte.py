"""BiTTE — Bidirectional Temporal Tree Encoder (plan §13).

The old fixed 1021D positional adjacency signature is deliberately **not**
used (plan §13): BiTTE reads the frozen 384D semantic vector plus ten
current-snapshot structural/time scalars per node and runs exactly two
bidirectional message-passing layers over the snapshot's parent/child edges.

The module is padding-independent: padded nodes and padded edges are masked
out, so a sample's output never depends on other samples' padding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.pilot_config import (BITTE_DROPOUT, BITTE_HIDDEN, BITTE_LAYERS,
                                   BITTE_SEMANTIC_HIDDEN, BITTE_STRUCT_HIDDEN,
                                   SEMANTIC_DIM, STRUCT_SCALAR_DIM)


class MessageLayer(nn.Module):
    """One layer of plan §13.3 with its own W_self / W_parent / W_child."""

    def __init__(self, hidden: int = BITTE_HIDDEN, dropout: float = BITTE_DROPOUT):
        super().__init__()
        self.w_self = nn.Linear(hidden, hidden, bias=False)
        self.w_parent = nn.Linear(hidden, hidden, bias=False)
        self.w_child = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, parent_idx, edge_child, edge_parent, edge_mask, mask):
        b, n, d = h.shape
        m_self = self.w_self(h)
        m_parent = torch.zeros_like(h)
        has_parent = parent_idx >= 0
        if has_parent.any():
            idx = parent_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, d)
            gathered = torch.gather(h, 1, idx)
            m_parent = self.w_parent(gathered) * has_parent.unsqueeze(-1).float()
        m_child = torch.zeros_like(h)
        if edge_child.numel():
            flat_parent = (edge_parent + torch.arange(
                b, device=h.device).unsqueeze(1) * n).reshape(-1)
            flat_child = (edge_child + torch.arange(
                b, device=h.device).unsqueeze(1) * n).reshape(-1)
            flat_mask = edge_mask.reshape(-1)
            flat_parent = flat_parent[flat_mask]
            flat_child = flat_child[flat_mask]
            if flat_child.numel():
                h_flat = h.reshape(b * n, d)
                child_msgs = self.w_child(h_flat)
                agg = torch.zeros(b * n, d, device=h.device, dtype=h.dtype)
                cnt = torch.zeros(b * n, 1, device=h.device, dtype=h.dtype)
                agg.index_add_(0, flat_parent, child_msgs[flat_child])
                cnt.index_add_(0, flat_parent,
                               torch.ones(flat_child.shape[0], 1,
                                          device=h.device, dtype=h.dtype))
                agg = agg / cnt.clamp(min=1.0)
                m_child = agg.reshape(b, n, d)
        tilde = F.gelu(m_self + m_parent + m_child)
        out = self.norm(h + self.dropout(tilde))
        return out * mask.unsqueeze(-1).float()


class BiTTE(nn.Module):
    """Bidirectional Temporal Tree Encoder (plan §13)."""

    def __init__(self, semantic_dim: int = SEMANTIC_DIM,
                 struct_dim: int = STRUCT_SCALAR_DIM,
                 hidden: int = BITTE_HIDDEN,
                 layers: int = BITTE_LAYERS,
                 dropout: float = BITTE_DROPOUT):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.struct_dim = struct_dim
        self.hidden = hidden
        self.semantic = nn.Sequential(
            nn.Linear(semantic_dim, BITTE_SEMANTIC_HIDDEN),
            nn.LayerNorm(BITTE_SEMANTIC_HIDDEN),
            nn.GELU(),
        )
        self.structure = nn.Sequential(
            nn.Linear(struct_dim, BITTE_STRUCT_HIDDEN),
            nn.GELU(),
            nn.Linear(BITTE_STRUCT_HIDDEN, BITTE_STRUCT_HIDDEN),
            nn.LayerNorm(BITTE_STRUCT_HIDDEN),
        )
        concat_dim = BITTE_SEMANTIC_HIDDEN + BITTE_STRUCT_HIDDEN  # 320
        self.project = nn.Sequential(
            nn.Linear(concat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [MessageLayer(hidden, dropout) for _ in range(layers)])

    def forward(self, sem, struct, mask, parent_idx, edge_child, edge_parent,
                edge_mask, source_idx):
        """Encode one batch of snapshots.

        ``sem`` (B,N,384), ``struct`` (B,N,10), ``mask`` (B,N) bool,
        ``parent_idx`` (B,N) long with -1 for "none", edges (B,E) long with an
        ``edge_mask`` (B,E) bool, ``source_idx`` (B,) long.
        Returns ``(h, h_src)`` with shapes (B,N,256) and (B,256).
        """
        mask_f = mask.unsqueeze(-1).float()
        h = self.project(torch.cat([self.semantic(sem), self.structure(struct)],
                                   dim=-1))
        h = h * mask_f
        for layer in self.layers:
            h = layer(h, parent_idx, edge_child, edge_parent, edge_mask, mask)
        h = h * mask_f
        b = h.shape[0]
        h_src = h[torch.arange(b, device=h.device), source_idx]
        return h, h_src

    def output_dim(self) -> int:
        return self.hidden


def pack_graph_batch(graphs, device="cpu"):
    """Pad a list of per-snapshot graph dicts into BiTTE batch tensors.

    Each graph dict needs keys ``sem`` (N,384), ``struct`` (N,10),
    ``parent_idx`` (N,), ``edges`` (E,2 child,parent) and ``source_idx``.
    """
    b = len(graphs)
    n_max = max(g["sem"].shape[0] for g in graphs)
    e_max = max((g["edges"].shape[0] for g in graphs), default=0)
    sem = torch.zeros(b, n_max, SEMANTIC_DIM, device=device)
    struct = torch.zeros(b, n_max, STRUCT_SCALAR_DIM, device=device)
    mask = torch.zeros(b, n_max, dtype=torch.bool, device=device)
    parent = torch.full((b, n_max), -1, dtype=torch.long, device=device)
    edge_child = torch.zeros(b, max(e_max, 1), dtype=torch.long, device=device)
    edge_parent = torch.zeros(b, max(e_max, 1), dtype=torch.long, device=device)
    edge_mask = torch.zeros(b, max(e_max, 1), dtype=torch.bool, device=device)
    source_idx = torch.zeros(b, dtype=torch.long, device=device)
    for i, g in enumerate(graphs):
        n = g["sem"].shape[0]
        sem[i, :n] = g["sem"].to(device)
        struct[i, :n] = g["struct"].to(device)
        mask[i, :n] = True
        parent[i, :n] = g["parent_idx"].to(device)
        e = g["edges"].shape[0]
        if e:
            edge_child[i, :e] = g["edges"][:, 0].to(device)
            edge_parent[i, :e] = g["edges"][:, 1].to(device)
            edge_mask[i, :e] = True
        source_idx[i] = g["source_idx"]
    return {"sem": sem, "struct": struct, "mask": mask, "parent_idx": parent,
            "edge_child": edge_child, "edge_parent": edge_parent,
            "edge_mask": edge_mask, "source_idx": source_idx}
