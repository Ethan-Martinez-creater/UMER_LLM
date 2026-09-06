"""Selector training proxy (plan §18).

Training never calls the LLM. alpha = softmax(u / tau) with frozen tau=0.5;
z_sel = sum_i alpha_i h_i; proxy classifier Linear(1536,256) -> GELU ->
Dropout(0.1) -> Linear(256,2) over [z_sel; h_source].

Loss: L = L_cls + 1.0 * L_fid + 0.05 * L_div with
  L_cls = CE(p_sel, y)
  L_fid = KL(stopgrad(p_full) || p_sel)
  L_div = sum_{i != j} alpha_i alpha_j max(0, cos(e_i, e_j))
V2 adds no other loss term.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.schema import (LOSS_WEIGHT_DIV, LOSS_WEIGHT_FID,
                             NODE_REPR_DIM, PROXY_INPUT_DIM, TAU)


class SelectorProxy(nn.Module):

    def __init__(self, node_dim=NODE_REPR_DIM, input_dim=PROXY_INPUT_DIM,
                 hidden_dim=256, dropout=0.1, tau=TAU):
        super().__init__()
        self.tau = tau
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h_nodes, h_source, u, candidate_mask=None):
        """Return (alpha, z_sel, p_sel).

        u rows that are -inf (masked out, e.g. the source row if a caller
        passed one) get alpha exactly 0 through the softmax.
        """
        alpha = F.softmax(u / self.tau, dim=0)
        if candidate_mask is not None:
            alpha = alpha * candidate_mask.to(alpha.dtype)
            alpha = alpha / alpha.sum().clamp_min(1e-12)
        z_sel = alpha.unsqueeze(1) * h_nodes
        z_sel = z_sel.sum(dim=0)
        logits = self.head(torch.cat([z_sel, h_source], dim=-1))
        return alpha, z_sel, logits


def proxy_loss(p_sel, y, p_full, alpha, sem_nodes, candidate_mask=None):
    """Assemble L = L_cls + 1.0 L_fid + 0.05 L_div for one event snapshot."""
    l_cls = F.cross_entropy(p_sel.unsqueeze(0), y.unsqueeze(0))

    log_q = F.log_softmax(p_sel, dim=-1)
    p_ref = F.softmax(p_full.detach(), dim=-1)
    l_fid = -(p_ref * log_q).sum()  # KL(stopgrad(p_full) || p_sel)

    # cos(e_i, e_j); normalize defensively (MiniLM rows are already L2-normalized)
    sem = F.normalize(sem_nodes, dim=1)
    cos = sem @ sem.t()
    if candidate_mask is not None:
        keep = candidate_mask.to(cos.dtype)
        w = torch.outer(alpha * keep, alpha * keep)
    else:
        w = torch.outer(alpha, alpha)
    eye = torch.eye(cos.size(0), device=cos.device)
    pair = (w * (1.0 - eye) * cos.clamp_min(0.0)).sum()
    l_div = pair

    total = l_cls + LOSS_WEIGHT_FID * l_fid + LOSS_WEIGHT_DIV * l_div
    return total, {"l_cls": float(l_cls), "l_fid": float(l_fid),
                   "l_div": float(l_div)}
