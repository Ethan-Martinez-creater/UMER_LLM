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

    def classify(self, h_source, z_sel):
        """Frozen proxy classification contract: [h_source ; z_sel] -> logits.

        Both the soft training path and the hard validation/baseline path
        must end here; nothing may concatenate [z_sel ; h_source] or
        substitute event_repr for h_source.
        """
        return self.head(torch.cat([h_source, z_sel], dim=-1))

    def forward(self, h_nodes, h_source, u, candidate_mask=None):
        """Return (alpha, z_sel, logits).

        Soft path: alpha = softmax(u / tau), z_sel = sum_i alpha_i h_i,
        logits = classify(h_source, z_sel). u rows that are -inf (masked
        out, e.g. the source row if a caller passed one) get alpha exactly
        0 through the softmax.
        """
        alpha = F.softmax(u / self.tau, dim=0)
        if candidate_mask is not None:
            alpha = alpha * candidate_mask.to(alpha.dtype)
            alpha = alpha / alpha.sum().clamp_min(1e-12)
        z_sel = alpha.unsqueeze(1) * h_nodes
        z_sel = z_sel.sum(dim=0)
        logits = self.classify(h_source, z_sel)
        return alpha, z_sel, logits


def classify_selected(proxy, h_source, selected_node_repr, alpha=None):
    """Unified classification for selected evidence.

    alpha is None  -> hard path: z_sel = mean(selected h_i) (validation /
                      baseline arms);
    alpha provided  -> soft path: z_sel = sum_i alpha_i h_i (training).

    Both paths end in ``proxy.classify(h_source, z_sel)`` so train and
    evaluation share one classification contract.
    """
    if alpha is None:
        z_sel = selected_node_repr.mean(dim=0)
    else:
        z_sel = (alpha.unsqueeze(1) * selected_node_repr).sum(dim=0)
    return proxy.classify(h_source, z_sel)


def proxy_loss(p_sel, y, p_full, alpha, sem_nodes, candidate_mask=None):
    """Assemble L = L_cls + 1.0 L_fid + 0.05 L_div for one event snapshot."""
    l_cls = F.cross_entropy(p_sel.unsqueeze(0), y.unsqueeze(0))

    # true KL(stopgrad(p_full) || p_sel) = sum p_ref (log p_ref - log q);
    # numerically gradient-equivalent to the previous cross-entropy form
    # (p_ref is detached) but the logged value is now the actual divergence.
    p_ref = F.softmax(p_full.detach(), dim=-1)
    log_p_ref = F.log_softmax(p_full.detach(), dim=-1)
    log_q = F.log_softmax(p_sel, dim=-1)
    l_fid = (p_ref * (log_p_ref - log_q)).sum()

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
