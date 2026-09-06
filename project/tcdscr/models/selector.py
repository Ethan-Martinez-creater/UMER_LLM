"""Static utility selector (plan §17).

z_i = [h_i; g_t; r_i; struct3_i] (1540-D) with r_i = cos(e_i, e_source) and
struct3 = [norm_degree, norm_depth, norm_time]. Scorer is frozen:
LayerNorm(1540) -> Linear(1540,256) -> GELU -> Dropout(0.1) -> Linear(256,1).

No attention, no GNN, no Transformer, no LLM features may be added. The
source never participates in the candidate ranking: callers pass candidate
rows only, and a defensive ``candidate_mask`` lets a padded batch mark the
source row False so it can never be selected.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.schema import SELECTOR_INPUT_DIM


class StaticUtilitySelector(nn.Module):

    def __init__(self, input_dim=SELECTOR_INPUT_DIM, hidden_dim=256,
                 dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_nodes, event_repr, semantic_nodes, semantic_source,
                struct3, candidate_mask=None):
        """Score candidate replies.

        h_nodes (N,768), event_repr (768,), semantic nodes/source (N,384)/
        (384,), struct3 (N,3). Returns u_i (N,). Rows with
        ``candidate_mask == False`` receive -inf (never selectable).
        """
        if h_nodes.ndim != 2:
            raise ValueError("h_nodes must be (N, hidden)")
        n = h_nodes.size(0)
        relevance = F.cosine_similarity(
            semantic_nodes, semantic_source.unsqueeze(0).expand_as(semantic_nodes),
            dim=-1).unsqueeze(1)
        z = torch.cat([
            h_nodes,
            event_repr.unsqueeze(0).expand(n, -1),
            relevance,
            struct3,
        ], dim=-1)
        if z.size(1) != SELECTOR_INPUT_DIM:
            raise ValueError(
                f"selector input must be {SELECTOR_INPUT_DIM}, got {z.size(1)}")
        u = self.scorer(z).squeeze(-1)
        if candidate_mask is not None:
            u = u.masked_fill(~candidate_mask, float("-inf"))
        return u
