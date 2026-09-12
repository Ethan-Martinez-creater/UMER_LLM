"""B0 — Text Only predictor (plan §17).

Inputs (no graph at all)::

    reply semantic 384
    source semantic 384
    semantic cosine
    token cost
    context size
    cutoff scalar

Outputs a continuous utility estimate and a HELPFUL/NEUTRAL/HARMFUL sign
logit vector, matching the B3 report surface so the comparison in §17/§25 P3
is like-for-like.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config.pilot_config import (BASELINE_HIDDEN, BUDGET_REF,
                                   SEMANTIC_DIM, SIGN_CLASSES,
                                   TEXT_FEATURE_DIM)


def text_features(e_reply, e_source, cosine_value: float, token_cost: int,
                  context_size: int, cutoff_minutes) -> list:
    """The plan §17 B0 input vector (772D)."""
    reply = e_reply.tolist() if hasattr(e_reply, "tolist") else list(e_reply)
    source = e_source.tolist() if hasattr(e_source, "tolist") else list(e_source)
    if len(reply) != SEMANTIC_DIM or len(source) != SEMANTIC_DIM:
        raise ValueError(f"semantic vectors must be {SEMANTIC_DIM}D")
    cutoff = max(float(cutoff_minutes), 1.0)
    return reply + source + [
        float(cosine_value),
        float(token_cost) / float(BUDGET_REF),
        float(context_size),
        math.log1p(cutoff) / math.log1p(360.0),
    ]


class TextOnlyPredictor(nn.Module):
    """B0 MLP head over the text-only feature vector."""

    def __init__(self, in_dim: int = TEXT_FEATURE_DIM,
                 hidden: int = BASELINE_HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
        )
        self.regress = nn.Linear(hidden // 2, 1)
        self.sign = nn.Linear(hidden // 2, len(SIGN_CLASSES))

    def forward(self, x):
        h = self.trunk(x)
        return self.regress(h).squeeze(-1), self.sign(h)
