"""Atomic utility representation and shared + reader-residual heads (plan §14–§15).

``z_i = [h_i; h_src; h_ctx; h_i* h_ctx; |h_i - h_ctx|; q_i]`` has dimension
``256*5 + 6 = 1286`` and is projected to a 256D utility embedding.

The utility model is ``u_hat_r(i) = mu_i + delta_{r,i}``: a shared head over the
projection plus a reader-residual head conditioned on a learned 16D reader
embedding. The held-out reader has **no** embedding, so at transfer time only
``mu`` (and the shared-only arm) is available — this is what makes the
leave-one-reader-out protocol honest (plan §15, §20).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config.pilot_config import (BITTE_HIDDEN, BUDGET_REF, LOSS_W_READER,
                                   LOSS_W_RESID, LOSS_W_SHARED, LOSS_W_SIGN,
                                   READER_EMBED_DIM, RESIDUAL_HEAD_HIDDEN,
                                   SHARED_HEAD_HIDDEN, SIGN_CLASSES,
                                   UTILITY_PROJ_HIDDEN, UTILITY_Q_DIM,
                                   UTILITY_THRESHOLD, UTILITY_Z_DIM)

SIGN_TO_INDEX = {name: i for i, name in enumerate(SIGN_CLASSES)}


def q_vector(cosine_value: float, token_cost: int, rank_percentile: float,
             cutoff_minutes, selected_count: int, token_utilization: float):
    """The six frozen scalars of plan §14, all bounded to a stable range."""
    cutoff = max(float(cutoff_minutes), 1.0)
    return [
        float(cosine_value),
        float(token_cost) / float(BUDGET_REF),
        float(rank_percentile),
        math.log1p(cutoff) / math.log1p(360.0),
        min(float(selected_count), 20.0) / 20.0,
        float(token_utilization),
    ]


def build_atomic_z(h, h_src, h_ctx, q):
    """Assemble ``z_i`` for every node (plan §14). Shapes: (N,256)/(N,6)."""
    if h.shape[0] != q.shape[0]:
        raise ValueError("h and q must share the node dimension")
    if h_ctx.dim() == 1:
        h_ctx = h_ctx.unsqueeze(0).expand_as(h)
    prod = h * h_ctx
    diff = torch.abs(h - h_ctx)
    h_src_rep = h_src.unsqueeze(0).expand_as(h)
    z = torch.cat([h, h_src_rep, h_ctx, prod, diff, q], dim=-1)
    if z.shape[-1] != UTILITY_Z_DIM:
        raise ValueError(
            f"z_i must be {UTILITY_Z_DIM}D, got {z.shape[-1]}")
    return z


class AtomicUtilityEncoder(nn.Module):
    """``1286 -> 256 -> 256`` with GELU / Dropout(0.1) / LayerNorm (§14)."""

    def __init__(self, in_dim: int = UTILITY_Z_DIM,
                 hidden: int = UTILITY_PROJ_HIDDEN,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )

    def forward(self, z):
        return self.net(z)


class SharedResidualUtility(nn.Module):
    """Shared head + per-training-reader residual + auxiliary sign head (§15)."""

    def __init__(self, n_training_readers: int,
                 hidden: int = UTILITY_PROJ_HIDDEN,
                 embed_dim: int = READER_EMBED_DIM,
                 dropout: float = 0.1):
        super().__init__()
        if n_training_readers < 1:
            raise ValueError("need at least one training reader")
        self.n_training_readers = n_training_readers
        self.encoder = AtomicUtilityEncoder(hidden=hidden, dropout=dropout)
        self.shared = nn.Sequential(
            nn.Linear(hidden, SHARED_HEAD_HIDDEN),
            nn.GELU(),
            nn.Linear(SHARED_HEAD_HIDDEN, 1),
        )
        self.reader_embed = nn.Embedding(n_training_readers, embed_dim)
        self.residual = nn.Sequential(
            nn.Linear(hidden + embed_dim, RESIDUAL_HEAD_HIDDEN),
            nn.GELU(),
            nn.Linear(RESIDUAL_HEAD_HIDDEN, 1),
        )
        self.sign = nn.Sequential(
            nn.Linear(hidden + embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, len(SIGN_CLASSES)),
        )

    def encode(self, z):
        return self.encoder(z)

    def shared_mu(self, z):
        return self.shared(self.encode(z)).squeeze(-1)

    def forward(self, z, reader_index):
        """``(mu, delta, u_hat, sign_logits)`` for one reader (plan §15)."""
        emb = self.encode(z)
        mu = self.shared(emb).squeeze(-1)
        r = self.reader_embed(reader_index)
        if r.dim() == 1:
            r = r.unsqueeze(0).expand(emb.shape[0], -1)
        joint = torch.cat([emb, r], dim=-1)
        delta = self.residual(joint).squeeze(-1)
        logits = self.sign(joint)
        return mu, delta, mu + delta, logits

    def shared_only(self, z, reader_index):
        """μ plus a zero residual — used for the S4 shared-only arm."""
        emb = self.encode(z)
        mu = self.shared(emb).squeeze(-1)
        logits = self.sign(torch.cat(
            [emb, self.reader_embed(reader_index).unsqueeze(0).expand(
                emb.shape[0], -1)], dim=-1))
        return mu, logits


def utility_loss(u_hat, u_target, mu, shared_target, sign_logits, sign_target,
                 delta):
    """The frozen plan §16 objective (weights 1.0 / 0.5 / 0.5 / 0.01)."""
    reader = torch.nn.functional.smooth_l1_loss(u_hat, u_target)
    shared = torch.nn.functional.smooth_l1_loss(mu, shared_target)
    sign = torch.nn.functional.cross_entropy(sign_logits, sign_target)
    resid = (delta ** 2).mean()
    total = (LOSS_W_READER * reader + LOSS_W_SHARED * shared
             + LOSS_W_SIGN * sign + LOSS_W_RESID * resid)
    return total, {"reader": float(reader.detach()),
                   "shared": float(shared.detach()),
                   "sign": float(sign.detach()),
                   "resid": float(resid.detach())}


def tri_class_label(utility: float, correct_before: bool, correct_after: bool):
    """Plan §10.1 tri-class label with correctness transitions overriding."""
    if correct_before and not correct_after:
        return "HELPFUL"
    if not correct_before and correct_after:
        return "HARMFUL"
    if utility >= UTILITY_THRESHOLD:
        return "HELPFUL"
    if utility <= -UTILITY_THRESHOLD:
        return "HARMFUL"
    return "NEUTRAL"


def utility_of(p_before: float, p_after: float) -> float:
    """``u_r(A) = p_r(y|C_ref) - p_r(y|C_ref \\ A)`` (plan §10)."""
    return float(p_before) - float(p_after)


def utility_record(gold_label: int, before_out: dict, after_out: dict) -> dict:
    """Full per-intervention utility record (plan §10, §32).

    ``before_out`` / ``after_out`` are :func:`sequence_scorer.ab_scores`
    outputs. Gold is used only to build utility supervision.
    """
    from ..readers.sequence_scorer import gold_probability, is_correct
    p_before = gold_probability(before_out, gold_label)
    p_after = gold_probability(after_out, gold_label)
    correct_before = is_correct(before_out, gold_label)
    correct_after = is_correct(after_out, gold_label)
    utility = utility_of(p_before, p_after)
    return {
        "score_A_before": before_out["score_A"],
        "score_B_before": before_out["score_B"],
        "score_A_after": after_out["score_A"],
        "score_B_after": after_out["score_B"],
        "p_rumor_before": before_out["p_rumor"],
        "p_nonrumor_before": before_out["p_nonrumor"],
        "p_rumor_after": after_out["p_rumor"],
        "p_nonrumor_after": after_out["p_nonrumor"],
        "prediction_before": before_out["prediction"],
        "prediction_after": after_out["prediction"],
        "correct_before": correct_before,
        "correct_after": correct_after,
        "label_flip": before_out["prediction"] != after_out["prediction"],
        "gold_probability_before": p_before,
        "gold_probability_after": p_after,
        "utility": utility,
        "sign": tri_class_label(utility, correct_before, correct_after),
    }
