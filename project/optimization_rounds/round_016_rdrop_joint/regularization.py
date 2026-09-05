"""R-Drop helpers for the unified graph/text/retrieval model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Mean bidirectional KL between two dropout-induced predictions."""
    log_a = F.log_softmax(logits_a, dim=-1)
    log_b = F.log_softmax(logits_b, dim=-1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    kl_ab = F.kl_div(log_a, prob_b, reduction="none").sum(dim=-1)
    kl_ba = F.kl_div(log_b, prob_a, reduction="none").sum(dim=-1)
    return 0.5 * (kl_ab + kl_ba).mean()


def rdrop_loss(
    model, outputs_a, outputs_b, labels, weight: float, class_weights=None
):
    """Average supervised branch losses and regularize final predictions."""
    losses_a = model.loss(outputs_a, labels, class_weights=class_weights)
    losses_b = model.loss(outputs_b, labels, class_weights=class_weights)
    losses = {
        name: 0.5 * (losses_a[name] + losses_b[name])
        for name in ("total", "final", "graph", "text", "cognitive")
    }
    consistency = symmetric_kl(outputs_a["logits"], outputs_b["logits"])
    losses["total"] = losses["total"] + float(weight) * consistency
    losses["consistency"] = consistency
    return losses
