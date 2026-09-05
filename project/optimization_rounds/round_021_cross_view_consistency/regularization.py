"""Prediction consistency across two distinct event text views."""

import torch


def cross_view_consistency_loss(view_logits: torch.Tensor) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence across event views."""
    if view_logits.ndim != 3 or view_logits.size(-1) != 2:
        raise ValueError("view_logits must have shape [batch, views, 2]")
    if view_logits.size(1) < 2:
        raise ValueError("cross-view consistency requires at least two views")
    log_probabilities = torch.log_softmax(view_logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    mixture = probabilities.mean(dim=1, keepdim=True).clamp_min(1e-8)
    divergence = probabilities * (log_probabilities - mixture.log())
    return divergence.sum(dim=-1).mean()
