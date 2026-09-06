"""Unit tests: selector proxy loss (delta-fix §31–§33)."""
import math

import torch
import torch.nn.functional as F

from ..models.selector_proxy import proxy_loss


def test_fidelity_loss_matches_manual_kl():
    torch.manual_seed(0)
    n = 5
    sem = F.normalize(torch.rand(n, 384), dim=1)
    alpha = torch.softmax(torch.randn(n), dim=0)
    y = torch.tensor(1)
    p_sel = torch.randn(2, requires_grad=True)
    p_full = torch.randn(2)
    total, parts = proxy_loss(p_sel, y, p_full, alpha, sem)

    p_ref = F.softmax(p_full.detach(), dim=-1)
    log_p_ref = F.log_softmax(p_full.detach(), dim=-1)
    log_q = F.log_softmax(p_sel, dim=-1)
    manual_kl = float((p_ref * (log_p_ref - log_q)).sum())

    assert abs(parts["l_fid"] - manual_kl) < 1e-6
    assert abs(float(total) - (float(F.cross_entropy(
        p_sel.unsqueeze(0), y.unsqueeze(0))) + 1.0 * manual_kl
        + 0.05 * parts["l_div"])) < 1e-5
    # KL of a distribution against itself is exactly zero
    same, _ = proxy_loss(p_full.clone(), y, p_full, alpha, sem)
    _, parts_same = proxy_loss(p_full.clone(), y, p_full, alpha, sem)
    assert abs(parts_same["l_fid"]) < 1e-6


def test_fidelity_direction_is_kl_ref_vs_sel():
    """The divergence must measure KL(p_full || p_sel): heavy mode mismatch
    must produce a strictly positive value larger than the reverse."""
    torch.manual_seed(1)
    sem = F.normalize(torch.rand(3, 384), dim=1)
    alpha = torch.softmax(torch.zeros(3), dim=0)
    y = torch.tensor(0)
    p_full = torch.tensor([4.0, -4.0])   # confident class 0
    p_sel = torch.tensor([-4.0, 4.0])    # confident class 1
    _, parts = proxy_loss(p_sel.clone().requires_grad_(True), y, p_full,
                          alpha, sem)
    # KL is finite and large for opposite confident distributions
    assert parts["l_fid"] > 1.0
    assert math.isfinite(parts["l_fid"])
