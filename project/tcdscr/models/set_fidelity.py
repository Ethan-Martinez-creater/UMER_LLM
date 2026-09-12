"""MF-TSR set-level fidelity objective (Dynamic V2 design §6-§9).

For an evidence set S inside the current causal snapshot G_t, the set
representation is the uniform mean of the frozen encoder node
representations (zeros(768) for the empty set), the proxy distribution is

    q(S) = softmax(Proxy.classify(h_source, z_S))

and the set distortion is the FORWARD KL

    D(S) = KL(stopgrad(p_full) || q(S)),

with p_full = softmax(full-snapshot encoder logits) computed from the
current G_t only.  The objective never sees gold labels, future graphs or
event-level truth: it compares the frozen teacher distribution against the
proxy distribution of the selected set only.  No trainable parameter lives
in this module.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NODE_REPR_DIM = 768  # encoder hidden dim (frozen proxy input is 2x this)


def set_representation(selected_node_repr: torch.Tensor) -> torch.Tensor:
    """z_S = mean(node_repr[i] for i in S); zeros(768) for the empty set."""
    if selected_node_repr is None or selected_node_repr.shape[0] == 0:
        return torch.zeros(NODE_REPR_DIM, dtype=torch.float32)
    return selected_node_repr.float().mean(dim=0)


def distortion_from_log_probs(p_full_probs: torch.Tensor,
                              logits: torch.Tensor) -> torch.Tensor:
    """KL(p_full || softmax(logits)) per row; p_full is stop-grad input."""
    p = p_full_probs.detach().float().clamp_min(1e-12)
    p = p / p.sum(dim=-1, keepdim=True)
    log_p = p.log()
    log_q = F.log_softmax(logits.float(), dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


def set_distortion(p_full_probs: torch.Tensor, h_source: torch.Tensor,
                   selected_node_repr: torch.Tensor, proxy):
    """Scalar D(S) for one evidence set through the frozen proxy head."""
    z_sel = set_representation(selected_node_repr).to(h_source.device)
    logits = proxy.classify(h_source.float(), z_sel)
    return float(distortion_from_log_probs(
        p_full_probs.view(1, -1).to(h_source.device), logits).item())


def batch_set_distortions(p_full_probs: torch.Tensor, h_source: torch.Tensor,
                          node_repr: torch.Tensor,
                          masks: torch.Tensor, proxy) -> torch.Tensor:
    """D(S_k) for K sets at once through one batched proxy forward.

    ``masks`` is (K, M) with 1.0 marking set members among ``node_repr``
    rows; an all-zero row denotes the empty set (z_S = 0).  The encoder is
    NEVER re-run here: only the small frozen proxy MLP is evaluated.
    """
    device = node_repr.device
    counts = masks.sum(dim=1)
    z = (masks.to(node_repr.dtype) @ node_repr.float()) \
        / counts.clamp_min(1.0).unsqueeze(1)
    z = z * (counts > 0).unsqueeze(1).to(z.dtype)
    k = masks.shape[0]
    hs = h_source.float().unsqueeze(0).expand(k, -1)
    logits = proxy.classify(hs, z)
    p = p_full_probs.detach().float().view(1, -1).to(device).expand(k, -1)
    return distortion_from_log_probs(p, logits)
