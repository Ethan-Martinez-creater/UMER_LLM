"""MS-TSR sufficiency primitives (Dynamic V3 design §5-§9).

V3 stops optimising agreement with the full-encoder teacher distribution and
instead asks whether a candidate evidence set still preserves the *Static
reader's decision* with enough room to spare:

  reference label       y_ref      = argmax q(S_static)          (§6)
  signed logit margin   m(S)       = l_{y_ref}(S) - l_{1-y_ref}(S)  (§7)
  static margin         m_static   = m(S_static)

  Sufficient(S; alpha)  <=>  argmax q(S) == y_ref                        (C1)
                        and  m(S) >= alpha * m_static                    (C2)

with the V3-only hyper-parameter alpha in {0.80, 0.90, 0.95, 1.00} (§8,
§16).  The Dual-View Consensus Gate (§6) allows compression only when the
two independent decision views already agree:

  argmax p_full == argmax q_static

otherwise the Static set is kept untouched.  No trainable parameter lives
here; nothing compares q(S) to p_full.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .set_fidelity import set_representation

NODE_REPR_DIM = 768
ALPHA_GRID = (0.80, 0.90, 0.95, 1.00)


def proxy_logits(proxy, h_source: torch.Tensor,
                 selected_node_repr: torch.Tensor) -> torch.Tensor:
    """Frozen proxy logits for [h_source ; mean(selected h)] (§7)."""
    z_sel = set_representation(selected_node_repr).to(h_source.device)
    return proxy.classify(h_source.float(), z_sel)


def batch_proxy_logits(proxy, h_source: torch.Tensor, node_repr: torch.Tensor,
                       masks: torch.Tensor) -> torch.Tensor:
    """Batched proxy logits for K sets at once (used by the ADD search).

    ``masks`` is (K, M) with 1.0 marking set membership; an all-zero row is
    the empty set (z_S = 0).  The encoder is never re-run here.
    """
    device = node_repr.device
    counts = masks.sum(dim=1)
    z = (masks.to(node_repr.dtype) @ node_repr.float()) \
        / counts.clamp_min(1.0).unsqueeze(1)
    z = z * (counts > 0).unsqueeze(1).to(z.dtype)
    k = masks.shape[0]
    hs = h_source.float().unsqueeze(0).expand(k, -1)
    return proxy.classify(hs, z)


def decision_margin(logits: torch.Tensor, ref_label: int) -> float:
    """Signed margin towards ``ref_label`` (positive = correct side)."""
    lg = logits.view(-1)
    return float(lg[ref_label] - lg[1 - ref_label])


def batch_decision_margins(logits: torch.Tensor,
                           ref_label: int) -> torch.Tensor:
    """(K, 2) logits -> (K,) signed margins towards ``ref_label``."""
    return logits[:, ref_label] - logits[:, 1 - ref_label]


def dual_view_agreement(p_full_probs: torch.Tensor,
                        q_static_probs: torch.Tensor) -> bool:
    """Dual-View Consensus Gate (§6): both decision views must agree."""
    return int(p_full_probs.view(-1).argmax()) == \
        int(q_static_probs.view(-1).argmax())


def sufficiency_conditions(logits: torch.Tensor, ref_label: int,
                           static_margin: float, alpha: float):
    """(sufficient, margin, c1) for one candidate set (§8).

    C1: argmax q(S) == y_ref; C2: m(S) >= alpha * m_static.
    """
    margin = decision_margin(logits, ref_label)
    c1 = int(logits.view(-1).argmax()) == ref_label
    c2 = margin >= alpha * static_margin
    return (c1 and c2), margin, c1


def reader_efficiency(delta_margin: float, token_cost: int) -> float:
    """Gain_i = Delta m_i / TokenCost(i) (§11)."""
    return delta_margin / max(int(token_cost), 1)
