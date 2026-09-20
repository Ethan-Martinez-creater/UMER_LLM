"""B4 ZERO-TOUCH behaviorally conditioned utility model (M1 plan §10).

Evidence encoder over E0+E1+E2, reader encoder over the behavioral
fingerprint, interaction ``[e, r, e * r]``, two heads (continuous utility +
3-class sign). No reader-ID embedding anywhere — the reader is represented
**only** by its label-free behavioral fingerprint, which is what makes the
leave-one-reader-out evaluation honest (M1 plan §1, §10).
"""
from __future__ import annotations

from ..config import protocol as P
from .baselines import TrainRefused, count_parameters


def build_conditioned_net(evidence_dim: int, fingerprint_dim: int,
                          embed: int = P.B4_EMBED_DIM,
                          dropout: float = 0.0):
    """The frozen B4 architecture (target trainable parameters < 250k)."""
    try:
        import torch
        nn = torch.nn
    except ImportError as exc:  # pragma: no cover
        raise TrainRefused(f"torch unavailable: {exc}") from exc

    class _ConditionedUtilityNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.evidence_encoder = nn.Sequential(
                nn.Linear(evidence_dim, embed),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed, embed),
                nn.GELU(),
            )
            self.reader_encoder = nn.Sequential(
                nn.Linear(fingerprint_dim, embed),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed, embed),
                nn.GELU(),
            )
            self.utility_head = nn.Linear(3 * embed, 1)
            self.sign_head = nn.Linear(3 * embed, len(P.SIGN_CLASSES))

        def forward(self, x_e, x_r):
            e = self.evidence_encoder(x_e)
            r = self.reader_encoder(x_r)
            joint = torch.cat([e, r, e * r], dim=-1)
            return self.utility_head(joint).squeeze(-1), \
                self.sign_head(joint)

    model = _ConditionedUtilityNet()
    n_params = count_parameters(model)
    if n_params >= P.B4_MAX_PARAMETERS:
        raise TrainRefused(
            f"B4 has {n_params} parameters >= {P.B4_MAX_PARAMETERS}")
    return model
