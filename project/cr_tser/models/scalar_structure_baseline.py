"""B1 — Text + Scalar Structure/Time predictor (plan §17).

B0 plus the ten fixed structural/time scalars, with **no** message passing.
This is the baseline that isolates whether BiTTE's learned propagation
encoding adds anything over simply handing a generic model the same scalars.
"""
from __future__ import annotations

from ..config.pilot_config import (BASELINE_HIDDEN, SCALAR_STRUCTURE_FEATURE_DIM,
                                   STRUCT_SCALAR_DIM)
from .text_baseline import TextOnlyPredictor


def scalar_structure_features(text_feature, structural_scalars) -> list:
    """B0 features concatenated with the ten §13.1 scalars (782D)."""
    scalars = list(structural_scalars)
    if len(scalars) != STRUCT_SCALAR_DIM:
        raise ValueError(f"expected {STRUCT_SCALAR_DIM} scalars, "
                         f"got {len(scalars)}")
    return list(text_feature) + [float(s) for s in scalars]


class ScalarStructurePredictor(TextOnlyPredictor):
    """B1: same head shape as B0, wider input, still no graph encoding."""

    def __init__(self, in_dim: int = SCALAR_STRUCTURE_FEATURE_DIM,
                 hidden: int = BASELINE_HIDDEN, dropout: float = 0.1):
        super().__init__(in_dim=in_dim, hidden=hidden, dropout=dropout)
