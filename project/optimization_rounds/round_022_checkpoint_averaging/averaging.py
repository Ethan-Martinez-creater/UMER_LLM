"""Validation-only checkpoint ranking and parameter averaging.

This module contains no training or test-set access.  It is kept separate from
the Round 021 runner so preparing the next candidate cannot alter an active
formal five-fold run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


METRIC_NAMES = ("accuracy", "weighted_f1", "macro_f1", "rumor_f1")


def validation_key(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    """Match the strict checkpoint ordering used by the joint runner."""
    missing = [name for name in METRIC_NAMES if name not in metrics]
    if missing:
        raise KeyError(f"Missing validation metrics: {missing}")
    return (
        min(float(metrics[name]) for name in METRIC_NAMES),
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
    )


def top_k_indices(
    metrics_by_epoch: Sequence[Mapping[str, float]], k: int
) -> list[int]:
    """Return epoch-list indices ordered from strongest to weakest."""
    if k <= 0:
        raise ValueError("k must be positive")
    ranked = sorted(
        range(len(metrics_by_epoch)),
        key=lambda index: validation_key(metrics_by_epoch[index]),
        reverse=True,
    )
    return ranked[:k]


def average_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average floating tensors and require identical discrete tensors.

    The update is incremental in the original floating dtype, avoiding a
    float64 copy of a DeBERTa-large checkpoint.  Returned tensors are detached
    CPU clones and never alias an input checkpoint.
    """
    if not state_dicts:
        raise ValueError("At least one state_dict is required")
    expected_keys = tuple(state_dicts[0].keys())
    expected_key_set = set(expected_keys)
    for index, state in enumerate(state_dicts[1:], start=1):
        if set(state.keys()) != expected_key_set:
            raise ValueError(f"state_dict {index} has different keys")

    averaged: dict[str, torch.Tensor] = {}
    for name in expected_keys:
        first = state_dicts[0][name].detach().cpu()
        tensors = [state[name].detach().cpu() for state in state_dicts]
        if any(tensor.shape != first.shape for tensor in tensors[1:]):
            raise ValueError(f"Shape mismatch for parameter {name}")
        if any(tensor.dtype != first.dtype for tensor in tensors[1:]):
            raise ValueError(f"Dtype mismatch for parameter {name}")
        if first.is_floating_point() or first.is_complex():
            value = first.clone()
            for count, tensor in enumerate(tensors[1:], start=2):
                value.mul_((count - 1) / count).add_(tensor, alpha=1 / count)
            averaged[name] = value
        else:
            if any(not torch.equal(first, tensor) for tensor in tensors[1:]):
                raise ValueError(f"Discrete tensor mismatch for {name}")
            averaged[name] = first.clone()
    return averaged
