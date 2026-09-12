"""Pilot event split (plan §5) — seed 7319, 80/50/15/25, label-balanced.

``foundation_train`` receives no required LLM intervention labels;
``utility_train`` trains the predictors; ``utility_dev`` controls early
stopping using training-reader utility only; ``utility_eval`` is the final
feasibility evaluation and may not be used for any hyperparameter choice.

All remaining events stay untouched and are returned separately so a caller
can never accidentally treat them as part of the pilot.
"""
from __future__ import annotations

import random
from collections import defaultdict

from ..config.pilot_config import PARTITION_SEED, SPLIT_SIZES, SPLIT_TOTAL


class SplitError(ValueError):
    """Raised when the event pool cannot satisfy the frozen split sizes."""


def _interleave_by_label(event_labels: dict, seed: int):
    """Deterministically interleave the two label classes.

    Shuffling each class independently and then dealing round-robin keeps the
    resulting sequence approximately label-balanced at every prefix, so any
    contiguous slice of the frozen sizes stays balanced "whenever possible"
    (plan §5) without a per-split quota search.
    """
    by_label = defaultdict(list)
    for eid in sorted(event_labels):
        by_label[event_labels[eid]].append(eid)
    rng = random.Random(seed)
    for label in sorted(by_label):
        rng.shuffle(by_label[label])
    order = sorted(by_label)
    sequence = []
    while any(by_label[label] for label in order):
        for label in order:
            if by_label[label]:
                sequence.append(by_label[label].pop())
    return sequence


def build_pilot_split(event_labels: dict, sizes=None,
                      seed: int = PARTITION_SEED) -> dict:
    """Split events into the four frozen pilot subsets plus ``unused``.

    Returns ``{"foundation_train": [...], "utility_train": [...],
    "utility_dev": [...], "utility_eval": [...], "unused": [...],
    "label_counts": {name: {0: n, 1: n}}}``.
    """
    sizes = dict(SPLIT_SIZES if sizes is None else sizes)
    total = sum(sizes.values())
    if len(event_labels) < total:
        raise SplitError(
            f"need at least {total} events for the pilot split, "
            f"only {len(event_labels)} available")
    unknown = [eid for eid in event_labels
               if event_labels[eid] not in (0, 1)]
    if unknown:
        raise SplitError(f"labels must be 0/1; offenders: {unknown[:5]}")

    sequence = _interleave_by_label(event_labels, seed)
    out = {}
    cursor = 0
    for name in ("foundation_train", "utility_train", "utility_dev",
                 "utility_eval"):
        n = sizes[name]
        out[name] = sorted(sequence[cursor:cursor + n])
        cursor += n
    out["unused"] = sorted(sequence[cursor:])
    out["label_counts"] = {
        name: {label: sum(1 for e in out[name] if event_labels[e] == label)
               for label in (0, 1)}
        for name in sizes
    }
    return out


def assert_event_disjoint(split: dict) -> None:
    """Every pilot subset must be pairwise event-disjoint (plan §5, §34)."""
    names = [n for n in split if n not in ("label_counts",)]
    seen = {}
    for name in names:
        for eid in split[name]:
            if eid in seen:
                raise SplitError(
                    f"event {eid!r} appears in both {seen[eid]!r} and {name!r}")
            seen[eid] = name


def split_summary(split: dict) -> dict:
    """Compact, JSON-safe summary of a resolved split."""
    return {
        "sizes": {n: len(split[n]) for n in
                  ("foundation_train", "utility_train", "utility_dev",
                   "utility_eval", "unused")},
        "label_counts": split["label_counts"],
        "total_planned": SPLIT_TOTAL,
    }
