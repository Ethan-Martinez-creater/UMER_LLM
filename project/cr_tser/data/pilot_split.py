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

from ..config.pilot_config import (CUTOFFS_MIN, PARTITION_SEED, SPLIT_SIZES,
                                   SPLIT_TOTAL)
from .snapshot_bridge import count_parent_cycles


class SplitError(ValueError):
    """Raised when the event pool cannot satisfy the frozen split sizes."""


def viable_event_ids(events, cutoffs=CUTOFFS_MIN):
    """Event ids able to produce a valid Reply–Parent unit (plan §5, V2 §8).

    This filter must run **before** the split: a label registry can reference
    events whose released data yields nothing, and sampling from it first
    would put unusable events into the pilot.

    Amendment V2 §7/§8 tightens the rule: a usable reply must be ``VALID``
    (not ``EMPTY_TEXT`` / ``MISSING_PARENT`` / ``EXTERNAL_PARENT`` /
    ``TEMPORAL_INVALID_NODE``), it must have a resolved in-event parent, and
    both must fall inside the same causal cutoff. Only true timestamps and the
    causal rule are used — never row/node order.
    """
    viable = []
    for event in events:
        t0 = event["source_timestamp"]
        by_id = {node["node_id"]: node for node in event["nodes"]}
        source = by_id.get(event["source_id"])
        if source is None or source["status"] != "VALID":
            continue
        if count_parent_cycles(event) > 0:
            continue
        units = []
        for node in event["nodes"]:
            if node["node_id"] == event["source_id"] \
                    or node["status"] != "VALID":
                continue
            parent_id = node["parent_id"]
            if parent_id is None or parent_id not in by_id:
                continue
            parent = by_id[parent_id]
            if parent["timestamp"] <= node["timestamp"]:
                units.append((node["timestamp"], parent["timestamp"]))
        for cutoff in cutoffs:
            limit = t0 + int(cutoff) * 60
            if any(reply_ts <= limit and parent_ts <= limit
                   for reply_ts, parent_ts in units):
                viable.append(event["event_id"])
                break
    return viable


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
