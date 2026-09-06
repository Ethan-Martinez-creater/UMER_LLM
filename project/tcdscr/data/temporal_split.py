"""Frozen data-split protocols (plan §5).

Protocol A — event-level stratified 5-fold; folds are assigned to *events*
first, and every snapshot of an event inherits its event's fold forever.
Randomly partitioning snapshots is forbidden.

Protocol B — PHEME chronological 70/15/15 by absolute source timestamp
(temporal + emerging-topic stress test only).

Protocol C — Ma-Weibo chronological diagnostic tables only; no training,
no boundary tuning to manufacture class balance.
"""
from __future__ import annotations

import random
from collections import defaultdict

FOLD_FRACTIONS = (0.70, 0.15, 0.15)


def stratified_event_folds(event_labels: dict, k: int = 5,
                           seed: int = 3090) -> dict:
    """Assign each event_id a fold in [0, k) stratified by label.

    Deterministic: events are shuffled per class with the given seed, then
    dealt round-robin so folds stay label-balanced.
    """
    by_class = defaultdict(list)
    for eid in sorted(event_labels):
        by_class[event_labels[eid]].append(eid)
    rng = random.Random(seed)
    folds = {}
    for label in sorted(by_class):
        events = by_class[label][:]
        rng.shuffle(events)
        for i, eid in enumerate(events):
            folds[eid] = i % k
    return folds


def event_folds_to_snapshot_folds(snapshot_event_ids: dict) -> dict:
    """Propagate event folds to every snapshot key (§5.1 same-fold rule).

    ``snapshot_event_ids`` maps a snapshot key (event_id, cutoff) to its
    event_id; the result maps the same keys to fold indices.
    """
    return {key: eid for key, eid in snapshot_event_ids.items()}


def chronological_split(event_times: dict, fractions=FOLD_FRACTIONS) -> dict:
    """Protocol B: 70/15/15 by absolute source timestamp.

    ``event_times`` maps event_id -> source unix timestamp. Boundary ratios
    are frozen; class composition is reported as-is (never tuned).
    """
    ordered = sorted(event_times, key=lambda e: (event_times[e], e))
    n = len(ordered)
    a = int(n * fractions[0])
    b = int(n * (fractions[0] + fractions[1]))
    return {
        "train": ordered[:a],
        "validation": ordered[a:b],
        "test": ordered[b:],
    }


def chronological_diagnostic(event_records: list) -> dict:
    """Protocol C diagnostic table for Ma-Weibo chronological splits.

    ``event_records``: list of dicts with event_id/label/source_ts. Reports
    date and class distribution per segment — nothing more (plan §5.3, §45).
    """
    from datetime import datetime, timezone
    times = {r["event_id"]: r["source_ts"] for r in event_records}
    labels = {r["event_id"]: r["label"] for r in event_records}
    segs = chronological_split(times)
    out = {"fractions": list(FOLD_FRACTIONS), "segments": {}}
    for name, events in segs.items():
        rumors = sum(1 for e in events if labels[e] == 1)
        out["segments"][name] = {
            "events": len(events),
            "rumor": rumors,
            "nonrumor": len(events) - rumors,
            "rumor_ratio": rumors / max(len(events), 1),
            "ts_min_utc": datetime.fromtimestamp(
                times[events[0]], tz=timezone.utc).isoformat() if events else None,
            "ts_max_utc": datetime.fromtimestamp(
                times[events[-1]], tz=timezone.utc).isoformat() if events else None,
        }
    return out
