"""Frozen data-split protocols (plan §5; delta-fix §4–§8).

Protocol A — event-level stratified 5-fold with an inner stratified
validation split; folds are assigned to *events* first, and every snapshot of
an event inherits its event's fold forever. Randomly partitioning snapshots
is forbidden.

Protocol B — PHEME chronological 70/15/15 by absolute source timestamp
(temporal + emerging-topic stress test only).

Protocol C — Ma-Weibo chronological diagnostic tables only; no training,
no boundary tuning to manufacture class balance.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

FOLD_FRACTIONS = (0.70, 0.15, 0.15)
N_OUTER_FOLDS = 5
PARTITION_SEED = 3090
VALIDATION_FRACTION = 0.10


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


def event_folds_to_snapshot_folds(snapshot_event_ids: dict,
                                  event_folds: dict) -> dict:
    """Propagate event folds to every snapshot key (§5.1 same-fold rule).

    ``snapshot_event_ids`` maps a snapshot key (event_id, cutoff) to its
    event_id; ``event_folds`` maps event_id -> fold id. The result maps each
    snapshot key to its **fold id** — never the event id. An event missing
    from ``event_folds`` raises instead of being silently dropped.
    """
    out = {}
    for key, eid in snapshot_event_ids.items():
        if eid not in event_folds:
            raise ValueError(
                f"snapshot {key!r} references event {eid!r} which has no "
                "fold assignment")
        out[key] = event_folds[eid]
    return out


def build_primary_fold_split(event_labels: dict, fold_index: int,
                             seed: int = PARTITION_SEED,
                             validation_fraction: float = VALIDATION_FRACTION,
                             n_splits: int = N_OUTER_FOLDS) -> dict:
    """Formal Protocol A split for one outer fold (delta-fix §6).

    Outer split: ``StratifiedKFold(n_splits=5, shuffle=True,
    random_state=3090)`` over events; fold ``fold_index`` is the outer test
    set. The remaining four folds are re-split with a stratified
    ``validation_fraction`` (10%) draw (same random_state) to obtain the
    inner validation set. Splits operate on event ids only — snapshots are
    never split independently.

    Returns {"train": [...], "validation": [...], "test": [...]} with the
    three sets pairwise disjoint and covering all events.
    """
    if fold_index not in range(n_splits):
        raise ValueError(f"fold_index must be in [0, {n_splits}), "
                         f"got {fold_index}")
    event_ids = sorted(event_labels)
    labels = np.array([event_labels[e] for e in event_ids])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=seed)
    splits = list(skf.split(np.zeros(len(event_ids)), labels))
    test_idx = splits[fold_index][1]
    trainval_idx = splits[fold_index][0]

    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=validation_fraction,
                                 random_state=seed)
    # sklearn returns (train_index, test_index): the 10% test side is the
    # inner validation set
    train_rel, val_rel = next(sss.split(np.zeros(len(trainval_idx)),
                                        labels[trainval_idx]))
    val_idx = trainval_idx[val_rel]
    train_idx = trainval_idx[train_rel]

    train = [event_ids[i] for i in sorted(train_idx)]
    validation = [event_ids[i] for i in sorted(val_idx)]
    test = [event_ids[i] for i in sorted(test_idx)]

    train_set, val_set, test_set = set(train), set(validation), set(test)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("primary fold split produced overlapping sets")
    if len(train) + len(validation) + len(test) != len(event_ids):
        raise ValueError("primary fold split does not cover all events")
    return {"train": train, "validation": validation, "test": test}


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
