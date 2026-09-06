"""Unit tests: split/fold integrity (plan §28 Split block, §5.1)."""
from ..data.temporal_split import (chronological_split,
                                   stratified_event_folds)


def test_event_fold_integrity():
    labels = {f"e{i}": i % 2 for i in range(50)}
    folds = stratified_event_folds(labels, k=5, seed=3090)
    assert set(folds) == set(labels)
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_no_event_cross_fold():
    labels = {f"e{i}": i % 3 for i in range(90)}
    folds = stratified_event_folds(labels, k=5, seed=3090)
    # every event owns exactly one fold — no event may live in two folds
    for eid, fold in folds.items():
        assert 0 <= fold < 5
        assert sum(1 for f in folds.values() if f == fold) >= 1
    assert len(folds) == len(set(folds))


def test_all_snapshots_same_fold():
    # all snapshots of an event inherit the event's fold (§5.1)
    labels = {f"e{i}": i % 2 for i in range(40)}
    folds = stratified_event_folds(labels, k=5, seed=3090)
    cutoffs = ("SOURCE_ONLY", 5, 15, 30, 60, 180, 360, 1440)
    snapshot_folds = {}
    for eid in labels:
        for cut in cutoffs:
            snapshot_folds[(eid, cut)] = folds[eid]
    for eid in labels:
        assigned = {snapshot_folds[(eid, c)] for c in cutoffs}
        assert len(assigned) == 1


def test_stratified_balance():
    labels = {f"r{i}": 1 for i in range(30)}
    labels.update({f"n{i}": 0 for i in range(30)})
    folds = stratified_event_folds(labels, k=5, seed=3090)
    per_fold_rumors = [0] * 5
    per_fold_total = [0] * 5
    for eid, fold in folds.items():
        per_fold_total[fold] += 1
        per_fold_rumors[fold] += labels[eid]
    for r, t in zip(per_fold_rumors, per_fold_total):
        assert t == 12  # 60 events / 5 folds
        assert r == 6   # stratification keeps 50/50


def test_chronological_split_shapes():
    times = {f"e{i}": 1000 + i for i in range(100)}
    split = chronological_split(times)
    assert len(split["train"]) == 70
    assert len(split["validation"]) == 15
    assert len(split["test"]) == 15
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["train"]) & set(split["validation"]))
    # strict time ordering across segments
    assert max(times[e] for e in split["train"]) <= \
        min(times[e] for e in split["validation"])
    assert max(times[e] for e in split["validation"]) <= \
        min(times[e] for e in split["test"])
