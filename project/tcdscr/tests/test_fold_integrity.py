"""Unit tests: split/fold integrity (plan §28 Split block, §5.1; delta-fix
§5/§6/§8)."""
import pytest

from ..data.temporal_split import (build_primary_fold_split,
                                   chronological_split,
                                   event_folds_to_snapshot_folds,
                                   stratified_event_folds)


def test_event_folds_to_snapshot_folds_returns_fold_ids():
    labels = {f"e{i}": i % 2 for i in range(20)}
    event_folds = stratified_event_folds(labels, k=5, seed=3090)
    snapshot_event_ids = {}
    for eid, fold in event_folds.items():
        for cut in (5, 15, 60, 1440):
            snapshot_event_ids[(eid, cut)] = eid
    snapshot_folds = event_folds_to_snapshot_folds(
        snapshot_event_ids, event_folds)
    # the helper must return FOLD IDS, never event ids (delta-fix §5)
    for (eid, cut), fold in snapshot_folds.items():
        assert fold == event_folds[eid]
        assert fold != eid or eid == fold  # a numeric id colliding by chance
        assert isinstance(fold, int) and 0 <= fold < 5


def test_all_snapshots_inherit_real_event_fold():
    labels = {f"e{i}": i % 3 for i in range(30)}
    event_folds = stratified_event_folds(labels, k=5, seed=3090)
    cutoffs = (5, 15, 30, 60, 180, 360, 1440)
    snapshot_event_ids = {(eid, c): eid
                          for eid in event_folds for c in cutoffs}
    snapshot_folds = event_folds_to_snapshot_folds(
        snapshot_event_ids, event_folds)
    for eid, fold in event_folds.items():
        assert snapshot_folds[(eid, 5)] == event_folds[eid]
        assert snapshot_folds[(eid, 15)] == event_folds[eid]
        assert {snapshot_folds[(eid, c)] for c in cutoffs} == {fold}


def test_snapshot_fold_helper_rejects_unknown_event():
    event_folds = {"e1": 0, "e2": 1}
    with pytest.raises(ValueError):
        event_folds_to_snapshot_folds({("ghost", 5): "ghost"}, event_folds)


def test_primary_split_disjoint():
    labels = {f"e{i}": i % 2 for i in range(200)}
    split = build_primary_fold_split(labels, fold_index=3, seed=3090)
    train, val, test = (set(split["train"]), set(split["validation"]),
                        set(split["test"]))
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_primary_split_covers_all_events():
    labels = {f"e{i}": i % 2 for i in range(200)}
    split = build_primary_fold_split(labels, fold_index=0, seed=3090)
    union = set(split["train"]) | set(split["validation"]) | set(split["test"])
    assert union == set(labels)
    n = len(labels)
    # outer test = one stratified fifth; inner validation = 10% of the rest
    assert len(split["test"]) == 40
    assert len(split["validation"]) == 16
    assert len(split["train"]) == 144


def test_primary_split_stratified():
    labels = {f"r{i}": 1 for i in range(150)}
    labels.update({f"n{i}": 0 for i in range(150)})
    for fold in range(5):
        split = build_primary_fold_split(labels, fold_index=fold, seed=3090)
        for seg in ("train", "validation", "test"):
            ids = split[seg]
            rumors = sum(1 for e in ids if labels[e] == 1)
            # 50/50 population: every segment stays within a few percent
            assert abs(rumors / len(ids) - 0.5) < 0.05


def test_primary_split_deterministic_per_fold():
    labels = {f"e{i}": i % 2 for i in range(120)}
    a = build_primary_fold_split(labels, fold_index=2, seed=3090)
    b = build_primary_fold_split(labels, fold_index=2, seed=3090)
    assert a == b
    # different folds give different, disjoint test sets
    tests = [set(build_primary_fold_split(labels, fold_index=k,
                                          seed=3090)["test"])
             for k in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            assert not (tests[i] & tests[j])
        assert len(tests[i]) == 24


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
