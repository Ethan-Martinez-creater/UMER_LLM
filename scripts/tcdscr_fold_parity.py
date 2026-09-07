#!/usr/bin/env python
"""Formal E1 finalization — UMER / TC-DSCR fold-ID parity helpers.

Reconstructs the historical UMER primary split exactly as the old training
runner did (round_006 screen_fold.strict_indices): outer
``StratifiedKFold(n_splits=5, shuffle=True, random_state=partition_seed)``
over the *file order* of ``splits/all_event_ids.txt``, then an inner
``train_test_split(test_size=0.10, random_state=partition_seed, shuffle=True,
stratify=...)`` — i.e. sklearn's StratifiedShuffleSplit path. The TC-DSCR
side is ``build_primary_fold_split`` (sorted event registry + the same two
sklearn calls). These helpers compare the two per outer fold.

The two implementations differ only in event ordering (file order vs sorted)
and label source; every sklearn call and seed is identical, which is what
``reconstruct_old_umer_split`` pins down so the audit is reproducible and
unit-testable without the server tree.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

N_OUTER_FOLDS = 5
PARTITION_SEED = 3090
VALIDATION_FRACTION = 0.10


def reconstruct_old_umer_split(event_ids, labels, fold_index: int,
                               partition_seed: int = PARTITION_SEED,
                               validation_fraction: float =
                               VALIDATION_FRACTION) -> dict:
    """Rebuild one UMER outer fold exactly as ``screen_fold.strict_indices``.

    ``event_ids`` and ``labels`` are parallel lists in the historical file
    order (``splits/all_event_ids.txt`` + the label manifest map). The outer
    fold ``fold_index`` becomes test; the remaining folds are re-split with
    a stratified 10% inner validation draw under the same seed.

    Returns {"train": [...], "validation": [...], "test": [...]} of event ids
    in index order.
    """
    if fold_index not in range(N_OUTER_FOLDS):
        raise ValueError(f"fold_index must be in [0, {N_OUTER_FOLDS})")
    labels_arr = np.asarray(labels, dtype=np.int64)
    indices = np.arange(len(event_ids), dtype=np.int64)
    outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                            random_state=int(partition_seed))
    train_val_idx, test_idx = list(
        outer.split(indices, labels_arr))[int(fold_index)]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=float(validation_fraction),
        random_state=int(partition_seed), shuffle=True,
        stratify=labels_arr[train_val_idx])
    train = [event_ids[i] for i in train_idx.tolist()]
    validation = [event_ids[i] for i in val_idx.tolist()]
    test = [event_ids[i] for i in test_idx.tolist()]
    return {"train": train, "validation": validation, "test": test}


def compare_fold_parity(old_split: dict, new_split: dict,
                        max_diff_ids: int = 50) -> dict:
    """Compare old-UMER and TC-DSCR event sets for one fold.

    Reports counts, exact-match flags, cross-set overlaps (train->test and
    train->validation leakage candidates on the new side) and symmetric
    only-in counts, plus up to ``max_diff_ids`` differing event ids per set.
    """
    old = {k: set(v) for k, v in old_split.items()}
    new = {k: set(v) for k, v in new_split.items()}
    out = {
        "old_train_count": len(old["train"]),
        "new_train_count": len(new["train"]),
        "old_val_count": len(old["validation"]),
        "new_val_count": len(new["validation"]),
        "old_test_count": len(old["test"]),
        "new_test_count": len(new["test"]),
        "train_exact_match": old["train"] == new["train"],
        "validation_exact_match": old["validation"] == new["validation"],
        "test_exact_match": old["test"] == new["test"],
        "old_train_intersect_new_test": len(old["train"] & new["test"]),
        "old_train_intersect_new_validation":
            len(old["train"] & new["validation"]),
        "old_validation_intersect_new_test":
            len(old["validation"] & new["test"]),
        "train_only_in_old_count": len(old["train"] - new["train"]),
        "train_only_in_new_count": len(new["train"] - old["train"]),
        "validation_only_in_old_count":
            len(old["validation"] - new["validation"]),
        "validation_only_in_new_count":
            len(new["validation"] - old["validation"]),
        "test_only_in_old_count": len(old["test"] - new["test"]),
        "test_only_in_new_count": len(new["test"] - old["test"]),
    }
    diffs = {}
    for k in ("train", "validation", "test"):
        only_old = sorted(old[k] - new[k])[:max_diff_ids]
        only_new = sorted(new[k] - old[k])[:max_diff_ids]
        diffs[k] = {"only_in_old_first": only_old,
                    "only_in_new_first": only_new}
    out["diff_ids_first_50"] = diffs
    return out


def parity_pass(comparison: dict) -> bool:
    """PASS iff every set matches exactly and no old-train/old-validation
    event leaked into the new test / new validation."""
    return (comparison["train_exact_match"]
            and comparison["validation_exact_match"]
            and comparison["test_exact_match"]
            and comparison["old_train_intersect_new_test"] == 0
            and comparison["old_train_intersect_new_validation"] == 0
            and comparison["old_validation_intersect_new_test"] == 0)


def label_consistency(old_split_ids, new_split_ids, old_labels, new_labels):
    """Per-set label sequence agreement (diagnostic, not a PASS condition)."""
    out = {}
    for k in ("train", "validation", "test"):
        old_lab = Counter(old_labels[e] for e in old_split_ids[k])
        new_lab = Counter(new_labels[e] for e in new_split_ids[k])
        out[k] = {"old_label_counts": dict(old_lab),
                  "new_label_counts": dict(new_lab)}
    return out