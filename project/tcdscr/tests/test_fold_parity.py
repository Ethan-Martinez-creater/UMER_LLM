"""Unit tests: Formal E1 finalization fold-parity helpers.

``reconstruct_old_umer_split`` must deterministically reproduce the
historical UMER split (StratifiedKFold 5 folds seed 3090 + stratified 10%
inner draw, same seed) over any event ordering, and ``compare_fold_parity``
must detect exact matches and cross-set overlap (old-train leaking into the
new test/validation) so an audit cannot phantom-PASS.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tcdscr_fold_parity import (  # noqa: E402
    compare_fold_parity,
    parity_pass,
    reconstruct_old_umer_split,
)


def _fake_registry(n=120, rumor_ratio=0.4, seed=7):
    import random
    rng = random.Random(seed)
    ids = [f"e{i:04d}" for i in range(n)]
    labels = {}
    for eid in ids:
        labels[eid] = 1 if rng.random() < rumor_ratio else 0
    return ids, labels


def test_old_umer_split_reconstruction_deterministic():
    ids, labels = _fake_registry()
    label_seq = [labels[e] for e in ids]
    a = reconstruct_old_umer_split(ids, label_seq, fold_index=2)
    b = reconstruct_old_umer_split(ids, label_seq, fold_index=2)
    # deterministic
    assert a == b
    # disjoint and covering
    for x, y in (("train", "validation"), ("train", "test"),
                 ("validation", "test")):
        assert not (set(a[x]) & set(a[y])), (x, y)
    assert (set(a["train"]) | set(a["validation"]) | set(a["test"])
            == set(ids))
    assert set(a["test"]) <= set(ids)
    # sizes follow the protocol: test ~20%, val = ceil(0.10 * train+val)
    n = len(ids)
    assert abs(len(a["test"]) - n // 5) <= 1
    expected_val = math.ceil((n - len(a["test"])) * 0.10)
    assert len(a["validation"]) == expected_val


def test_fold_parity_audit_detects_overlap():
    ids, labels = _fake_registry()
    label_seq = [labels[e] for e in ids]
    old = reconstruct_old_umer_split(ids, label_seq, fold_index=0)
    # identical split -> PASS and zero intersections
    c = compare_fold_parity(old, old)
    assert parity_pass(c)
    assert c["old_train_intersect_new_test"] == 0
    # adversarial shift: swap one train event into the new test set
    new = {k: list(v) for k, v in old.items()}
    victim = new["train"][0]
    new["train"].remove(victim)
    new["test"].append(victim)
    c2 = compare_fold_parity(old, new)
    assert not c2["train_exact_match"]
    assert not c2["test_exact_match"]
    assert c2["old_train_intersect_new_test"] == 1
    assert not parity_pass(c2)
    # fields the audit report relies on are all present
    for key in ("old_train_count", "new_test_count",
                "validation_exact_match", "old_validation_intersect_new_test",
                "train_only_in_old_count", "test_only_in_new_count",
                "diff_ids_first_50"):
        assert key in c2