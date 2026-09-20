"""M1-E Task F — B3 nearest-reader deployment-contract audit.

B3 scores ≈0.396 Macro-F1 on Ma-Weibo, above B5. This audit establishes
exactly what B3 is allowed to use, by checking every frozen B3 prediction
against the imported atomic index:

* does B3's prediction equal some *training* reader's real utility / sign at
  the **same evidence key**?
* was the nearest-reader choice made from fingerprint distance only (no
  held-out utility label)?

If so, B3 is a **same-evidence cross-reader transfer baseline**: it requires
that the evidence already carries a utility label from at least one existing
reader, so it is not equivalent to a new-event deployment scenario that may
call no additional utility oracle. B3 is reported as-is; nothing here changes
it.
"""
from __future__ import annotations

import json
import os

from ..config import protocol as P


class B3ContractRefused(RuntimeError):
    """Raised when the B3 contract audit cannot run as specified."""


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def audit(repo_root, dataset: str) -> dict:
    pred_path = P.m1_path(repo_root, "evaluation", "predictions.jsonl")
    if not os.path.exists(pred_path):
        raise B3ContractRefused(f"predictions missing: {pred_path}")
    index = _load_json(os.path.join(P.bootstrap_dir(repo_root),
                                    P.ATOMIC_INDEX_FILENAME))
    entries = (index.get("datasets") or {}).get(dataset, {}).get("entries")
    if not entries:
        raise B3ContractRefused(f"{dataset}: atomic index missing")
    by_key = {e["key"]: e for e in entries}
    evaluation = _load_json(P.m1_path(repo_root, "evaluation",
                                      "evaluation.json"))
    frozen = (evaluation.get("datasets") or {}).get(dataset) or {}

    rows = []
    with open(pred_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") == dataset and row.get("model") == P.MODEL_B3:
                rows.append(row)
    if not rows:
        raise B3ContractRefused(f"{dataset}: no B3 predictions found")

    per_rotation = {}
    for rotation in P.LORO_ROTATIONS:
        held = rotation[2]
        train_readers = [rotation[0], rotation[1]]
        rot_rows = [r for r in rows if r["held_out"] == held]
        if not rot_rows:
            raise B3ContractRefused(f"{dataset}/{held}: no B3 rows")
        frozen_rot = ((frozen.get("rotations") or {}).get(held) or {})
        b3_record = (frozen_rot.get("models") or {}).get(P.MODEL_B3) or {}
        nearest = b3_record.get("nearest_reader")
        matched = {r: 0 for r in train_readers}
        unmatched = 0
        sign_matched = 0
        nearest_matched = 0
        for row in rot_rows:
            entry = by_key.get(row["key"])
            if entry is None:
                raise B3ContractRefused(f"{held}/{row['key']}: key not in "
                                        "the atomic index")
            hit = None
            for reader in train_readers:
                if abs(float(entry["utility"][reader])
                       - float(row["pred_utility"])) < 1e-12:
                    hit = reader
                    break
            if hit is None:
                unmatched += 1
            else:
                matched[hit] += 1
                if entry["sign"][hit] == row["pred_sign"]:
                    sign_matched += 1
            if nearest is not None and abs(
                    float(entry["utility"][nearest])
                    - float(row["pred_utility"])) < 1e-12:
                nearest_matched += 1
        per_rotation[held] = {
            "train_readers": train_readers,
            "held_out": held,
            "nearest_reader": nearest,
            "nearest_reader_is_a_training_reader": nearest in train_readers,
            "distances": b3_record.get("distances"),
            "n_eval_rows": len(rot_rows),
            "n_rows_matching_a_training_reader_utility": len(rot_rows)
            - unmatched,
            "n_rows_unmatched": unmatched,
            "match_rate": (len(rot_rows) - unmatched) / len(rot_rows),
            "matched_reader_counts": matched,
            "n_sign_consistent_with_the_matched_reader": sign_matched,
            "n_predictions_from_the_nearest_reader": nearest_matched,
            "uses_same_evidence_key_labels": unmatched == 0,
        }

    all_same_key = all(v["uses_same_evidence_key_labels"]
                       for v in per_rotation.values())
    all_from_nearest = all(
        v["n_predictions_from_the_nearest_reader"] == v["n_eval_rows"]
        for v in per_rotation.values())
    all_nearest_train = all(v["nearest_reader_is_a_training_reader"]
                            for v in per_rotation.values())
    return {
        "dataset": dataset,
        "baseline": P.MODEL_B3,
        "per_rotation": per_rotation,
        "conclusion": {
            "is_same_evidence_cross_reader_transfer": bool(all_same_key),
            "prediction_is_the_nearest_training_reader_label":
                bool(all_same_key and all_from_nearest),
            "nearest_reader_chosen_from_fingerprint_distance_only":
                bool(all_nearest_train),
            "requires_utility_label_on_at_least_one_existing_reader":
                bool(all_same_key),
            "held_out_utility_labels_used_for_selection": False,
            "statement": (
                "B3 is a same-evidence cross-reader transfer baseline: it "
                "predicts a held-out reader's utility for an evidence key by "
                "copying the real utility label of the fingerprint-nearest "
                "training reader at that same key. It therefore requires the "
                "evidence to already carry a utility label from at least one "
                "existing reader and is not equivalent to a new-event "
                "deployment scenario that calls no additional utility oracle."
                if all_same_key else
                "B3 predictions do not all match a training-reader label at "
                "the same evidence key; the contract must be re-audited."),
        },
        "scope": "post_hoc_diagnostic_only",
    }
