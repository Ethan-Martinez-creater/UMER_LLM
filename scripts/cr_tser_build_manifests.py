#!/usr/bin/env python
"""Freeze the CR-TSER pilot manifests (plan §31).

Writes ``results/cr_tser/manifests/``: ``event_split.json``,
``snapshot_manifest.jsonl``, ``intervention_manifest.jsonl`` and
``hashes.json``. Manifests become immutable after the first reader utility
call; the ``--force`` flag exists only for a pre-freeze re-run and refuses
once a utility-label file is present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_tser_common as common  # noqa: E402

from cr_tser.config.pilot_config import (CUTOFFS_MIN, LORO_ROTATIONS,  # noqa: E402
                                         READER_KEYS, READER_MODEL_IDS,
                                         SPLIT_SIZES)
from cr_tser.data.pilot_split import assert_event_disjoint, build_pilot_split  # noqa: E402
from cr_tser.readers.base_reader import (ReaderSpec, model_weight_hash,  # noqa: E402
                                         tokenizer_hash)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_manifests(dataset, paths, out_root, normalized_paths=None,
                    force=False):
    label_file = os.path.join(out_root, "utility_labels", f"{dataset}.jsonl")
    if os.path.exists(label_file) and not force:
        raise RuntimeError(
            "utility labels already exist; manifests are immutable after the "
            "first reader call (plan §31). Pass --force only before any call.")

    registry = common.dataset_registry(dataset, paths)
    split = build_pilot_split(registry)
    assert_event_disjoint(split)
    manifests = os.path.join(out_root, "manifests")
    common.write_json(os.path.join(manifests, "event_split.json"), {
        **{k: split[k] for k in ("foundation_train", "utility_train",
                                 "utility_dev", "utility_eval", "unused")},
        "label_counts": split["label_counts"],
        "sizes": dict(SPLIT_SIZES),
    })

    # Only the three LLM-labeled segments produce snapshot/intervention rows.
    labeled_ids = set(split["utility_train"]) | set(split["utility_dev"]) | \
        set(split["utility_eval"])
    events = [e for e in common.load_dataset_events(dataset, paths,
                                                    normalized_paths)
              if e["event_id"] in labeled_ids]
    encoder = common.CrSemanticEncoder(paths.semantic_model, dataset)
    tokenizer = common.canonical_tokenizer(paths.canonical_tokenizer)

    snap_path = os.path.join(manifests, "snapshot_manifest.jsonl")
    iv_path = os.path.join(manifests, "intervention_manifest.jsonl")
    for path in (snap_path, iv_path):
        if os.path.exists(path):
            os.remove(path)
    n_snap = n_iv = n_zero = 0
    for event, cutoff in common.iter_events_cutoffs(events, CUTOFFS_MIN):
        art = common.snapshot_artifacts(event, cutoff, encoder, tokenizer)
        snap = art["snapshot"]
        t0 = event["source_timestamp"]
        limit = t0 + cutoff * 60
        future_leak = any(ts > limit for ts in snap["timestamps"])
        common.append_jsonl(snap_path, {
            "event_id": event["event_id"], "cutoff": cutoff,
            "num_nodes": len(snap["node_ids"]),
            "num_replies": len(art["units"]),
            "zero_reply": art["zero_reply"],
            "edge_count": len(snap["edge_index"]),
            "source_timestamp": t0,
            "max_timestamp": max(snap["timestamps"]) if snap["timestamps"] else None,
            "future_node_leak": bool(future_leak),
            "used_original_order_as_time": False,
            "src_selected": len(art["src"]["selected_node_ids"]) if art["src"] else 0,
            "src_tokens": art["src"]["total_tokens"] if art["src"] else 0,
        })
        n_snap += 1
        if art["zero_reply"]:
            n_zero += 1
            continue
        for iv in art["interventions"]:
            common.append_jsonl(iv_path, {
                "event_id": event["event_id"], "cutoff": cutoff,
                "intervention_id": iv["intervention_id"],
                "type": iv["type"], "status": iv["status"],
                "remove_node_ids": iv["remove_node_ids"],
                "valid_in_gt": all(nid in snap["node_ids"]
                                   for nid in iv["remove_node_ids"]),
            })
            n_iv += 1

    readers = {}
    for key in READER_KEYS:
        path = paths.reader_path(key)
        readers[key] = {
            "model_id": READER_MODEL_IDS[key], "path": path,
            "weight_hash": model_weight_hash(path),
            "tokenizer_hash": tokenizer_hash(path),
        }
    hashes = {
        "dataset": dataset,
        "event_split_sha256": _hash_text(json.dumps(
            {k: split[k] for k in SPLIT_SIZES}, sort_keys=True)),
        "cutoffs": list(CUTOFFS_MIN),
        "loro_rotations": [list(r) for r in LORO_ROTATIONS],
        "readers": readers,
        "n_snapshots": n_snap, "n_interventions": n_iv,
        "n_zero_reply_snapshots": n_zero,
    }
    common.write_json(os.path.join(manifests, "hashes.json"), hashes)
    return {"manifests": manifests, "n_snapshots": n_snap,
            "n_interventions": n_iv, "n_zero_reply": n_zero}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pheme", "weibo22"), required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--normalized-events", default=None,
                    help="Weibo22 normalized JSONL export (timestamps)")
    ap.add_argument("--force", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    out_root = args.out_root or os.path.join(paths.out_root or
                                             str(common.REPO / "results" /
                                                 "cr_tser"))
    result = build_manifests(args.dataset, paths, out_root,
                             normalized_paths=args.normalized_events,
                             force=args.force)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
