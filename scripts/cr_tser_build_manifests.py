#!/usr/bin/env python
"""Freeze the CR-TSER pilot manifests (plan §5, §31).

Writes ``results/cr_tser/manifests/<dataset>/`` so PHEME and Weibo22 can
coexist: ``event_split.json``, ``snapshot_manifest.jsonl``,
``intervention_manifest.jsonl`` and ``hashes.json``.

Two rules from the review are enforced here:

* **viability before split** — the 80/50/15/25 seed-7319 split is drawn from
  the events that can actually produce a non-empty snapshot, never from a
  label registry that may reference unusable events (plan §5);
* **immutable after labels** — once any formal utility-label cache exists for
  the dataset, the manifests cannot be rewritten at all. ``--force`` only
  permits a *pre-freeze* rebuild while no labels exist.
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
from cr_tser.data.pilot_split import (assert_event_disjoint,  # noqa: E402
                                      build_pilot_split, viable_event_ids)
from cr_tser.data.source_manifest import (assert_same_source,  # noqa: E402
                                          load_frozen_source,
                                          source_fingerprint,
                                          write_frozen_source)
from cr_tser.readers.base_reader import (model_weight_hash,  # noqa: E402
                                         tokenizer_hash)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def labels_path(out_root, dataset):
    namespaced = os.path.join(out_root, "utility_labels", dataset,
                              "labels.jsonl")
    legacy = os.path.join(out_root, "utility_labels", f"{dataset}.jsonl")
    for path in (namespaced, legacy):
        if os.path.exists(path):
            return path
    return None


def assert_manifests_mutable(out_root, dataset, force):
    """Refuse to touch frozen manifests once any label cache exists (plan §31)."""
    existing = labels_path(out_root, dataset)
    if existing:
        raise RuntimeError(
            f"utility labels already exist at {existing}; manifests are "
            "immutable after the first reader utility call (plan §31). "
            "--force cannot bypass this.")
    if not force:
        target = os.path.join(out_root, "manifests", dataset,
                              "event_split.json")
        if os.path.exists(target):
            raise RuntimeError(
                f"{target} already exists; pass --force for a pre-freeze "
                "rebuild (allowed only while no utility labels exist).")


def build_manifests(dataset, paths, out_root, force=False):
    """Manifest build; the source of record is the effective paths only.

    There is deliberately no ``--normalized-events`` override: the same
    ``CRTSER_WEIBO22_NORMALIZED`` path drives event loading, validation, the
    fingerprint and ``source.json`` (plan §31 review fix).
    """
    assert_manifests_mutable(out_root, dataset, force)

    events = common.load_dataset_events(dataset, paths)
    # ---- viability filtering happens BEFORE any split (plan §5) ----
    viable = set(viable_event_ids(events, CUTOFFS_MIN))
    registry = {e["event_id"]: int(e["label"]) for e in events
                if e["event_id"] in viable}
    if not viable:
        raise RuntimeError(
            f"{dataset}: no viable events (a viability gate must precede the "
            "split; plan §5)")
    split = build_pilot_split(registry)
    assert_event_disjoint(split)

    manifests = os.path.join(out_root, "manifests", dataset)
    source = source_fingerprint(dataset, paths)
    previous = load_frozen_source(manifests)
    if previous:
        assert_same_source(previous, source,
                           context=f"build_manifests/{dataset}")
    write_frozen_source(manifests, source)
    common.write_json(os.path.join(manifests, "event_split.json"), {
        **{k: split[k] for k in ("foundation_train", "utility_train",
                                 "utility_dev", "utility_eval", "unused")},
        "label_counts": split["label_counts"],
        "sizes": dict(SPLIT_SIZES),
        "dataset": dataset,
        "source": {k: source[k] for k in ("kind", "path", "sha256",
                                          "n_files", "exists")},
        "viable_event_count": len(viable),
        "total_events": len(events),
        "viability_filtered": len(events) - len(viable),
        "split_seed": 7319,
    })

    labeled_ids = set(split["utility_train"]) | set(split["utility_dev"]) | \
        set(split["utility_eval"])
    labeled = [e for e in events if e["event_id"] in labeled_ids]
    encoder = common.CrSemanticEncoder(paths.semantic_model, dataset)
    tokenizer = common.canonical_tokenizer(paths.canonical_tokenizer)

    snap_path = os.path.join(manifests, "snapshot_manifest.jsonl")
    iv_path = os.path.join(manifests, "intervention_manifest.jsonl")
    for path in (snap_path, iv_path):
        if os.path.exists(path):
            os.remove(path)
    n_snap = n_iv = n_zero = 0
    max_nodes_seen = 0
    for event, cutoff in common.iter_events_cutoffs(labeled, CUTOFFS_MIN):
        art = common.snapshot_artifacts(event, cutoff, encoder, tokenizer)
        snap = art["snapshot"]
        t0 = event["source_timestamp"]
        limit = t0 + cutoff * 60
        future_leak = any(ts > limit for ts in snap["timestamps"])
        max_nodes_seen = max(max_nodes_seen, len(snap["node_ids"]))
        common.append_jsonl(snap_path, {
            "dataset": dataset,
            "event_id": event["event_id"], "cutoff": cutoff,
            "num_nodes": len(snap["node_ids"]),
            "num_replies": len(art["units"]),
            "zero_reply": art["zero_reply"],
            "edge_count": len(snap["edge_index"]),
            "source_timestamp": t0,
            "max_timestamp": max(snap["timestamps"]) if snap["timestamps"] else None,
            "future_node_leak": bool(future_leak),
            "used_original_order_as_time": False,
            "cap_hit": bool(snap.get("cap_hit")),
            "src_selected": len(art["src"]["selected_node_ids"]) if art["src"] else 0,
            "src_tokens": art["src"]["total_tokens"] if art["src"] else 0,
        })
        n_snap += 1
        if art["zero_reply"]:
            n_zero += 1
            continue
        for iv in art["interventions"]:
            common.append_jsonl(iv_path, {
                "dataset": dataset,
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
        "source": source,
        "event_split_sha256": _sha_text(json.dumps(
            {k: split[k] for k in SPLIT_SIZES}, sort_keys=True)),
        "cutoffs": list(CUTOFFS_MIN),
        "loro_rotations": [list(r) for r in LORO_ROTATIONS],
        "readers": readers,
        "viable_events": len(viable),
        "n_snapshots": n_snap, "n_interventions": n_iv,
        "n_zero_reply_snapshots": n_zero,
        "max_snapshot_nodes": max_nodes_seen,
        "snapshot_cap": None,
    }
    common.write_json(os.path.join(manifests, "hashes.json"), hashes)
    return {"dataset": dataset, "manifests": manifests, "n_snapshots": n_snap,
            "n_interventions": n_iv, "n_zero_reply": n_zero,
            "viable_events": len(viable), "max_snapshot_nodes": max_nodes_seen}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pheme", "weibo22"), required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--force", action="store_true",
                    help="pre-freeze rebuild only; refused once labels exist")
    ap.add_argument("--smoke", action="store_true",
                    help="write into the smoke namespace, never formal")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    out_root = args.out_root or os.path.join(paths.out_root or
                                             str(common.REPO / "results" /
                                                 "cr_tser"))
    if args.smoke:
        out_root = common.smoke_root(out_root)
    result = build_manifests(args.dataset, paths, out_root, force=args.force)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
