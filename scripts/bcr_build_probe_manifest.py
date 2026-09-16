#!/usr/bin/env python
"""Freeze the 72-item behavioral probe manifest (plan §6.1, §11.7).

LOCAL-side: consumes the SERVER-produced ``probe_availability.json`` and the
frozen ``event_split.json`` and writes ``probe_manifest.json`` with the exact
36 items per dataset (12 per cutoff), the overlap audit against
utility_train/dev/eval and the frozen context contract.

It never calls a reader.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
PROJECT = REPO / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from bcr_utility.config import protocol as P                       # noqa: E402
from bcr_utility.data import historical_import                     # noqa: E402
from bcr_utility.probes import probe_contexts, probe_manifest      # noqa: E402


def build(repo_root: str, availability: dict, datasets=P.DATASETS) -> dict:
    out = {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "M0",
        "source_split": P.PROBE_SOURCE_SPLIT,
        "seed": P.PROBE_SEED,
        "design": {
            "events_per_dataset": P.PROBE_EVENTS_PER_DATASET,
            "events_per_cutoff": P.PROBE_EVENTS_PER_CUTOFF,
            "items_total": P.PROBE_ITEMS_TOTAL,
            "cutoffs_min": list(P.CUTOFFS_MIN),
            "event_disjoint_across_cutoffs": True,
            "min_visible_units": P.PROBE_MIN_VISIBLE_UNITS,
            "min_src_selected": P.PROBE_MIN_SRC_SELECTED,
            "gold_labels_used_for": "probe-event sampling balance only",
            "gold_labels_stored_per_item": False,
            **probe_contexts.context_plan(),
        },
        "datasets": {},
        "overlap_audit": {},
    }
    for dataset in datasets:
        entry = (availability.get("datasets") or {}).get(dataset)
        if entry is None:
            raise probe_manifest.ProbeManifestRefused(
                f"{dataset}: {P.PROBE_AVAILABILITY_FILENAME} has no entry")
        split_path = Path(repo_root) / P.HISTORICAL_RESULTS_ROOT / "manifests" \
            / dataset / "event_split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        labels = entry["gold_labels_for_balance_only"]
        manifest = probe_manifest.build_probe_manifest(
            split, dataset, entry["availability"], labels,
            seed=P.PROBE_SEED)
        probe_manifest.validate_probe_manifest(manifest, dataset)
        manifest["source_availability"] = {
            "event_split_sha256": entry["event_split_sha256"],
            "availability_digest": probe_manifest.availability_digest(
                entry["availability"]),
            "n_foundation_events": entry["n_foundation_events"],
        }
        manifest["overlap_audit"] = probe_manifest.audit_overlap(
            manifest, split)
        out["datasets"][dataset] = manifest
        out["overlap_audit"][dataset] = manifest["overlap_audit"]
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--availability", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", action="append", choices=list(P.DATASETS))
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    bootstrap = Path(args.repo_root) / P.RESULTS_ROOT / P.BOOTSTRAP_DIRNAME
    availability = json.loads(Path(
        args.availability or bootstrap / P.PROBE_AVAILABILITY_FILENAME
    ).read_text(encoding="utf-8"))
    payload = build(args.repo_root, availability,
                    datasets=tuple(args.dataset or P.DATASETS))
    target = args.out or str(bootstrap / P.PROBE_MANIFEST_FILENAME)
    historical_import.write_json(target, payload)
    print(json.dumps({
        "wrote": target,
        "counts": {d: {"items": v["n_items"],
                       "per_cutoff": v["per_cutoff_counts"],
                       "label_balance": v["label_balance"],
                       "candidate_pool": v["candidate_pool"]}
                   for d, v in payload["datasets"].items()},
        "overlap_audit": payload["overlap_audit"],
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
