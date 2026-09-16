#!/usr/bin/env python
"""SERVER-only: measure probe availability on the raw datasets (plan §6.1).

The frozen probe manifest needs one fact that only the raw data can answer:
for every ``foundation_train`` event and cutoff, is there at least one valid
visible evidence unit **and** at least one SRC-selected unit?

This stage therefore rebuilds the causal snapshots of the foundation events with
the frozen CR-TSER orchestration (``snapshot_artifacts``: causal snapshot →
Reply–Parent evidence units → SRC) and records counts only. It calls no reader,
loads no model weights and writes no utility label.

Output: ``results/bcr_utility_v1/bootstrap/probe_availability.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
PROJECT = REPO / "project"
SCRIPTS = REPO / "scripts"
for path in (str(PROJECT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import cr_tser_common as common                                    # noqa: E402

from bcr_utility.config import protocol as P                       # noqa: E402
from bcr_utility.data import historical_import                     # noqa: E402


def scan_dataset(dataset: str, paths, split_path: Path, limit: int = None,
                 progress_every: int = 10) -> dict:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    foundation = [str(e) for e in split[P.PROBE_SOURCE_SPLIT]]
    wanted = set(foundation)
    events = [e for e in common.load_dataset_events(dataset, paths)
              if e["event_id"] in wanted]
    missing = sorted(wanted - {e["event_id"] for e in events})
    if missing:
        raise RuntimeError(
            f"{dataset}: {len(missing)} foundation events are absent from the "
            f"dataset source, e.g. {missing[:3]}")

    encoder = common.CrSemanticEncoder(paths.semantic_model, dataset)
    tokenizer = common.canonical_tokenizer(paths.canonical_tokenizer)
    registry = common.dataset_registry(dataset, paths)

    availability = {}
    started = time.time()
    for n, event in enumerate(sorted(events, key=lambda e: e["event_id"]), 1):
        per_event = {}
        for cutoff in P.CUTOFFS_MIN:
            art = common.snapshot_artifacts(event, cutoff, encoder, tokenizer)
            src = art.get("src")
            per_event[str(cutoff)] = {
                "n_units": len(art.get("units") or []),
                "src_selected": len(src["selected_node_ids"]) if src else 0,
                "n_visible_nodes": len(art["snapshot"]["node_ids"]),
                "zero_reply": bool(art.get("zero_reply")),
            }
        availability[str(event["event_id"])] = per_event
        if progress_every and n % progress_every == 0:
            print(f"  {dataset}: {n}/{len(events)} events "
                  f"({time.time() - started:.0f}s)", flush=True)
        if limit and n >= limit:
            break

    labels = {eid: int(registry[eid]) for eid in foundation
              if eid in registry}
    if len(labels) != len(foundation):
        raise RuntimeError(
            f"{dataset}: gold labels resolved for {len(labels)} of "
            f"{len(foundation)} foundation events")
    return {
        "n_foundation_events": len(foundation),
        "event_split_sha256": historical_import.canonical_sha256_file(
            str(split_path)),
        "gold_labels_for_balance_only": labels,
        "availability": availability,
        "scan_seconds": round(time.time() - started, 1),
    }


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", choices=list(P.DATASETS))
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--historical-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="scan only the first N events (timing probe only)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    historical_root = Path(args.historical_root or
                           P.historical_root(args.repo_root))
    datasets = tuple(args.dataset or P.DATASETS)
    payload = {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "M0",
        "source_split": P.PROBE_SOURCE_SPLIT,
        "generated_by": "scripts/bcr_probe_scan.py",
        "environment": "SERVER/DGPA",
        "cutoffs_min": list(P.CUTOFFS_MIN),
        "datasets": {},
    }
    for dataset in datasets:
        print(f"scanning {dataset} ...", flush=True)
        payload["datasets"][dataset] = scan_dataset(
            dataset, paths,
            historical_root / "manifests" / dataset / "event_split.json",
            limit=args.limit)
    target = args.out or str(Path(args.repo_root) / P.RESULTS_ROOT
                             / P.BOOTSTRAP_DIRNAME
                             / P.PROBE_AVAILABILITY_FILENAME)
    historical_import.write_json(target, payload)
    print(json.dumps({"wrote": target, "datasets": {
        d: {"n_foundation_events": v["n_foundation_events"],
            "scan_seconds": v["scan_seconds"]}
        for d, v in payload["datasets"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
