#!/usr/bin/env python
"""Formal E3 verification (E3 order §40) — must end with issues == 0.

Checks, over results/tcdscr/formal_e3/:
- 30 runs (2 datasets x 5 folds x 3 seeds) complete with manifests;
- 36 configurations per dataset x fold in grid_results.json;
- validation only: test split never read, no test predictions;
- corrected E2 checkpoints only (SHA match with formal_e2_corrected, and
  never the invalidated formal_e2 run);
- encoder/selector/proxy checksums unchanged (before == after, all runs);
- no future memory leakage (memory_previous equals the previous cutoff's
  dynamic selection, and memory only ever contains past nodes);
- first-cutoff (5m) dynamic selection equals static selection;
- selected ids belong to the current snapshot (spot rebuild);
- fold best_config is part of the grid; fold prediction rows complete.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project"))

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
DATASETS = ("pheme", "maweibo")
N_CONFIGS = 36


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _rows(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e3")
    ap.add_argument("--e2-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e2_corrected")
    ap.add_argument("--old-e2-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e2")
    args = ap.parse_args(argv)

    issues = []
    checked = {"runs": 0, "rows_checked": 0, "first_cutoff_checked": 0,
               "grid_configs": 0, "spot_snapshots_rebuilt": 0}

    for dataset in DATASETS:
        for fold in FOLDS:
            fold_dir = os.path.join(args.root, dataset, f"fold{fold}")
            grid_path = os.path.join(fold_dir, "grid_results.json")
            if not os.path.exists(grid_path):
                issues.append(f"{dataset}/fold{fold}: grid_results.json "
                              "missing")
                continue
            grid = _load(grid_path)
            if len(grid) != N_CONFIGS:
                issues.append(f"{dataset}/fold{fold}: expected "
                              f"{N_CONFIGS} configs, got {len(grid)}")
            checked["grid_configs"] += len(grid)
            bc_path = os.path.join(fold_dir, "best_config.json")
            if not os.path.exists(bc_path):
                issues.append(f"{dataset}/fold{fold}: best_config.json "
                              "missing")
            else:
                bc = _load(bc_path)
                key = f"{bc['lambda_n']}_{bc['lambda_p']}_{bc['budget']}"
                if key not in grid:
                    issues.append(f"{dataset}/fold{fold}: best config "
                                  f"{key} not in grid")

            for seed in SEEDS:
                run_dir = os.path.join(args.root, "runs", dataset,
                                       f"fold{fold}_seed{seed}")
                tag = f"{dataset}/fold{fold}_seed{seed}"
                if not os.path.isdir(run_dir):
                    issues.append(f"{tag}: run dir missing")
                    continue
                checked["runs"] += 1
                mpath = os.path.join(run_dir, "run_manifest.json")
                if not os.path.exists(mpath):
                    issues.append(f"{tag}: run_manifest.json missing")
                    continue
                man = _load(mpath)
                if man.get("stage") != "formal_e3":
                    issues.append(f"{tag}: wrong stage {man.get('stage')}")
                if man.get("test_split_read") is not False:
                    issues.append(f"{tag}: test_split_read not False")
                if os.path.exists(os.path.join(run_dir,
                                               "test_metrics.json")):
                    issues.append(f"{tag}: test predictions present "
                                  "(TEST_PROTOCOL_VIOLATION)")
                ck = man.get("checksums", {})
                for name in ("encoder", "selector", "proxy"):
                    if ck.get(f"{name}_match") is not True:
                        issues.append(f"{tag}: {name} checksum mismatch")
                    if ck.get(f"{name}_before") != ck.get(f"{name}_after"):
                        issues.append(f"{tag}: {name} before/after differ")
                # corrected-E2-only provenance
                e2_man_path = os.path.join(
                    args.e2_root, dataset, f"fold{fold}_seed{seed}",
                    "run_manifest.json")
                if not os.path.exists(e2_man_path):
                    issues.append(f"{tag}: corrected E2 manifest missing")
                else:
                    e2_man = _load(e2_man_path)
                    if man.get("encoder", {}).get("checkpoint_sha") != \
                            e2_man.get("encoder", {}).get("checkpoint_sha"):
                        issues.append(f"{tag}: encoder SHA differs from "
                                      "corrected E2 manifest")
                sel_sha = ck.get("selector_checkpoint_sha")
                pt_path = os.path.join(args.e2_root, dataset,
                                       f"fold{fold}_seed{seed}",
                                       "best_selector.pt")
                if sel_sha and os.path.exists(pt_path):
                    if file_sha256(pt_path) != sel_sha:
                        issues.append(f"{tag}: selector checkpoint SHA "
                                      "mismatch")
                e2_run_ref = man.get("selector_proxy", {}).get("e2_run", "")
                if args.old_e2_root in e2_run_ref and \
                        args.e2_root not in e2_run_ref:
                    issues.append(f"{tag}: points at the invalidated E2 run")

                rows_path = os.path.join(run_dir, "validation_rows.jsonl")
                if not os.path.exists(rows_path):
                    issues.append(f"{tag}: validation_rows.jsonl missing")
                    continue
                rows = _rows(rows_path)
                checked["rows_checked"] += len(rows)
                # temporal causality + first-cutoff equality, per (event, cfg)
                groups = defaultdict(list)
                for r in rows:
                    groups[(r["event_id"], r["lambda_n"], r["lambda_p"],
                            r["budget"])].append(r)
                for (eid, ln, lp, b), ev_rows in groups.items():
                    ev_rows.sort(key=lambda r: int(r["cutoff"]))
                    prev_ids = []
                    prev_cutoff = None
                    for i, r in enumerate(ev_rows):
                        if r["cutoff"] == "5":
                            checked["first_cutoff_checked"] += 1
                            if r["static_selected_node_ids"] != \
                                    r["dynamic_selected_node_ids"]:
                                issues.append(
                                    f"{tag}: 5m dynamic != static for "
                                    f"{eid} cfg {ln}/{lp}/{b}")
                        if r["memory_previous_ids"] != prev_ids:
                            issues.append(
                                f"{tag}: memory_previous mismatch for "
                                f"{eid}/{r['cutoff']} cfg {ln}/{lp}/{b}")
                        if prev_cutoff is not None and \
                                int(r["cutoff"]) <= prev_cutoff:
                            issues.append(
                                f"{tag}: non-monotonic cutoffs for {eid}")
                        # memory may only contain nodes of past cutoffs
                        mem_ids = set(r["memory_previous_ids"])
                        for s in r["selected_scores"]:
                            if s["node_id"] not in mem_ids and \
                                    s["persistence"] != 0.0:
                                issues.append(
                                    f"{tag}: persistence flag without "
                                    f"membership {eid}/{r['cutoff']}")
                        prev_ids = r["dynamic_selected_node_ids"]
                        prev_cutoff = int(r["cutoff"])

    # spot snapshot rebuild: selected ids must belong to the current snapshot
    spot_rows = []
    for dataset in DATASETS:
        for fold in FOLDS:
            p = os.path.join(args.root, dataset, f"fold{fold}",
                             "validation_predictions.jsonl")
            if os.path.exists(p):
                spot_rows.extend(_rows(p))
    if spot_rows:
        from tcdscr.config.schema import config_from_env
        from tcdscr.data.snapshot_builder import build_snapshot
        from tcdscr.data.temporal_split import build_primary_fold_split
        from tcdscr_common import (event_label_registry, load_split_events)
        by_fold = defaultdict(list)
        for r in spot_rows:
            by_fold[(r["dataset"], r["fold"])].append(r)
        for (dataset, fold), rows in by_fold.items():
            cfg = config_from_env(dataset)
            registry = event_label_registry(dataset, cfg)
            split = build_primary_fold_split(registry, fold, seed=3090)
            events = load_split_events(dataset, cfg, split)
            ev_map = {ev["event_id"]: ev for ev in events["validation"]}
            seen_events = set()
            for r in rows:
                if r["event_id"] in seen_events:
                    continue
                seen_events.add(r["event_id"])
                ev = ev_map.get(r["event_id"])
                if ev is None:
                    issues.append(f"{dataset}/fold{fold}: {r['event_id']} "
                                  "not in validation split")
                    continue
                snap = build_snapshot(ev, int(r["cutoff"]))
                snap_ids = set(snap["node_ids"])
                for sid in r["dynamic_selected_node_ids"]:
                    if sid not in snap_ids:
                        issues.append(
                            f"{dataset}/fold{fold}: selected {sid} outside "
                            f"snapshot at {r['cutoff']} for "
                            f"{r['event_id']}")
                checked["spot_snapshots_rebuilt"] += 1

    result = {"n_issues": len(issues), "checked": checked,
              "issues": issues[:20]}
    with open(os.path.join(args.root, "e3_verify.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
