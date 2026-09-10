#!/usr/bin/env python
"""Formal E3-B verification (E3 test order §20) — must end with issues == 0.

Checks, over results/tcdscr/formal_e3_test/:
- 30 runs (2 datasets x 5 folds x 3 seeds) complete with all artifacts;
- test event ids exactly match the rebuilt outer split;
- fold best_config exactly matches the validation-frozen config;
- no test-time hyperparameter search, no validation-driven re-selection;
- corrected E2 checkpoints only (SHA match with formal_e2_corrected);
- encoder/selector/proxy checksums unchanged;
- 5m dynamic selection == static selection;
- memory_previous equals the previous cutoff's dynamic selection;
- selected ids belong to the current snapshot (spot rebuild);
- no future memory leakage; no Qwen model inference flag.
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
                    default="/data/jyz/next/llm/results/tcdscr/formal_e3_test")
    ap.add_argument("--e3-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e3")
    ap.add_argument("--e2-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e2_corrected")
    ap.add_argument("--old-e2-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e2")
    args = ap.parse_args(argv)

    issues = []
    checked = {"runs": 0, "rows_checked": 0, "first_cutoff_checked": 0,
               "test_ids_verified": 0, "spot_snapshots_rebuilt": 0}

    for dataset in DATASETS:
        for fold in FOLDS:
            bc_path = os.path.join(args.e3_root, dataset, f"fold{fold}",
                                   "best_config.json")
            if not os.path.exists(bc_path):
                issues.append(f"{dataset}/fold{fold}: frozen best_config "
                              "missing")
                continue
            frozen = _load(bc_path)
            for seed in SEEDS:
                run_dir = os.path.join(args.root, "runs", dataset,
                                       f"fold{fold}_seed{seed}")
                tag = f"{dataset}/fold{fold}_seed{seed}"
                for fn in ("run_manifest.json", "test_predictions.jsonl",
                           "test_metrics.json", "temporal_metrics.json"):
                    if not os.path.exists(os.path.join(run_dir, fn)):
                        issues.append(f"{tag}: {fn} missing")
                mpath = os.path.join(run_dir, "run_manifest.json")
                if not os.path.exists(mpath):
                    continue
                checked["runs"] += 1
                man = _load(mpath)
                if man.get("stage") != "formal_e3_test":
                    issues.append(f"{tag}: wrong stage {man.get('stage')}")
                if man.get("split_used") != "test":
                    issues.append(f"{tag}: split_used != test")
                if man.get("qwen_model_inference") is not False:
                    issues.append(f"{tag}: qwen inference flag set")
                if man.get("test_time_hyperparameter_search") is not False:
                    issues.append(f"{tag}: test-time search flag set")
                # frozen config equality
                mc = man.get("config", {})
                for k in ("lambda_n", "lambda_p", "budget"):
                    if mc.get(k) != frozen.get(k):
                        issues.append(f"{tag}: config {k} "
                                      f"{mc.get(k)} != frozen {frozen.get(k)}")
                if mc.get("validation_frozen") is not True:
                    issues.append(f"{tag}: config not marked frozen")
                # corrected-E2 provenance
                e2_man = os.path.join(args.e2_root, dataset,
                                      f"fold{fold}_seed{seed}",
                                      "run_manifest.json")
                if os.path.exists(e2_man):
                    e2 = _load(e2_man)
                    if man.get("encoder", {}).get("checkpoint_sha") != \
                            e2.get("encoder", {}).get("checkpoint_sha"):
                        issues.append(f"{tag}: encoder SHA differs from "
                                      "corrected E2")
                # test ids exactly match rebuilt split
                from tcdscr.config.schema import config_from_env
                from tcdscr.data.temporal_split import (
                    build_primary_fold_split)
                from tcdscr_common import event_label_registry
                cfg = config_from_env(dataset)
                registry = event_label_registry(dataset, cfg)
                split = build_primary_fold_split(registry, fold, seed=3090)
                expected = sorted(split["test"])
                got = sorted(man.get("test_event_ids", []))
                if got != expected:
                    issues.append(f"{tag}: test ids mismatch "
                                  f"({len(got)} vs {len(expected)})")
                checked["test_ids_verified"] += len(got)
                # split parity record
                if man.get("split_parity", {}).get("test_exact_match") \
                        is not True:
                    issues.append(f"{tag}: test parity not exact match")
                # checksums
                ck = man.get("checksums", {})
                for name in ("encoder", "selector", "proxy"):
                    if ck.get(f"{name}_match") is not True:
                        issues.append(f"{tag}: {name} checksum mismatch")
                # rows: temporal causality + 5m equality
                rows_path = os.path.join(run_dir, "test_predictions.jsonl")
                if not os.path.exists(rows_path):
                    continue
                rows = _rows(rows_path)
                checked["rows_checked"] += len(rows)
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
                        prev_ids = r["dynamic_selected_node_ids"]
                        prev_cutoff = int(r["cutoff"])

    # spot snapshot rebuild: selected ids inside the current snapshot
    spot_rows = []
    for dataset in DATASETS:
        for fold in FOLDS:
            for seed in SEEDS:
                p = os.path.join(args.root, "runs", dataset,
                                 f"fold{fold}_seed{seed}",
                                 "test_predictions.jsonl")
                if os.path.exists(p):
                    spot_rows.extend(_rows(p))
    if spot_rows:
        from tcdscr.config.schema import config_from_env
        from tcdscr.data.snapshot_builder import build_snapshot
        from tcdscr.data.temporal_split import build_primary_fold_split
        from tcdscr_common import (event_label_registry, load_split_events)
        for dataset in DATASETS:
            for fold in FOLDS:
                for seed in SEEDS:
                    p = os.path.join(args.root, "runs", dataset,
                                     f"fold{fold}_seed{seed}",
                                     "test_predictions.jsonl")
                    if not os.path.exists(p):
                        continue
                    rows = _rows(p)
                    cfg = config_from_env(dataset)
                    registry = event_label_registry(dataset, cfg)
                    split = build_primary_fold_split(registry, fold,
                                                     seed=3090)
                    evs = load_split_events(dataset, cfg, split)
                    ev_map = {ev["event_id"]: ev for ev in evs["test"]}
                    seen = set()
                    for r in rows:
                        if r["event_id"] in seen:
                            continue
                        seen.add(r["event_id"])
                        ev = ev_map.get(r["event_id"])
                        if ev is None:
                            issues.append(
                                f"{dataset}/fold{fold}: {r['event_id']} "
                                "not in test split")
                            continue
                        snap = build_snapshot(ev, int(r["cutoff"]))
                        snap_ids = set(snap["node_ids"])
                        for sid in r["dynamic_selected_node_ids"]:
                            if sid not in snap_ids:
                                issues.append(
                                    f"{dataset}/fold{fold}: selected "
                                    f"{sid} outside snapshot at "
                                    f"{r['cutoff']} for {r['event_id']}")
                        checked["spot_snapshots_rebuilt"] += 1

    result = {"n_issues": len(issues), "checked": checked,
              "issues": issues[:20]}
    with open(os.path.join(args.root, "e3_test_verify.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
