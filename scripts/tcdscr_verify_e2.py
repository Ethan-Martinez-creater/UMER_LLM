#!/usr/bin/env python
"""Formal E2 completeness verification (E2 order §27).

Checks (issues must be 0):
 1. 30 run directories exist;
 2. per-run artifacts complete (manifest/history/validation_metrics/
    validation_predictions/best_selector/diagnostics);
 3. encoder checkpoint matches fold/seed and is the Random-init E1 encoder;
 4. encoder frozen: manifest checksum_match true and best_selector.pt
    records identical before/after checksums;
 5. validation event ids equal the fold's validation split (rebuilt with
    build_primary_fold_split, partition_seed=3090);
 6. no test artifacts before readiness: test_metrics.json only present
    when the global readiness summary says PASS (and even then they are
    never used for any decision);
 7. no future leakage (spot check with snapshot rebuilds): every
    selected_node_id belongs to the current snapshot of its row;
 8. metric range [0,1] for all four metrics;
 9. selected node sets are subsets of the snapshot nodes (same check as 7);
10. source never selected (selected_node_ids never contain source_id);
11. evidence_tokens within the 1024 budget, counts consistent.

Readiness summary is read from <root>/readiness/e2_summary.json if present.
Writes <root>/e2_verify.json (issues must be 0).
"""
import argparse
import json
import os

import tcdscr_common  # noqa: F401  (registers the project package)

from tcdscr.config.schema import config_from_env  # noqa: E402
from tcdscr_common import event_label_registry  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
DATASETS = ("pheme", "maweibo")
CUTOFFS = (5, 15, 30, 60, 180, 360)
BUDGET = 1024
REQUIRED_FILES = ("run_manifest.json", "history.json",
                  "validation_metrics.json",
                  "validation_predictions.jsonl", "best_selector.pt",
                  "diagnostics.json")
METRICS = ("accuracy", "macro_f1", "weighted_f1", "rumor_f1")
SPOT_RUNS = [("pheme", 0, 2000), ("maweibo", 0, 2000)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e2")
    args = ap.parse_args(argv)

    issues = []
    n_runs = 0
    checked = {}
    missing_by_run = {}
    readiness = None
    rpath = os.path.join(args.root, "readiness", "e2_summary.json")
    if os.path.exists(rpath):
        readiness = json.load(open(rpath, encoding="utf-8"))
    for dataset in DATASETS:
        cfg = config_from_env(dataset)
        registry = event_label_registry(dataset, cfg)
        from tcdscr.data.temporal_split import build_primary_fold_split
        for fold in FOLDS:
            split = build_primary_fold_split(registry, fold, seed=3090)
            val_ids = set(split["validation"])
            for seed in SEEDS:
                n_runs += 1
                run_dir = os.path.join(args.root, dataset,
                                       f"fold{fold}_seed{seed}")
                if not os.path.isdir(run_dir):
                    issues.append(f"{dataset}/fold{fold}_seed{seed}: "
                                  "missing run dir")
                    continue
                for fname in REQUIRED_FILES:
                    if not os.path.exists(os.path.join(run_dir, fname)):
                        issues.append(f"{os.path.basename(run_dir)}: "
                                      f"missing {fname}")
                manifest = json.load(open(os.path.join(
                    run_dir, "run_manifest.json"), encoding="utf-8"))
                # 3. encoder fold/seed + random init
                enc = manifest.get("encoder", {})
                ckpt = enc.get("checkpoint_path", "")
                expected = os.path.join(
                    args.root.replace("formal_e2", "formal_e1"),
                    dataset, f"fold{fold}_random_seed{seed}")
                if f"fold{fold}_random_seed{seed}" not in ckpt:
                    issues.append(f"{os.path.basename(run_dir)}: encoder "
                                  "checkpoint fold/seed mismatch")
                if enc.get("type") != \
                        "random_init_tcdscr_causal_social_encoder":
                    issues.append(f"{os.path.basename(run_dir)}: encoder "
                                  "type not random-init")
                if not os.path.exists(ckpt):
                    issues.append(f"{os.path.basename(run_dir)}: encoder "
                                  f"checkpoint missing {ckpt}")
                # 4. frozen checksums
                if not enc.get("checksum_match"):
                    issues.append(f"{os.path.basename(run_dir)}: encoder "
                                  "checksum mismatch in manifest")
                pt = torch_load(os.path.join(run_dir, "best_selector.pt"))
                if pt.get("encoder_checksum_before") != \
                        enc.get("checksum_before") or \
                        pt.get("encoder_checksum_after") != \
                        enc.get("checksum_after"):
                    issues.append(f"{os.path.basename(run_dir)}: "
                                  "best_selector checksums disagree with "
                                  "manifest")
                if pt.get("encoder_checkpoint_sha") != enc.get(
                        "checkpoint_sha"):
                    issues.append(f"{os.path.basename(run_dir)}: "
                                  "best_selector encoder SHA disagrees "
                                  "with manifest")
                # 5. validation ids: rows must be a subset of the split;
                # events missing from the rows must be verified (with
                # snapshot rebuilds) to have no reply candidates at any
                # cutoff — see spot_check_leakage
                rows = []
                with open(os.path.join(run_dir,
                                       "validation_predictions.jsonl"),
                          encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            rows.append(json.loads(line))
                row_ids = {r["event_id"] for r in rows}
                if not row_ids <= val_ids:
                    issues.append(
                        f"{os.path.basename(run_dir)}: validation rows "
                        f"include ids outside the fold validation split "
                        f"({len(row_ids - val_ids)} stray)")
                missing_by_run[f"{dataset}/{os.path.basename(run_dir)}"] = \
                    sorted(val_ids - row_ids)
                # 6. test gating
                has_test = os.path.exists(os.path.join(
                    run_dir, "test_metrics.json"))
                gate = readiness and readiness.get("status") == "PASS"
                if has_test and not gate:
                    issues.append(f"{os.path.basename(run_dir)}: test "
                                  "metrics present although readiness is "
                                  "not PASS")
                # 8. metric range
                vm = json.load(open(os.path.join(
                    run_dir, "validation_metrics.json"), encoding="utf-8"))
                for arm, by_cut in vm.get("arms", {}).items():
                    for c, m in by_cut.items():
                        for mk in METRICS:
                            v = m.get(mk)
                            if v is None or not (0.0 <= v <= 1.0):
                                issues.append(
                                    f"{os.path.basename(run_dir)}: metric "
                                    f"{arm}/{c}/{mk}={v} out of range")
                # 10/11. source never selected + budget + counts
                for r in rows:
                    sel = r.get("selected_node_ids") or []
                    if r.get("source_id") in sel:
                        issues.append(
                            f"{os.path.basename(run_dir)}: source selected "
                            f"in {r['event_id']}/{r['cutoff']}/{r['method']}")
                    if r.get("evidence_tokens", 0) > BUDGET:
                        issues.append(
                            f"{os.path.basename(run_dir)}: budget "
                            f"exceeded {r['event_id']}/{r['cutoff']}/"
                            f"{r['method']}: {r['evidence_tokens']}")
                    if len(sel) != r.get("selected_count"):
                        issues.append(
                            f"{os.path.basename(run_dir)}: selected_count "
                            f"inconsistent {r['event_id']}/{r['cutoff']}")
    checked["n_runs"] = n_runs
    checked["required_files"] = REQUIRED_FILES
    checked["readiness_summary_present"] = readiness is not None

    # 7/9. future-leakage spot check + missing-event no-candidate check
    leakage = spot_check_leakage(args.root, missing_by_run)
    checked["spot_leakage"] = leakage
    if leakage["failures"]:
        issues.append(f"future-leakage spot check: "
                      f"{leakage['failures']} failures")
    if leakage["missing_with_candidates"]:
        issues.append(f"missing events with candidates: "
                      f"{leakage['missing_with_candidates']}")

    out = {"issues": issues, "n_issues": len(issues), "checked": checked,
           "spot_runs": SPOT_RUNS}
    with open(os.path.join(args.root, "e2_verify.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"n_issues": len(issues), "checked": checked},
                     indent=1))
    for i in issues[:30]:
        print(" -", i)
    return 0 if not issues else 1


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def spot_check_leakage(root, missing_by_run=None):
    """Rebuild snapshots for a few runs and check that every selected node
    of every arm belongs to its row's snapshot node set. Also verifies (for
    every run) that validation events absent from the prediction rows have
    no reply candidates at any primary cutoff — i.e. their omission is the
    runner's no-candidate skip, not a data loss."""
    from tcdscr.data.maweibo_adapter import load_event as load_mw
    from tcdscr.data.pheme_adapter import load_event as load_pheme
    from tcdscr.data import pheme_adapter, maweibo_adapter
    from tcdscr.data.snapshot_builder import build_snapshot
    failures = 0
    rows_checked = 0
    for dataset, fold, seed in SPOT_RUNS:
        cfg = config_from_env(dataset)
        run_dir = os.path.join(root, dataset, f"fold{fold}_seed{seed}")
        if not os.path.isdir(run_dir):
            failures += 1
            continue
        from tcdscr.data.temporal_split import build_primary_fold_split
        registry = event_label_registry(dataset, cfg)
        split = build_primary_fold_split(registry, fold, seed=3090)
        val_ids = set(split["validation"])
        cache = {}
        node_sets = {}
        for ev_id in val_ids:
            if dataset == "pheme":
                by_id = {eid: (topic, label, folder)
                         for eid, topic, label, folder
                         in pheme_adapter.event_ids(cfg.raw_dir)}
                ev = load_pheme(*by_id[ev_id])
            else:
                labels = dict(event_label_registry(dataset, cfg))
                ev = load_mw(ev_id, labels[ev_id],
                             f"{cfg.raw_dir.rstrip('/')}/{ev_id}.json")
            for c in CUTOFFS:
                snap = build_snapshot(ev, c)
                node_sets[(ev_id, str(c))] = set(snap["node_ids"])
        with open(os.path.join(run_dir, "validation_predictions.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                allowed = node_sets.get((r["event_id"], r["cutoff"]))
                if allowed is None:
                    continue
                rows_checked += 1
                sel = set(r.get("selected_node_ids") or [])
                if not sel <= allowed:
                    failures += 1
    # missing events must be no-candidate at every primary cutoff
    missing_with_candidates = 0
    missing_checked = 0
    if missing_by_run:
        for key, missing_ids in missing_by_run.items():
            if not missing_ids:
                continue
            dataset = key.split("/")[0]
            cfg = config_from_env(dataset)
            registry = event_label_registry(dataset, cfg)
            for ev_id in missing_ids:
                if dataset == "pheme":
                    by_id = {eid: (topic, label, folder)
                             for eid, topic, label, folder
                             in pheme_adapter.event_ids(cfg.raw_dir)}
                    ev = load_pheme(*by_id[ev_id])
                else:
                    ev = load_mw(ev_id, registry[ev_id],
                                 f"{cfg.raw_dir.rstrip('/')}/{ev_id}.json")
                for c in CUTOFFS:
                    snap = build_snapshot(ev, c)
                    missing_checked += 1
                    if len(snap["node_ids"]) - 1 > 0:
                        missing_with_candidates += 1
    return {"rows_checked": rows_checked, "failures": failures,
            "missing_events_checked": missing_checked,
            "missing_with_candidates": missing_with_candidates,
            "rule": "selected_node_ids subset of the current snapshot's "
                    "node set (rebuilt with the frozen builder); absent "
                    "validation rows must be no-candidate events"}


if __name__ == "__main__":
    import sys
    sys.exit(main())