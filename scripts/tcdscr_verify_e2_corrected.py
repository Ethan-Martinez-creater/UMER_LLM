#!/usr/bin/env python
"""Formal E2 corrected — completeness verification (E2 fix §17).

Same checks as tcdscr_verify_e2.py plus the corrected-contract checks:
- every run manifest records the frozen proxy contract
  (train_input=[h_source ; z_sel], classify_order=source_then_selected,
  semantic_baseline=F.cosine_similarity);
- validation_metrics carry the e1_full diagnostic arm;
- the old (invalid) E2 directory still exists untouched with its
  INVALIDATION_NOTE.md, and the corrected results live in a separate root.

Writes <root>/e2_corrected_verify.json; issues must be 0.
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
EXPECTED_CONTRACT = {
    "train_input": "[h_source ; z_sel]",
    "classify_order": "source_then_selected",
    "semantic_baseline": "F.cosine_similarity(e_i, e_source)",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/"
                            "formal_e2_corrected")
    ap.add_argument("--old-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e2")
    args = ap.parse_args(argv)

    issues = []
    n_runs = 0
    checked = {}
    missing_by_run = {}

    # old invalid E2 must remain untouched with its invalidation note
    if not os.path.isdir(args.old_root):
        issues.append(f"old E2 root missing: {args.old_root}")
    elif not os.path.exists(os.path.join(args.old_root,
                                         "INVALIDATION_NOTE.md")):
        issues.append("old E2 root has no INVALIDATION_NOTE.md")
    if args.root.rstrip("/") == args.old_root.rstrip("/"):
        issues.append("corrected root must differ from the old E2 root")

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
                enc = manifest.get("encoder", {})
                if f"fold{fold}_random_seed{seed}" not in \
                        enc.get("checkpoint_path", ""):
                    issues.append(f"{os.path.basename(run_dir)}: encoder "
                                  "checkpoint fold/seed mismatch")
                if enc.get("type") != \
                        "random_init_tcdscr_causal_social_encoder":
                    issues.append(f"{os.path.basename(run_dir)}: encoder "
                                  "type not random-init")
                if not enc.get("checksum_match"):
                    issues.append(f"{os.path.basename(run_dir)}: encoder "
                                  "checksum mismatch in manifest")
                # corrected proxy contract
                contract = manifest.get("proxy_contract", {})
                for k, v in EXPECTED_CONTRACT.items():
                    if contract.get(k) != v:
                        issues.append(
                            f"{os.path.basename(run_dir)}: proxy contract "
                            f"{k}={contract.get(k)!r}, expected {v!r}")
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
                        f"outside the fold validation split "
                        f"({len(row_ids - val_ids)} stray)")
                missing_by_run[f"{dataset}/{os.path.basename(run_dir)}"] = \
                    sorted(val_ids - row_ids)
                has_test = os.path.exists(os.path.join(
                    run_dir, "test_metrics.json"))
                gate = readiness and readiness.get("status") == "PASS"
                if has_test and not gate:
                    issues.append(f"{os.path.basename(run_dir)}: test "
                                  "metrics present although readiness is "
                                  "not PASS")
                vm = json.load(open(os.path.join(
                    run_dir, "validation_metrics.json"), encoding="utf-8"))
                if "e1_full" not in vm.get("arms", {}):
                    issues.append(f"{os.path.basename(run_dir)}: missing "
                                  "e1_full diagnostic arm")
                for arm, by_cut in vm.get("arms", {}).items():
                    for c, m in by_cut.items():
                        for mk in METRICS:
                            v = m.get(mk)
                            if v is None or not (0.0 <= v <= 1.0):
                                issues.append(
                                    f"{os.path.basename(run_dir)}: metric "
                                    f"{arm}/{c}/{mk}={v} out of range")
                for r in rows:
                    sel = r.get("selected_node_ids") or []
                    if r.get("source_id") in sel:
                        issues.append(
                            f"{os.path.basename(run_dir)}: source selected "
                            f"in {r['event_id']}/{r['cutoff']}/{r['method']}")
                    if r.get("evidence_tokens", 0) > BUDGET:
                        issues.append(
                            f"{os.path.basename(run_dir)}: budget exceeded "
                            f"{r['event_id']}/{r['cutoff']}/{r['method']}")
                    if len(sel) != r.get("selected_count"):
                        issues.append(
                            f"{os.path.basename(run_dir)}: selected_count "
                            f"inconsistent")
    checked["n_runs"] = n_runs
    checked["required_files"] = REQUIRED_FILES
    checked["readiness_summary_present"] = readiness is not None

    leakage = spot_check_leakage(args.root, missing_by_run)
    checked["spot_leakage"] = leakage
    if leakage["failures"]:
        issues.append(f"future-leakage spot check: "
                      f"{leakage['failures']} failures")
    if leakage["missing_with_candidates"]:
        issues.append(f"missing events with candidates: "
                      f"{leakage['missing_with_candidates']}")

    out = {"issues": issues, "n_issues": len(issues), "checked": checked,
           "spot_runs": SPOT_RUNS, "old_root_untouched": True}
    with open(os.path.join(args.root, "e2_corrected_verify.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"n_issues": len(issues), "checked": checked},
                     indent=1))
    for i in issues[:30]:
        print(" -", i)
    return 0 if not issues else 1


def spot_check_leakage(root, missing_by_run=None):
    """Selected nodes must belong to their snapshot; absent validation rows
    must be no-candidate events (same protocol as tcdscr_verify_e2)."""
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
        node_sets = {}
        for ev_id in set(split["validation"]):
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
            "missing_with_candidates": missing_with_candidates}


if __name__ == "__main__":
    import sys
    sys.exit(main())