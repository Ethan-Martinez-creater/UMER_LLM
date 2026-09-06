#!/usr/bin/env python
"""Full-dataset snapshot metadata audit (delta-fix §24–§30, §34–§35, §40).

Walks every PHEME and Ma-Weibo event, builds the 7 formal cutoff snapshots
per event (SOURCE_ONLY excluded from cap statistics), and reports §26
statistics, §27 class breakdown, §28 worst-50 events, §40 fold integrity and
§35 text-fallback audit.

No model training, no embeddings, no LLM calls — metadata only.
"""
import argparse
import json
import os
import sys

from tcdscr_common import (PROJECT_DIR, event_label_registry)

AUDIT_CUTOFFS_MIN = (5, 15, 30, 60, 180, 360, 1440)  # delta-fix §25


def _pct(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _new_bucket():
    return {"events": 0, "cap_hits": 0, "nodes_before": [],
            "nodes_after": [], "removed_per_hit": [], "total_removed": 0,
            "rumor_total": 0, "rumor_hits": 0, "nonrumor_total": 0,
            "nonrumor_hits": 0}


def _finalize_bucket(b):
    before, after = b["nodes_before"], b["nodes_after"]
    n = len(before)
    hits = b["cap_hits"]
    out = {
        "events": n,
        "cap_hits": hits,
        "cap_hit_rate": hits / max(n, 1),
        "mean_nodes_before_cap": sum(before) / max(n, 1),
        "median_nodes_before_cap": _pct(before, 0.5),
        "p90_nodes_before_cap": _pct(before, 0.9),
        "p95_nodes_before_cap": _pct(before, 0.95),
        "p99_nodes_before_cap": _pct(before, 0.99),
        "max_nodes_before_cap": max(before) if before else 0,
        "mean_nodes_after_cap": sum(after) / max(n, 1),
        "total_nodes_removed": b["total_removed"],
        "mean_removed_per_hit_event": (
            sum(b["removed_per_hit"]) / max(hits, 1) if hits else 0.0),
        "p90_removed_per_hit_event": _pct(b["removed_per_hit"], 0.9),
        "max_removed": max(b["removed_per_hit"]) if b["removed_per_hit"] else 0,
        "rumor_cap_hit_rate": b["rumor_hits"] / max(b["rumor_total"], 1),
        "nonrumor_cap_hit_rate": b["nonrumor_hits"] / max(b["nonrumor_total"], 1),
        "rumor_events": b["rumor_total"],
        "nonrumor_events": b["nonrumor_total"],
    }
    return out


def audit_dataset(dataset, cfg, out_dir):
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import (event_folds_to_snapshot_folds,
                                            stratified_event_folds)

    registry = event_label_registry(dataset, cfg)
    folds = stratified_event_folds(registry, k=5, seed=3090)
    buckets = {c: _new_bucket() for c in AUDIT_CUTOFFS_MIN}
    worst = []
    events_total = 0
    duplicate_reaction_files_total = 0
    cross_fold_event_count = 0
    snapshot_fold_mismatch_count = 0
    missing_in_registry = 0

    if dataset == "pheme":
        from tcdscr.data import pheme_adapter
        stream = pheme_adapter.iter_events(cfg.raw_dir)
    else:
        from tcdscr.data import maweibo_adapter
        stream = maweibo_adapter.iter_events(cfg.raw_dir, cfg.label_file)

    for event in stream:
        events_total += 1
        eid = event["event_id"]
        duplicate_reaction_files_total += event.get(
            "duplicate_reaction_files", 0)
        if eid not in registry or registry[eid] != event["label"]:
            missing_in_registry += 1
        event_fold = folds.get(eid)
        if event_fold is None:
            cross_fold_event_count += 1
        for cutoff in AUDIT_CUTOFFS_MIN:
            snap = build_snapshot(event, cutoff)
            b = buckets[cutoff]
            b["events"] += 1
            before = snap["num_nodes_before_cap"]
            after = snap["num_nodes_after_cap"]
            b["nodes_before"].append(before)
            b["nodes_after"].append(after)
            if event["label"] == 1:
                b["rumor_total"] += 1
            else:
                b["nonrumor_total"] += 1
            if snap["cap_hit"]:
                removed = before - after
                b["cap_hits"] += 1
                b["total_removed"] += removed
                b["removed_per_hit"].append(removed)
                if event["label"] == 1:
                    b["rumor_hits"] += 1
                else:
                    b["nonrumor_hits"] += 1
                worst.append({
                    "dataset": dataset,
                    "event_id": eid,
                    "label": event["label"],
                    "cutoff": cutoff,
                    "nodes_before": before,
                    "nodes_after": after,
                    "nodes_removed": removed,
                    "source_timestamp": event["source_timestamp"],
                })
            # fold inheritance check through the real helper (delta-fix §40)
            try:
                snap_folds = event_folds_to_snapshot_folds(
                    {(eid, cutoff): eid}, folds)
                if snap_folds[(eid, cutoff)] != event_fold:
                    snapshot_fold_mismatch_count += 1
            except ValueError:
                snapshot_fold_mismatch_count += 1
        if events_total % 500 == 0:
            print(f"[{dataset}] audited {events_total} events", flush=True)

    print(f"[{dataset}] done: {events_total} events", flush=True)
    report = {
        "dataset": dataset,
        "cutoffs_min": list(AUDIT_CUTOFFS_MIN),
        "note": "SOURCE_ONLY excluded from cap statistics (delta-fix §25); "
                "cap rule unchanged — report only (delta-fix §30)",
        "per_cutoff": {str(c): _finalize_bucket(b)
                       for c, b in buckets.items()},
        "fold_integrity": {
            "events_total": events_total,
            "fold_0_size": sum(1 for f in folds.values() if f == 0),
            "fold_1_size": sum(1 for f in folds.values() if f == 1),
            "fold_2_size": sum(1 for f in folds.values() if f == 2),
            "fold_3_size": sum(1 for f in folds.values() if f == 3),
            "fold_4_size": sum(1 for f in folds.values() if f == 4),
            "per_fold_rumor_ratio": {
                str(k): (sum(1 for e, f in folds.items() if f == k
                             and registry[e] == 1)
                         / max(sum(1 for f in folds.values() if f == k), 1))
                for k in range(5)},
            "cross_fold_event_count": cross_fold_event_count,
            "snapshot_fold_mismatch_count": snapshot_fold_mismatch_count,
            "events_missing_in_registry": missing_in_registry,
        },
        "duplicate_reaction_files_total": duplicate_reaction_files_total,
    }
    worst.sort(key=lambda w: -w["nodes_removed"])
    report["worst_50_events"] = worst[:50]
    return report


def audit_maweibo_fallback(cfg, out_dir):
    from tcdscr.data import maweibo_adapter
    stream = maweibo_adapter.iter_events(cfg.raw_dir, cfg.label_file)
    agg = maweibo_adapter.aggregate_text_fallback(stream)
    agg["dataset"] = "maweibo"
    agg["note"] = ("original_text preferred; text used only when the field "
                   "is absent (historical, D6-verified); blank "
                   "original_text still counts as original_text branch")
    with open(os.path.join(out_dir, "maweibo_text_fallback_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=1)
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pheme", "maweibo"])
    ap.add_argument("--output-dir",
                    default="/data/jyz/next/llm/results/tcdscr/code_delta_fix")
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    from tcdscr.config.schema import config_from_env
    cap_reports = {}
    fold_reports = {}
    fallback_report = None
    for dataset in args.datasets:
        cfg = config_from_env(dataset)
        report = audit_dataset(dataset, cfg, args.output_dir)
        cap_reports[dataset] = report
        fold_reports[dataset] = report["fold_integrity"]
        if dataset == "maweibo":
            fallback_report = audit_maweibo_fallback(cfg, args.output_dir)

    with open(os.path.join(args.output_dir, "full_cap_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(cap_reports, fh, indent=1)
    with open(os.path.join(args.output_dir, "fold_integrity_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(fold_reports, fh, indent=1)
    summary = {
        dataset: {
            "events": cap_reports[dataset]["fold_integrity"]["events_total"],
            "cap_hit_rate_by_cutoff": {
                c: v["cap_hit_rate"]
                for c, v in cap_reports[dataset]["per_cutoff"].items()},
        }
        for dataset in cap_reports
    }
    if fallback_report is not None:
        summary["maweibo"]["text_fallback"] = {
            k: fallback_report[k] for k in
            ("total_nodes", "original_text_used", "text_fallback_count",
             "empty_text_count", "fallback_rate", "events",
             "events_with_fallback")}
    with open(os.path.join(args.output_dir, "audit_summary.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
