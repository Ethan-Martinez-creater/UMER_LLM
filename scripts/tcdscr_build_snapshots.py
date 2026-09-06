#!/usr/bin/env python
"""STEP 3/5 entry point: build causal snapshots + features and the §27 cap
report for a dataset sample.

Usage: python tcdscr_build_snapshots.py --dataset pheme [--events 16]
"""
import argparse
import json
import os

from tcdscr_common import (SemanticBackend, build_event_snapshots,
                           load_events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pheme", "maweibo"))
    ap.add_argument("--events", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3090)
    args = ap.parse_args()

    from tcdscr.config.schema import config_from_env
    from tcdscr.data.manifests import AdapterAuditLog, CapStatistics
    cfg = config_from_env(args.dataset)
    events = load_events(args.dataset, cfg, limit=args.events, seed=args.seed)
    backend = SemanticBackend(cfg)

    cap = CapStatistics()
    audit = AdapterAuditLog()
    out_dir = os.path.join(cfg.output_dir, "snapshots")
    os.makedirs(out_dir, exist_ok=True)
    for event in events:
        snaps = build_event_snapshots(event, cfg.all_cutoffs_min)
        for snap in snaps.values():
            backend.snapshot_features(event, snap)
            cap.add(snap)
        audit.add_event(event, list(snaps.values()))

    report = {
        "dataset": args.dataset,
        "events": len(events),
        "cutoffs": ["SOURCE_ONLY"] + list(cfg.all_cutoffs_min),
        "cap_report": cap.to_dict(),
        "audit_summary": audit.summary(),
    }
    audit.save(os.path.join(out_dir, f"{args.dataset}_audit.jsonl"))
    with open(os.path.join(out_dir, f"{args.dataset}_cap_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report["cap_report"], indent=1))


if __name__ == "__main__":
    main()
