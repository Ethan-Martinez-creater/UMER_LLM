#!/usr/bin/env python
"""STEP 4 entry point: build the 384D semantic snapshot cache (§7/§15).

Usage: python tcdscr_build_cache.py --dataset pheme [--events N] [--all]
"""
import argparse
import json

from tcdscr_common import SemanticBackend, load_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pheme", "maweibo"))
    ap.add_argument("--events", type=int, default=None,
                    help="sample size; omit with --all for every event")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=3090)
    args = ap.parse_args()

    from tcdscr.config.schema import config_from_env
    cfg = config_from_env(args.dataset)
    limit = None if args.all else args.events
    events = load_events(args.dataset, cfg, limit=limit, seed=args.seed)
    backend = SemanticBackend(cfg)

    n_snapshots = 0
    from tcdscr_common import build_event_snapshots
    for event in events:
        snaps = build_event_snapshots(event, cfg.all_cutoffs_min)
        for cutoff, snap in snaps.items():
            backend.snapshot_semantics(event, snap)
            n_snapshots += 1
    print(json.dumps({"dataset": args.dataset, "events": len(events),
                      "snapshots_cached": n_snapshots,
                      "cache_dir": backend.cache.root}))


if __name__ == "__main__":
    main()
