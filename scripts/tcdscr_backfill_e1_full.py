#!/usr/bin/env python
"""Backfill the e1_full diagnostic arm into completed formal_e2_corrected runs.

The corrected E2 runner computes the E1 full-encoder sanity view inside
evaluate_arms_on_items but the persisted validation_metrics.json only stored
the three selection arms (static/random/semantic). This script replays each
run's validation split through the frozen Random-init E1 encoder with the
identical item builder and candidate filter used by the runner, recomputes
per-cutoff e1_full classification metrics, and merges them into
validation_metrics.json["arms"]["e1_full"].

The e1_full arm is diagnostic only (Proxy Sanity vs E1 table); it never
enters any readiness decision. Runs are skipped when their
validation_metrics.json already contains e1_full or the run is not complete.
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, build_light_item,
                           classification_metrics, encoder_forward_batch,
                           file_sha256, load_random_e1_encoder,
                           mean_primary_macro_f1)

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
DATASETS = ("pheme", "maweibo")


def rebuild_validation_items(dataset, fold, seed):
    from tcdscr.config.schema import config_from_env
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split

    from tcdscr_common import (EventSemanticStore, event_label_registry,
                               load_split_events)

    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    split = build_primary_fold_split(registry, fold, seed=3090)
    events = load_split_events(dataset, cfg, split)
    store = EventSemanticStore(cfg)
    cache = {}
    items = []
    for ev in events["validation"]:
        for c in PRIMARY_CUTOFFS:
            snap = build_snapshot(ev, c)
            items.append(build_light_item(ev, snap,
                                          store.get_store(ev, cache)))
    return items


def backfill_run(root, dataset, fold, seed, e1_root, device, dry_run=False):
    run_dir = os.path.join(root, dataset, f"fold{fold}_seed{seed}")
    vm_path = os.path.join(run_dir, "validation_metrics.json")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(vm_path) or not os.path.exists(manifest_path):
        return "skip", "run not complete"
    with open(vm_path, encoding="utf-8") as fh:
        vm = json.load(fh)
    arms = vm.get("arms", {})
    if "e1_full" in arms:
        return "skip", "already has e1_full"
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    items = rebuild_validation_items(dataset, fold, seed)
    encoder, ckpt_path, _e1 = load_random_e1_encoder(
        dataset, fold, seed, e1_root, device)
    run_sha = manifest["encoder"]["checkpoint_sha"]
    loaded_sha = file_sha256(ckpt_path)
    if loaded_sha != run_sha:
        raise RuntimeError(
            f"E1 checkpoint sha mismatch for {run_dir}: manifest {run_sha} "
            f"vs loaded {loaded_sha}")

    pairs = {str(c): [] for c in PRIMARY_CUTOFFS}
    for start in range(0, len(items), 32):
        chunk = items[start:start + 32]
        outs = encoder_forward_batch(encoder, chunk, device)
        for item, (_nr, _er, logits_full) in zip(chunk, outs):
            n = item["num_nodes"]
            src = item["source_pos"]
            cand = [i for i in range(n) if i != src]
            if not cand:
                continue
            pairs[str(item["cutoff_minutes"])].append(
                (item["label"], int(logits_full.argmax(dim=-1))))
        del outs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    e1_metrics = {str(c): classification_metrics(pairs[str(c)])
                  for c in PRIMARY_CUTOFFS}
    mean_primary = mean_primary_macro_f1(
        {str(c): e1_metrics[str(c)] for c in PRIMARY_CUTOFFS})

    if not dry_run:
        arms["e1_full"] = e1_metrics
        tmp = vm_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(vm, fh, indent=1)
        os.replace(tmp, vm_path)
    print(f"[{dataset} fold{fold} s{seed}] e1_full mean primary "
          f"{mean_primary:.4f} ({len(items)} val rows) "
          f"{'(dry-run)' if dry_run else 'written'}", flush=True)
    return "ok", {"dataset": dataset, "fold": fold, "seed": seed,
                  "mean_primary": mean_primary,
                  "per_cutoff": e1_metrics}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e2_corrected")
    ap.add_argument("--e1-root", default="/data/jyz/next/llm/results/tcdscr/formal_e1")
    ap.add_argument("--dataset", choices=DATASETS, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = (args.dataset,) if args.dataset else DATASETS
    summary = {"root": args.root, "device": device, "runs": []}
    done = skipped = 0
    for dataset in datasets:
        for fold in FOLDS:
            for seed in SEEDS:
                status, payload = backfill_run(
                    args.root, dataset, fold, seed, args.e1_root, device,
                    dry_run=args.dry_run)
                if status == "ok":
                    summary["runs"].append(payload)
                    done += 1
                else:
                    summary["runs"].append({"dataset": dataset, "fold": fold,
                                            "seed": seed, "skipped": payload})
                    skipped += 1
    summary["backfilled"] = done
    summary["skipped"] = skipped
    out_path = os.path.join(args.root, "readiness", "e1_full_backfill.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"backfilled {done}, skipped {skipped} -> {out_path}", flush=True)
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    main()
