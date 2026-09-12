#!/usr/bin/env python
"""Final Evidence Closure — Gap A: Static Utility all-event held-out.

Runs the frozen corrected-E2 (Static utility / Random / Semantic) selection
arms on the OUTER TEST split, all events at every real cutoff (no-candidate
snapshots included, empty selection -> z_sel = zero vector).  Frozen
checkpoints only; no retraining, no checkpoint selection, budget fixed at
1024.  Outputs go to results/tcdscr/final_evidence/gap_a/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "project")
for _path in (HERE, PROJECT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, build_light_item,
                           mean_primary_macro_f1, state_sha256)

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
ARM_NAMES = ("static", "random", "semantic")
BUDGET = 1024
PARTITION_SEED = 3090

E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"
E2_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e2_corrected"
OUT_ROOT = "/data/jyz/next/llm/results/tcdscr/final_evidence/gap_a"


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--fold", type=int, choices=FOLDS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--e1-root", default=E1_ROOT)
    ap.add_argument("--e2-root", default=E2_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true",
                    help="run every dataset x fold x seed")
    return ap


def run_one(args, dataset, fold, seed):
    import torch
    from transformers import AutoTokenizer

    from tcdscr.config.schema import config_from_env
    from tcdscr.context.token_budget import EvidenceBudgetSelector
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr_common import (EventSemanticStore, event_label_registry,
                               load_split_events)
    from tcdscr_run_e2 import evaluate_arms_on_items, file_sha256
    from tcdscr_run_e3 import load_frozen_components

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    split = build_primary_fold_split(registry, fold, seed=PARTITION_SEED)
    events = load_split_events(dataset, cfg, split)
    test_events = events["test"]
    if args.limit:
        test_events = test_events[:args.limit]

    encoder, selector, proxy, checksums = load_frozen_components(
        dataset, fold, seed, args.e1_root, args.e2_root, device)
    after = {"encoder": state_sha256(encoder.state_dict()),
             "selector": state_sha256(selector.state_dict()),
             "proxy": state_sha256(proxy.state_dict())}
    match = {"encoder": after["encoder"] == checksums["encoder_before"],
             "selector": after["selector"] == checksums["selector_before"],
             "proxy": after["proxy"] == checksums["proxy_before"]}
    trainable = sum(p.numel() for m in (encoder, selector, proxy)
                    for p in m.parameters() if p.requires_grad)
    if trainable:
        raise RuntimeError("frozen components still require grad")

    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path,
                                              local_files_only=True)
    store = EventSemanticStore(cfg)
    cache = {}
    items = []
    for ev in test_events:
        for c in PRIMARY_CUTOFFS:
            snap = build_snapshot(ev, c)
            items.append(build_light_item(ev, snap, store.get_store(ev,
                                                                    cache)))
    bs = EvidenceBudgetSelector(tokenizer, BUDGET)
    metrics, rows, _diag = evaluate_arms_on_items(
        encoder, selector, proxy, items, bs, device, dataset, seed)
    mean_primary = {a: mean_primary_macro_f1({c: metrics[c][a]
                                              for c in metrics})
                    for a in ARM_NAMES}
    n_events = len({r["event_id"] for r in rows})
    expected_rows = n_events * len(PRIMARY_CUTOFFS) * len(ARM_NAMES)
    if len(rows) != expected_rows:
        raise RuntimeError(f"all-event coverage violated: {len(rows)} rows "
                           f"for {n_events} events x 6 cutoffs x 3 arms")
    empty_rows = sum(1 for r in rows if r["selected_count"] == 0)
    if empty_rows == 0:
        raise RuntimeError("no no-candidate (empty-selection) row produced")

    run_dir = os.path.join(args.out_root, "runs", dataset,
                           f"fold{fold}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "predictions.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in rows:
            r_full = dict(r)
            r_full.update({"dataset": dataset, "fold": fold, "seed": seed})
            fh.write(json.dumps(r_full) + "\n")
    manifest = {
        "stage": "final_evidence_gap_a",
        "dataset": dataset, "fold": fold, "seed": seed,
        "split_used": "test",
        "protocol": "all events at every real cutoff (candidate_count == 0 "
                    "included, empty selection)",
        "budget": BUDGET, "arms": list(ARM_NAMES),
        "partition_seed": PARTITION_SEED,
        "n_test_events": n_events, "n_rows": len(rows),
        "expected_rows": expected_rows,
        "empty_selection_rows": empty_rows,
        "mean_primary_macro_f1": mean_primary,
        "frozen_checksum_match": match,
        "new_trainable_parameters": 0,
        "test_time_hyperparameter_search": False,
        "source_checkpoints": {
            "e1_root": args.e1_root, "e2_root": args.e2_root,
            "encoder_ckpt_sha": checksums.get("encoder_ckpt_sha")
            or checksums.get("e2_manifest_encoder_sha"),
            "selector_checkpoint_sha": checksums.get(
                "selector_checkpoint_sha"),
            "e2_manifest_sha256": file_sha256(
                os.path.join(args.e2_root, dataset,
                             f"fold{fold}_seed{seed}",
                             "run_manifest.json"))
            if os.path.exists(os.path.join(
                args.e2_root, dataset, f"fold{fold}_seed{seed}",
                "run_manifest.json")) else None,
        },
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps({"gap": "A", "dataset": dataset, "fold": fold,
                      "seed": seed, "n_events": n_events,
                      "n_rows": len(rows), "empty_rows": empty_rows,
                      "mean_primary": {k: round(v, 4)
                                       for k, v in mean_primary.items()}}),
          flush=True)
    del encoder, selector, proxy
    return manifest


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.all:
        for dataset in DATASETS:
            for fold in FOLDS:
                for seed in SEEDS:
                    run_one(args, dataset, fold, seed)
        return 0
    if args.dataset is None or args.fold is None or args.seed is None:
        raise SystemExit("--dataset/--fold/--seed required unless --all")
    run_one(args, args.dataset, args.fold, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
