#!/usr/bin/env python
"""STEP 6 entry point: train the causal social encoder (delta-fix §7.1).

Formal Protocol A entry: the split is computed over the full event registry
first (event-level stratified 5-fold + 10% inner validation), then only the
train events are loaded and used for optimization. Validation events are
never loaded in this stage (reserved for model selection infrastructure);
test events are completely frozen — only their ids/label counts appear in
the split manifest.

Usage: python tcdscr_train_encoder.py --dataset pheme --fold 0..4
       [--partition-seed 3090] [--events N (smoke cap)] [--epochs 1]
       [--umer-init PATH]
"""
import argparse
import json
import os

import torch

from tcdscr_common import (SemanticBackend, build_event_snapshots,
                           event_label_registry, label_counts,
                           resolve_train_events)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pheme", "maweibo"))
    ap.add_argument("--fold", type=int, required=True,
                    choices=[0, 1, 2, 3, 4],
                    help="outer fold index 0..4 (Protocol A)")
    ap.add_argument("--partition-seed", type=int, default=3090)
    ap.add_argument("--events", type=int, default=None,
                    help="cap on train events loaded (smoke runs only)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--umer-init", default=None,
                    help="path to a UMER checkpoint for §16.1 init")
    return ap


def make_split_manifest(dataset, fold, partition_seed, split, registry):
    return {
        "dataset": dataset,
        "fold": fold,
        "partition_seed": partition_seed,
        "train_event_ids": sorted(split["train"]),
        "validation_event_ids": sorted(split["validation"]),
        "test_event_ids": sorted(split["test"]),
        "train_label_counts": label_counts(split["train"], registry),
        "validation_label_counts": label_counts(split["validation"], registry),
        "test_label_counts": label_counts(split["test"], registry),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)

    from tcdscr.config.schema import config_from_env
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr.models.causal_social_encoder import (CausalSocialEncoder,
                                                     load_umer_init)
    from tcdscr.training.checkpointing import file_sha256, save_checkpoint
    from tcdscr.training.train_encoder import train_encoder
    cfg = config_from_env(args.dataset)

    registry = event_label_registry(args.dataset, cfg)
    split = build_primary_fold_split(registry, args.fold,
                                     seed=args.partition_seed)
    manifest = make_split_manifest(args.dataset, args.fold,
                                   args.partition_seed, split, registry)
    splits_dir = os.path.join(cfg.output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    with open(os.path.join(splits_dir,
                           f"split_manifest_{args.dataset}_fold{args.fold}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    # only train events are ever loaded; validation/test stay untouched
    events = resolve_train_events(args.dataset, cfg, split,
                                  limit=args.events, seed=args.seed)
    per_event = prepare_features(cfg, args.dataset, events)
    labels = {e["event_id"]: e["label"] for e in events}

    encoder = CausalSocialEncoder()
    init_info = {"mode": "random"}
    if args.umer_init:
        init_info = load_umer_init(encoder, args.umer_init)
        init_info["mode"] = "umer"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = train_encoder(encoder, per_event, labels, epochs=args.epochs,
                            seed=args.seed, device=device)

    ckpt_path = os.path.join(cfg.output_dir, "checkpoints",
                             f"encoder_{args.dataset}_fold{args.fold}.pt")
    save_checkpoint({"model_state_dict": encoder.state_dict(),
                     "init": init_info, "history": history,
                     "fold": args.fold,
                     "partition_seed": args.partition_seed,
                     "train_event_ids": sorted(e["event_id"]
                                               for e in events)},
                    ckpt_path)
    print(json.dumps({"dataset": args.dataset, "fold": args.fold,
                      "init": init_info["mode"], "epochs": args.epochs,
                      "n_train_loaded": len(events),
                      "history": history, "checkpoint": ckpt_path,
                      "checkpoint_sha16": file_sha256(ckpt_path)}))


if __name__ == "__main__":
    main()
