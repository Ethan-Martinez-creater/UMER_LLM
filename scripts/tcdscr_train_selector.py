#!/usr/bin/env python
"""STEP 7/8 entry point: train the static utility selector + proxy
(delta-fix §7.2).

Same Protocol A discipline as the encoder entry: the split is computed over
the full registry; selector training reads train event ids only; validation
is reserved for model selection; test stays completely frozen.

Usage: python tcdscr_train_selector.py --dataset pheme --fold 0..4
       [--partition-seed 3090] [--events N] [--epochs 1]
       [--encoder-ckpt PATH]
"""
import argparse
import json
import os

import torch

from tcdscr_common import (SemanticBackend, build_event_snapshots,
                           event_label_registry, resolve_train_events)


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
    ap.add_argument("--encoder-ckpt", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    from tcdscr.config.schema import config_from_env
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr.models.causal_social_encoder import CausalSocialEncoder
    from tcdscr.models.selector import StaticUtilitySelector
    from tcdscr.models.selector_proxy import SelectorProxy
    from tcdscr.training.checkpointing import file_sha256, save_checkpoint
    from tcdscr.training.train_selector import train_selector
    cfg = config_from_env(args.dataset)

    registry = event_label_registry(args.dataset, cfg)
    split = build_primary_fold_split(registry, args.fold,
                                     seed=args.partition_seed)
    from tcdscr_train_encoder import make_split_manifest
    manifest = make_split_manifest(args.dataset, args.fold,
                                   args.partition_seed, split, registry)
    splits_dir = os.path.join(cfg.output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    with open(os.path.join(splits_dir,
                           f"split_manifest_{args.dataset}_fold{args.fold}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    events = resolve_train_events(args.dataset, cfg, split,
                                  limit=args.events, seed=args.seed)
    per_event = prepare_features(cfg, args.dataset, events)
    labels = {e["event_id"]: e["label"] for e in events}

    encoder = CausalSocialEncoder()
    if args.encoder_ckpt:
        blob = torch.load(args.encoder_ckpt, map_location="cpu",
                          weights_only=True)
        encoder.load_state_dict(blob["model_state_dict"])
    selector = StaticUtilitySelector()
    proxy = SelectorProxy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = train_selector(encoder, selector, proxy, per_event, labels,
                             epochs=args.epochs, seed=args.seed,
                             device=device)

    out_dir = os.path.join(cfg.output_dir, "checkpoints")
    sel_path = os.path.join(
        out_dir, f"selector_{args.dataset}_fold{args.fold}.pt")
    save_checkpoint({
        "selector_state_dict": selector.state_dict(),
        "proxy_state_dict": proxy.state_dict(),
        "encoder_checkpoint": args.encoder_ckpt,
        "fold": args.fold,
        "partition_seed": args.partition_seed,
        "train_event_ids": sorted(e["event_id"] for e in events),
        "history": history,
    }, sel_path)
    print(json.dumps({"dataset": args.dataset, "fold": args.fold,
                      "epochs": args.epochs, "n_train_loaded": len(events),
                      "history": history, "checkpoint": sel_path,
                      "checkpoint_sha16": file_sha256(sel_path)}))


if __name__ == "__main__":
    main()
