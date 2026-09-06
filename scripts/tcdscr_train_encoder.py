#!/usr/bin/env python
"""STEP 6 entry point: train the causal social encoder.

Usage: python tcdscr_train_encoder.py --dataset pheme [--events 16]
       [--epochs 1] [--umer-init PATH]
"""
import argparse
import json
import os

import torch

from tcdscr_common import (SemanticBackend, build_event_snapshots,
                           load_events)


def prepare_features(cfg, dataset, events):
    backend = SemanticBackend(cfg)
    dynamic = [c for c in cfg.primary_cutoffs_min]
    per_event = {}
    for event in events:
        snaps = build_event_snapshots(event, dynamic)
        per_event[event["event_id"]] = {
            cut: backend.snapshot_features(event, snap)
            for cut, snap in snaps.items() if cut != "SOURCE_ONLY"}
    return per_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pheme", "maweibo"))
    ap.add_argument("--events", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--umer-init", default=None,
                    help="path to a UMER checkpoint for §16.1 init")
    args = ap.parse_args()

    from tcdscr.config.schema import config_from_env
    from tcdscr.models.causal_social_encoder import (CausalSocialEncoder,
                                                     load_umer_init)
    from tcdscr.training.checkpointing import file_sha256, save_checkpoint
    from tcdscr.training.train_encoder import train_encoder
    cfg = config_from_env(args.dataset)
    events = load_events(args.dataset, cfg, limit=args.events, seed=3090)
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
                             f"encoder_{args.dataset}_smoke.pt")
    save_checkpoint({"model_state_dict": encoder.state_dict(),
                     "init": init_info, "history": history}, ckpt_path)
    print(json.dumps({"dataset": args.dataset, "init": init_info["mode"],
                      "epochs": args.epochs, "history": history,
                      "checkpoint": ckpt_path,
                      "checkpoint_sha16": file_sha256(ckpt_path)}))


if __name__ == "__main__":
    main()
