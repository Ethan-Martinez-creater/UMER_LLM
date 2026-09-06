#!/usr/bin/env python
"""STEP 7/8 entry point: train the static utility selector + proxy.

Usage: python tcdscr_train_selector.py --dataset pheme [--events 16]
       [--epochs 1] [--encoder-ckpt PATH]
"""
import argparse
import json
import os

import torch

from tcdscr_common import (SemanticBackend, build_event_snapshots,
                           load_events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pheme", "maweibo"))
    ap.add_argument("--events", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--encoder-ckpt", default=None)
    args = ap.parse_args()

    from tcdscr.config.schema import config_from_env
    from tcdscr.models.causal_social_encoder import CausalSocialEncoder
    from tcdscr.models.selector import StaticUtilitySelector
    from tcdscr.models.selector_proxy import SelectorProxy
    from tcdscr.training.checkpointing import file_sha256, save_checkpoint
    from tcdscr.training.train_selector import train_selector
    cfg = config_from_env(args.dataset)
    events = load_events(args.dataset, cfg, limit=args.events, seed=3090)

    from tcdscr_train_encoder import prepare_features
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
    sel_path = os.path.join(out_dir, f"selector_{args.dataset}_smoke.pt")
    save_checkpoint({
        "selector_state_dict": selector.state_dict(),
        "proxy_state_dict": proxy.state_dict(),
        "encoder_checkpoint": args.encoder_ckpt,
        "history": history,
    }, sel_path)
    print(json.dumps({"dataset": args.dataset, "epochs": args.epochs,
                      "history": history, "checkpoint": sel_path,
                      "checkpoint_sha16": file_sha256(sel_path)}))


if __name__ == "__main__":
    main()
