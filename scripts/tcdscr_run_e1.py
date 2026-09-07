#!/usr/bin/env python
"""Formal E1 — Causal Encoder (plan §36/§37).

One run = (dataset, fold, init, seed). Protocol A: the split is resolved
over the full registry first (StratifiedKFold 5 folds seed 3090 + stratified
10% inner validation); only that fold's events are loaded. Training samples
one dynamic snapshot per train event per epoch (SOURCE_ONLY excluded);
model selection uses the inner validation set (fixed per-event snapshot
draw, macro-F1, patience-based early stopping); the test set is evaluated
once with the best checkpoint on every causal view (SOURCE_ONLY, 5m..6h,
24h).

Hyperparameters follow the frozen UMER five-fold training recipe
(batch 32, max 60 epochs, patience 7, AdamW lr 1e-4, weight decay 0.05,
label smoothing 0.1) and are recorded in every run manifest.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

from tcdscr_common import (PROJECT_DIR, EventSemanticStore,
                           event_label_registry, load_split_events)

PRIMARY_CUTOFFS = (5, 15, 30, 60, 180, 360)
EVAL_CUTOFFS = ("SOURCE_ONLY",) + PRIMARY_CUTOFFS + (1440,)

# frozen UMER five-fold training recipe
HPARAMS = {
    "batch_size": 32,
    "max_epochs": 60,
    "patience": 7,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
}

UMER_CKPT_TEMPLATE = ("/data/jyz/next/llm/checkpoints/{dataset}/"
                      "fold_{fold1}_train_2000/best_joint_model.pt")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pheme", "maweibo"))
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--init", required=True,
                    choices=("random", "umer", "both"))
    ap.add_argument("--seeds", default="2000,2001,2002")
    ap.add_argument("--epochs", type=int, default=HPARAMS["max_epochs"])
    ap.add_argument("--batch-size", type=int, default=HPARAMS["batch_size"])
    ap.add_argument("--patience", type=int, default=HPARAMS["patience"])
    ap.add_argument("--lr", type=float, default=HPARAMS["lr"])
    ap.add_argument("--out-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e1")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap per split for pipeline smoke only")
    return ap


def light_features(event, snapshot, store):
    """Light per-snapshot item: O(N+E) data only (semantics row stack,
    edges, 3D summary). The 1021D signature is densified on GPU in
    collate_light_semantic with the exact historical formula."""
    import torch

    from tcdscr.data.structural_features import structural_summary
    sem = torch.stack([store[nid] for nid in snapshot["node_ids"]])
    edge = torch.tensor(snapshot["edge_index"], dtype=torch.long).t() \
        if snapshot["edge_index"] else torch.empty((2, 0), dtype=torch.long)
    return {
        "event_id": event["event_id"],
        "label": event["label"],
        "num_nodes": len(snapshot["node_ids"]),
        "sem": sem,
        "edge_index": edge,
        "summary": structural_summary(snapshot),
        "source_pos": snapshot["node_ids"].index(snapshot["source_id"]),
    }


def collate_light_semantic(items, device):
    """Batch light items; build the 1021D adjacency signature on GPU with
    the historical construction (adj[parent, child]=1, self-loop diagonal,
    row-normalized by its own out-degree + 1e-8, columns to 1021)."""
    import torch

    batch = len(items)
    max_nodes = max(it["num_nodes"] for it in items)
    node_feat = torch.zeros(batch, max_nodes, 384)
    summary = torch.zeros(batch, max_nodes, 3)
    num_nodes = torch.tensor([it["num_nodes"] for it in items],
                             dtype=torch.long)
    labels = torch.tensor([it["label"] for it in items], dtype=torch.long)
    for b, it in enumerate(items):
        n = it["num_nodes"]
        node_feat[b, :n] = it["sem"]
        summary[b, :n] = it["summary"]
    adj = torch.zeros(batch, max_nodes, 1021, device=device)
    edge_list = []
    adj_edges = []
    for b, it in enumerate(items):
        if it["edge_index"].numel():
            e = it["edge_index"]
            # global [child; parent] rows for the returned edge_index
            edge_list.append(e + b * max_nodes)
            # for the signature: parent indexes the flattened batch rows,
            # child stays a LOCAL column index inside [0, 1021)
            adj_edges.append(torch.stack([e[1] + b * max_nodes, e[0]]))
    edge_index = torch.cat(edge_list, dim=1) if edge_list \
        else torch.empty((2, 0), dtype=torch.long)
    if adj_edges:
        pe = torch.cat(adj_edges, dim=1).to(device)
        # 2-D advanced indexing on the flattened batch rows: pe[0] = global
        # parent row in [0, batch*max_nodes), pe[1] = child column < 1021
        adj.view(batch * max_nodes, 1021)[pe[0], pe[1]] = 1.0
    node_range = torch.arange(max_nodes, device=device)
    for b, it in enumerate(items):
        n = it["num_nodes"]
        adj[b, node_range[:n], node_range[:n]] = 1.0
    adj = adj / (adj.sum(dim=2, keepdim=True) + 1e-8)
    struct = torch.cat(
        [adj, summary.to(device)], dim=2)
    return (node_feat.to(device), struct, num_nodes.to(device),
            edge_index, labels.to(device))


def evaluate_view(encoder, items, device, batch_size=64):
    """Forward a list of light items; returns per-item p_rumor + prediction."""
    import torch

    encoder.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            node_feat, struct, num_nodes, _edges, _y = collate_light_semantic(
                chunk, device)
            _h, _g, logits = encoder(node_feat, struct, num_nodes)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(dim=-1)
            outs.extend(zip(prob.tolist(), pred.tolist()))
    return outs


def macro_f1_of(pairs):
    """Quick macro-F1 over (pred, gold) pairs for model selection."""
    tp1 = sum(1 for p, g in pairs if p == 1 and g == 1)
    fp1 = sum(1 for p, g in pairs if p == 0 and g == 1)
    fn1 = sum(1 for p, g in pairs if p == 1 and g == 0)
    tp0 = sum(1 for p, g in pairs if p == 0 and g == 0)
    fp0 = sum(1 for p, g in pairs if p == 1 and g == 0)
    fn0 = sum(1 for p, g in pairs if p == 0 and g == 1)

    def f1(tp, fp, fn):
        pr = tp / (tp + fp) if tp + fp else 0.0
        re = tp / (tp + fn) if tp + fn else 0.0
        return 2 * pr * re / (pr + re) if pr + re else 0.0

    return (f1(tp1, fp1, fn1) + f1(tp0, fp0, fn0)) / 2


def run_one(args, dataset, cfg, fold, init_mode, seed, store, registry):
    import torch

    from tcdscr.data.snapshot_builder import build_snapshot, build_source_only
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr.models.causal_social_encoder import CausalSocialEncoder, load_umer_init
    from tcdscr.training.checkpointing import file_sha256
    from tcdscr.training.sampler import SnapshotSampler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = build_primary_fold_split(registry, fold, seed=3090)
    events = load_split_events(dataset, cfg, split)
    if args.limit:
        for seg in events:
            events[seg] = events[seg][:args.limit]

    run_dir = os.path.join(args.out_root, dataset,
                           f"fold{fold}_{init_mode}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed)
    random.seed(seed)
    encoder = CausalSocialEncoder()
    init_info = {"mode": init_mode}
    if init_mode == "umer":
        ckpt = UMER_CKPT_TEMPLATE.format(dataset=dataset, fold1=fold + 1)
        init_info.update(load_umer_init(encoder, ckpt))
    encoder.to(device)

    # fixed validation draw: one dynamic snapshot per val event
    val_rng = random.Random(5000 + fold)
    val_snapshots = {}
    for ev in events["validation"]:
        cuts = sorted(c for c in PRIMARY_CUTOFFS)
        val_snapshots[ev["event_id"]] = val_rng.choice(cuts)

    def train_items_epoch(epoch_rng):
        order = sorted(events["train"], key=lambda e: e["event_id"])
        epoch_rng.shuffle(order)
        items = []
        for ev in order:
            cut = epoch_rng.choice(PRIMARY_CUTOFFS)
            snap = build_snapshot(ev, cut)
            items.append(light_features(ev, snap, store.get_store(ev, cache)))
        return items

    cache = {}
    val_items = []
    for ev in events["validation"]:
        snap = build_snapshot(ev, val_snapshots[ev["event_id"]])
        val_items.append(light_features(ev, snap, store.get_store(ev, cache)))
    val_pairs = [(ev["event_id"], ev["label"]) for ev in events["validation"]]

    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr,
                            weight_decay=HPARAMS["weight_decay"])
    best_val, best_state, best_epoch, wait = -1.0, None, -1, 0
    history = []
    epoch_rng = random.Random(seed)
    for epoch in range(args.epochs):
        encoder.train()
        items = train_items_epoch(epoch_rng)
        order = list(range(len(items)))
        epoch_rng.shuffle(order)
        total_loss, n_steps = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            chunk = [items[j] for j in order[i:i + args.batch_size]]
            node_feat, struct, num_nodes, _e, y = collate_light_semantic(
                chunk, device)
            logits = encoder(node_feat, struct, num_nodes)[2]
            loss = torch.nn.functional.cross_entropy(
                logits, y, label_smoothing=HPARAMS["label_smoothing"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss)
            n_steps += 1
        # validation model selection
        encoder.eval()
        outs = evaluate_view(encoder, val_items, device)
        pairs = [(pred, val_pairs[k][1]) for k, (_p, pred) in
                 enumerate(outs)]
        val_f1 = macro_f1_of(pairs)
        history.append({"epoch": epoch, "train_loss": total_loss
                        / max(n_steps, 1), "val_macro_f1": val_f1})
        if val_f1 > best_val:
            best_val, best_epoch, wait = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in encoder.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
        print(f"[{dataset} fold{fold} {init_mode} s{seed}] "
              f"epoch {epoch} loss={history[-1]['train_loss']:.4f} "
              f"val_f1={val_f1:.4f} best={best_val:.4f}", flush=True)

    if best_state is not None:
        encoder.load_state_dict(best_state)
    ckpt_path = os.path.join(run_dir, "best_encoder.pt")
    torch.save({"model_state_dict": encoder.state_dict(),
                "init": init_info, "fold": fold, "seed": seed,
                "best_epoch": best_epoch, "best_val_macro_f1": best_val},
               ckpt_path)

    # test evaluation on every causal view
    rows = []
    metrics = {}
    for cutoff in EVAL_CUTOFFS:
        items = []
        for ev in events["test"]:
            snap = build_source_only(ev) if cutoff == "SOURCE_ONLY" \
                else build_snapshot(ev, cutoff)
            items.append(light_features(ev, snap, store.get_store(ev, cache)))
        outs = evaluate_view(encoder, items, device)
        for ev, (p_rumor, pred) in zip(events["test"], outs):
            rows.append({"event_id": ev["event_id"], "cutoff": str(cutoff),
                         "gold": ev["label"], "pred": int(pred),
                         "p_rumor": p_rumor})
        pairs = [(ev["label"], pred) for ev, (_p, pred)
                 in zip(events["test"], outs)]
        metrics[str(cutoff)] = classification_metrics_full(pairs)

    manifest = {
        "stage": "formal_e1",
        "dataset": dataset,
        "fold": fold,
        "init": init_mode,
        "seed": seed,
        "hparams": {**HPARAMS, "epochs_run": args.epochs, "lr": args.lr,
                    "batch_size": args.batch_size,
                    "patience": args.patience},
        "split": {k: len(v) for k, v in split.items()},
        "partition_seed": 3090,
        "init_info": {k: v for k, v in init_info.items()
                      if k != "copied"} | {"n_copied_tensors":
                                           len(init_info.get("copied", []))},
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "device": device,
        "checkpoint_sha16": file_sha256(ckpt_path),
        "encoder_umer_ckpt": UMER_CKPT_TEMPLATE.format(
            dataset=dataset, fold1=fold + 1) if init_mode == "umer" else None,
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    with open(os.path.join(run_dir, "metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)
    with open(os.path.join(run_dir, "predictions.jsonl"), "w",
              encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    with open(os.path.join(run_dir, "history.json"), "w",
              encoding="utf-8") as fh:
        json.dump(history, fh, indent=1)
    print("RUN DONE", run_dir, "best_epoch", best_epoch,
          "val", round(best_val, 4), flush=True)
    return manifest


def classification_metrics_full(pairs):
    """pairs: (gold, pred) -> Accuracy / Macro-F1 / Weighted-F1 / Rumor-F1."""
    n = max(len(pairs), 1)
    acc = sum(1 for g, p in pairs if g == p) / n
    tp1 = sum(1 for g, p in pairs if g == 1 and p == 1)
    fp1 = sum(1 for g, p in pairs if g == 0 and p == 1)
    fn1 = sum(1 for g, p in pairs if g == 1 and p == 0)
    tp0 = sum(1 for g, p in pairs if g == 0 and p == 0)
    fp0 = sum(1 for g, p in pairs if g == 1 and p == 0)
    fn0 = sum(1 for g, p in pairs if g == 0 and p == 1)

    def prf(tp, fp, fn):
        pr = tp / (tp + fp) if tp + fp else 0.0
        re = tp / (tp + fn) if tp + fn else 0.0
        return pr, re, (2 * pr * re / (pr + re) if pr + re else 0.0)

    _, _, f1_rumor = prf(tp1, fp1, fn1)
    _, _, f1_non = prf(tp0, fp0, fn0)
    s1, s0 = tp1 + fn1, tp0 + fn0
    return {
        "n": len(pairs),
        "accuracy": acc,
        "macro_f1": (f1_rumor + f1_non) / 2,
        "weighted_f1": (f1_rumor * s1 + f1_non * s0) / max(s1 + s0, 1),
        "rumor_f1": f1_rumor,
        "support_rumor": s1,
        "support_nonrumor": s0,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    from tcdscr.config.schema import config_from_env
    cfg = config_from_env(args.dataset)
    registry = event_label_registry(args.dataset, cfg)
    store = EventSemanticStore(cfg)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    init_modes = ["random", "umer"] if args.init == "both" else [args.init]
    for init_mode in init_modes:
        for seed in seeds:
            run_one(args, args.dataset, cfg, args.fold, init_mode, seed,
                    store, registry)


if __name__ == "__main__":
    main()
