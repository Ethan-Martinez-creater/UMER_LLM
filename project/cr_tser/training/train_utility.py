"""Utility predictor training (plan §16, §20, §33).

Frozen protocol::

    AdamW, lr 1e-3, weight_decay 1e-4
    batch 128 reader-intervention rows
    max_epochs 100, patience 10, grad_clip 1.0
    seeds 7319 / 7320 / 7321 (averaged, never selected)

Early stopping uses the **mean utility_dev Spearman across the two training
readers**; when Spearman is undefined (constant predictions or fewer than two
distinct rows) it falls back to negative validation SmoothL1 — exactly the
plan's fallback. Held-out reader labels never enter this module's inputs
(plan §20, §33).
"""
from __future__ import annotations

import math
import random

import torch

from ..config.pilot_config import (ADAMW_LR, ADAMW_WEIGHT_DECAY, BATCH_SIZE,
                                   GRAD_CLIP, MAX_EPOCHS, PATIENCE)
from ..evaluation.bootstrap import spearman_rank as spearman
from ..models.utility_heads import (SIGN_TO_INDEX, SharedResidualUtility,
                                    build_atomic_z, utility_loss)
from .utility_dataset import group_as_graph, row_sign_index


def group_z(bitte, group, device):
    """BiTTE-encode one snapshot and build ``z`` for its labeled units."""
    from ..models.bitte import pack_graph_batch
    batch = pack_graph_batch([group_as_graph(group)], device=device)
    h, h_src = bitte(**batch)
    h, h_src = h[0], h_src[0]
    idx = [r["unit_index"] for r in group["unit_rows"]]
    if group["ctx_indices"]:
        h_ctx = h[group["ctx_indices"]].mean(dim=0)
    else:
        h_ctx = torch.zeros_like(h_src)
    return build_atomic_z(h[idx], h_src, h_ctx, group["q"][idx])


def _batch_forward(bitte, model, groups, reader_index_map, device):
    z_list, reader_idx, targets, signs, keys = [], [], [], [], []
    for g in groups:
        z = group_z(bitte, g, device)
        z_list.append(z)
        for row in g["unit_rows"]:
            reader_idx.append(reader_index_map[row["reader"]])
            targets.append(float(row["target"]))
            signs.append(row_sign_index(row["sign"]))
            keys.append(row["unit_key"])
    if not z_list:
        return None
    z = torch.cat(z_list, dim=0)
    r = torch.tensor(reader_idx, dtype=torch.long, device=device)
    u = torch.tensor(targets, dtype=torch.float32, device=device)
    s = torch.tensor(signs, dtype=torch.long, device=device)
    return z, r, u, s, keys


def shared_targets(keys, targets):
    """``ubar_i`` = mean utility over the training readers (plan §16)."""
    agg = {}
    for key, value in zip(keys, targets):
        agg.setdefault(key, []).append(float(value))
    return [sum(agg[k]) / len(agg[k]) for k in keys]


def _predict_rows(bitte, model, groups, reader_index_map, reader_key, device):
    """Per-unit predicted utility for one reader on the given groups."""
    out = []
    ridx = reader_index_map[reader_key]
    with torch.no_grad():
        for g in groups:
            z = group_z(bitte, g, device)
            r = torch.full((z.shape[0],), ridx, dtype=torch.long, device=device)
            mu, delta, u_hat, _ = model(z, r)
            for row, value in zip(g["unit_rows"], u_hat.tolist()):
                if row["reader"] == reader_key:
                    out.append((row["unit_key"], float(value)))
    return out


def _dev_metric(bitte, model, dev_dataset, reader_keys, device):
    """Mean dev Spearman across training readers; falls back to -SmoothL1."""
    scores, losses = [], []
    with torch.no_grad():
        for key in reader_keys:
            preds, targets = [], []
            for g in dev_dataset.groups:
                z = group_z(bitte, g, device)
                ridx = dev_dataset.reader_index_map[key]
                r = torch.full((z.shape[0],), ridx, dtype=torch.long,
                               device=device)
                _, _, u_hat, _ = model(z, r)
                for row, value in zip(g["unit_rows"], u_hat.tolist()):
                    if row["reader"] == key:
                        preds.append(float(value))
                        targets.append(float(row["target"]))
            if len(preds) >= 2:
                rho = spearman(preds, targets)
                if not math.isnan(rho):
                    scores.append(rho)
                losses.append(sum((p - t) ** 2 for p, t in
                                  zip(preds, targets)) / len(preds))
    if scores:
        return sum(scores) / len(scores), {"metric": "dev_spearman"}
    if losses:
        return -sum(losses) / len(losses), {"metric": "neg_dev_smoothl1_fallback"}
    return float("-inf"), {"metric": "undefined"}


def train_one_seed(train_dataset, dev_dataset, reader_keys, seed, device="cpu",
                   max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
                   batch_size: int = BATCH_SIZE, verbose: bool = False):
    """Train one (rotation, seed) model; returns model + history."""
    from ..models.bitte import BiTTE
    torch.manual_seed(seed)
    bitte = BiTTE().to(device)
    model = SharedResidualUtility(n_training_readers=len(reader_keys)).to(device)
    params = list(bitte.parameters()) + list(model.parameters())
    opt = torch.optim.AdamW(params, lr=ADAMW_LR,
                            weight_decay=ADAMW_WEIGHT_DECAY)
    best = {"score": float("-inf"), "epoch": -1,
            "bitte": None, "model": None}
    history = []
    stale = 0
    for epoch in range(max_epochs):
        bitte.train()
        model.train()
        epoch_rows = 0
        for batch in train_dataset.iter_batches(batch_size, seed=seed + epoch):
            packed = _batch_forward(bitte, model, batch,
                                    train_dataset.reader_index_map, device)
            if packed is None:
                continue
            z, r, u, s, keys = packed
            mu, delta, u_hat, logits = model(z, r)
            shared = torch.tensor(shared_targets(keys, u.tolist()),
                                  dtype=torch.float32, device=device)
            loss, parts = utility_loss(u_hat, u, mu, shared, logits, s,
                                       delta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            epoch_rows += z.shape[0]
        bitte.eval()
        model.eval()
        score, meta = _dev_metric(bitte, model, dev_dataset, reader_keys, device)
        history.append({"epoch": epoch, "rows": epoch_rows, "dev": score,
                        "metric": meta["metric"]})
        if verbose:
            print(f"seed {seed} epoch {epoch} dev {score:.4f} "
                  f"({meta['metric']})")
        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "bitte": {k: v.detach().clone()
                              for k, v in bitte.state_dict().items()},
                    "model": {k: v.detach().clone()
                              for k, v in model.state_dict().items()},
                    "metric": meta["metric"]}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best["bitte"] is not None:
        bitte.load_state_dict(best["bitte"])
        model.load_state_dict(best["model"])
    return {"bitte": bitte, "model": model, "best": best, "history": history,
            "seed": seed}


def train_rotation(train_dataset, dev_dataset, reader_keys, seeds, device="cpu",
                   verbose: bool = False):
    """Train all seeds of one rotation and return them for averaging (§20)."""
    runs = [train_one_seed(train_dataset, dev_dataset, reader_keys, seed,
                           device=device, verbose=verbose) for seed in seeds]
    return {"reader_keys": list(reader_keys), "runs": runs}


def average_predictions(runs, groups, reader_index_map, reader_key, device):
    """Average the three seeds' predictions per unit (never select a seed)."""
    accum = {}
    for run in runs:
        for unit_key, value in _predict_rows(run["bitte"], run["model"], groups,
                                             reader_index_map, reader_key,
                                             device):
            accum.setdefault(unit_key, []).append(value)
    return {k: sum(v) / len(v) for k, v in accum.items()}


def shared_predictions(runs, groups, reader_index_map, device):
    """Average the shared head μ across seeds (S4 arm)."""
    accum = {}
    key_any = next(iter(reader_index_map))
    for run in runs:
        bitte, model = run["bitte"], run["model"]
        with torch.no_grad():
            for g in groups:
                z = group_z(bitte, g, device)
                mu, _ = model.shared_only(z, reader_index_map[key_any])
                for row, value in zip(g["unit_rows"], mu.tolist()):
                    accum.setdefault(row["unit_key"], []).append(float(value))
    return {k: sum(v) / len(v) for k, v in accum.items()}


# --------------------------------------------------------------------------
# B0 / B1 row-level baselines (plan §17) — same protocol, no graph encoder
# --------------------------------------------------------------------------
def _rows_tensor(rows, device):
    x = torch.tensor([r["x"] for r in rows], dtype=torch.float32,
                     device=device)
    y = torch.tensor([float(r["target"]) for r in rows], dtype=torch.float32,
                     device=device)
    signs = torch.tensor([row_sign_index(r["sign"]) for r in rows],
                         dtype=torch.long, device=device)
    return x, y, signs


def _baseline_dev_metric(model, dev_rows, reader_keys, device):
    if not dev_rows:
        return float("-inf"), "undefined"
    scores = []
    with torch.no_grad():
        x, y, _ = _rows_tensor(dev_rows, device)
        pred, _ = model(x)
        pred = pred.tolist()
        targets = y.tolist()
        for key in reader_keys:
            idx = [i for i, r in enumerate(dev_rows) if r["reader"] == key]
            if len(idx) >= 2:
                rho = spearman([pred[i] for i in idx], [targets[i] for i in idx])
                if not math.isnan(rho):
                    scores.append(rho)
    if scores:
        return sum(scores) / len(scores), "dev_spearman"
    mse = sum((p - t) ** 2 for p, t in zip(pred, targets)) / len(pred)
    return -mse, "neg_dev_smoothl1_fallback"


def train_baseline(train_rows, dev_rows, model_factory, seeds, device="cpu",
                   reader_keys=None, batch_size: int = BATCH_SIZE):
    """Train B0/B1 with the plan §16 optimizer/early-stopping protocol."""
    reader_keys = list(reader_keys or sorted({r["reader"]
                                              for r in train_rows}))
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = model_factory().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=ADAMW_LR,
                                weight_decay=ADAMW_WEIGHT_DECAY)
        best = {"score": float("-inf"), "epoch": -1, "state": None,
                "metric": "undefined"}
        stale = 0
        for epoch in range(MAX_EPOCHS):
            model.train()
            order = list(range(len(train_rows)))
            random.Random(seed + epoch).shuffle(order)
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                batch = [train_rows[i] for i in idx]
                x, y, signs = _rows_tensor(batch, device)
                pred, logits = model(x)
                loss = (torch.nn.functional.smooth_l1_loss(pred, y)
                        + 0.5 * torch.nn.functional.cross_entropy(logits, signs))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            model.eval()
            score, metric = _baseline_dev_metric(model, dev_rows, reader_keys,
                                                 device)
            if score > best["score"]:
                best = {"score": score, "epoch": epoch,
                        "state": {k: v.detach().clone()
                                  for k, v in model.state_dict().items()},
                        "metric": metric}
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        runs.append({"model": model, "seed": seed, "best": best})
    return runs


def predict_baseline(runs, rows, device="cpu"):
    """Mean over seeds of the baseline's continuous utility prediction."""
    if not rows:
        return {}
    accum = {}
    x = torch.tensor([r["x"] for r in rows], dtype=torch.float32,
                     device=device)
    for run in runs:
        run["model"].eval()
        with torch.no_grad():
            pred, _ = run["model"](x)
        for row, value in zip(rows, pred.tolist()):
            accum.setdefault(row["key"], []).append(float(value))
    return {k: sum(v) / len(v) for k, v in accum.items()}
