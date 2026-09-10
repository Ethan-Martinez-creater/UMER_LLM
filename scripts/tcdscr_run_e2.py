#!/usr/bin/env python
"""Formal E2 — Static Utility Selector (plan §38; E2 order §1–§30).

One run = (dataset, fold, seed). The Random-init E1 causal encoder is
loaded from the matching E1 checkpoint (a UMER-init checkpoint is refused),
frozen (eval + requires_grad_(False), parameter checksums before/after),
and used to train the Static Utility Selector and Proxy Classifier on the
E1-inherited train split with L = L_cls + 1.0*L_fid + 0.05*L_div, where
L_fid is the true KL(stopgrad(p_full) || p_sel).

Each epoch draws one dynamic snapshot per train event (SnapshotSampler,
seed derived from the run seed; SOURCE_ONLY/24h never used). Model
selection uses the STATIC arm's validation mean Macro-F1 over the six
primary cutoffs (patience early stopping; frozen recipe: AdamW lr 1e-4,
weight decay 0.01, max 30 epochs, patience 5).

Validation additionally scores the two frozen simple baselines — Random
(deterministic per dataset/event/cutoff/run-seed) and Semantic
(cos(e_i, e_source)) — under the identical 1024-evidence-token budget
(Qwen3-8B tokenizer, Reply-Parent pairs, atomic pair rule, no top-k). All
three arms classify through the same proxy head with uniform alpha over
their selected units, so the selection strategy is the only difference.

The test split is never evaluated in the training pass: a separate
--evaluate-test-only run (only after both datasets pass readiness) loads
best_selector.pt and scores the test split. No LLM is called anywhere.
"""
import argparse
import hashlib
import json
import os
import random
import time

import torch

from tcdscr_common import (EventSemanticStore, event_label_registry,
                           load_split_events)

PRIMARY_CUTOFFS = (5, 15, 30, 60, 180, 360)
ARM_NAMES = ("static", "random", "semantic")
METRIC_KEYS = ("accuracy", "macro_f1", "weighted_f1", "rumor_f1")

# frozen E2 selector recipe (§9): no separate selector recipe exists in the
# V2 formal code beyond the smoke loop, so these values are fixed here
E2_HPARAMS = {
    "lr": 1e-4,
    "weight_decay": 0.01,
    "max_epochs": 30,
    "patience": 5,
    "budget_tokens": 1024,
    "tau": 0.5,
    "loss_weights": {"fid": 1.0, "div": 0.05},
}

E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pheme", "maweibo"))
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", default="2000,2001,2002")
    ap.add_argument("--epochs", type=int, default=E2_HPARAMS["max_epochs"])
    ap.add_argument("--patience", type=int, default=E2_HPARAMS["patience"])
    ap.add_argument("--lr", type=float, default=E2_HPARAMS["lr"])
    ap.add_argument("--weight-decay", type=float,
                    default=E2_HPARAMS["weight_decay"])
    ap.add_argument("--budget", type=int, default=E2_HPARAMS["budget_tokens"])
    ap.add_argument("--e1-root", default=E1_ROOT)
    ap.add_argument("--out-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e2")
    ap.add_argument("--evaluate-test-only", action="store_true",
                    help="skip training; score the test split with the saved "
                         "best_selector.pt (E2-B, only after readiness PASS)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap per split for pipeline smoke only")
    return ap


def state_sha256(state_dict) -> str:
    blob = b"".join(v.detach().cpu().contiguous().numpy().tobytes()
                    for v in state_dict.values())
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classification_metrics(pairs):
    n = max(len(pairs), 1)
    acc = sum(1 for g, p in pairs if g == p) / n
    tp1 = sum(1 for g, p in pairs if g == 1 and p == 1)
    fp1 = sum(1 for g, p in pairs if g == 0 and p == 1)
    fn1 = sum(1 for g, p in pairs if g == 1 and p == 0)
    tp0 = sum(1 for g, p in pairs if g == 0 and p == 0)
    fp0 = sum(1 for g, p in pairs if g == 1 and p == 0)
    fn0 = sum(1 for g, p in pairs if g == 0 and p == 1)

    def f1(tp, fp, fn):
        pr = tp / (tp + fp) if tp + fp else 0.0
        re = tp / (tp + fn) if tp + fn else 0.0
        return 2 * pr * re / (pr + re) if pr + re else 0.0

    s1, s0 = tp1 + fn1, tp0 + fn0
    return {"n": len(pairs), "accuracy": acc,
            "macro_f1": (f1(tp1, fp1, fn1) + f1(tp0, fp0, fn0)) / 2,
            "weighted_f1": (f1(tp1, fp1, fn1) * s1
                            + f1(tp0, fp0, fn0) * s0) / max(s1 + s0, 1),
            "rumor_f1": f1(tp1, fp1, fn1)}


def mean_primary_macro_f1(metrics_by_cutoff):
    vals = [metrics_by_cutoff[str(c)]["macro_f1"]
            for c in PRIMARY_CUTOFFS
            if str(c) in metrics_by_cutoff]
    return sum(vals) / len(vals) if vals else 0.0


def readiness_pass(static_mean, best_baseline_mean, threshold=0.005):
    """Readiness gate (§15): static must beat the best simple baseline by at
    least 0.005 = 0.5 Macro-F1 *percentage point* (NOT +0.5 absolute F1)."""
    return (static_mean - best_baseline_mean) >= threshold


def random_budget_scores(event_id, cutoff, n_units, run_seed, dataset):
    """Deterministic per event/snapshot/run seed (§12)."""
    digest = hashlib.sha256(
        f"{dataset}:{event_id}:{cutoff}:{run_seed}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return [rng.random() for _ in range(n_units)]


def build_light_item(event, snapshot, store_rows):
    """Per-snapshot item: light tensors plus the snapshot fields needed to
    build evidence units and the selection context."""
    import torch

    from tcdscr.data.structural_features import structural_summary
    sem = torch.stack([store_rows[nid] for nid in snapshot["node_ids"]])
    edge = torch.tensor(snapshot["edge_index"], dtype=torch.long).t() \
        if snapshot["edge_index"] else torch.empty((2, 0), dtype=torch.long)
    return {
        "event_id": event["event_id"],
        "label": event["label"],
        "num_nodes": len(snapshot["node_ids"]),
        "sem": sem,
        "edge_index": edge,
        "summary": structural_summary(snapshot),
        "node_ids": list(snapshot["node_ids"]),
        "texts": list(snapshot["texts"]),
        "depths": list(snapshot["depths"]),
        "elapsed": list(snapshot["elapsed_seconds"]),
        "parent_ids": list(snapshot["parent_ids"]),
        "source_id": snapshot["source_id"],
        "cutoff_minutes": snapshot["cutoff_minutes"],
        "source_pos": snapshot["node_ids"].index(snapshot["source_id"]),
    }


def collate_light(items, device):
    """Batch light items; 1021D adjacency signature built on GPU with the
    historical formula (same as the E1 collate path)."""
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
    adj_edges = []
    for b, it in enumerate(items):
        if it["edge_index"].numel():
            e = it["edge_index"]
            adj_edges.append(torch.stack([e[1] + b * max_nodes, e[0]]))
    if adj_edges:
        pe = torch.cat(adj_edges, dim=1).to(device)
        adj.view(batch * max_nodes, 1021)[pe[0], pe[1]] = 1.0
    node_range = torch.arange(max_nodes, device=device)
    for b, it in enumerate(items):
        n = it["num_nodes"]
        adj[b, node_range[:n], node_range[:n]] = 1.0
    adj = adj / (adj.sum(dim=2, keepdim=True) + 1e-8)
    struct = torch.cat([adj, summary.to(device)], dim=2)
    return (node_feat.to(device), struct, num_nodes.to(device),
            labels.to(device))


def encoder_forward_batch(encoder, items, device, batch_size=32):
    """Frozen encoder forward over items, batched (E1-style) to bound GPU
    memory; returns per-item (node_repr, event_repr, logits_full). Callers
    must consume each batch promptly (see evaluate/train loops) so GPU
    memory stays bounded for large-snapshot datasets."""
    out = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        node_feat, struct, num_nodes, _ = collate_light(chunk, device)
        with torch.no_grad():
            node_repr, event_repr, logits = encoder(node_feat, struct,
                                                    num_nodes)
        for b, it in enumerate(chunk):
            n = it["num_nodes"]
            out.append((node_repr[b, :n], event_repr[b], logits[b]))
    return out


def build_units(item):
    from tcdscr.context.evidence_unit import build_evidence_units
    return build_evidence_units({
        "node_ids": item["node_ids"],
        "texts": item["texts"],
        "depths": item["depths"],
        "elapsed_seconds": item["elapsed"],
        "parent_ids": item["parent_ids"],
        "source_id": item["source_id"],
    })


def evaluate_arms_on_items(encoder, selector, proxy, items, budget_selector,
                           device, dataset, run_seed):
    """Score the three arms over (event, cutoff) items.

    Classification: every arm selects a budgeted unit set; the proxy head
    classifies [mean(selected h); h_source] with uniform alpha, so the
    selection strategy is the only difference between arms. Returns
    (metrics_by_cutoff, prediction_rows, diagnostics).
    """
    import torch.nn.functional as F

    from tcdscr.models.selector_proxy import classify_selected

    results = {str(c): {a: [] for a in ARM_NAMES} | {"e1_full": []}
               for c in PRIMARY_CUTOFFS}
    rows = []
    diag = {str(c): {"score_mean": [], "score_std": [], "entropy": [],
                     "selected_count": [], "n_units": [],
                     "evidence_tokens": [], "jaccard": []}
            for c in PRIMARY_CUTOFFS}

    def process_one(item, node_repr, event_repr, logits_full):
        cutoff = str(item["cutoff_minutes"])
        n = item["num_nodes"]
        src = item["source_pos"]
        cand = [i for i in range(n) if i != src]
        if not cand:
            return
        cand_idx = torch.tensor(cand, dtype=torch.long, device=device)
        h_cand = node_repr[cand_idx]
        sem = item["sem"].to(device)
        struct3 = item["summary"].to(device)
        units = build_units(item)
        unit_index = {u2["node_id"]: i for i, u2 in enumerate(units)}
        # E1 full-encoder sanity view (diagnostic only, not an arm)
        results[cutoff]["e1_full"].append(
            (item["label"], int(logits_full.argmax(dim=-1))))
        u = selector(h_cand, event_repr, sem[cand_idx], sem[src],
                     struct3[cand_idx])
        alpha = F.softmax(u / E2_HPARAMS["tau"], dim=0)
        entropy = -float((alpha * (alpha + 1e-12).log()).sum())
        u_list = u.detach().cpu().tolist()
        mean_u = sum(u_list) / len(u_list)
        std_u = (sum((x - mean_u) ** 2 for x in u_list)
                 / len(u_list)) ** 0.5
        cos_scores = F.cosine_similarity(
            sem[cand_idx], sem[src].unsqueeze(0).expand_as(sem[cand_idx]),
            dim=-1).detach().cpu().tolist()
        cat_scores = {"static": u_list, "semantic": cos_scores}
        selections = {}
        h_source = node_repr[src]
        for arm in ARM_NAMES:
            scores = (random_budget_scores(item["event_id"],
                                           item["cutoff_minutes"],
                                           len(units), run_seed, dataset)
                      if arm == "random" else cat_scores[arm])
            selected, tokens = budget_selector.select(units, scores)
            selections[arm] = (selected, tokens)
            if selected:
                sel_idx = [cand[unit_index[u2["node_id"]]] for u2 in selected]
                sel_repr = node_repr[torch.tensor(sel_idx, dtype=torch.long,
                                                  device=device)]
                logits = classify_selected(proxy, h_source, sel_repr)
            else:
                z_sel = torch.zeros(768, device=device)
                logits = proxy.classify(h_source, z_sel)
            p = F.softmax(logits, dim=-1)
            pred = int(logits.argmax(dim=-1))
            results[cutoff][arm].append((item["label"], pred))
            rows.append({
                "event_id": item["event_id"], "cutoff": cutoff,
                "gold": item["label"], "method": arm, "pred": pred,
                "p_rumor": float(p[1]),
                "selected_node_ids": [u2["node_id"] for u2 in selected],
                "selected_count": len(selected),
                "evidence_tokens": tokens,
                "source_id": item["source_id"],
            })
        s_set = {u2["node_id"] for u2 in selections["static"][0]}
        m_set = {u2["node_id"] for u2 in selections["semantic"][0]}
        union = s_set | m_set
        if not union:
            return
        diag[cutoff]["jaccard"].append(len(s_set & m_set) / len(union))
        diag[cutoff]["score_mean"].append(mean_u)
        diag[cutoff]["score_std"].append(std_u)
        diag[cutoff]["entropy"].append(entropy)
        diag[cutoff]["selected_count"].append(len(selections["static"][0]))
        diag[cutoff]["evidence_tokens"].append(selections["static"][1])
        diag[cutoff]["n_units"].append(len(units))

    # forward + consume in chunks so GPU memory stays bounded
    for start in range(0, len(items), 32):
        chunk = items[start:start + 32]
        outs = encoder_forward_batch(encoder, chunk, device)
        for item, (node_repr, event_repr, logits_full) in zip(chunk, outs):
            process_one(item, node_repr, event_repr, logits_full)
        del outs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    metrics_by_cutoff = {c: {a: classification_metrics(results[c][a])
                             for a in ARM_NAMES} | {
                             "e1_full": classification_metrics(
                                 results[c]["e1_full"])}
                         for c in results}
    diagnostics = {}
    for c, d in diag.items():
        k = len(d["entropy"])
        diagnostics[c] = {
            "n": k,
            "mean_selected_count": (sum(d["selected_count"]) / k if k else 0.0),
            "mean_evidence_tokens": (sum(d["evidence_tokens"]) / k if k else 0.0),
            "mean_n_units": (sum(d["n_units"]) / k if k else 0.0),
            "selection_score_mean": (sum(d["score_mean"]) / k if k else 0.0),
            "selection_score_std": (sum(d["score_std"]) / k if k else 0.0),
            "mean_selection_entropy": (sum(d["entropy"]) / k if k else 0.0),
            "mean_jaccard_static_semantic":
                (sum(d["jaccard"]) / len(d["jaccard"]) if d["jaccard"] else 0.0),
        }
    return metrics_by_cutoff, rows, diagnostics


def load_random_e1_encoder(dataset, fold, seed, e1_root, device):
    """Load and freeze the Random-init E1 encoder; refuse UMER-init."""
    from tcdscr.models.causal_social_encoder import CausalSocialEncoder
    e1_run = os.path.join(e1_root, dataset, f"fold{fold}_random_seed{seed}")
    manifest_path = os.path.join(e1_run, "run_manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        e1_manifest = json.load(fh)
    if e1_manifest.get("init") != "random":
        raise RuntimeError(
            f"E2 requires the Random-init E1 encoder, found "
            f"init={e1_manifest.get('init')} at {e1_run}")
    ckpt_path = os.path.join(e1_run, "best_encoder.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    encoder = CausalSocialEncoder()
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, ckpt_path, e1_manifest


def check_split_parity(dataset, fold, seed, split, events, e1_root,
                       registry):
    """Verify E1/E2 split exact match.

    E2 rebuilds the identical builder + registry call as E1 (same function,
    partition_seed=3090), so this is an audit, not an independent draw:
    (a) E2 set sizes must equal the E1 manifest counts; (b) the E2 test ids
    must equal the E1 test ids recovered from the *independent* E1
    predictions artifact; (c) E2 sets must be pairwise disjoint and cover
    the registry (E1 train/validation ids are not stored in E1 artifacts,
    so their equality follows from the identical builder+registry, the
    counts, and the disjoint complete coverage given (b)).
    """
    e1_run = os.path.join(e1_root, dataset, f"fold{fold}_random_seed{seed}")
    with open(os.path.join(e1_run, "run_manifest.json"),
              encoding="utf-8") as fh:
        e1_manifest = json.load(fh)
    counts_ok = e1_manifest.get("split") == \
        {k: len(v) for k, v in split.items()}
    e1_test_ids = set()
    with open(os.path.join(e1_run, "predictions.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                e1_test_ids.add(json.loads(line)["event_id"])
    test_ok = set(split["test"]) == e1_test_ids
    tr, va, te = (set(split["train"]), set(split["validation"]),
                  set(split["test"]))
    disjoint = not (tr & va or tr & te or va & te)
    cover = tr | va | te == set(registry)
    loaded_ok = all({ev["event_id"] for ev in events[seg]} <= set(split[seg])
                    for seg in ("train", "validation", "test"))
    train_val_ok = counts_ok and disjoint and cover and test_ok and loaded_ok
    return {
        "e1_train_count": e1_manifest["split"]["train"],
        "e2_train_count": len(split["train"]),
        "e1_val_count": e1_manifest["split"]["validation"],
        "e2_val_count": len(split["validation"]),
        "e1_test_count": e1_manifest["split"]["test"],
        "e2_test_count": len(split["test"]),
        "train_exact_match": train_val_ok,
        "validation_exact_match": train_val_ok,
        "test_exact_match": test_ok,
        "e1_test_ids_recovered": len(e1_test_ids),
        "disjoint": disjoint,
        "covers_registry": cover,
        "note": ("E1 train/validation ids are not stored in the E1 "
                 "artifacts; equality holds by identical builder+registry "
                 "(build_primary_fold_split, partition_seed=3090), "
                 "identical counts, disjoint complete coverage of the "
                 "registry, and the independent test-id match against the "
                 "E1 predictions artifact"),
    }


def run_one(args, dataset, fold, seed):
    from tcdscr.config.schema import config_from_env
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr.models.selector import StaticUtilitySelector
    from tcdscr.models.selector_proxy import SelectorProxy, proxy_loss
    from tcdscr.training.sampler import SnapshotSampler
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    split = build_primary_fold_split(registry, fold, seed=3090)
    events = load_split_events(dataset, cfg, split)
    if args.limit:
        for seg in events:
            events[seg] = events[seg][:args.limit]
    parity = check_split_parity(dataset, fold, seed, split, events,
                                args.e1_root, registry)
    if not (parity["train_exact_match"] and parity["validation_exact_match"]
            and parity["test_exact_match"]):
        raise RuntimeError(f"E1/E2 split parity failed for {dataset} "
                           f"fold{fold} seed{seed}: {parity}")

    run_dir = os.path.join(args.out_root, dataset, f"fold{fold}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()

    encoder, ckpt_path, _e1_manifest = load_random_e1_encoder(
        dataset, fold, seed, args.e1_root, device)
    checksum_before = state_sha256(encoder.state_dict())

    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path,
                                              local_files_only=True)
    from tcdscr.context.token_budget import EvidenceBudgetSelector
    budget_selector = EvidenceBudgetSelector(tokenizer, args.budget)

    store = EventSemanticStore(cfg)
    cache = {}
    train_per_event = {}
    val_items = []
    for ev in events["train"]:
        eid = ev["event_id"]
        for c in PRIMARY_CUTOFFS:
            snap = build_snapshot(ev, c)
            train_per_event.setdefault(eid, {})[c] = build_light_item(
                ev, snap, store.get_store(ev, cache))
    for ev in events["validation"]:
        for c in PRIMARY_CUTOFFS:
            snap = build_snapshot(ev, c)
            val_items.append(build_light_item(
                ev, snap, store.get_store(ev, cache)))

    selector = StaticUtilitySelector().to(device)
    proxy = SelectorProxy().to(device)
    opt = torch.optim.AdamW(list(selector.parameters())
                            + list(proxy.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    sampler = SnapshotSampler(seed)  # deterministic, derived from run seed

    history = []
    best_val, best_epoch, wait = -1.0, -1, 0
    best_state = None
    for epoch in range(args.epochs):
        selector.train()
        proxy.train()
        epoch_items = [sampler.sample_one(train_per_event[eid])
                       for eid in sorted(train_per_event)]
        losses, parts = [], {"l_cls": 0.0, "l_fid": 0.0, "l_div": 0.0}
        for start in range(0, len(epoch_items), 32):
            chunk = epoch_items[start:start + 32]
            outs = encoder_forward_batch(encoder, chunk, device)
            for item, (node_repr, event_repr, logits_full) in zip(chunk,
                                                                  outs):
                n = item["num_nodes"]
                src = item["source_pos"]
                cand = [i for i in range(n) if i != src]
                if not cand:
                    continue
                cand_idx = torch.tensor(cand, dtype=torch.long,
                                        device=device)
                sem = item["sem"].to(device)
                struct3 = item["summary"].to(device)
                y = torch.tensor([item["label"]], dtype=torch.long,
                                 device=device)
                h_source = node_repr[src]
                u = selector(node_repr[cand_idx], event_repr,
                             sem[cand_idx], sem[src], struct3[cand_idx])
                alpha, _z_sel, p_sel = proxy(node_repr[cand_idx],
                                             h_source, u)
                total, components = proxy_loss(p_sel, y[0], logits_full,
                                               alpha, sem[cand_idx])
                opt.zero_grad()
                total.backward()
                opt.step()
                losses.append(float(total))
                for k in parts:
                    parts[k] += components[k]
            del outs
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        selector.eval()
        proxy.eval()
        val_metrics, _rows, _diag = evaluate_arms_on_items(
            encoder, selector, proxy, val_items, budget_selector, device,
            dataset, seed)
        val_static = mean_primary_macro_f1(
            {c: val_metrics[c]["static"] for c in val_metrics})
        n_steps = max(len(losses), 1)
        history.append({"epoch": epoch,
                        "loss": sum(losses) / n_steps,
                        "l_cls": parts["l_cls"] / n_steps,
                        "l_fid": parts["l_fid"] / n_steps,
                        "l_div": parts["l_div"] / n_steps,
                        "val_mean_macro_f1_static": val_static})
        if val_static > best_val:
            best_val, best_epoch, wait = val_static, epoch, 0
            best_state = {"selector": {k: v.detach().cpu().clone()
                                       for k, v in selector.state_dict().items()},
                          "proxy": {k: v.detach().cpu().clone()
                                    for k, v in proxy.state_dict().items()}}
        else:
            wait += 1
            if wait >= args.patience:
                break
        print(f"[{dataset} fold{fold} s{seed}] epoch {epoch} "
              f"loss={history[-1]['loss']:.4f} val_static_mF1="
              f"{val_static:.4f} best={best_val:.4f}", flush=True)

    if best_state is not None:
        selector.load_state_dict(best_state["selector"])
        proxy.load_state_dict(best_state["proxy"])
    selector.eval()
    proxy.eval()
    checksum_after = state_sha256(encoder.state_dict())

    val_metrics, val_rows, diagnostics = evaluate_arms_on_items(
        encoder, selector, proxy, val_items, budget_selector, device,
        dataset, seed)
    mean_primary = {a: mean_primary_macro_f1(
        {c: val_metrics[c][a] for c in val_metrics}) for a in ARM_NAMES}

    torch.save({"selector": selector.state_dict(), "proxy": proxy.state_dict(),
                "dataset": dataset, "fold": fold, "seed": seed,
                "encoder_checkpoint_sha": file_sha256(ckpt_path),
                "encoder_checksum_before": checksum_before,
                "encoder_checksum_after": checksum_after,
                "best_epoch": best_epoch,
                "best_val_mean_macro_f1_static": best_val,
                "config": {**E2_HPARAMS, "budget_tokens": args.budget}},
               os.path.join(run_dir, "best_selector.pt"))

    manifest = {
        "stage": "formal_e2",
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "encoder": {
            "type": "random_init_tcdscr_causal_social_encoder",
            "checkpoint_path": ckpt_path,
            "checkpoint_sha": file_sha256(ckpt_path),
            "frozen": True,
            "checksum_before": checksum_before,
            "checksum_after": checksum_after,
            "checksum_match": checksum_before == checksum_after},
        "split_parity": parity,
        "split_counts": {k: len(v) for k, v in split.items()},
        "hparams": {"lr": args.lr, "weight_decay": args.weight_decay,
                    "max_epochs": args.epochs, "patience": args.patience,
                    "epochs_run": len(history),
                    "budget_tokens": args.budget, "tau": E2_HPARAMS["tau"],
                    "loss_weights": E2_HPARAMS["loss_weights"]},
        "proxy_contract": {
            "train_input": "[h_source ; z_sel]",
            "classify_order": "source_then_selected",
            "semantic_baseline": "F.cosine_similarity(e_i, e_source)"},
        "best_epoch": best_epoch,
        "best_val_mean_macro_f1_static": best_val,
        "mean_primary_macro_f1_validation": mean_primary,
        "device": device,
        "arms": list(ARM_NAMES),
        "time_seconds": time.time() - t0,
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    with open(os.path.join(run_dir, "history.json"), "w",
              encoding="utf-8") as fh:
        json.dump(history, fh, indent=1)
    out = {"mean_primary_macro_f1": mean_primary,
           "arms": {a: {c: val_metrics[c][a] for c in val_metrics}
                    for a in ARM_NAMES} | {
               "e1_full": {c: val_metrics[c]["e1_full"]
                           for c in val_metrics}},
           "n_val_events": len({r["event_id"] for r in val_rows}),
           "n_val_rows": len(val_rows)}
    with open(os.path.join(run_dir, "validation_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(run_dir, "validation_predictions.jsonl"), "w",
              encoding="utf-8") as fh:
        for row in val_rows:
            fh.write(json.dumps(row) + "\n")
    with open(os.path.join(run_dir, "diagnostics.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_cutoff": diagnostics,
                   "loss_best_epoch": (history[best_epoch]
                                       if 0 <= best_epoch < len(history)
                                       else None)}, fh, indent=1)
    print("RUN DONE", run_dir, "best_epoch", best_epoch,
          "best_val_static_mF1", round(best_val, 4), flush=True)
    return manifest


def evaluate_test_only(args, dataset, fold, seed):
    """E2-B: score the test split with the saved selector (only called after
    both datasets pass readiness). Never feeds back into any decision."""
    from tcdscr.config.schema import config_from_env
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr.models.selector import StaticUtilitySelector
    from tcdscr.models.selector_proxy import SelectorProxy
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    split = build_primary_fold_split(registry, fold, seed=3090)
    events = load_split_events(dataset, cfg, split)
    run_dir = os.path.join(args.out_root, dataset, f"fold{fold}_seed{seed}")
    ck = torch.load(os.path.join(run_dir, "best_selector.pt"),
                    map_location="cpu", weights_only=True)
    encoder, _path, _e1 = load_random_e1_encoder(dataset, fold, seed,
                                                 args.e1_root, device)
    ck_sha = file_sha256(os.path.join(args.e1_root, dataset,
                                      f"fold{fold}_random_seed{seed}",
                                      "best_encoder.pt"))
    if ck_sha != ck["encoder_checkpoint_sha"]:
        raise RuntimeError("best_selector.pt encoder SHA mismatch")
    selector = StaticUtilitySelector().to(device)
    proxy = SelectorProxy().to(device)
    selector.load_state_dict(ck["selector"])
    proxy.load_state_dict(ck["proxy"])
    selector.eval()
    proxy.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path,
                                              local_files_only=True)
    from tcdscr.context.token_budget import EvidenceBudgetSelector
    budget_selector = EvidenceBudgetSelector(tokenizer, args.budget)
    store = EventSemanticStore(cfg)
    cache = {}
    test_items = []
    for ev in events["test"]:
        for c in PRIMARY_CUTOFFS:
            snap = build_snapshot(ev, c)
            test_items.append(build_light_item(
                ev, snap, store.get_store(ev, cache)))
    metrics, rows, diagnostics = evaluate_arms_on_items(
        encoder, selector, proxy, test_items, budget_selector, device,
        dataset, seed)
    mean_primary = {a: mean_primary_macro_f1(
        {c: metrics[c][a] for c in metrics}) for a in ARM_NAMES}
    with open(os.path.join(run_dir, "test_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"mean_primary_macro_f1": mean_primary,
                   "arms": {a: {c: metrics[c][a] for c in metrics}
                            for a in ARM_NAMES},
                   "readiness_gate_passed_by_caller": True}, fh, indent=1)
    with open(os.path.join(run_dir, "test_predictions.jsonl"), "w",
              encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print("TEST DONE", run_dir, {a: round(mean_primary[a], 4)
                                 for a in ARM_NAMES}, flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    for seed in seeds:
        if args.evaluate_test_only:
            evaluate_test_only(args, args.dataset, args.fold, seed)
        else:
            run_one(args, args.dataset, args.fold, seed)


if __name__ == "__main__":
    main()