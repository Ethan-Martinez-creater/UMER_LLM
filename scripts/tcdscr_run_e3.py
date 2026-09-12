#!/usr/bin/env python
"""Formal E3 — Dynamic Evidence Memory (validation only; E3 order §1-§44).

One run = (dataset, fold, seed): the corrected E2 frozen components
(Random-init E1 encoder + Static Utility Selector + Proxy) are loaded from
results/tcdscr/formal_e2_corrected/, fully frozen (eval + no grad, before/
after checksums), and every validation event is rolled forward along the
true temporal trajectory SOURCE_ONLY -> 5m -> 15m -> 30m -> 1h -> 3h -> 6h
for all 36 (lambda_n, lambda_p, budget) configurations.

Static and Dynamic share the identical snapshot, tokenizer, budget, evidence
unit and pair token accounting; the only difference is the ranking score
(u_i vs d_i = u_i + lambda_n*novelty_i + lambda_p*persistence_i). Both rank
through the same frozen proxy head with [h_source; mean(selected h)].

The test split is never read. No Qwen model is ever called (tokenizer only).
"""
import argparse
import json
import os
import sys

import torch

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, build_light_item,
                           build_units, classification_metrics,
                           encoder_forward_batch, file_sha256,
                           load_random_e1_encoder, mean_primary_macro_f1,
                           state_sha256)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tcdscr.models.dynamic_memory import (dynamic_scores, memory_embeddings,
                                          novelty_scores, persistence_flags,
                                          select_with_costs)

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
DATASETS = ("pheme", "maweibo")

LAMBDA_N_GRID = (0.0, 0.25, 0.5, 1.0)
LAMBDA_P_GRID = (0.0, 0.1, 0.25)
BUDGET_GRID = (512, 1024, 2048)
CONFIGS = [(ln, lp, b) for ln in LAMBDA_N_GRID
           for lp in LAMBDA_P_GRID for b in BUDGET_GRID]
CONFIG_KEY = lambda ln, lp, b: f"{ln}_{lp}_{b}"  # noqa: E731

E2_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e2_corrected"
E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"

NODE_REPR_DIM = 768  # encoder hidden dim, used for the empty-selection z_sel
MAX_NODES = 1021


def load_frozen_components(dataset, fold, seed, e1_root, e2_root, device):
    """Load E1 encoder + corrected E2 selector/proxy, freeze, return checksums.

    Refuses anything but the corrected E2 checkpoint for the matching
    (dataset, fold, seed): the E2 run manifest's encoder checkpoint SHA must
    match the loaded encoder, and best_selector.pt's recorded
    encoder_checkpoint_sha must match as well.
    """
    from tcdscr.models.selector import StaticUtilitySelector
    from tcdscr.models.selector_proxy import SelectorProxy

    e2_run = os.path.join(e2_root, dataset, f"fold{fold}_seed{seed}")
    manifest_path = os.path.join(e2_run, "run_manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("stage") != "formal_e2":
        raise RuntimeError(f"not a formal E2 manifest: {manifest_path}")
    encoder, ckpt_path, _ = load_random_e1_encoder(
        dataset, fold, seed, e1_root, device)
    if file_sha256(ckpt_path) != manifest["encoder"]["checkpoint_sha"]:
        raise RuntimeError(
            f"E1 encoder SHA mismatch for {dataset}/fold{fold}_seed{seed}: "
            f"manifest {manifest['encoder']['checkpoint_sha']} vs "
            f"loaded {file_sha256(ckpt_path)}")

    ckpt_path_e2 = os.path.join(e2_run, "best_selector.pt")
    ckpt = torch.load(ckpt_path_e2, map_location="cpu", weights_only=True)
    if ckpt.get("encoder_checkpoint_sha") != \
            manifest["encoder"]["checkpoint_sha"]:
        raise RuntimeError(
            f"best_selector.pt encoder SHA does not match the E2 manifest "
            f"for {dataset}/fold{fold}_seed{seed}")
    selector = StaticUtilitySelector().to(device)
    proxy = SelectorProxy().to(device)
    selector.load_state_dict(ckpt["selector"])
    proxy.load_state_dict(ckpt["proxy"])
    for model in (encoder, selector, proxy):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    checksums = {
        "encoder_before": state_sha256(encoder.state_dict()),
        "selector_before": state_sha256(selector.state_dict()),
        "proxy_before": state_sha256(proxy.state_dict()),
        "e2_manifest_encoder_sha": manifest["encoder"]["checkpoint_sha"],
        "selector_checkpoint_sha": file_sha256(ckpt_path_e2),
        "e2_checksums": {
            "encoder_match": manifest["encoder"]["checksum_match"],
            "best_epoch": ckpt.get("best_epoch"),
            "best_val_static": ckpt.get("best_val_mean_macro_f1_static"),
        },
    }
    return encoder, selector, proxy, checksums


def run_event_trajectory(event_rows, config, classify_fn):
    """Roll one event through the ordered cutoffs for one config.

    ``event_rows`` must be in ascending cutoff order, each row holding:
      cutoff, cand_node_ids, u (tensor, candidate order), sem_c (tensor),
      units, costs, node_repr (full tensor), src_pos, num_nodes.
    ``classify_fn(h_source, sel_repr_or_None) -> logits`` injects the frozen
    proxy head so this core stays unit-testable.

    Returns (rows, first_cutoff_mismatch).
    """
    lambda_n, lambda_p, budget = config
    memory_ids, memory_embs = [], None
    out_rows = []
    mismatch = 0
    for idx, er in enumerate(event_rows):
        cand_ids = er["cand_node_ids"]
        u = er["u"].float()
        sem_c = er["sem_c"].float()
        nov = novelty_scores(sem_c, memory_embs)
        per = persistence_flags(cand_ids, memory_ids)
        d = dynamic_scores(u, nov, per, lambda_n, lambda_p)
        sel_dyn, tok_dyn = select_with_costs(er["units"], d.tolist(),
                                             er["costs"], budget)
        sel_sta, tok_sta = select_with_costs(er["units"], u.tolist(),
                                             er["costs"], budget)
        dyn_ids = [u2["node_id"] for u2 in sel_dyn]
        sta_ids = [u2["node_id"] for u2 in sel_sta]
        if idx == 0 and dyn_ids != sta_ids:
            mismatch += 1
        node_repr = er["node_repr"]
        h_source = node_repr[er["src_pos"]]
        sel_idx = [er["units_index"][nid] for nid in dyn_ids]
        logits_dyn = classify_fn(h_source, node_repr[sel_idx]
                                 if sel_idx else None)
        sel_idx_sta = [er["units_index"][nid] for nid in sta_ids]
        logits_sta = classify_fn(h_source, node_repr[sel_idx_sta]
                                 if sel_idx_sta else None)
        sel_set = set(dyn_ids)
        out_rows.append({
            "lambda_n": lambda_n, "lambda_p": lambda_p, "budget": budget,
            "event_id": er["event_id"],
            "cutoff": str(er["cutoff"]),
            "gold": er["label"],
            "n_candidates": len(cand_ids),
            "cap_hit": er["num_nodes"] >= MAX_NODES,
            "static_prediction": int(logits_sta.argmax(dim=-1)),
            "dynamic_prediction": int(logits_dyn.argmax(dim=-1)),
            "static_selected_node_ids": sta_ids,
            "dynamic_selected_node_ids": dyn_ids,
            "static_evidence_tokens": tok_sta,
            "dynamic_evidence_tokens": tok_dyn,
            "memory_previous_ids": list(memory_ids),
            "memory_current_ids": list(dyn_ids),
            "selected_scores": [
                {"node_id": u2["node_id"],
                 "base_score": float(u[i]),
                 "novelty": float(nov[i]),
                 "persistence": float(per[i]),
                 "dynamic_score": float(d[i])}
                for i, u2 in enumerate(er["units"])
                if u2["node_id"] in sel_set],
        })
        memory_ids = dyn_ids
        memory_embs = memory_embeddings(sel_dyn, er["sem_all"],
                                        er["node_ids"])
    return out_rows, mismatch


def count_unit_costs(tokenizer, units):
    """Per-unit evidence token count, cached on rendered text."""
    from tcdscr.context.evidence_unit import render_evidence
    cache = {}
    costs = []
    for unit in units:
        text = render_evidence(unit, 1)
        if text not in cache:
            cache[text] = len(tokenizer(text, add_special_tokens=False)
                                ["input_ids"])
        costs.append(cache[text])
    return costs


def collect_event_data(encoder, selector, items, device):
    """Per (event, cutoff) frozen forwards -> per-event trajectory rows."""
    from tcdscr.context.token_budget import EvidenceBudgetSelector

    by_event = {}
    for start in range(0, len(items), 32):
        chunk = items[start:start + 32]
        outs = encoder_forward_batch(encoder, chunk, device)
        for item, (node_repr, event_repr, _logits_full) in zip(chunk, outs):
            n = item["num_nodes"]
            src = item["source_pos"]
            cand = [i for i in range(n) if i != src]
            # Protocol correction (Dynamic V2 §2): no-candidate snapshots
            # are kept in the trajectory (empty selection, z_sel = 0
            # prediction) instead of being skipped out of the metrics.
            cand_idx = torch.tensor(cand, dtype=torch.long, device=device)
            sem = item["sem"].to(device)
            struct3 = item["summary"].to(device)
            units = build_units(item)
            units_index = {u2["node_id"]: cand[i]
                           for i, u2 in enumerate(units)}
            h_cand = node_repr[cand_idx]
            if not cand:
                u = torch.empty(0, device=device)
            else:
                u = selector(h_cand, event_repr, sem[cand_idx], sem[src],
                             struct3[cand_idx]).detach()
            by_event.setdefault(item["event_id"], []).append({
                "cutoff": item["cutoff_minutes"],
                "label": item["label"],
                "event_id": item["event_id"],
                "src_pos": src,
                "num_nodes": n,
                "node_repr": node_repr.detach(),
                "sem_all": sem.detach().cpu(),
                "node_ids": item["node_ids"],
                "cand_node_ids": [item["node_ids"][i] for i in cand],
                "u": u.cpu(),
                "sem_c": sem[cand_idx].detach().cpu(),
                "units": units,
                "units_index": units_index,
                "costs": None,  # filled after the tokenizer is available
            })
        del outs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    for ev_rows in by_event.values():
        ev_rows.sort(key=lambda r: r["cutoff"])
    return by_event


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def summarize_config_rows(rows):
    """Per-config aggregation over a run's rows (all cutoffs present).

    rows is one config's ordered per-(event, cutoff) records; records of the
    same event appear in ascending cutoff order. Returns per-cutoff metrics
    plus temporal diagnostics.
    """
    by_cut = {str(c): {"static": [], "dynamic": []} for c in PRIMARY_CUTOFFS}
    for r in rows:
        by_cut[r["cutoff"]]["static"].append((r["gold"], r["static_prediction"]))
        by_cut[r["cutoff"]]["dynamic"].append((r["gold"], r["dynamic_prediction"]))
    metrics = {c: {"static": classification_metrics(by_cut[c]["static"]),
                   "dynamic": classification_metrics(by_cut[c]["dynamic"])}
               for c in by_cut}

    # per-event temporal aggregates
    by_event = {}
    for r in rows:
        by_event.setdefault(r["event_id"], []).append(r)
    flips = {"static": 0, "dynamic": 0}
    n_transitions = 0
    turnovers, retention, novelty_sel, tok_dyn, tok_sta, sel_count = (
        [], [], [], [], [], [])
    sat_denom = 0
    for ev, ev_rows in by_event.items():
        for i in range(1, len(ev_rows)):
            n_transitions += 1
            prev_s, cur_s = ev_rows[i - 1], ev_rows[i]
            if prev_s["static_prediction"] != cur_s["static_prediction"]:
                flips["static"] += 1
            if prev_s["dynamic_prediction"] != cur_s["dynamic_prediction"]:
                flips["dynamic"] += 1
            turnovers.append(1.0 - _jaccard(
                cur_s["memory_previous_ids"], cur_s["memory_current_ids"]))
            prev_ids = set(prev_s["memory_current_ids"])
            cur_ids = cur_s["memory_current_ids"]
            if prev_ids:
                retention.append(
                    len(prev_ids & set(cur_ids)) / len(prev_ids))
        for r in ev_rows:
            sel_count.append(len(r["dynamic_selected_node_ids"]))
            tok_dyn.append(r["dynamic_evidence_tokens"])
            tok_sta.append(r["static_evidence_tokens"])
            sat_denom += r["n_candidates"]
            for s in r["selected_scores"]:
                novelty_sel.append(s["novelty"])
    novelty_sel = sorted(novelty_sel)
    n = len(novelty_sel)
    return {
        "per_cutoff": metrics,
        "mean_primary_macro_f1": {
            "static": mean_primary_macro_f1(
                {c: metrics[c]["static"] for c in metrics}),
            "dynamic": mean_primary_macro_f1(
                {c: metrics[c]["dynamic"] for c in metrics})},
        "flip_rate": {
            "static": (flips["static"] / n_transitions
                       if n_transitions else 0.0),
            "dynamic": (flips["dynamic"] / n_transitions
                        if n_transitions else 0.0)},
        "n_transitions": n_transitions,
        "mean_evidence_tokens": {
            "static": (sum(tok_sta) / len(tok_sta) if tok_sta else 0.0),
            "dynamic": (sum(tok_dyn) / len(tok_dyn) if tok_dyn else 0.0)},
        "median_evidence_tokens": {
            "static": (sorted(tok_sta)[len(tok_sta) // 2] if tok_sta else 0.0),
            "dynamic": (sorted(tok_dyn)[len(tok_dyn) // 2]
                        if tok_dyn else 0.0)},
        "mean_selected_units": (sum(sel_count) / len(sel_count)
                                if sel_count else 0.0),
        "mean_turnover": (sum(turnovers) / len(turnovers)
                          if turnovers else 0.0),
        "memory_retention": (sum(retention) / len(retention)
                             if retention else 0.0),
        "selected_novelty": {
            "mean": (sum(novelty_sel) / n) if n else 0.0,
            "median": novelty_sel[n // 2] if n else 0.0,
            "p25": novelty_sel[n // 4] if n else 0.0,
            "p75": novelty_sel[3 * n // 4] if n else 0.0},
        "selection_saturation": {
            "mean": (sum(len(r["dynamic_selected_node_ids"])
                         for r in rows) / sat_denom if sat_denom else 0.0)},
    }


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--fold", type=int, choices=FOLDS, required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, required=True)
    ap.add_argument("--out-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e3")
    ap.add_argument("--e1-root", default=E1_ROOT)
    ap.add_argument("--e2-root", default=E2_ROOT)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N validation events")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from tcdscr.config.schema import config_from_env
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split
    from transformers import AutoTokenizer

    from tcdscr_common import (EventSemanticStore, event_label_registry,
                               load_split_events)

    cfg = config_from_env(args.dataset)
    registry = event_label_registry(args.dataset, cfg)
    split = build_primary_fold_split(registry, args.fold, seed=3090)
    events = load_split_events(args.dataset, cfg, split)
    if args.limit:
        events["validation"] = events["validation"][:args.limit]

    encoder, selector, proxy, checksums = load_frozen_components(
        args.dataset, args.fold, args.seed, args.e1_root, args.e2_root,
        device)
    checksums["encoder_after"] = state_sha256(encoder.state_dict())
    checksums["selector_after"] = state_sha256(selector.state_dict())
    checksums["proxy_after"] = state_sha256(proxy.state_dict())
    checksums["encoder_match"] = (checksums["encoder_before"]
                                  == checksums["encoder_after"])
    checksums["selector_match"] = (checksums["selector_before"]
                                   == checksums["selector_after"])
    checksums["proxy_match"] = (checksums["proxy_before"]
                                == checksums["proxy_after"])

    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path,
                                              local_files_only=True)
    store = EventSemanticStore(cfg)
    cache = {}
    items = []
    for ev in events["validation"]:
        for c in PRIMARY_CUTOFFS:
            snap = build_snapshot(ev, c)
            items.append(build_light_item(ev, snap, store.get_store(ev,
                                                                    cache)))

    by_event = collect_event_data(encoder, selector, items, device)
    for ev_rows in by_event.values():
        for er in ev_rows:
            er["costs"] = count_unit_costs(tokenizer, er["units"])

    from tcdscr.models.selector_proxy import classify_selected

    def classify_fn(h_source, sel_repr):
        if sel_repr is None:
            z_sel = torch.zeros(NODE_REPR_DIM, device=device)
            return proxy.classify(h_source, z_sel)
        return classify_selected(proxy, h_source, sel_repr)

    n_events = len(by_event)
    rows_by_config = {CONFIG_KEY(*c): [] for c in CONFIGS}
    first_cutoff_mismatch = 0
    for ev_rows in by_event.values():
        for config in CONFIGS:
            r, mm = run_event_trajectory(ev_rows, config, classify_fn)
            rows_by_config[CONFIG_KEY(*config)].extend(r)
            first_cutoff_mismatch += mm

    grid_metrics = {}
    for config in CONFIGS:
        key = CONFIG_KEY(*config)
        grid_metrics[key] = summarize_config_rows(rows_by_config[key])
        grid_metrics[key]["config"] = {
            "lambda_n": config[0], "lambda_p": config[1],
            "budget": config[2]}

    run_dir = os.path.join(args.out_root, "runs", args.dataset,
                           f"fold{args.fold}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "stage": "formal_e3",
            "dataset": args.dataset, "fold": args.fold, "seed": args.seed,
            "encoder": {"checkpoint_sha":
                        checksums["e2_manifest_encoder_sha"]},
            "selector_proxy": {
                "e2_run": os.path.join(
                    args.e2_root, args.dataset,
                    f"fold{args.fold}_seed{args.seed}"),
                "selector_checkpoint_sha":
                    checksums["selector_checkpoint_sha"],
                "best_epoch": checksums["e2_checksums"]["best_epoch"]},
            "frozen": {"encoder": True, "selector": True, "proxy": True},
            "checksums": checksums,
            "grid": {"lambda_n": list(LAMBDA_N_GRID),
                     "lambda_p": list(LAMBDA_P_GRID),
                     "budget": list(BUDGET_GRID),
                     "n_configs": len(CONFIGS)},
            "n_validation_events": n_events,
            "first_cutoff_mismatch": first_cutoff_mismatch,
            "test_split_read": False,
        }, fh, indent=1)
    with open(os.path.join(run_dir, "checksums.json"), "w",
              encoding="utf-8") as fh:
        json.dump(checksums, fh, indent=1)
    with open(os.path.join(run_dir, "grid_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(grid_metrics, fh, indent=1)
    with open(os.path.join(run_dir, "validation_rows.jsonl"), "w",
              encoding="utf-8") as fh:
        for key in sorted(rows_by_config):
            for r in rows_by_config[key]:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps({"dataset": args.dataset, "fold": args.fold,
                      "seed": args.seed, "n_events": n_events,
                      "n_rows": sum(len(v) for v in rows_by_config.values()),
                      "first_cutoff_mismatch": first_cutoff_mismatch,
                      "checksum_match": (checksums["encoder_match"]
                                         and checksums["selector_match"]
                                         and checksums["proxy_match"])},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
