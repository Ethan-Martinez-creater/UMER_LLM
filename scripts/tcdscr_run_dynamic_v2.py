#!/usr/bin/env python
"""Dynamic V2 (MF-TSR) formal validation runner (execution protocol §1-§42).

One run = (dataset, fold, seed).  The corrected E2 frozen components
(Random-init E1 encoder + Static Utility Selector + Proxy) are loaded from
results/tcdscr/formal_e2_corrected/, fully frozen (eval + requires_grad
False, before/after checksums, torch.no_grad throughout), each validation
event is rolled forward along 5m -> 15m -> 30m -> 1h -> 3h -> 6h and every
snapshot gets BOTH arms:

  Static   : corrected E2 greedy pack by Static Utility at budget 1024;
  MF-TSR   : marginal-fidelity temporal set refinement (warm start from the
             previous final set, set-level REMOVE/ADD/SWAP under the
             forward-KL teacher distortion, static fallback).

Protocol-corrected (§2): a no-candidate snapshot is never skipped -- it
yields an empty selected set, z_sel = 0 vector through the frozen proxy,
and a normal prediction row with candidate_count = 0; memory M_t = empty.

The three epsilons {0, 0.01, 0.05} share one encoder/selector forward pass
but keep INDEPENDENT memory chains (M_t depends on epsilon).  Budget is
frozen at 1024; nothing else is searched.  The test split is never read;
no Qwen model is ever called (tokenizer only).  No training anywhere.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, build_light_item, build_units,
                           classification_metrics, encoder_forward_batch,
                           file_sha256, mean_primary_macro_f1, state_sha256)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tcdscr.models.marginal_set_refiner import (BUDGET_TOKENS, EPSILON_GRID,
                                                MAX_REFINEMENT_STEPS,
                                                PRUNE_TOP, refine_set)
from tcdscr.models.set_fidelity import set_distortion

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
NODE_REPR_DIM = 768
MAX_NODES = 1021
E2_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e2_corrected"
E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"


def load_frozen_components(dataset, fold, seed, e1_root, e2_root, device):
    """Same provenance-checked, fully frozen loader as the E3 runner."""
    from tcdscr_run_e3 import load_frozen_components as _lfc
    return _lfc(dataset, fold, seed, e1_root, e2_root, device)


def memory_positions_for(memory_ids, cand_node_ids):
    """M_previous ids -> positions inside the CURRENT snapshot's candidate
    list; historical evidence no longer present simply drops out (§11)."""
    pos = {nid: i for i, nid in enumerate(cand_node_ids)}
    return [pos[nid] for nid in memory_ids if nid in pos]


def _selected_novelty(sem_c, mem_sem, positions):
    """Diagnostic-only novelty (1 - max cosine to previous memory; 1.0 when
    empty), never part of the MF-TSR objective (§22, §32)."""
    if not positions:
        return None
    if mem_sem is None or mem_sem.shape[0] == 0:
        return [1.0 for _ in positions]
    cos = F.cosine_similarity(sem_c[positions].unsqueeze(1),
                              mem_sem.unsqueeze(0), dim=-1).clamp(-1.0, 1.0)
    return [float(1.0 - cos[k].max().item()) for k in range(len(positions))]


def run_event_trajectory_v2(event_rows, epsilon, proxy, device,
                            budget=BUDGET_TOKENS, prune_top=PRUNE_TOP,
                            max_steps=MAX_REFINEMENT_STEPS):
    """Roll one event through ascending cutoffs for one epsilon.

    ``event_rows`` fields (built by collect_event_data_v2): event_id,
    cutoff, label, cand_node_ids, u, costs, order, node_repr (M,768),
    h_source (768,), p_full (2,), sem_c (M,384), num_nodes.
    ``proxy`` is the frozen SelectorProxy.  Returns prediction rows
    (execution-protocol §42) with an independent memory chain per call.
    """
    memory_ids = []
    mem_sem = None
    out_rows = []
    with torch.no_grad():
        for er in event_rows:
            cand_node_ids = er["cand_node_ids"]
            m = len(cand_node_ids)
            node_repr = er["node_repr"].to(device)
            h_source = er["h_source"].to(device)
            p_full = er["p_full"].to(device)
            u_list = er["u"]
            costs = er["costs"]
            order = er["order"]
            sem_c = er["sem_c"].to(device)
            pos = {nid: i for i, nid in enumerate(cand_node_ids)}
            mem_pos = memory_positions_for(memory_ids, cand_node_ids)
            if m == 0:
                # Protocol-corrected no-candidate snapshot (§2): empty
                # evidence set, z_sel = 0, prediction still produced.
                z0 = torch.zeros(NODE_REPR_DIM, device=device)
                pred = int(proxy.classify(h_source, z0).argmax(dim=-1))
                d_empty = set_distortion(p_full, h_source,
                                         torch.empty(0, NODE_REPR_DIM,
                                                     device=device), proxy)
                res = {"selected_node_ids": [], "evidence_tokens": 0,
                       "distortion": d_empty, "static_node_ids": [],
                       "static_tokens": 0, "static_distortion": d_empty,
                       "accepted_moves": [], "fallback_to_static": False,
                       "max_step_hit": False, "init_source": "empty",
                       "warm_start_positions": []}
                pred_static = pred_dyn = pred
            else:
                res = refine_set(node_repr, h_source, p_full, u_list, costs,
                                 order, cand_node_ids, mem_pos, proxy,
                                 budget=budget, epsilon=epsilon,
                                 max_steps=max_steps, prune_top=prune_top)
                z_s = node_repr[res["static_positions"]].mean(0) \
                    if res["static_positions"] else \
                    torch.zeros(NODE_REPR_DIM, device=device)
                z_f = node_repr[res["selected_positions"]].mean(0) \
                    if res["selected_positions"] else \
                    torch.zeros(NODE_REPR_DIM, device=device)
                pred_static = int(proxy.classify(h_source, z_s).argmax(-1))
                pred_dyn = int(proxy.classify(h_source, z_f).argmax(-1))
            # diagnostic novelty vs the PREVIOUS memory only
            sel_pos = [pos[n] for n in res["selected_node_ids"]]
            sta_pos = [pos[n] for n in res["static_node_ids"]]
            admitted = [mv["added_node_id"] for mv in res["accepted_moves"]
                        if mv["added_node_id"] is not None]
            adm_pos = [pos[n] for n in admitted if n in pos]
            nov_sel = _selected_novelty(sem_c, mem_sem, sel_pos)
            nov_sta = _selected_novelty(sem_c, mem_sem, sta_pos)
            nov_adm = _selected_novelty(sem_c, mem_sem, adm_pos)
            out_rows.append({
                "event_id": er["event_id"],
                "cutoff": str(er["cutoff"]),
                "gold": er["label"],
                "epsilon": epsilon,
                "budget": budget,
                "candidate_count": m,
                "cap_hit": er["num_nodes"] >= MAX_NODES,
                "static_prediction": pred_static,
                "mf_tsr_prediction": pred_dyn,
                "static_selected_node_ids": list(res["static_node_ids"]),
                "mf_tsr_selected_node_ids": list(res["selected_node_ids"]),
                "memory_previous_ids": list(memory_ids),
                "memory_current_ids": list(res["selected_node_ids"]),
                "static_distortion": res["static_distortion"],
                "mf_tsr_distortion": res["distortion"],
                "warm_start_distortion": res.get("warm_start_distortion"),
                "init_source": res["init_source"],
                "accepted_moves": res["accepted_moves"],
                "fallback_to_static": res["fallback_to_static"],
                "max_step_hit": res["max_step_hit"],
                "pool_sizes": res.get("pool_sizes", []),
                "static_evidence_tokens": res["static_tokens"],
                "mf_tsr_evidence_tokens": res["evidence_tokens"],
                "evidence_tokens": res["evidence_tokens"],
                "diagnostic_static_selected_novelty": nov_sta,
                "diagnostic_mf_tsr_selected_novelty": nov_sel,
                "diagnostic_admitted_novelty": nov_adm,
            })
            memory_ids = list(res["selected_node_ids"])
            mem_sem = sem_c[[pos[n] for n in memory_ids]] \
                if memory_ids else None
    return out_rows


def collect_event_data_v2(encoder, selector, items, tokenizer, device):
    """One frozen encoder+selector pass per snapshot -> per-event rows.

    Protocol-corrected: no-candidate snapshots are KEPT (empty candidate
    arrays), never skipped.  Heavy tensors are moved to CPU here; the
    trajectory lifts per-snapshot views back to the device.
    """
    by_event = {}
    for start in range(0, len(items), 32):
        chunk = items[start:start + 32]
        outs = encoder_forward_batch(encoder, chunk, device)
        for item, (node_repr, event_repr, logits_full) in zip(chunk, outs):
            n = item["num_nodes"]
            src = item["source_pos"]
            cand = [i for i in range(n) if i != src]
            cand_idx = torch.tensor(cand, dtype=torch.long, device=device) \
                if cand else torch.empty(0, dtype=torch.long,
                                         device=device)
            sem = item["sem"].to(device)
            units = build_units(item)
            h_cand = node_repr[cand_idx]
            if cand:
                struct3 = item["summary"].to(device)
                u = selector(h_cand, event_repr, sem[cand_idx], sem[src],
                             struct3[cand_idx]).detach()
            else:
                u = torch.empty(0)
            p_full = F.softmax(logits_full.float(), dim=-1).detach().cpu()
            by_event.setdefault(item["event_id"], []).append({
                "event_id": item["event_id"],
                "cutoff": item["cutoff_minutes"],
                "label": item["label"],
                "num_nodes": n,
                "cand_node_ids": [item["node_ids"][i] for i in cand],
                "u": u.detach().cpu().tolist(),
                "node_repr": h_cand.detach().cpu(),
                "h_source": node_repr[src].detach().cpu(),
                "p_full": p_full,
                "sem_c": sem[cand_idx].detach().cpu(),
                "order": [u2["order"] for u2 in units],
                "costs": count_unit_costs(tokenizer, units),
            })
        del outs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    for ev_rows in by_event.values():
        ev_rows.sort(key=lambda r: r["cutoff"])
    return by_event


def count_unit_costs(tokenizer, units):
    """Per-unit evidence token count (Qwen tokenizer, rendered pair text)."""
    from tcdscr.context.evidence_unit import render_evidence
    cache = getattr(count_unit_costs, "_cache", None)
    if cache is None:
        cache = count_unit_costs._cache = {}
    costs = []
    for unit in units:
        text = render_evidence(unit, 1)
        if text not in cache:
            cache[text] = len(tokenizer(text, add_special_tokens=False)
                                ["input_ids"])
        costs.append(cache[text])
    return costs


def summarize_run_rows(rows):
    """Per-epsilon aggregation over one run's rows (all cutoffs present)."""
    by_cut = {str(c): {"static": [], "mf_tsr": []} for c in PRIMARY_CUTOFFS}
    for r in rows:
        by_cut[r["cutoff"]]["static"].append(
            (r["gold"], r["static_prediction"]))
        by_cut[r["cutoff"]]["mf_tsr"].append(
            (r["gold"], r["mf_tsr_prediction"]))
    metrics = {c: {a: classification_metrics(by_cut[c][a])
                   for a in ("static", "mf_tsr")} for c in by_cut}
    by_event = {}
    for r in rows:
        by_event.setdefault(r["event_id"], []).append(r)
    flips = {"static": 0, "mf_tsr": 0}
    n_trans = 0
    survival, exact, jac = [], [], []
    move_counts, gains, fb, hits = [], [], 0, 0
    d_st, d_dyn, tok_st, tok_dyn, sel_ct, cand_ct = [], [], [], [], [], []
    nov_sel, nov_sta, nov_adm = [], [], []
    added = removed = swapped = 0
    for ev, ev_rows in by_event.items():
        ev_rows = sorted(ev_rows, key=lambda r: PRIMARY_CUTOFFS.index(
            int(r["cutoff"])))
        for i in range(1, len(ev_rows)):
            n_trans += 1
            if ev_rows[i - 1]["static_prediction"] != \
                    ev_rows[i]["static_prediction"]:
                flips["static"] += 1
            if ev_rows[i - 1]["mf_tsr_prediction"] != \
                    ev_rows[i]["mf_tsr_prediction"]:
                flips["mf_tsr"] += 1
            prev = set(ev_rows[i - 1]["memory_current_ids"])
            cur = set(ev_rows[i]["memory_current_ids"])
            if prev:
                survival.append(len(prev & cur) / len(prev))
        for r in ev_rows:
            s_set = set(r["static_selected_node_ids"])
            d_set = set(r["mf_tsr_selected_node_ids"])
            exact.append(1.0 if s_set == d_set else 0.0)
            union = s_set | d_set
            jac.append(len(s_set & d_set) / len(union) if union else 1.0)
            mv = r["accepted_moves"]
            move_counts.append(len(mv))
            gains.extend(x["relative_gain"] for x in mv)
            fb += int(r["fallback_to_static"])
            hits += int(r["max_step_hit"])
            for x in mv:
                if x["move_type"] == "ADD":
                    added += 1
                elif x["move_type"] == "REMOVE":
                    removed += 1
                else:
                    swapped += 1
            d_st.append(r["static_distortion"])
            d_dyn.append(r["mf_tsr_distortion"])
            tok_st.append(r["static_evidence_tokens"])
            tok_dyn.append(r["mf_tsr_evidence_tokens"])
            sel_ct.append(len(d_set))
            cand_ct.append(r["candidate_count"])
            for key, bag in (("diagnostic_static_selected_novelty", nov_sta),
                             ("diagnostic_mf_tsr_selected_novelty", nov_sel),
                             ("diagnostic_admitted_novelty", nov_adm)):
                if r[key]:
                    bag.extend(r[key])

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    n_snap = max(len(rows), 1)
    return {
        "per_cutoff": metrics,
        "mean_primary_macro_f1": {
            "static": mean_primary_macro_f1(
                {c: metrics[c]["static"] for c in metrics}),
            "mf_tsr": mean_primary_macro_f1(
                {c: metrics[c]["mf_tsr"] for c in metrics})},
        "flip_rate": {"static": (flips["static"] / n_trans
                                 if n_trans else 0.0),
                      "mf_tsr": (flips["mf_tsr"] / n_trans
                                 if n_trans else 0.0)},
        "n_transitions": n_trans,
        "memory_survival_rate": mean(survival),
        "set_mechanism": {
            "exact_match_rate": mean(exact),
            "mean_jaccard": mean(jac),
            "add_count": added, "remove_count": removed,
            "swap_count": swapped,
            "mean_moves_per_snapshot": mean(move_counts),
            "mean_accepted_relative_gain": mean(gains),
            "fallback_to_static_rate": fb / n_snap,
            "max_step_hit_rate": hits / n_snap,
            "static_mean_distortion": mean(d_st),
            "mf_tsr_mean_distortion": mean(d_dyn),
            "distortion_reduction": mean(d_st) - mean(d_dyn)},
        "budget": {
            "mean_evidence_tokens": {"static": mean(tok_st),
                                     "mf_tsr": mean(tok_dyn)},
            "median_evidence_tokens": {
                "static": (sorted(tok_st)[len(tok_st) // 2] if tok_st else 0),
                "mf_tsr": (sorted(tok_dyn)[len(tok_dyn) // 2]
                           if tok_dyn else 0)},
            "budget_utilization": mean(tok_dyn) / BUDGET_TOKENS,
            "mean_candidate_count": mean(cand_ct),
            "mean_selected_count": mean(sel_ct)},
        "novelty_diagnostics": {
            "static_selected_mean": mean(nov_sta),
            "mf_tsr_selected_mean": mean(nov_sel),
            "admitted_mean": mean(nov_adm)},
    }


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--fold", type=int, choices=FOLDS, required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, required=True)
    ap.add_argument("--out-root",
                    default="/data/jyz/next/llm/results/tcdscr/dynamic_v2")
    ap.add_argument("--e1-root", default=E1_ROOT)
    ap.add_argument("--e2-root", default=E2_ROOT)
    ap.add_argument("--budget", type=int, default=BUDGET_TOKENS)
    ap.add_argument("--limit", type=int, default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.budget != BUDGET_TOKENS:
        raise RuntimeError("budget is frozen at 1024 (protocol §10/§46)")
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
    trainable = sum(p.numel() for p in selector.parameters()
                    if p.requires_grad) + sum(
        p.numel() for p in proxy.parameters() if p.requires_grad) + sum(
        p.numel() for p in encoder.parameters() if p.requires_grad)
    if trainable:
        raise RuntimeError("frozen components still require grad")

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
    by_event = collect_event_data_v2(encoder, selector, items, tokenizer,
                                     device)
    n_expected = len(events["validation"]) * len(PRIMARY_CUTOFFS)
    n_collected = sum(len(v) for v in by_event.values())
    if n_collected != n_expected:
        raise RuntimeError(f"protocol §2 violated: {n_collected} snapshot "
                           f"rows collected for {n_expected} event-cutoffs")

    all_rows, grid = [], {}
    for epsilon in EPSILON_GRID:
        rows = []
        for ev in events["validation"]:
            rows.extend(run_event_trajectory_v2(
                by_event[ev["event_id"]], epsilon, proxy, device,
                budget=args.budget))
        for r in rows:
            r_full = {"dataset": args.dataset, "fold": args.fold,
                      "seed": args.seed}
            r_full.update(r)
            all_rows.append(r_full)
        grid[str(epsilon)] = summarize_run_rows(rows)

    run_dir = os.path.join(args.out_root, "runs", args.dataset,
                           f"fold{args.fold}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "validation_predictions.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(run_dir, "grid_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(grid, fh, indent=1)
    with open(os.path.join(run_dir, "run_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "stage": "dynamic_v2_mfts",
            "dataset": args.dataset, "fold": args.fold, "seed": args.seed,
            "protocol_correction": "no-candidate snapshots included (§2)",
            "encoder": {"checkpoint_sha":
                        checksums["e2_manifest_encoder_sha"]},
            "selector_proxy": {
                "e2_run": os.path.join(args.e2_root, args.dataset,
                                       f"fold{args.fold}_seed{args.seed}"),
                "selector_checkpoint_sha":
                    checksums["selector_checkpoint_sha"]},
            "frozen": {"encoder": True, "selector": True, "proxy": True},
            "new_trainable_parameters": 0,
            "checksums": checksums,
            "budget": args.budget,
            "epsilon_grid": list(EPSILON_GRID),
            "prune_top": PRUNE_TOP,
            "max_refinement_steps": MAX_REFINEMENT_STEPS,
            "n_validation_events": len(by_event),
            "n_snapshot_rows": n_collected,
            "test_split_read": False,
        }, fh, indent=1)
    print(json.dumps({
        "dataset": args.dataset, "fold": args.fold, "seed": args.seed,
        "n_events": len(by_event), "n_rows": len(all_rows),
        "checksum_match": (checksums["encoder_match"]
                           and checksums["selector_match"]
                           and checksums["proxy_match"]),
        "mf_tsr_mean_primary_macro_f1": {
            str(e): grid[str(e)]["mean_primary_macro_f1"]["mf_tsr"]
            for e in EPSILON_GRID},
        "static_mean_primary_macro_f1": {
            str(e): grid[str(e)]["mean_primary_macro_f1"]["static"]
            for e in EPSILON_GRID}}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
