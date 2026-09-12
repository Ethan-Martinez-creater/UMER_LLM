#!/usr/bin/env python
"""Dynamic V3 (MS-TSR) validation runner — Stage V3-A (design §1-§30).

One run = (dataset, fold, seed).  The corrected E2 frozen components
(Random-init E1 encoder + Static Utility Selector + Proxy) are loaded,
fully frozen (eval + requires_grad False + before/after checksums +
torch.no_grad), each validation event is rolled forward along
5m -> 15m -> 30m -> 1h -> 3h -> 6h under the all-event protocol, and both
arms are scored on every snapshot:

  Static : corrected E2 greedy pack by Static Utility at budget 1024;
  MS-TSR : Minimal-Sufficient Temporal Set Refinement — Dual-View Consensus
           Gate, forward ADD by reader-margin gain per token, backward
           redundant REMOVE, Static fallback, memory as pool/tie-break only.

The four alphas {0.80, 0.90, 0.95, 1.00} share one encoder/selector pass
but keep independent memory chains.  Nothing else is searched, the test
split is never read, no LLM is called (Qwen tokenizer only), and no model
is trained anywhere.
"""
import argparse
import json
import os
import sys

import torch

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, classification_metrics,
                           mean_primary_macro_f1, state_sha256)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tcdscr.models.marginal_set_refiner import BUDGET_TOKENS, static_pack
from tcdscr.models.minimal_set_refiner import (TOP_K_POOL,
                                               refine_minimal_set)
from tcdscr.models.sufficiency import ALPHA_GRID

from tcdscr_run_dynamic_v2 import (DATASETS, FOLDS, MAX_NODES,
                                   NODE_REPR_DIM, SEEDS, collect_event_data_v2,
                                   load_frozen_components,
                                   memory_positions_for)

E2_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e2_corrected"
E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"


def _empty_row_stats(proxy, h_source, p_full, device):
    """No-candidate snapshot (§15): empty sets, zero vector, prediction."""
    z0 = torch.zeros(NODE_REPR_DIM, device=device)
    logits = proxy.classify(h_source, z0)
    pred = int(logits.argmax(dim=-1))
    margin = float(logits[pred] - logits[1 - pred])
    return {
        "static_prediction": pred, "ms_prediction": pred,
        "static_selected_node_ids": [], "ms_selected_node_ids": [],
        "static_evidence_tokens": 0, "ms_evidence_tokens": 0,
        "static_margin": margin, "ms_margin": margin,
        "margin_retention": 1.0,
        "static_logits": [float(x) for x in logits.view(-1)],
        "ms_logits": [float(x) for x in logits.view(-1)],
        "ref_label": int(logits.argmax(dim=-1)),
        "full_prediction": int(p_full.view(-1).argmax()),
        "dual_view_agree": (int(logits.argmax(dim=-1))
                            == int(p_full.view(-1).argmax())),
        "compression_attempted": False, "sufficiency_reached": True,
        "fallback_to_static": True, "fallback_reason": "no_candidates",
        "pool_size": 0, "pool_size_before_prune": 0,
        "accepted_adds": [], "accepted_removes": [],
        "add_count": 0, "remove_count": 0,
        "static_selected_count": 0, "ms_selected_count": 0,
    }


def run_event_trajectory_v3(event_rows, alpha, proxy, device,
                            budget=BUDGET_TOKENS, top_k=TOP_K_POOL):
    """Roll one event through ascending cutoffs for one alpha.

    ``event_rows`` comes from collect_event_data_v2 (event_id, cutoff,
    label, cand_node_ids, u, costs, order, node_repr, h_source, p_full,
    sem_c, num_nodes).  Returns prediction rows; the MS-TSR memory chain is
    internal to this call (§14: M_t = S_t^MS, ids only).
    """
    memory_ids = []
    out_rows = []
    with torch.no_grad():
        for er in event_rows:
            cand_node_ids = er["cand_node_ids"]
            m = len(cand_node_ids)
            node_repr = er["node_repr"].to(device)
            h_source = er["h_source"].to(device)
            p_full = er["p_full"].to(device)
            if m == 0:
                res = _empty_row_stats(proxy, h_source, p_full, device)
            else:
                mem_pos = memory_positions_for(memory_ids, cand_node_ids)
                static_positions, _st = static_pack(er["costs"], er["u"],
                                                    er["order"], budget)
                res = refine_minimal_set(
                    node_repr, h_source, p_full, er["u"], er["costs"],
                    er["order"], cand_node_ids, static_positions, mem_pos,
                    proxy, budget=budget, alpha=alpha, top_k=top_k)
                sel_pos = res["selected_positions"]
                z_ms = node_repr[sel_pos].mean(0) if sel_pos else \
                    torch.zeros(NODE_REPR_DIM, device=device)
                res["ms_prediction"] = int(
                    proxy.classify(h_source, z_ms).argmax(dim=-1))
                res["static_prediction"] = int(
                    torch.tensor(res["static_logits"]).argmax())
                res["static_selected_count"] = len(res["static_positions"])
                res["ms_selected_count"] = len(sel_pos)
                res["ms_selected_node_ids"] = list(res["selected_node_ids"])
                res["static_selected_node_ids"] = list(
                    res["static_node_ids"])
                res["static_evidence_tokens"] = res["static_tokens"]
                res["ms_evidence_tokens"] = res["evidence_tokens"]
            row = {
                "event_id": er["event_id"],
                "cutoff": str(er["cutoff"]),
                "gold": er["label"],
                "alpha": alpha,
                "budget": budget,
                "candidate_count": m,
                "num_nodes": er["num_nodes"],
                "cap_hit": er["num_nodes"] >= MAX_NODES,
                "memory_previous_ids": list(memory_ids),
                "memory_current_ids": list(res["ms_selected_node_ids"]),
            }
            for key in ("static_prediction", "ms_prediction",
                        "static_selected_node_ids", "ms_selected_node_ids",
                        "static_evidence_tokens", "ms_evidence_tokens",
                        "static_margin", "ms_margin", "margin_retention",
                        "static_logits", "ms_logits", "ref_label",
                        "full_prediction", "dual_view_agree",
                        "compression_attempted", "sufficiency_reached",
                        "fallback_to_static", "fallback_reason",
                        "pool_size", "pool_size_before_prune",
                        "accepted_adds", "accepted_removes", "add_count",
                        "remove_count", "static_selected_count",
                        "ms_selected_count"):
                row[key] = res[key]
            row["pool_node_ids"] = [cand_node_ids[i]
                                    for i in res.get("pool_positions", [])]
            row["evidence_tokens"] = res["ms_evidence_tokens"]
            out_rows.append(row)
            memory_ids = list(res["ms_selected_node_ids"])
    return out_rows


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[len(s) // 2])


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[min(int(q * len(s)), len(s) - 1)])


def summarize_run_rows_v3(rows):
    """Per-alpha aggregation for one run under the all-event protocol."""
    by_cut = {str(c): {"static": [], "ms": []} for c in PRIMARY_CUTOFFS}
    for r in rows:
        by_cut[r["cutoff"]]["static"].append(
            (r["gold"], r["static_prediction"]))
        by_cut[r["cutoff"]]["ms"].append((r["gold"], r["ms_prediction"]))
    metrics = {c: {a: classification_metrics(by_cut[c][a])
                   for a in ("static", "ms")} for c in by_cut}
    by_event = {}
    for r in rows:
        by_event.setdefault(r["event_id"], []).append(r)
    flips = {"static": 0, "ms": 0}
    n_trans = 0
    survival, jac = [], []
    tok_st, tok_ms, unit_st, unit_ms, pool_sizes = [], [], [], [], []
    adds, removes = [], []
    agree = attempt = success = fallback = 0
    success_attempted = 0
    n_nonempty = 0
    m_st, m_ms, ret = [], [], []
    for ev, ev_rows in by_event.items():
        ev_rows = sorted(ev_rows, key=lambda r: PRIMARY_CUTOFFS.index(
            int(r["cutoff"])))
        for i in range(1, len(ev_rows)):
            n_trans += 1
            if ev_rows[i - 1]["static_prediction"] != \
                    ev_rows[i]["static_prediction"]:
                flips["static"] += 1
            if ev_rows[i - 1]["ms_prediction"] != \
                    ev_rows[i]["ms_prediction"]:
                flips["ms"] += 1
            prev = set(ev_rows[i - 1]["memory_current_ids"])
            cur = set(ev_rows[i]["memory_current_ids"])
            union = prev | cur
            if union:
                jac.append(len(prev & cur) / len(union))
            if prev:
                survival.append(len(prev & cur) / len(prev))
        for r in ev_rows:
            agree += int(r["dual_view_agree"])
            attempt += int(r["compression_attempted"])
            success += int(r["sufficiency_reached"])
            fallback += int(r["fallback_to_static"])
            if r["candidate_count"] > 0:
                n_nonempty += 1
            if r["compression_attempted"] and r["sufficiency_reached"]:
                success_attempted += 1
            tok_st.append(r["static_evidence_tokens"])
            tok_ms.append(r["ms_evidence_tokens"])
            unit_st.append(r["static_selected_count"])
            unit_ms.append(r["ms_selected_count"])
            pool_sizes.append(r["pool_size"])
            adds.append(r["add_count"])
            removes.append(r["remove_count"])
            m_st.append(r["static_margin"])
            m_ms.append(r["ms_margin"])
            if r["margin_retention"] is not None:
                ret.append(r["margin_retention"])
    n = max(len(rows), 1)
    red = [1.0 - (b / a) if a > 0 else 0.0
           for a, b in zip(tok_st, tok_ms)]
    return {
        "per_cutoff": metrics,
        "mean_primary_macro_f1": {
            "static": mean_primary_macro_f1(
                {c: metrics[c]["static"] for c in metrics}),
            "ms": mean_primary_macro_f1(
                {c: metrics[c]["ms"] for c in metrics})},
        "flip_rate": {"static": (flips["static"] / n_trans
                                 if n_trans else 0.0),
                      "ms": (flips["ms"] / n_trans if n_trans else 0.0)},
        "n_transitions": n_trans,
        "compression": {
            "mean_static_tokens": _mean(tok_st),
            "mean_ms_tokens": _mean(tok_ms),
            "median_ms_tokens": _median(tok_ms),
            "p25_ms_tokens": _pct(tok_ms, 0.25),
            "p75_ms_tokens": _pct(tok_ms, 0.75),
            "mean_token_reduction": _mean(red),
            "median_token_reduction": _median(red),
            "p25_token_reduction": _pct(red, 0.25),
            "p75_token_reduction": _pct(red, 0.75),
            "mean_static_units": _mean(unit_st),
            "mean_ms_units": _mean(unit_ms),
            "unit_reduction": (1.0 - _mean(unit_ms) / _mean(unit_st)
                               if _mean(unit_st) > 0 else 0.0),
            "budget_utilization_ms": _mean(tok_ms) / BUDGET_TOKENS},
        "sufficiency": {
            "dual_view_agreement_rate": agree / n,
            "compression_attempt_rate": attempt / n,
            "sufficiency_success_rate": (success_attempted / attempt
                                         if attempt else 0.0),
            "static_fallback_rate": (fallback / n_nonempty
                                     if n_nonempty else 0.0)},
        "temporal": {
            "memory_survival_rate": _mean(survival),
            "adjacent_set_jaccard": _mean(jac),
            "context_churn": 1.0 - _mean(jac)},
        "reader_margin": {
            "mean_static_margin": _mean(m_st),
            "mean_ms_margin": _mean(m_ms),
            "mean_margin_retention": _mean(ret)},
        "search": {
            "mean_add_count": _mean(adds),
            "mean_remove_count": _mean(removes),
            "max_pool_size": max(pool_sizes) if pool_sizes else 0,
            "mean_pool_size": _mean(pool_sizes)},
    }


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--fold", type=int, choices=FOLDS, required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, required=True)
    ap.add_argument("--out-root",
                    default="/data/jyz/next/llm/results/tcdscr/dynamic_v3")
    ap.add_argument("--e1-root", default=E1_ROOT)
    ap.add_argument("--e2-root", default=E2_ROOT)
    ap.add_argument("--budget", type=int, default=BUDGET_TOKENS)
    ap.add_argument("--limit", type=int, default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.budget != BUDGET_TOKENS:
        raise RuntimeError("budget is frozen at 1024 (design §9/§16)")
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
    trainable = sum(p.numel() for p in encoder.parameters()
                    if p.requires_grad) + sum(
        p.numel() for p in selector.parameters() if p.requires_grad) + sum(
        p.numel() for p in proxy.parameters() if p.requires_grad)
    if trainable:
        raise RuntimeError("frozen components still require grad")

    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path,
                                              local_files_only=True)
    store = EventSemanticStore(cfg)
    cache = {}
    from tcdscr_run_e2 import build_light_item
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
        raise RuntimeError(f"all-event protocol violated: {n_collected} "
                           f"snapshot rows for {n_expected} event-cutoffs")

    all_rows, grid = [], {}
    for alpha in ALPHA_GRID:
        rows = []
        for ev in events["validation"]:
            rows.extend(run_event_trajectory_v3(by_event[ev["event_id"]],
                                                alpha, proxy, device,
                                                budget=args.budget))
        for r in rows:
            r_full = {"dataset": args.dataset, "fold": args.fold,
                      "seed": args.seed}
            r_full.update(r)
            all_rows.append(r_full)
        grid[str(alpha)] = summarize_run_rows_v3(rows)

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
            "stage": "dynamic_v3_msts",
            "dataset": args.dataset, "fold": args.fold, "seed": args.seed,
            "protocol": "all validation events at all 6 cutoffs; "
                        "no-candidate snapshots included",
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
            "alpha_grid": list(ALPHA_GRID),
            "top_k_pool": TOP_K_POOL,
            "moves": "ADD+REMOVE only (no SWAP, design §13)",
            "n_validation_events": len(by_event),
            "n_snapshot_rows": n_collected,
            "test_split_read": False,
            "qwen_called": False,
        }, fh, indent=1)
    print(json.dumps({
        "dataset": args.dataset, "fold": args.fold, "seed": args.seed,
        "n_events": len(by_event), "n_rows": len(all_rows),
        "checksum_match": (checksums["encoder_match"]
                           and checksums["selector_match"]
                           and checksums["proxy_match"]),
        "ms_mean_primary_macro_f1": {
            str(a): grid[str(a)]["mean_primary_macro_f1"]["ms"]
            for a in ALPHA_GRID},
        "static_mean_primary_macro_f1": {
            str(a): grid[str(a)]["mean_primary_macro_f1"]["static"]
            for a in ALPHA_GRID},
        "mean_token_reduction": {
            str(a): grid[str(a)]["compression"]["mean_token_reduction"]
            for a in ALPHA_GRID}}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
