#!/usr/bin/env python
"""Stage V2-A: no-candidate protocol correction artifacts (§3-§5).

Three independent deliverables, all VALIDATION ONLY (the outer test split
is never read):

  audit  -> dynamic_v2_protocol/no_candidate_audit.json
      per dataset x cutoff: events_total / with / without candidates /
      no_candidate_rate.

  e2     -> e2_corrected_all_event_metrics.{json,md}
      Re-run corrected E2 validation INFERENCE (frozen E1 encoder +
      corrected E2 selector/proxy checkpoints, no retraining) with the
      protocol-corrected evaluate_arms_on_items so every validation event
      at every cutoff enters the metrics, including candidate_count == 0
      (empty selection, z_sel = 0).  Budget 1024, three arms
      (random-budget, semantic-budget, static).  Old corrected-E2 numbers
      are READ and reported alongside, never modified.

  e3v1   -> e3_v1_all_event_diagnostic.json
      With the frozen per-fold Dynamic V1 configs (no retraining, no new
      lambda search) re-run only the missing no-candidate predictions so
      the old "no real benefit" conclusion can be re-checked under the
      all-event protocol.  Diagnostic only; V1 is not re-validated as a
      method.
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, build_light_item,
                           classification_metrics, mean_primary_macro_f1)

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
BUDGET = 1024
DISPLAY = {"pheme": "PHEME", "maweibo": "Ma-Weibo"}
OUT_ROOT = "/data/jyz/next/llm/results/tcdscr/dynamic_v2_protocol"
E2_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e2_corrected"
E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"
E3_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e3"


def _validation_events(dataset):
    from tcdscr.config.schema import config_from_env
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr_common import event_label_registry, load_split_events
    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    out = {}
    for fold in FOLDS:
        split = build_primary_fold_split(registry, fold, seed=3090)
        events = load_split_events(dataset, cfg, split)
        out[fold] = events["validation"]
    return out


def run_audit(args):
    from tcdscr.data.snapshot_builder import build_snapshot
    result = {}
    for dataset in DATASETS:
        per_fold = _validation_events(dataset)
        # fold0 validation events are the dataset's full validation pool
        # per fold; audit across ALL folds is fold-specific.  The protocol
        # asks for a dataset-level audit: use each fold's validation pool
        # and report per fold + pooled across the 5 folds (unique events
        # differ per fold; pooled uses fold0 as the canonical view AND
        # per-fold detail).
        detail = {}
        for fold in FOLDS:
            cutoff_stats = {}
            for c in PRIMARY_CUTOFFS:
                total = with_c = 0
                for ev in per_fold[fold]:
                    snap = build_snapshot(ev, c)
                    total += 1
                    if len(snap["node_ids"]) > 1:
                        with_c += 1
                cutoff_stats[str(c)] = {
                    "events_total": total,
                    "events_with_candidates": with_c,
                    "events_without_candidates": total - with_c,
                    "no_candidate_rate": (total - with_c) / max(total, 1)}
            detail[f"fold{fold}"] = cutoff_stats
        result[dataset] = {"scope": "validation events only",
                           "per_fold": detail,
                           "pooled_over_folds": {
                               str(c): {
                                   "events_total": sum(
                                       detail[f"fold{f}"][str(c)]
                                       ["events_total"] for f in FOLDS),
                                   "events_with_candidates": sum(
                                       detail[f"fold{f}"][str(c)]
                                       ["events_with_candidates"]
                                       for f in FOLDS),
                                   "events_without_candidates": sum(
                                       detail[f"fold{f}"][str(c)]
                                       ["events_without_candidates"]
                                       for f in FOLDS),
                                   "no_candidate_rate": sum(
                                       detail[f"fold{f}"][str(c)]
                                       ["events_without_candidates"]
                                       for f in FOLDS) / max(sum(
                                       detail[f"fold{f}"][str(c)]
                                       ["events_total"] for f in FOLDS), 1)}
                               for c in PRIMARY_CUTOFFS}}
    os.makedirs(args.out_root, exist_ok=True)
    path = os.path.join(args.out_root, "no_candidate_audit.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1)[:4000], flush=True)
    return result


def _load_val_items(dataset, fold):
    from tcdscr.config.schema import config_from_env
    from tcdscr.data.snapshot_builder import build_snapshot
    from transformers import AutoTokenizer
    from tcdscr_common import EventSemanticStore
    cfg = config_from_env(dataset)
    events = _validation_events(dataset)[fold]
    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path,
                                              local_files_only=True)
    store = EventSemanticStore(cfg)
    cache = {}
    items = []
    for ev in events:
        for c in PRIMARY_CUTOFFS:
            snap = build_snapshot(ev, c)
            items.append(build_light_item(ev, snap, store.get_store(ev,
                                                                    cache)))
    return items, tokenizer


def run_e2(args):
    """Per-run all-event E2 validation metrics + aggregated json/md."""
    from tcdscr.context.token_budget import EvidenceBudgetSelector
    from tcdscr_run_e2 import evaluate_arms_on_items
    from tcdscr_run_e3 import load_frozen_components

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs_root = os.path.join(args.out_root, "runs")
    os.makedirs(runs_root, exist_ok=True)
    per_run = {}
    for dataset in DATASETS:
        for fold in FOLDS:
            items, tokenizer = _load_val_items(dataset, fold)
            for seed in SEEDS:
                path = os.path.join(runs_root,
                                    f"e2_{dataset}_fold{fold}_seed{seed}"
                                    ".json")
                if os.path.exists(path):
                    continue
                encoder, selector, proxy, checksums = \
                    load_frozen_components(dataset, fold, seed,
                                           args.e1_root, args.e2_root,
                                           device)
                from tcdscr_run_e2 import state_sha256
                frozen_ok = (
                    state_sha256(encoder.state_dict()) ==
                    checksums["encoder_before"] and
                    state_sha256(selector.state_dict()) ==
                    checksums["selector_before"] and
                    state_sha256(proxy.state_dict()) ==
                    checksums["proxy_before"])
                bs = EvidenceBudgetSelector(tokenizer, BUDGET)
                metrics, rows, _diag = evaluate_arms_on_items(
                    encoder, selector, proxy, items, bs, device, dataset,
                    seed)
                mean_primary = {a: mean_primary_macro_f1(
                    {c: metrics[c][a] for c in metrics})
                    for a in ("static", "random", "semantic")}
                n_events = len({r["event_id"] for r in rows})
                out = {"dataset": dataset, "fold": fold, "seed": seed,
                       "protocol": "all events at every cutoff (§2)",
                       "budget": BUDGET,
                       "n_val_events": n_events,
                       "n_val_rows": len(rows),
                       "per_cutoff": metrics,
                       "mean_primary_macro_f1": mean_primary,
                       "frozen_checksum_match": frozen_ok,
                       "new_trainable_parameters": 0}
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(out, fh, indent=1)
                print("E2 ALL-EVENT DONE", dataset, fold, seed,
                      {k: round(v, 4) for k, v in mean_primary.items()},
                      flush=True)
                del encoder, selector, proxy
    # aggregate
    for dataset in DATASETS:
        per_run[dataset] = {}
        for fold in FOLDS:
            per_run[dataset][f"fold{fold}"] = {}
            for seed in SEEDS:
                path = os.path.join(
                    runs_root,
                    f"e2_{dataset}_fold{fold}_seed{seed}.json")
                with open(path, encoding="utf-8") as fh:
                    per_run[dataset][f"fold{fold}"][str(seed)] = json.load(
                        fh)
    agg = {"protocol": "new metric: ALL events at the real cutoff; "
                       "old corrected-E2 metric: conditioned on "
                       "candidate_count > 0",
           "budget": BUDGET, "datasets": {}}
    for dataset in DATASETS:
        old_mean = {"static": [], "random": [], "semantic": []}
        new_mean = {"static": [], "random": [], "semantic": []}
        per_fold = {}
        for fold in FOLDS:
            seeds_out = {}
            for seed in SEEDS:
                run = per_run[dataset][f"fold{fold}"][str(seed)]
                mp = run["mean_primary_macro_f1"]
                for a in new_mean:
                    new_mean[a].append(mp[a])
                old_path = os.path.join(args.e2_root, dataset,
                                        f"fold{fold}_seed{seed}",
                                        "validation_metrics.json")
                with open(old_path, encoding="utf-8") as fh:
                    old = json.load(fh)
                omp = old["mean_primary_macro_f1"]
                for a in old_mean:
                    old_mean[a].append(omp[a])
                seeds_out[str(seed)] = {
                    "new_all_event": mp,
                    "old_candidates_only": omp}
            per_fold[f"fold{fold}"] = seeds_out
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
        agg["datasets"][dataset] = {
            "mean_over_runs_new": {a: mean(v) for a, v in new_mean.items()},
            "mean_over_runs_old": {a: mean(v) for a, v in old_mean.items()},
            "delta_new_minus_old": {
                a: mean(new_mean[a]) - mean(old_mean[a])
                for a in new_mean},
            "per_fold": per_fold}
    agg_path = os.path.join(args.out_root,
                            "e2_corrected_all_event_metrics.json")
    with open(agg_path, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=1)
    with open(os.path.join(args.out_root,
                           "e2_corrected_all_event_metrics.md"), "w",
              encoding="utf-8") as fh:
        fh.write("# Corrected E2 — All-Event Validation Metrics "
                 "(budget 1024)\n\n")
        fh.write("old metric: conditioned on candidate_count > 0\n\n")
        fh.write("new metric: all events at the real cutoff "
                 "(no-candidate snapshots classify with the zero "
                 "evidence vector)\n\n")
        for dataset in DATASETS:
            d = agg["datasets"][dataset]
            fh.write(f"## {DISPLAY[dataset]}\n\n")
            fh.write("| method | old mean-primary Macro-F1 | new "
                     "all-event mean-primary Macro-F1 | delta |\n")
            fh.write("|---|---:|---:|---:|\n")
            for a in ("random", "semantic", "static"):
                fh.write(f"| {a} | {d['mean_over_runs_old'][a]:.4f} | "
                         f"{d['mean_over_runs_new'][a]:.4f} | "
                         f"{d['delta_new_minus_old'][a]:+.4f} |\n")
            fh.write("\n")
    print("E2 AGGREGATE WRITTEN", agg_path, flush=True)
    return agg


def run_e3v1(args):
    """Old Dynamic V1 configs re-scored under the all-event protocol."""
    from tcdscr_run_e3 import (collect_event_data, count_unit_costs,
                               load_frozen_components, run_event_trajectory)
    from tcdscr_run_e2 import mean_primary_macro_f1 as mpmf  # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {}
    from tcdscr.models.selector_proxy import classify_selected
    for dataset in DATASETS:
        result[dataset] = {"per_fold": {}, "pooled": {
            "static": [], "dynamic": [], "flip_static": [],
            "flip_dynamic": []}}
        for fold in FOLDS:
            with open(os.path.join(args.e3_root, dataset, f"fold{fold}",
                                   "best_config.json"),
                      encoding="utf-8") as fh:
                bc = json.load(fh)
            config = (bc["lambda_n"], bc["lambda_p"], bc["budget"])
            items, tokenizer = _load_val_items(dataset, fold)
            seeds_out = {}
            for seed in SEEDS:
                encoder, selector, proxy, _ck = load_frozen_components(
                    dataset, fold, seed, args.e1_root, args.e2_root,
                    device)

                def classify_fn(h_source, sel_repr):
                    if sel_repr is None:
                        z = torch.zeros(768, device=device)
                        return proxy.classify(h_source, z)
                    return classify_selected(proxy, h_source, sel_repr)

                by_event = collect_event_data(encoder, selector, items,
                                              device)
                for ev_rows in by_event.values():
                    for er in ev_rows:
                        er["costs"] = count_unit_costs(tokenizer,
                                                       er["units"])
                rows = []
                mismatch = 0
                for ev_rows in by_event.values():
                    r, mm = run_event_trajectory(ev_rows, config,
                                                 classify_fn)
                    rows.extend(r)
                    mismatch += mm
                by_cut = {str(c): {"static": [], "dynamic": []}
                          for c in PRIMARY_CUTOFFS}
                for r in rows:
                    by_cut[r["cutoff"]]["static"].append(
                        (r["gold"], r["static_prediction"]))
                    by_cut[r["cutoff"]]["dynamic"].append(
                        (r["gold"], r["dynamic_prediction"]))
                metrics = {c: {a: classification_metrics(by_cut[c][a])
                               for a in ("static", "dynamic")}
                           for c in by_cut}
                mp = {"static": mpmf({c: metrics[c]["static"]
                                      for c in metrics}),
                      "dynamic": mpmf({c: metrics[c]["dynamic"]
                                       for c in metrics})}
                by_event_rows = {}
                for r in rows:
                    by_event_rows.setdefault(r["event_id"], []).append(r)
                flips = {"static": 0, "dynamic": 0}
                n_trans = 0
                for ev_rows2 in by_event_rows.values():
                    ev_rows2.sort(key=lambda x: PRIMARY_CUTOFFS.index(
                        int(x["cutoff"])))
                    for i in range(1, len(ev_rows2)):
                        n_trans += 1
                        if ev_rows2[i - 1]["static_prediction"] != \
                                ev_rows2[i]["static_prediction"]:
                            flips["static"] += 1
                        if ev_rows2[i - 1]["dynamic_prediction"] != \
                                ev_rows2[i]["dynamic_prediction"]:
                            flips["dynamic"] += 1
                seeds_out[str(seed)] = {
                    "mean_primary_macro_f1": mp,
                    "delta_dynamic_minus_static":
                        mp["dynamic"] - mp["static"],
                    "flip_rate": {"static": (flips["static"] / n_trans
                                             if n_trans else 0.0),
                                  "dynamic": (flips["dynamic"] / n_trans
                                              if n_trans else 0.0)},
                    "first_cutoff_mismatch": mismatch,
                    "n_rows": len(rows)}
                for arm in ("static", "dynamic"):
                    result[dataset]["pooled"][arm].append(mp[arm])
                    result[dataset]["pooled"][f"flip_{arm}"].append(
                        seeds_out[str(seed)]["flip_rate"][arm])
                print("E3V1 ALL-EVENT DONE", dataset, fold, seed,
                      round(mp["dynamic"] - mp["static"], 5), flush=True)
                del encoder, selector, proxy
            result[dataset]["per_fold"][f"fold{fold}"] = {
                "frozen_config": {"lambda_n": config[0],
                                  "lambda_p": config[1],
                                  "budget": config[2]},
                "seeds": seeds_out}
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
        result[dataset]["all_event_mean"] = {
            "static": mean(result[dataset]["pooled"]["static"]),
            "dynamic": mean(result[dataset]["pooled"]["dynamic"]),
            "delta": mean(result[dataset]["pooled"]["dynamic"]) -
                     mean(result[dataset]["pooled"]["static"]),
            "flip_static": mean(result[dataset]["pooled"]["flip_static"]),
            "flip_dynamic":
                mean(result[dataset]["pooled"]["flip_dynamic"])}
        del result[dataset]["pooled"]
    result["conclusion_check"] = {}
    for dataset in DATASETS:
        d = result[dataset]["all_event_mean"]["delta"]
        result["conclusion_check"][dataset] = {
            "all_event_delta_dynamic_minus_static": d,
            "direction_same_as_held_out_weak_positive": abs(d) < 0.005}
    path = os.path.join(args.out_root, "e3_v1_all_event_diagnostic.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("E3V1 DIAGNOSTIC WRITTEN", path, flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("audit", "e2", "e3v1", "all"),
                    default="all")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--e1-root", default=E1_ROOT)
    ap.add_argument("--e2-root", default=E2_ROOT)
    ap.add_argument("--e3-root", default=E3_ROOT)
    args = ap.parse_args(argv)
    if args.stage in ("audit", "all"):
        run_audit(args)
    if args.stage in ("e2", "all"):
        run_e2(args)
    if args.stage in ("e3v1", "all"):
        run_e3v1(args)
    with open(os.path.join(args.out_root, "protocol_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"stage": "dynamic_v2_protocol_correction",
                   "scope": "validation events only",
                   "stages_run": args.stage,
                   "budget": BUDGET,
                   "no_retraining": True,
                   "test_split_read": False,
                   "encoder_selector_proxy": "frozen corrected E2 "
                                             "checkpoints only"}, fh,
                  indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
