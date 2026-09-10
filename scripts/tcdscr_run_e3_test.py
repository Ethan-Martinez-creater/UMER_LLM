#!/usr/bin/env python
"""Formal E3-B — Held-out Test (E3 test order §1-§24).

One run = (dataset, fold, seed): the fold's validation-frozen best
configuration (read from results/tcdscr/formal_e3/{dataset}/fold{k}/
best_config.json — never re-searched) is applied to the outer test split.
All components (Random-init E1 encoder, corrected E2 selector and proxy)
stay frozen with before/after checksums; Static and Dynamic share the same
snapshot / candidate set / evidence units / tokenizer / token costs /
budget / proxy — only the ranking score differs (u_i vs d_i).

The test split is the only split read. No Qwen model is ever called.
"""
import argparse
import json
import os
import sys

import torch

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, build_light_item,
                           check_split_parity, classification_metrics,
                           encoder_forward_batch, load_random_e1_encoder,
                           mean_primary_macro_f1, state_sha256)

from tcdscr_run_e3 import (FOLDS, SEEDS, DATASETS, collect_event_data,
                           count_unit_costs, load_frozen_components,
                           run_event_trajectory)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"
E2_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e2_corrected"
E3_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e3"
OUT_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e3_test"


def load_best_config(e3_root, dataset, fold):
    """Authoritative frozen config; the JSON is the source of truth."""
    p = os.path.join(e3_root, dataset, f"fold{fold}", "best_config.json")
    with open(p, encoding="utf-8") as fh:
        bc = json.load(fh)
    config = (float(bc["lambda_n"]), float(bc["lambda_p"]),
              int(bc["budget"]))
    return config, bc, p


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--fold", type=int, choices=FOLDS, required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, required=True)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--e1-root", default=E1_ROOT)
    ap.add_argument("--e2-root", default=E2_ROOT)
    ap.add_argument("--e3-root", default=E3_ROOT)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N test events")
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

    config, best_config, bc_path = load_best_config(
        args.e3_root, args.dataset, args.fold)

    cfg = config_from_env(args.dataset)
    registry = event_label_registry(args.dataset, cfg)
    split = build_primary_fold_split(registry, args.fold, seed=3090)
    events = load_split_events(args.dataset, cfg, split)
    if args.limit:
        events["test"] = events["test"][:args.limit]
    parity = check_split_parity(args.dataset, args.fold, args.seed, split,
                                events, args.e1_root, registry)
    test_event_ids = sorted(ev["event_id"] for ev in events["test"])

    encoder, selector, proxy, checksums = load_frozen_components(
        args.dataset, args.fold, args.seed, args.e1_root, args.e2_root,
        device)
    for name, model in (("encoder", encoder), ("selector", selector),
                        ("proxy", proxy)):
        checksums[f"{name}_after"] = state_sha256(model.state_dict())
        checksums[f"{name}_match"] = (checksums[f"{name}_before"]
                                      == checksums[f"{name}_after"])

    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path,
                                              local_files_only=True)
    store = EventSemanticStore(cfg)
    cache = {}
    items = []
    for ev in events["test"]:
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
            z_sel = torch.zeros(768, device=device)
            return proxy.classify(h_source, z_sel)
        return classify_selected(proxy, h_source, sel_repr)

    rows = []
    first_cutoff_mismatch = 0
    for ev_rows in by_event.values():
        r, mm = run_event_trajectory(ev_rows, config, classify_fn)
        rows.extend(r)
        first_cutoff_mismatch += mm
    for r in rows:
        r["dataset"] = args.dataset
        r["fold"] = args.fold
        r["seed"] = args.seed

    by_cut = {str(c): {"static": [], "dynamic": []} for c in PRIMARY_CUTOFFS}
    for r in rows:
        by_cut[r["cutoff"]]["static"].append((r["gold"],
                                              r["static_prediction"]))
        by_cut[r["cutoff"]]["dynamic"].append((r["gold"],
                                               r["dynamic_prediction"]))
    metrics = {c: {"static": classification_metrics(by_cut[c]["static"]),
                   "dynamic": classification_metrics(by_cut[c]["dynamic"])}
               for c in by_cut}
    mean_primary = {"static": mean_primary_macro_f1(
        {c: metrics[c]["static"] for c in metrics}),
        "dynamic": mean_primary_macro_f1(
            {c: metrics[c]["dynamic"] for c in metrics})}

    from tcdscr_summarize_e3 import temporal_metrics as agg_temporal
    temporal, dynamics = agg_temporal(rows)

    run_dir = os.path.join(args.out_root, "runs", args.dataset,
                           f"fold{args.fold}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "stage": "formal_e3_test",
            "dataset": args.dataset, "fold": args.fold, "seed": args.seed,
            "config": {"lambda_n": config[0], "lambda_p": config[1],
                       "budget": config[2],
                       "source_best_config": bc_path,
                       "validation_frozen": True,
                       "test_time_search": False},
            "encoder": {"checkpoint_sha":
                        checksums["e2_manifest_encoder_sha"]},
            "selector_proxy": {
                "selector_checkpoint_sha":
                    checksums["selector_checkpoint_sha"],
                "best_epoch": checksums["e2_checksums"]["best_epoch"]},
            "checksums": checksums,
            "split_parity": parity,
            "n_test_events": len(test_event_ids),
            "test_event_ids": test_event_ids,
            "n_rows": len(rows),
            "first_cutoff_mismatch": first_cutoff_mismatch,
            "split_used": "test",
            "qwen_model_inference": False,
            "test_time_hyperparameter_search": False,
        }, fh, indent=1)
    with open(os.path.join(run_dir, "test_predictions.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(run_dir, "test_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"mean_primary_macro_f1": mean_primary,
                   "per_cutoff": metrics,
                   "n_test_events": len(test_event_ids)}, fh, indent=1)
    with open(os.path.join(run_dir, "temporal_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"temporal": temporal, "per_cutoff": dynamics},
                  fh, indent=1)
    print(json.dumps({"dataset": args.dataset, "fold": args.fold,
                      "seed": args.seed,
                      "config": {"lambda_n": config[0],
                                 "lambda_p": config[1],
                                 "budget": config[2]},
                      "n_test_events": len(test_event_ids),
                      "n_rows": len(rows),
                      "mean_primary_macro_f1": mean_primary,
                      "test_parity": parity.get("test_exact_match"),
                      "first_cutoff_mismatch": first_cutoff_mismatch,
                      "checksum_match": (checksums["encoder_match"]
                                         and checksums["selector_match"]
                                         and checksums["proxy_match"])},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
