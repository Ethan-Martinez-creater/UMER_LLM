#!/usr/bin/env python
"""Final Evidence Closure — Gap F: Dynamic V1 all-event held-out closure.

Re-evaluates the FROZEN per-fold Dynamic V1 configuration (read from the
validation-stage best_config.json; never re-searched) against the frozen
Static arm on the outer test split, with the all-event protocol: every test
event x every cutoff produces a row, including candidate_count == 0
(empty static and dynamic selection, prediction still generated).

No retraining, no test-time search.  Outputs go to
results/tcdscr/final_evidence/gap_f/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "project")
for _path in (HERE, PROJECT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, build_light_item,
                           classification_metrics, mean_primary_macro_f1,
                           state_sha256)

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
PARTITION_SEED = 3090

E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"
E2_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e2_corrected"
E3_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e3"
OUT_ROOT = "/data/jyz/next/llm/results/tcdscr/final_evidence/gap_f"


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--fold", type=int, choices=FOLDS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--e1-root", default=E1_ROOT)
    ap.add_argument("--e2-root", default=E2_ROOT)
    ap.add_argument("--e3-root", default=E3_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    return ap


def load_best_config(e3_root, dataset, fold):
    path = os.path.join(e3_root, dataset, f"fold{fold}", "best_config.json")
    with open(path, encoding="utf-8") as fh:
        bc = json.load(fh)
    return (float(bc["lambda_n"]), float(bc["lambda_p"]),
            int(bc["budget"])), bc, path


def run_one(args, dataset, fold, seed):
    import torch
    from transformers import AutoTokenizer

    from tcdscr.config.schema import config_from_env
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr_common import (EventSemanticStore, event_label_registry,
                               load_split_events)
    from tcdscr_run_e3 import (collect_event_data, count_unit_costs,
                               load_frozen_components, run_event_trajectory)
    from tcdscr_run_e2 import file_sha256

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, best_config, bc_path = load_best_config(args.e3_root, dataset,
                                                    fold)
    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    split = build_primary_fold_split(registry, fold, seed=PARTITION_SEED)
    events = load_split_events(dataset, cfg, split)
    test_events = events["test"]
    if args.limit:
        test_events = test_events[:args.limit]

    encoder, selector, proxy, checksums = load_frozen_components(
        dataset, fold, seed, args.e1_root, args.e2_root, device)
    match = {name: state_sha256(m.state_dict()) == checksums[f"{name}_before"]
             for name, m in (("encoder", encoder), ("selector", selector),
                             ("proxy", proxy))}
    trainable = sum(p.numel() for m in (encoder, selector, proxy)
                    for p in m.parameters() if p.requires_grad)
    if trainable:
        raise RuntimeError("frozen components still require grad")

    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path,
                                              local_files_only=True)
    store = EventSemanticStore(cfg)
    cache = {}
    items = []
    for ev in test_events:
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
    mismatch = 0
    for ev_rows in by_event.values():
        r, mm = run_event_trajectory(ev_rows, config, classify_fn)
        rows.extend(r)
        mismatch += mm
    for r in rows:
        r["dataset"] = dataset
        r["fold"] = fold
        r["seed"] = seed

    n_events = len({r["event_id"] for r in rows})
    expected = n_events * len(PRIMARY_CUTOFFS)
    if len(rows) != expected:
        raise RuntimeError(f"all-event coverage violated: {len(rows)} rows "
                           f"for {n_events} events x 6 cutoffs")
    empty_rows = sum(1 for r in rows if r["n_candidates"] == 0)
    if empty_rows == 0:
        raise RuntimeError("no no-candidate row produced")

    by_cut = {str(c): {"static": [], "dynamic": []} for c in PRIMARY_CUTOFFS}
    for r in rows:
        by_cut[r["cutoff"]]["static"].append((r["gold"],
                                              r["static_prediction"]))
        by_cut[r["cutoff"]]["dynamic"].append((r["gold"],
                                               r["dynamic_prediction"]))
    metrics = {c: {a: classification_metrics(by_cut[c][a])
                   for a in ("static", "dynamic")} for c in by_cut}
    mean_primary = {a: mean_primary_macro_f1({c: metrics[c][a]
                                              for c in metrics})
                    for a in ("static", "dynamic")}

    run_dir = os.path.join(args.out_root, "runs", dataset,
                           f"fold{fold}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "test_predictions.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(run_dir, "run_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "stage": "final_evidence_gap_f",
            "dataset": dataset, "fold": fold, "seed": seed,
            "split_used": "test",
            "protocol": "all events at every real cutoff (candidate_count "
                        "== 0 included)",
            "config": {"lambda_n": config[0], "lambda_p": config[1],
                       "budget": config[2],
                       "source_best_config": bc_path,
                       "validation_frozen": True,
                       "test_time_search": False,
                       "best_config_sha256": file_sha256(bc_path)},
            "partition_seed": PARTITION_SEED,
            "n_test_events": n_events, "n_rows": len(rows),
            "expected_rows": expected, "no_candidate_rows": empty_rows,
            "mean_primary_macro_f1": mean_primary,
            "per_cutoff": metrics,
            "first_cutoff_mismatch": mismatch,
            "frozen_checksum_match": match,
            "new_trainable_parameters": 0,
        }, fh, indent=1)
    print(json.dumps({"gap": "F", "dataset": dataset, "fold": fold,
                      "seed": seed, "n_events": n_events,
                      "n_rows": len(rows), "no_candidate_rows": empty_rows,
                      "mean_primary": {k: round(v, 4)
                                       for k, v in mean_primary.items()}}),
          flush=True)
    del encoder, selector, proxy
    return metrics


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.all:
        for dataset in DATASETS:
            for fold in FOLDS:
                for seed in SEEDS:
                    run_one(args, dataset, fold, seed)
        return 0
    if args.dataset is None or args.fold is None or args.seed is None:
        raise SystemExit("--dataset/--fold/--seed required unless --all")
    run_one(args, args.dataset, args.fold, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
