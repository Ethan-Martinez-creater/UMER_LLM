#!/usr/bin/env python
"""Final Evidence Closure — Gap B + Gap C reader context build.

Per (dataset, fold): sample 60 unique outer-test events (10 per cutoff, six
disjoint cutoff groups) from the outer test split, excluding every event id
that appeared in the V3-B validation reader pilot.  For each sampled
(event, cutoff) build four frozen contexts:

  STATIC_FULL              corrected Static Utility selection under budget 1024
  UTILITY_TOKEN_MATCHED    Static-utility order, scanned so the rendered social
                           block never exceeds the MS-TSR social-token count
  RANDOM_TOKEN_MATCHED     deterministic hash permutation, same token cap
  MS_TSR                   frozen V3-A MS-TSR (alpha 0.8, budget 1024)

Gold labels are never read here: sampling uses only fold membership, event
id, cutoff allocation, the prior-reader exclusion set and the fixed seed.

Outputs: results/tcdscr/final_evidence/reader/ (exclusion set, contexts,
prompts, sampling manifest + sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "project")
for _path in (HERE, PROJECT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tcdscr_run_e2 import PRIMARY_CUTOFFS, build_light_item, build_units

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
CUTOFFS = tuple(PRIMARY_CUTOFFS)
PER_CUTOFF = 10
PER_FOLD = PER_CUTOFF * len(CUTOFFS)
SAMPLING_SEED = 4096
CONTEXT_SOURCE_SEED = 2000
ALPHA = 0.8
BUDGET = 1024
PARTITION_SEED = 3090
ARMS = ("STATIC_FULL", "UTILITY_TOKEN_MATCHED", "RANDOM_TOKEN_MATCHED",
        "MS_TSR")

E1_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e1"
E2_ROOT = "/data/jyz/next/llm/results/tcdscr/formal_e2_corrected"
V3B_READER_ROOT = "/data/jyz/next/llm/results/tcdscr/dynamic_v3_reader"
OUT_ROOT = "/data/jyz/next/llm/results/tcdscr/final_evidence/reader"


def stable_seed(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8"))
    return int(h.hexdigest()[:16], 16)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--v3b-reader-root", default=V3B_READER_ROOT)
    ap.add_argument("--e1-root", default=E1_ROOT)
    ap.add_argument("--e2-root", default=E2_ROOT)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--folds", default=",".join(str(f) for f in FOLDS))
    return ap


def load_prior_excluded(v3b_root):
    """Every event id that appeared in the V3-B validation reader pilot."""
    path = os.path.join(v3b_root, "sampling_manifest.json")
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    ids = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("event_id", "event_ids") and isinstance(v, str):
                    ids.add(v)
                elif k in ("event_id", "event_ids") and isinstance(v, list):
                    ids.update(str(x) for x in v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(man)
    per_ds = {}
    for ds, block in (man.get("datasets") or {}).items() if isinstance(
            man.get("datasets"), dict) else []:
        per_ds[ds] = sorted({r.get("event_id") for r in
                             (block.get("samples") or []) if r.get("event_id")})
    return ids, per_ds


def allocate(available_ids, dataset, fold):
    """Deterministic 60-event allocation, 10 per cutoff, disjoint groups."""
    rng = random.Random(stable_seed("sample", dataset, fold, SAMPLING_SEED))
    pool = sorted(available_ids)
    rng.shuffle(pool)
    need = PER_FOLD
    picked = pool[:need]
    allocation = {}
    for i, c in enumerate(CUTOFFS):
        allocation[c] = picked[i * PER_CUTOFF:(i + 1) * PER_CUTOFF]
    shortage = max(0, need - len(picked))
    return allocation, shortage, len(pool)


def token_matched(units_by_id, ordered_ids, target_tokens, tokenizer,
                  order_of):
    """Scan in the given order, keep every unit that still fits the cap.

    Tokens are counted on the final presentation order (snapshot order), which
    is exactly what the reader sees.
    """
    from tcdscr.llm.reader_prompt import (count_tokens,
                                          render_evidence_block)
    chosen = []
    for nid in ordered_ids:
        trial = chosen + [nid]
        sel = sorted(trial, key=lambda n: order_of[n])
        block = render_evidence_block([units_by_id[n] for n in sel])
        if count_tokens(tokenizer, block) <= target_tokens:
            chosen = trial
    return chosen


def build_cell(dataset, fold, args, tokenizer, device):
    """Returns (manifest_rows, context_rows, prompt_rows) for one fold."""
    import torch

    from tcdscr.config.schema import config_from_env
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr.llm.reader_prompt import (build_arm_prompt,
                                          order_units_by_snapshot)
    from tcdscr_common import (EventSemanticStore, event_label_registry,
                               load_split_events)
    from tcdscr_run_dynamic_v2 import collect_event_data_v2
    from tcdscr_run_dynamic_v3 import run_event_trajectory_v3
    from tcdscr_run_e3 import load_frozen_components

    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    split = build_primary_fold_split(registry, fold, seed=PARTITION_SEED)
    events = load_split_events(dataset, cfg, split)
    test_events = events["test"]

    excluded, _ = load_prior_excluded(args.v3b_reader_root)
    available = {ev["event_id"] for ev in test_events} - excluded
    allocation, shortage, pool_size = allocate(available, dataset, fold)
    wanted = {eid for ids in allocation.values() for eid in ids}
    by_id = {ev["event_id"]: ev for ev in test_events}

    encoder, selector, proxy, _chk = load_frozen_components(
        dataset, fold, CONTEXT_SOURCE_SEED, args.e1_root, args.e2_root,
        device)
    store = EventSemanticStore(cfg)
    cache = {}

    manifest_rows, context_rows, prompt_rows = [], [], []
    for cutoff in CUTOFFS:
        for eid in allocation[cutoff]:
            ev = by_id[eid]
            items = [build_light_item(ev, build_snapshot(ev, c),
                                      store.get_store(ev, cache))
                     for c in CUTOFFS]
            by_event = collect_event_data_v2(encoder, selector, items,
                                             tokenizer, device)
            rows = run_event_trajectory_v3(by_event[eid], ALPHA, proxy,
                                           device, budget=BUDGET)
            row = next(r for r in rows if int(r["cutoff"]) == int(cutoff))
            src_item = next(i for i in items
                            if int(i["cutoff_minutes"]) == int(cutoff))
            source_text = src_item["texts"][src_item["source_pos"]]
            units = build_units(src_item)
            units_by_id = {u["node_id"]: u for u in units}
            collected = next(r for r in by_event[eid]
                             if int(r["cutoff"]) == int(cutoff))
            cand_ids = list(collected["cand_node_ids"])
            u_by_id = {nid: float(v) for nid, v in
                       zip(collected["cand_node_ids"], collected["u"])}

            static_ids = list(row.get("static_selected_node_ids") or [])
            ms_ids = list(row.get("ms_selected_node_ids") or [])
            order_of = {u["node_id"]: u["order"] for u in units}

            static_units = order_units_by_snapshot(units, static_ids)
            ms_units = order_units_by_snapshot(units, ms_ids)
            target = build_arm_prompt(tokenizer, source_text, cutoff,
                                      ms_units)["social_tokens"]

            util_order = sorted(cand_ids, key=lambda n: -u_by_id.get(n, 0.0))
            rng = random.Random(stable_seed("random_token_matched", dataset,
                                            fold, eid, cutoff, SAMPLING_SEED))
            rand_order = list(cand_ids)
            rng.shuffle(rand_order)
            util_chosen = token_matched(units_by_id, util_order, target,
                                        tokenizer, order_of)
            rand_chosen = token_matched(units_by_id, rand_order, target,
                                        tokenizer, order_of)

            arm_units = {
                "STATIC_FULL": static_units,
                "UTILITY_TOKEN_MATCHED": order_units_by_snapshot(
                    units, util_chosen),
                "RANDOM_TOKEN_MATCHED": order_units_by_snapshot(
                    units, rand_chosen),
                "MS_TSR": ms_units,
            }
            sample_id = f"{dataset}_fold{fold}_{eid}_{int(cutoff)}"
            tokens = {}
            for arm in ARMS:
                bundle = build_arm_prompt(tokenizer, source_text, cutoff,
                                          arm_units[arm])
                tokens[arm] = bundle["social_tokens"]
                prompt_rows.append({
                    "sample_id": sample_id, "arm": arm,
                    "dataset": dataset, "fold": fold, "event_id": eid,
                    "cutoff": int(cutoff),
                    "prompt": bundle["prompt"],
                    "system_prompt": bundle["system_prompt"],
                    "evidence_ids": bundle["evidence_ids"],
                    "social_tokens": bundle["social_tokens"],
                    "total_input_tokens": bundle["total_input_tokens"],
                    "n_evidence": bundle["n_evidence"],
                })
                context_rows.append({
                    "sample_id": sample_id, "arm": arm,
                    "dataset": dataset, "fold": fold, "event_id": eid,
                    "cutoff": int(cutoff),
                    "selected_node_ids": [u["node_id"]
                                          for u in arm_units[arm]],
                    "evidence_block": bundle["evidence_block"],
                    "social_tokens": bundle["social_tokens"],
                    "n_candidates": int(row["candidate_count"]),
                })
            if tokens["UTILITY_TOKEN_MATCHED"] > target or \
                    tokens["RANDOM_TOKEN_MATCHED"] > target:
                raise RuntimeError("token-matched arm exceeded the MS cap")
            arm_order = list(ARMS)
            random.Random(stable_seed("arm_order", dataset, fold, eid,
                                      cutoff, SAMPLING_SEED)).shuffle(arm_order)
            manifest_rows.append({
                "sample_id": sample_id,
                "dataset": dataset, "fold": fold, "event_id": eid,
                "cutoff": int(cutoff),
                "context_source_seed": CONTEXT_SOURCE_SEED,
                "prior_reader_excluded": bool(eid in excluded),
                "candidate_count": int(row["candidate_count"]),
                "fallback_to_static": bool(row.get("fallback_to_static",
                                                   False)),
                "static_selected_ids": static_ids,
                "ms_selected_ids": ms_ids,
                "utility_tm_selected_ids": [u["node_id"]
                                            for u in arm_units[
                                                "UTILITY_TOKEN_MATCHED"]],
                "random_tm_selected_ids": [u["node_id"]
                                           for u in arm_units[
                                               "RANDOM_TOKEN_MATCHED"]],
                "static_social_tokens": tokens["STATIC_FULL"],
                "ms_social_tokens": tokens["MS_TSR"],
                "utility_tm_social_tokens": tokens["UTILITY_TOKEN_MATCHED"],
                "random_tm_social_tokens": tokens["RANDOM_TOKEN_MATCHED"],
                "target_social_tokens": target,
                "utility_token_gap": target - tokens["UTILITY_TOKEN_MATCHED"],
                "random_token_gap": target - tokens["RANDOM_TOKEN_MATCHED"],
                "arm_order": arm_order,
                "allocation_shortage": bool(shortage),
            })
    del encoder, selector, proxy
    return manifest_rows, context_rows, prompt_rows, shortage, pool_size


def main(argv=None):
    args = build_parser().parse_args(argv)
    import torch
    from transformers import AutoTokenizer
    from tcdscr.config.schema import config_from_env
    from tcdscr_run_v3_reader import sha256_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = args.datasets.split(",")
    folds = [int(f) for f in args.folds.split(",")]
    os.makedirs(args.out_root, exist_ok=True)
    for sub in ("contexts", "prompts"):
        os.makedirs(os.path.join(args.out_root, sub), exist_ok=True)

    excluded, per_ds = load_prior_excluded(args.v3b_reader_root)
    with open(os.path.join(args.out_root, "prior_reader_exclusion.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"source": os.path.join(args.v3b_reader_root,
                                          "sampling_manifest.json"),
                   "n_excluded_event_ids": len(excluded),
                   "per_dataset": per_ds,
                   "excluded_event_ids": sorted(excluded)}, fh, indent=1)

    tokenizer = AutoTokenizer.from_pretrained(
        config_from_env(datasets[0]).qwen_model_path, local_files_only=True)
    manifest, contexts, prompts = [], [], []
    for dataset in datasets:
        for fold in folds:
            m, c, p, shortage, pool = build_cell(dataset, fold, args,
                                                 tokenizer, device)
            manifest.extend(m)
            contexts.extend(c)
            prompts.extend(p)
            print(json.dumps({"built": dataset, "fold": fold,
                              "samples": len(m), "shortage": shortage,
                              "pool": pool}), flush=True)
    for dataset in datasets:
        with open(os.path.join(args.out_root, "contexts",
                               f"{dataset}.jsonl"), "w",
                  encoding="utf-8") as fh:
            for r in contexts:
                if r["dataset"] == dataset:
                    fh.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out_root, "prompts",
                               f"{dataset}.jsonl"), "w",
                  encoding="utf-8") as fh:
            for r in prompts:
                if r["dataset"] == dataset:
                    fh.write(json.dumps(r) + "\n")
    man_path = os.path.join(args.out_root, "reader_sampling_manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump({"stage": "final_evidence_reader",
                   "prompt_version": "v3b-2",
                   "sampling_seed": SAMPLING_SEED,
                   "context_source_seed": CONTEXT_SOURCE_SEED,
                   "alpha": ALPHA, "budget": BUDGET,
                   "partition_seed": PARTITION_SEED,
                   "arms": list(ARMS),
                   "n_samples": len(manifest),
                   "samples": manifest}, fh, indent=1)
    digest = sha256_file(man_path)
    with open(os.path.join(args.out_root,
                           "reader_sampling_manifest.sha256"), "w",
              encoding="utf-8") as fh:
        fh.write(digest + "\n")
    print(json.dumps({"n_samples": len(manifest), "sha256": digest}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
