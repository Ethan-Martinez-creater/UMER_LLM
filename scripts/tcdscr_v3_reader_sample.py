#!/usr/bin/env python
"""V3-B STEP 2-4: validation sampling population, frozen manifest, prompts.

Builds the paired reader-transfer pilot population from the frozen V3-A
validation artifacts (alpha = 0.8, context source seed = 2000), draws a
deterministic stratified sample of 300 unique event-cutoff samples per dataset
(sampling seed 3090), renders the frozen Static / MS-TSR prompts and writes:

  results/tcdscr/dynamic_v3_reader/sampling_manifest.json
  results/tcdscr/dynamic_v3_reader/prompts/<dataset>.jsonl

Only the validation split is loaded; test events are never read (§2).  The
manifest is written before any generation and is not modified afterwards (§3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tcdscr_common import (PROJECT_DIR, config_from_env,  # noqa: E402,F401
                           event_label_registry)

from tcdscr.data import maweibo_adapter, pheme_adapter  # noqa: E402
from tcdscr.data.snapshot_builder import build_snapshot  # noqa: E402
from tcdscr.context.evidence_unit import build_evidence_units  # noqa: E402
from tcdscr.llm.reader_prompt import (PROMPT_VERSION,  # noqa: E402
                                      build_arm_prompt,
                                      order_units_by_snapshot)

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
CUTOFFS = (5, 15, 30, 60, 180, 360)
SAMPLE_SEED = 3090
CONTEXT_SOURCE_SEED = 2000
ALPHA = 0.8
BUDGET = 1024
N_PER_DATASET = 300
PER_CUTOFF_TARGET = 50
PARTITION_SEED = 3090
DEFAULT_RUNS = "/data/jyz/next/llm/results/tcdscr/dynamic_v3/runs"
DEFAULT_OUT = "/data/jyz/next/llm/results/tcdscr/dynamic_v3_reader"


def sample_id(dataset, event_id, cutoff):
    return f"{dataset}__{event_id}__{cutoff}"


def pressure_bin(candidate_count, static_selected_count):
    """§8 selection pressure; ``NO_CANDIDATE`` when nothing was available."""
    if int(candidate_count) == 0:
        return "NO_CANDIDATE"
    saturation = int(static_selected_count) / int(candidate_count)
    if saturation >= 0.8:
        return "LOW"
    if saturation >= 0.4:
        return "MEDIUM"
    return "HIGH"


def arm_order(dataset, event_id, cutoff, seed=SAMPLE_SEED):
    """§19 fixed-hash arm execution order."""
    key = f"{dataset}|{event_id}|{cutoff}|{seed}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return "static_first" if int(digest[:8], 16) % 2 == 0 else "ms_first"


def stratified_sample(rows, cutoffs=CUTOFFS, per_cutoff=PER_CUTOFF_TARGET,
                      total=N_PER_DATASET, seed=SAMPLE_SEED):
    """Cutoff-stratified deterministic sample with even backfill (§7).

    Returns ``(chosen, allocation, deviation, spare)`` where ``allocation`` is
    the per-cutoff count actually taken.  Under-populated cutoffs are kept in
    full and the shortfall is spread evenly over the cutoffs that still have
    spare rows, never by dropping 5m/15m.
    """
    rng = random.Random(seed)
    by_cut = {}
    for c in cutoffs:
        pool = [r for r in rows if int(r["cutoff"]) == int(c)]
        pool.sort(key=lambda r: (str(r["event_id"]), int(r["cutoff"])))
        by_cut[int(c)] = pool
    take = {c: min(per_cutoff, len(by_cut[c])) for c in cutoffs}
    spare = total - sum(take.values())
    while spare > 0:
        avail = [c for c in cutoffs if take[c] < len(by_cut[c])]
        if not avail:
            break
        share = max(1, spare // len(avail))
        progressed = False
        for c in avail:
            if spare == 0:
                break
            give = min(share, spare, len(by_cut[c]) - take[c])
            if give > 0:
                take[c] += give
                spare -= give
                progressed = True
        if not progressed:
            break
    chosen = []
    for c in cutoffs:
        pool = by_cut[c]
        if take[c] >= len(pool):
            picked = list(pool)
        else:
            picked = rng.sample(pool, take[c])
        picked.sort(key=lambda r: (int(r["cutoff"]), str(r["event_id"])))
        chosen.extend(picked)
    deviation = {str(c): take[c] - per_cutoff for c in cutoffs}
    return chosen, {str(c): take[c] for c in cutoffs}, deviation, spare


def load_population(dataset, runs_root, validation_ids_by_fold,
                    test_ids_by_fold):
    """All alpha=0.8 / seed=2000 validation rows for one dataset, deduplicated.

    Raises ``RuntimeError("TEST_PROTOCOL_VIOLATION")`` if any run row claims an
    event that is in the test split of its fold.
    """
    by_key = {}
    conflicts = []
    for fold in FOLDS:
        path = os.path.join(runs_root, dataset,
                            f"fold{fold}_seed{CONTEXT_SOURCE_SEED}",
                            "validation_predictions.jsonl")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if abs(float(row["alpha"]) - ALPHA) > 1e-9:
                    continue
                if int(row["seed"]) != CONTEXT_SOURCE_SEED:
                    continue
                if int(row["fold"]) != fold:
                    continue
                eid = row["event_id"]
                if eid in test_ids_by_fold[fold]:
                    raise RuntimeError("TEST_PROTOCOL_VIOLATION")
                if eid not in validation_ids_by_fold[fold]:
                    raise RuntimeError(f"event {eid} is not in fold{fold} "
                                       "validation")
                key = (eid, str(row["cutoff"]))
                if key in by_key:
                    conflicts.append({"event_id": eid,
                                      "cutoff": str(row["cutoff"]),
                                      "folds": [by_key[key]["fold"],
                                                int(row["fold"])]})
                    if int(row["fold"]) < int(by_key[key]["fold"]):
                        by_key[key] = row
                else:
                    by_key[key] = row
    rows = list(by_key.values())
    rows.sort(key=lambda r: (int(r["cutoff"]), str(r["event_id"])))
    return rows, conflicts


def load_validation_events(dataset, cfg, wanted_ids):
    """Load only the validation events actually sampled."""
    wanted = set(wanted_ids)
    if dataset == "pheme":
        by_id = {eid: (topic, label, folder)
                 for eid, topic, label, folder
                 in pheme_adapter.event_ids(cfg.raw_dir)}
        return {eid: pheme_adapter.load_event(*by_id[eid])
                for eid in sorted(wanted)}
    labels = dict(maweibo_adapter.event_ids(cfg.raw_dir, cfg.label_file))
    return {eid: maweibo_adapter.load_event(
        eid, labels[eid], f"{cfg.raw_dir.rstrip('/')}/{eid}.json")
        for eid in sorted(wanted)}


def build_sample_records(dataset, rows, events, tokenizer):
    """Manifest rows + prompt rows for the chosen event-cutoff samples."""
    manifest_rows = []
    prompt_rows = []
    for row in rows:
        eid = row["event_id"]
        cutoff = int(row["cutoff"])
        event = events[eid]
        snapshot = build_snapshot(event, cutoff)
        units = build_evidence_units(snapshot)
        pos = {nid: i for i, nid in enumerate(snapshot["node_ids"])}
        source_text = snapshot["texts"][pos[snapshot["source_id"]]]

        static_ids = list(row["static_selected_node_ids"])
        ms_ids = list(row["ms_selected_node_ids"])
        for nid in static_ids + ms_ids:
            if nid not in pos:
                raise RuntimeError(
                    f"{dataset}/{eid}/{cutoff}: selected node {nid} is not in "
                    "the snapshot")
        static_units = order_units_by_snapshot(units, static_ids)
        ms_units = order_units_by_snapshot(units, ms_ids)
        if len(static_units) != len(static_ids) or len(ms_units) != len(ms_ids):
            raise RuntimeError(
                f"{dataset}/{eid}/{cutoff}: evidence lost while ordering")

        static_arm = build_arm_prompt(tokenizer, source_text, cutoff,
                                      static_units)
        ms_arm = build_arm_prompt(tokenizer, source_text, cutoff, ms_units)
        sid = sample_id(dataset, eid, cutoff)
        arm = arm_order(dataset, eid, cutoff)

        if row["fallback_to_static"] and set(static_ids) != set(ms_ids):
            raise RuntimeError(
                f"{dataset}/{eid}/{cutoff}: fallback arm differs from Static")

        manifest_rows.append({
            "sample_id": sid,
            "seed": SAMPLE_SEED,
            "dataset": dataset,
            "event_id": eid,
            "cutoff": str(cutoff),
            "fold": int(row["fold"]),
            "context_source_seed": CONTEXT_SOURCE_SEED,
            "gold": int(row["gold"]),
            "candidate_count": int(row["candidate_count"]),
            "pressure_bin": pressure_bin(row["candidate_count"],
                                         row["static_selected_count"]),
            "static_margin": float(row["static_margin"]),
            "ms_margin": float(row["ms_margin"]),
            "dual_view_agree": bool(row["dual_view_agree"]),
            "fallback_to_static": bool(row["fallback_to_static"]),
            "static_selected_node_ids": static_ids,
            "ms_selected_node_ids": ms_ids,
            "static_selected_count": int(row["static_selected_count"]),
            "ms_selected_count": int(row["ms_selected_count"]),
            "v3a_static_units": int(row["static_evidence_tokens"]),
            "v3a_ms_units": int(row["ms_evidence_tokens"]),
            "static_social_tokens": int(static_arm["social_tokens"]),
            "ms_social_tokens": int(ms_arm["social_tokens"]),
            "static_total_input_tokens": int(
                static_arm["total_input_tokens"]),
            "ms_total_input_tokens": int(ms_arm["total_input_tokens"]),
            "arm_order": arm,
        })
        for arm_name, bundle in (("static", static_arm), ("ms", ms_arm)):
            prompt_rows.append({
                "sample_id": sid,
                "dataset": dataset,
                "event_id": eid,
                "cutoff": str(cutoff),
                "arm": arm_name,
                "prompt": bundle["prompt"],
                "evidence_ids": bundle["evidence_ids"],
                "n_evidence": bundle["n_evidence"],
                "social_tokens": bundle["social_tokens"],
                "total_input_tokens": bundle["total_input_tokens"],
            })
    return manifest_rows, prompt_rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from tcdscr.data.temporal_split import build_primary_fold_split
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "prompts"), exist_ok=True)

    summary = {"seed": SAMPLE_SEED, "alpha": ALPHA, "budget": BUDGET,
               "context_source_seed": CONTEXT_SOURCE_SEED,
               "n_per_dataset": N_PER_DATASET,
               "per_cutoff_target": PER_CUTOFF_TARGET,
               "prompt_version": PROMPT_VERSION,
               "runs_root": args.runs_root,
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
               "datasets": {}, "samples": []}

    probe_cfg = config_from_env(DATASETS[0])
    model_path = args.model_path or probe_cfg.qwen_model_path
    summary["model_path"] = model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=False)

    for dataset in DATASETS:
        cfg = config_from_env(dataset)
        registry = event_label_registry(dataset, cfg)
        val_ids, test_ids = {}, {}
        for fold in FOLDS:
            split = build_primary_fold_split(registry, fold,
                                             seed=PARTITION_SEED)
            val_ids[fold] = set(split["validation"])
            test_ids[fold] = set(split["test"])
        population, conflicts = load_population(
            dataset, args.runs_root, val_ids, test_ids)
        chosen, allocation, deviation, spare = stratified_sample(population)
        events = load_validation_events(
            dataset, cfg, [r["event_id"] for r in chosen])
        manifest_rows, prompt_rows = build_sample_records(
            dataset, chosen, events, tokenizer)

        with open(os.path.join(args.out, "prompts", f"{dataset}.jsonl"),
                  "w", encoding="utf-8") as fh:
            for r in prompt_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        pressure_counts = {}
        for r in manifest_rows:
            pressure_counts[r["pressure_bin"]] = \
                pressure_counts.get(r["pressure_bin"], 0) + 1
        summary["datasets"][dataset] = {
            "n": len(manifest_rows),
            "population": len(population),
            "population_by_cutoff": {
                str(c): sum(1 for r in population
                            if int(r["cutoff"]) == int(c))
                for c in CUTOFFS},
            "allocation": allocation,
            "allocation_deviation": deviation,
            "unfilled_slots": spare,
            "pressure_counts": pressure_counts,
            "cross_fold_conflicts": conflicts,
        }
        summary["samples"].extend(manifest_rows)
        print(json.dumps({"dataset": dataset, "n": len(manifest_rows),
                          "population": len(population),
                          "allocation": allocation,
                          "deviation": deviation, "spare": spare,
                          "pressure": pressure_counts}, indent=1))

    payload = json.dumps(summary["samples"], sort_keys=True,
                         ensure_ascii=False).encode("utf-8")
    summary["samples_sha256"] = hashlib.sha256(payload).hexdigest()
    with open(os.path.join(args.out, "sampling_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(json.dumps({"samples_sha256": summary["samples_sha256"],
                      "n_total": len(summary["samples"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
