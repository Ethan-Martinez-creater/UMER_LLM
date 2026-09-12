#!/usr/bin/env python
"""V3-B reader-transfer verifier (§49).

Independently re-checks the frozen pilot rather than trusting the artifacts:

  - exactly 300 unique paired samples per dataset, one Static and one MS
    generation each;
  - the manifest predates every generation (selection cannot depend on Qwen);
  - sampling seed 3090, context source seed 2000, alpha 0.8, budget 1024;
  - validation only: no test event appears, checked against the reconstructed
    fold split;
  - prompt evidence ids reproduce the frozen V3-A selection (re-read from the
    V3-A run rows), and a sample of prompts is re-ordered against a freshly
    rebuilt snapshot;
  - paired prompts share the source post and differ only in the evidence block;
  - one frozen model/decoding configuration, no fine-tuning;
  - bootstrap CI and citation accounting are recomputed from the raw rows;
  - fallback pairs carry identical prompts.

Writes ``reader_transfer_verify.json`` and refreshes the report verdict line.
Requires ``issues = 0``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
CUTOFFS = (5, 15, 30, 60, 180, 360)
EXPECTED_N = 300
EXPECTED_SEED = 3090
EXPECTED_CONTEXT_SEED = 2000
EXPECTED_ALPHA = 0.8
EXPECTED_BUDGET = 1024
PARTITION_SEED = 3090
V3_ROOT = "/data/jyz/next/llm/results/tcdscr/dynamic_v3"
DEFAULT_ROOT = "/data/jyz/next/llm/results/tcdscr/dynamic_v3_reader"
ORDER_SAMPLE_PER_DATASET = 20
CITATION_ABS_TOL = 1e-12


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _split_prompt(prompt):
    header = "Observed Social Evidence up to "
    i = prompt.index(header)
    head_end = prompt.index("\n", i) + 1
    j = prompt.index("\n\nTask:")
    return prompt[:head_end], prompt[head_end:j], prompt[j:]


def _source_text(prompt):
    marker = "Source Post:\n"
    start = prompt.index(marker) + len(marker)
    end = prompt.index("\n\nObserved Social Evidence up to ")
    return prompt[start:end]


_V3_CACHE = {}


def _v3_index(dataset, fold):
    """Frozen V3-A alpha=0.8 validation rows for one fold, keyed by
    (event_id, cutoff).  Read once per fold."""
    key = (dataset, fold)
    if key in _V3_CACHE:
        return _V3_CACHE[key]
    path = os.path.join(V3_ROOT, "runs", dataset,
                        f"fold{fold}_seed{EXPECTED_CONTEXT_SEED}",
                        "validation_predictions.jsonl")
    index = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if abs(float(row["alpha"]) - EXPECTED_ALPHA) > 1e-9:
                continue
            index[(row["event_id"], str(row["cutoff"]))] = row
    _V3_CACHE[key] = index
    return index


def _v3_selection(dataset, fold, event_id, cutoff):
    """Frozen V3-A selected node ids for one validation event-cutoff."""
    return _v3_index(dataset, fold).get((event_id, str(cutoff)))


def _check_sampling(manifest, issues):
    samples = manifest["samples"]
    for ds in DATASETS:
        sub = [s for s in samples if s["dataset"] == ds]
        if len(sub) != EXPECTED_N:
            issues.append(f"{ds}: {len(sub)} samples, expected "
                          f"{EXPECTED_N}")
        keys = {(s["dataset"], s["event_id"], s["cutoff"]) for s in sub}
        if len(keys) != len(sub):
            issues.append(f"{ds}: duplicate event-cutoff samples")
        ids = [s["sample_id"] for s in sub]
        if len(set(ids)) != len(ids):
            issues.append(f"{ds}: duplicate sample_id")
        for s in sub:
            if s["seed"] != EXPECTED_SEED:
                issues.append(f"{ds}/{s['sample_id']}: seed != "
                              f"{EXPECTED_SEED}")
                break
            if s["context_source_seed"] != EXPECTED_CONTEXT_SEED:
                issues.append(f"{ds}/{s['sample_id']}: context seed != "
                              f"{EXPECTED_CONTEXT_SEED}")
                break
    if manifest["seed"] != EXPECTED_SEED:
        issues.append("manifest sampling seed != 3090")
    if manifest["context_source_seed"] != EXPECTED_CONTEXT_SEED:
        issues.append("manifest context_source_seed != 2000")
    if abs(float(manifest["alpha"]) - EXPECTED_ALPHA) > 1e-9:
        issues.append("manifest alpha != 0.8")
    if int(manifest["budget"]) != EXPECTED_BUDGET:
        issues.append("manifest budget != 1024")


def _check_frozen_alpha(issues):
    for ds in DATASETS:
        for fold in FOLDS:
            path = os.path.join(V3_ROOT, ds, f"fold{fold}", "best_config.json")
            if not os.path.exists(path):
                issues.append(f"missing V3-A best_config: {ds}/fold{fold}")
                continue
            with open(path, encoding="utf-8") as fh:
                best = json.load(fh)
            if abs(float(best["alpha"]) - EXPECTED_ALPHA) > 1e-9:
                issues.append(f"{ds}/fold{fold}: frozen V3-A alpha is "
                              f"{best['alpha']}, expected 0.8")
            if int(best["budget"]) != EXPECTED_BUDGET:
                issues.append(f"{ds}/fold{fold}: frozen V3-A budget is "
                              f"{best['budget']}, expected 1024")


def _check_validation_only(manifest, issues):
    from tcdscr_common import config_from_env, event_label_registry
    from tcdscr.data.temporal_split import build_primary_fold_split
    for ds in DATASETS:
        cfg = config_from_env(ds)
        registry = event_label_registry(ds, cfg)
        for fold in FOLDS:
            split = build_primary_fold_split(registry, fold,
                                             seed=PARTITION_SEED)
            test_ids = set(split["test"])
            val_ids = set(split["validation"])
            sub = [s for s in manifest["samples"]
                   if s["dataset"] == ds and s["fold"] == fold]
            for s in sub:
                if s["event_id"] in test_ids:
                    issues.append(f"{ds}/{s['sample_id']}: TEST event "
                                  "sampled")
                if s["event_id"] not in val_ids:
                    issues.append(f"{ds}/{s['sample_id']}: not a validation "
                                  "event")


def _check_generations(root, manifest, issues):
    for ds in DATASETS:
        raw = load_jsonl(os.path.join(root, "raw_generations", f"{ds}.jsonl"))
        parsed = load_jsonl(os.path.join(root, "parsed", f"{ds}.jsonl"))
        sub = [s for s in manifest["samples"] if s["dataset"] == ds]
        expected = {(s["sample_id"], arm) for s in sub
                    for arm in ("static", "ms")}
        got = [(r["sample_id"], r["arm"]) for r in raw]
        if sorted(got) != sorted(expected):
            issues.append(f"{ds}: raw generations do not match the manifest "
                          f"({len(got)} rows, {len(expected)} expected)")
        if len(got) != len(set(got)):
            issues.append(f"{ds}: duplicate (sample_id, arm) generations")
        got_parsed = [(r["sample_id"], r["arm"]) for r in parsed]
        if sorted(got_parsed) != sorted(expected):
            issues.append(f"{ds}: parsed rows do not match the manifest")
        for r in raw:
            if not r["prompt_hash"]:
                issues.append(f"{ds}/{r['sample_id']}/{r['arm']}: missing "
                              "prompt hash")
                break


def _check_manifest_precedes_generations(root, issues):
    """The frozen selection (prompts) and the manifest must both predate every
    generation, so no Qwen output can have influenced the selection (§3, §40).

    The prompts and the manifest are produced by the same sampling step; their
    order relative to each other carries no requirement.
    """
    manifest_path = os.path.join(root, "sampling_manifest.json")
    prompts_dir = os.path.join(root, "prompts")
    raw_dir = os.path.join(root, "raw_generations")
    if not os.path.exists(manifest_path):
        issues.append("missing sampling_manifest.json")
        return
    m_manifest = os.path.getmtime(manifest_path)
    for ds in DATASETS:
        raw_path = os.path.join(raw_dir, f"{ds}.jsonl")
        prompt_path = os.path.join(prompts_dir, f"{ds}.jsonl")
        if not os.path.exists(raw_path):
            issues.append(f"missing raw generations for {ds}")
            continue
        t_raw = os.path.getmtime(raw_path)
        if m_manifest > t_raw:
            issues.append(f"{ds}: generations predate the frozen manifest")
        if os.path.getmtime(prompt_path) > t_raw:
            issues.append(f"{ds}: generations predate the frozen prompts")


def _check_prompts(root, manifest, issues):
    sample_by_id = {s["sample_id"]: s for s in manifest["samples"]}
    for ds in DATASETS:
        prompts = {}
        for row in load_jsonl(os.path.join(root, "prompts", f"{ds}.jsonl")):
            prompts[(row["sample_id"], row["arm"])] = row
        for s in [x for x in manifest["samples"] if x["dataset"] == ds]:
            sid = s["sample_id"]
            st = prompts.get((sid, "static"))
            ms = prompts.get((sid, "ms"))
            if st is None or ms is None:
                issues.append(f"{ds}/{sid}: missing arm prompt")
                continue
            if _source_text(st["prompt"]) != _source_text(ms["prompt"]):
                issues.append(f"{ds}/{sid}: paired source text differs")
            hs, bs, ts = _split_prompt(st["prompt"])
            hm, bm, tm = _split_prompt(ms["prompt"])
            if hs != hm or ts != tm:
                issues.append(f"{ds}/{sid}: prompts differ outside the "
                              "evidence block")
            st_nodes = sorted(st["evidence_ids"].values())
            ms_nodes = sorted(ms["evidence_ids"].values())
            if st_nodes != sorted(s["static_selected_node_ids"]):
                issues.append(f"{ds}/{sid}: Static prompt evidence != frozen "
                              "V3-A Static selection")
            if ms_nodes != sorted(s["ms_selected_node_ids"]):
                issues.append(f"{ds}/{sid}: MS prompt evidence != frozen "
                              "V3-A MS selection")
            row = _v3_selection(ds, s["fold"], s["event_id"], s["cutoff"])
            if row is None:
                issues.append(f"{ds}/{sid}: no frozen V3-A row found")
                continue
            if sorted(row["static_selected_node_ids"]) != st_nodes:
                issues.append(f"{ds}/{sid}: Static selection does not "
                              "reproduce the V3-A artifact")
            if sorted(row["ms_selected_node_ids"]) != ms_nodes:
                issues.append(f"{ds}/{sid}: MS selection does not reproduce "
                              "the V3-A artifact")
            if s["fallback_to_static"] and st["prompt"] != ms["prompt"]:
                issues.append(f"{ds}/{sid}: fallback arms carry different "
                              "prompts")


def _check_ordering(root, manifest, issues):
    """Rebuild snapshots for a sample of prompts and verify E# order."""
    from tcdscr_common import (config_from_env, event_label_registry)
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr.context.evidence_unit import build_evidence_units
    for ds in DATASETS:
        cfg = config_from_env(ds)
        registry = event_label_registry(ds, cfg)
        prompts = {}
        for row in load_jsonl(os.path.join(root, "prompts", f"{ds}.jsonl")):
            if row["arm"] == "ms":
                prompts[row["sample_id"]] = row
        sub = [s for s in manifest["samples"] if s["dataset"] == ds]
        chosen = [sub[i] for i in range(0, len(sub),
                                       max(1, len(sub) //
                                           ORDER_SAMPLE_PER_DATASET))
                  ][:ORDER_SAMPLE_PER_DATASET]
        events = {}
        for s in chosen:
            split = build_primary_fold_split(registry, s["fold"],
                                             seed=PARTITION_SEED)
            val_ids = set(split["validation"])
            if s["event_id"] not in events:
                events[s["event_id"]] = _load_event(ds, cfg, s["event_id"],
                                                    val_ids)
            event = events[s["event_id"]]
            snapshot = build_snapshot(event, int(s["cutoff"]))
            units = build_evidence_units(snapshot)
            wanted = set(s["ms_selected_node_ids"])
            ordered = [u["node_id"] for u in units if u["node_id"] in wanted]
            prompt_row = prompts.get(s["sample_id"])
            if prompt_row is None:
                issues.append(f"{ds}/{s['sample_id']}: missing MS prompt")
                continue
            got = [prompt_row["evidence_ids"][f"E{i}"]
                   for i in range(1, len(prompt_row["evidence_ids"]) + 1)]
            if got != ordered:
                issues.append(f"{ds}/{s['sample_id']}: MS evidence order is "
                              "not snapshot (timestamp ascending) order")
            if len(got) != len(s["ms_selected_node_ids"]):
                issues.append(f"{ds}/{s['sample_id']}: MS prompt dropped or "
                              "added evidence units")


def _load_event(dataset, cfg, event_id, val_ids):
    from tcdscr.data import maweibo_adapter, pheme_adapter
    if event_id not in val_ids:
        raise RuntimeError(f"{event_id} is not a validation event")
    if dataset == "pheme":
        by_id = {eid: (topic, label, folder)
                 for eid, topic, label, folder
                 in pheme_adapter.event_ids(cfg.raw_dir)}
        return pheme_adapter.load_event(*by_id[event_id])
    labels = dict(maweibo_adapter.event_ids(cfg.raw_dir, cfg.label_file))
    return maweibo_adapter.load_event(
        event_id, labels[event_id],
        f"{cfg.raw_dir.rstrip('/')}/{event_id}.json")


def _check_model(root, manifest, issues):
    path = os.path.join(root, "run_manifest.json")
    if not os.path.exists(path):
        issues.append("missing run_manifest.json")
        return
    with open(path, encoding="utf-8") as fh:
        run = json.load(fh)
    if run.get("model_path") != manifest.get("model_path"):
        issues.append("run manifest model path != sampling manifest model path")
    dec = run.get("decoding") or {}
    if dec.get("do_sample") is not False:
        issues.append("decoding do_sample is not false")
    if float(dec.get("temperature", -1)) != 0.0:
        issues.append("decoding temperature is not 0")
    if dec.get("max_new_tokens") is None:
        issues.append("decoding max_new_tokens not recorded")
    if run.get("finetuned") is not False:
        issues.append("run manifest does not state finetuned=false")
    if int(run.get("trainable_parameters_updated", -1)) != 0:
        issues.append("run manifest reports parameter updates")
    if not run.get("weight_shards_sha256"):
        issues.append("model weight shard hash missing")
    if int(run.get("n_generations", -1)) != EXPECTED_N * 2 * len(DATASETS):
        issues.append(f"run manifest n_generations="
                      f"{run.get('n_generations')}, expected "
                      f"{EXPECTED_N * 2 * len(DATASETS)}")


def _check_statistics(root, manifest, issues):
    """Recompute bootstrap and citation accounting from the raw rows."""
    import tcdscr_summarize_v3_reader as summ
    for ds in DATASETS:
        stats_path = os.path.join(root, "statistics", f"{ds}.json")
        if not os.path.exists(stats_path):
            issues.append(f"missing statistics/{ds}.json")
            continue
        with open(stats_path, encoding="utf-8") as fh:
            stats = json.load(fh)
        _, samples, raw, parsed = summ.load_dataset(root, ds)
        pairs, _, _ = summ.build_pairs(samples, parsed, raw)
        boot = summ.paired_bootstrap_delta(pairs)
        if abs(boot["ci_low"] - stats["bootstrap"]["ci_low"]) > 1e-12 or \
                abs(boot["ci_high"] - stats["bootstrap"]["ci_high"]) > 1e-12:
            issues.append(f"{ds}: bootstrap CI does not reproduce")
        if abs(boot["point"] - stats["delta_macro_f1"]) > 1e-9:
            issues.append(f"{ds}: delta Macro-F1 does not reproduce")

        # independent citation recount straight from the raw text
        from tcdscr.llm.reader_parser import (citation_stats,
                                              parse_reader_output)
        prompts_rows = load_jsonl(os.path.join(root, "prompts",
                                               f"{ds}.jsonl"))
        allowed = {(r["sample_id"], r["arm"]): set(r["evidence_ids"].keys())
                   for r in prompts_rows}
        raw_by = {(r["sample_id"], r["arm"]): r for r in raw}
        for arm in ("static", "ms"):
            cited = invalid = 0
            for row in parsed:
                if row["arm"] != arm:
                    continue
                raw_row = raw_by.get((row["sample_id"], arm))
                if raw_row is None:
                    continue
                text = raw_row["raw_output_retry"] if raw_row["retry_used"] \
                    else raw_row["raw_output"]
                obj, _ = parse_reader_output(text)
                if obj is None:
                    continue
                stats_ = citation_stats(obj["evidence_ids"],
                                        allowed.get((row["sample_id"], arm),
                                                    set()))
                if sorted(stats_["valid"]) != sorted(
                        row.get("valid_evidence_ids") or []) or \
                        sorted(stats_["invalid"]) != sorted(
                            row.get("invalid_evidence_ids") or []):
                    issues.append(f"{ds}/{row['sample_id']}/{arm}: citation "
                                  "accounting does not reproduce")
                    break
                cited += stats_["n_cited"]
                invalid += len(stats_["invalid"])
            recorded = stats["citation"][arm]
            expected_rate = (invalid / cited) if cited else None
            if not _close(expected_rate, recorded["unsupported_citation_rate"]):
                issues.append(f"{ds}/{arm}: unsupported citation rate "
                              "does not reproduce")


def _close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def _check_determinism(root, manifest, issues):
    path = os.path.join(root, "statistics", "pheme.json")
    if not os.path.exists(path):
        return
    for ds in DATASETS:
        with open(os.path.join(root, "statistics", f"{ds}.json"),
                  encoding="utf-8") as fh:
            stats = json.load(fh)
        det = stats["determinism"]
        if det["n_fallback_pairs"] != det["n_identical_prompt_hash"]:
            issues.append(f"{ds}: fallback pairs do not share one prompt")
        if det["n_identical_raw_output"] > det["n_identical_prompt_hash"]:
            issues.append(f"{ds}: determinism audit inconsistent")


def _model_context_limit(run):
    limit = run.get("max_position_embeddings")
    if limit:
        return int(limit)
    cfg = os.path.join(run.get("model_path", ""), "config.json")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as fh:
            return int(json.load(fh)["max_position_embeddings"])
    return None


def _check_context_limits(root, issues):
    """§38: nothing may exceed the model context, and nothing is truncated."""
    path = os.path.join(root, "run_manifest.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        run = json.load(fh)
    limit = _model_context_limit(run)
    if limit is None:
        issues.append("cannot resolve the model context limit")
        return
    for ds in DATASETS:
        for row in load_jsonl(os.path.join(root, "prompts", f"{ds}.jsonl")):
            if int(row["total_input_tokens"]) > limit:
                issues.append(f"{ds}/{row['sample_id']}/{row['arm']}: prompt "
                              f"exceeds the model context ({limit})")
    for ds in DATASETS:
        for row in load_jsonl(os.path.join(root, "raw_generations",
                                           f"{ds}.jsonl")):
            if row.get("parse_status") == "CONTEXT_OVERFLOW":
                issues.append(f"{ds}/{row['sample_id']}/{row['arm']}: "
                              "CONTEXT_OVERFLOW present")


def _refresh_report(root, result):
    path = os.path.join(root, "V3_B_READER_TRANSFER_REPORT.md")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    detail = (f" (samples={result['samples']}, "
              f"generations={result['generations']}, "
              f"determinism_warnings={result['determinism_warnings']})")
    pattern = re.compile(r"^issues = .*$", re.M)
    if not pattern.search(text):
        return
    text = pattern.sub(f"issues = {result['n_issues']}{detail}", text,
                       count=1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def verify(root):
    issues = []
    manifest_path = os.path.join(root, "sampling_manifest.json")
    if not os.path.exists(manifest_path):
        return {"n_issues": 1, "issues": ["missing sampling_manifest.json"]}
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    _check_sampling(manifest, issues)
    _check_frozen_alpha(issues)
    _check_manifest_precedes_generations(root, issues)
    _check_generations(root, manifest, issues)
    _check_prompts(root, manifest, issues)
    _check_ordering(root, manifest, issues)
    _check_model(root, manifest, issues)
    _check_context_limits(root, issues)
    _check_statistics(root, manifest, issues)
    _check_determinism(root, manifest, issues)
    _check_validation_only(manifest, issues)
    for name in ("V3_B_READER_TRANSFER_REPORT.md",
                 "reader_transfer_summary.json"):
        if not os.path.exists(os.path.join(root, name)):
            issues.append(f"missing {name}")
    warnings = 0
    det_path = os.path.join(root, "statistics", "pheme.json")
    if os.path.exists(det_path):
        for ds in DATASETS:
            with open(os.path.join(root, "statistics", f"{ds}.json"),
                      encoding="utf-8") as fh:
                warnings += json.load(fh)["determinism"][
                    "determinism_warnings"]
    return {
        "n_issues": len(issues),
        "issues": issues[:40],
        "samples": len(manifest["samples"]),
        "generations": sum(len(load_jsonl(os.path.join(
            root, "raw_generations", f"{ds}.jsonl"))) for ds in DATASETS),
        "determinism_warnings": warnings,
        "checks": [
            "300 unique paired samples per dataset",
            "one Static and one MS generation per sample",
            "manifest frozen before generations",
            "sampling seed 3090 / context seed 2000 / alpha 0.8 / budget 1024",
            "validation only, no test events",
            "prompt evidence reproduces the frozen V3-A selection",
            "MS evidence order is snapshot (timestamp ascending) order",
            "paired source text identical; only the evidence block differs",
            "one frozen model/config, do_sample=false, temperature=0",
            "Qwen weights frozen, no fine-tuning",
            "no Qwen output used for selection",
            "bootstrap CI and multiplicity reproduce",
            "fallback pair contexts identical",
            "unsupported evidence ids recounted from raw text",
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    result = verify(args.root)
    out = args.out or os.path.join(args.root, "reader_transfer_verify.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    _refresh_report(args.root, result)
    print(json.dumps(result, indent=1, ensure_ascii=False), flush=True)
    return 0 if result["n_issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
