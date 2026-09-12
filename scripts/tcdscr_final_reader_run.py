#!/usr/bin/env python
"""Final Evidence Closure — Gap B + Gap C reader inference.

Reads the frozen reader prompts and runs the frozen Qwen3-8B reader with the
V3-B decoding configuration (temperature 0, do_sample false, same
max_new_tokens).  Gold labels are never read here: only raw generations and
parsed outputs are written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "project")
for _path in (HERE, PROJECT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DATASETS = ("pheme", "maweibo")
ARMS = ("STATIC_FULL", "UTILITY_TOKEN_MATCHED", "RANDOM_TOKEN_MATCHED",
        "MS_TSR")
OUT_ROOT = "/data/jyz/next/llm/results/tcdscr/final_evidence/reader"
V3B_READER_ROOT = "/data/jyz/next/llm/results/tcdscr/dynamic_v3_reader"


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--v3b-reader-root", default=V3B_READER_ROOT)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: only the first N samples per dataset")
    return ap


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def model_manifest_matches(v3b_root, current):
    path = os.path.join(v3b_root, "run_manifest.json")
    if not os.path.exists(path):
        return None, "V3-B run manifest not found"
    with open(path, encoding="utf-8") as fh:
        prev = json.load(fh)
    prev_model = prev.get("model") or {}
    keys = ("weight_shards_sha256", "config_sha256", "index_sha256")
    for key in keys:
        if prev_model.get(key) and current.get(key) and \
                prev_model[key] != current[key]:
            return False, f"MODEL_MISMATCH on {key}"
    return True, None


def main(argv=None):
    args = build_parser().parse_args(argv)
    import torch

    from tcdscr.llm.reader_parser import parse_reader_output
    from tcdscr.llm.reader_prompt import RETRY_SUFFIX
    from tcdscr_run_v3_reader import (Reader, build_model_manifest,
                                      sha256_file, sha256_text,
                                      token_limit_exceeded)

    datasets = args.datasets.split(",")
    manifest_path = os.path.join(args.out_root,
                                "reader_sampling_manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert sha256_file(manifest_path) == open(os.path.join(
        args.out_root, "reader_sampling_manifest.sha256"),
        encoding="utf-8").read().strip(), "sampling manifest mutated"

    for sub in ("raw_generations", "parsed", "diagnostics"):
        os.makedirs(os.path.join(args.out_root, sub), exist_ok=True)

    model_path = args.model_path
    if model_path is None:
        from tcdscr.config.schema import config_from_env
        model_path = config_from_env(datasets[0]).qwen_model_path
    reader = Reader(model_path, device="cuda:0" if torch.cuda.is_available()
                    else "cpu")
    model = build_model_manifest(model_path, reader.tokenizer, reader.model)
    ok, why = model_manifest_matches(args.v3b_reader_root, model)
    if ok is False:
        raise SystemExit(f"{why} — STOP")
    with open(os.path.join(args.out_root, "diagnostics",
                           "reader_model_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"model": model, "matches_v3b": ok, "mismatch": why},
                  fh, indent=1)

    by_sample = {r["sample_id"]: r for r in manifest["samples"]}
    total = 0
    for dataset in datasets:
        prompts = read_jsonl(os.path.join(args.out_root, "prompts",
                                          f"{dataset}.jsonl"))
        if args.limit:
            keep = {p["sample_id"] for p in prompts[:0]}
            seen = []
            for p in prompts:
                if p["sample_id"] not in keep and len(seen) < args.limit:
                    keep.add(p["sample_id"])
                    seen.append(p["sample_id"])
            prompts = [p for p in prompts if p["sample_id"] in keep]
        done = {(r["sample_id"], r["arm"]) for r in read_jsonl(
            os.path.join(args.out_root, "raw_generations",
                         f"{dataset}.jsonl"))}
        raw_fh = open(os.path.join(args.out_root, "raw_generations",
                                   f"{dataset}.jsonl"), "a",
                      encoding="utf-8")
        parsed_fh = open(os.path.join(args.out_root, "parsed",
                                      f"{dataset}.jsonl"), "a",
                         encoding="utf-8")
        groups = {}
        for p in prompts:
            groups.setdefault(p["sample_id"], {})[p["arm"]] = p
        ordered_prompts = []
        for row in manifest["samples"]:
            if row["dataset"] != dataset:
                continue
            sid = row["sample_id"]
            g = groups.get(sid) or {}
            for arm in row["arm_order"]:
                if arm in g:
                    ordered_prompts.append(g[arm])
        try:
            for p in ordered_prompts:
                key = (p["sample_id"], p["arm"])
                if key in done:
                    continue
                t0 = time.time()
                raw = reader.generate(p["prompt"])
                latency = time.time() - t0
                parsed, errors = parse_reader_output(raw)
                retry_used = False
                if parsed is None:
                    raw2 = reader.generate(p["prompt"] + "\n\n" +
                                           RETRY_SUFFIX)
                    latency = time.time() - t0
                    parsed2, errors2 = parse_reader_output(raw2)
                    if parsed2 is not None:
                        raw, parsed, errors = raw2, parsed2, errors2
                    retry_used = True
                total += 1
                generated_tokens = len(reader.tokenizer(
                    raw, add_special_tokens=False)["input_ids"])
                raw_fh.write(json.dumps({
                    "sample_id": p["sample_id"], "arm": p["arm"],
                    "dataset": p["dataset"], "fold": p["fold"],
                    "event_id": p["event_id"], "cutoff": p["cutoff"],
                    "prompt_sha256": sha256_text(p["prompt"]),
                    "social_tokens": p["social_tokens"],
                    "total_input_tokens": p["total_input_tokens"],
                    "generated_tokens": generated_tokens,
                    "context_overflow": token_limit_exceeded(
                        p["total_input_tokens"],
                        getattr(reader.model.config,
                                "max_position_embeddings", None)),
                    "raw": raw, "latency_sec": latency,
                    "retry_used": retry_used,
                    "parse_ok": parsed is not None,
                    "parse_errors": errors,
                }) + "\n")
                raw_fh.flush()
                if parsed is not None:
                    parsed_fh.write(json.dumps({
                        "sample_id": p["sample_id"], "arm": p["arm"],
                        "dataset": p["dataset"], "fold": p["fold"],
                        "event_id": p["event_id"], "cutoff": p["cutoff"],
                        "label": parsed["label"],
                        "confidence": parsed["confidence"],
                        "evidence_ids": parsed["evidence_ids"],
                        "reason": parsed["reason"],
                        "supplied_evidence_ids": sorted(
                            p["evidence_ids"].keys()),
                    }) + "\n")
                    parsed_fh.flush()
                if total % 50 == 0:
                    print(json.dumps({"generated": total,
                                      "dataset": dataset}), flush=True)
        finally:
            raw_fh.close()
            parsed_fh.close()
    print(json.dumps({"generated_total": total}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
