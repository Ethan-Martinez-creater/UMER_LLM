#!/usr/bin/env python
"""V3-B STEP 5-8: deterministic paired Static/MS Qwen3-8B reader inference.

Reads the frozen sampling manifest and prompt file, runs both arms of every
paired sample through one frozen Qwen3-8B reader (greedy, no sampling), retries
at most once when the JSON schema fails, and writes:

  raw_generations/<dataset>.jsonl   §47
  parsed/<dataset>.jsonl            §48
  run_manifest.json                 model / decoding provenance

The reader never influences selection (§40): prompts are read verbatim from the
frozen file and the only difference between the two arms is the social evidence
block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tcdscr_common import PROJECT_DIR  # noqa: E402,F401

from tcdscr.llm.reader_parser import (citation_stats,  # noqa: E402
                                      parse_reader_output,
                                      reason_invalid_refs)
from tcdscr.llm.reader_prompt import (MAX_NEW_TOKENS, PROMPT_VERSION,  # noqa
                                      RETRY_SUFFIX,
                                      count_chat_tokens,
                                      count_tokens)

DATASETS = ("pheme", "maweibo")
GOLD_TO_LABEL = {1: "RUMOR", 0: "NON_RUMOR"}
DEFAULT_OUT = "/data/jyz/next/llm/results/tcdscr/dynamic_v3_reader"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_model_manifest(model_path, tokenizer, model):
    """Model provenance for §10/§49 (weights are never modified)."""
    import torch
    import transformers
    index = os.path.join(model_path, "model.safetensors.index.json")
    cfg = os.path.join(model_path, "config.json")
    shards = []
    if os.path.exists(index):
        with open(index, encoding="utf-8") as fh:
            weight_map = json.load(fh).get("weight_map", {})
        for name in sorted(set(weight_map.values())):
            p = os.path.join(model_path, name)
            shards.append({"file": name, "bytes": os.path.getsize(p),
                           "sha256": sha256_file(p)})
    return {
        "model_path": model_path,
        "prompt_version": PROMPT_VERSION,
        "config_sha256": sha256_file(cfg) if os.path.exists(cfg) else None,
        "index_sha256": sha256_file(index) if os.path.exists(index) else None,
        "shards": shards,
        "weight_shards_sha256": sha256_text(
            json.dumps(shards, sort_keys=True)),
        "model_class": type(model).__name__,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "decoding": {"do_sample": False, "num_beams": 1,
                     "max_new_tokens": MAX_NEW_TOKENS,
                     "temperature": 0.0, "top_p": None, "top_k": None,
                     "temperature_note":
                         "greedy decoding; do_sample=false makes temperature "
                         "inactive, recorded as 0 for the frozen protocol"},
        "chat_template_sha256": sha256_text(
            str(getattr(tokenizer, "chat_template", ""))),
        "max_position_embeddings": getattr(
            getattr(model, "config", None), "max_position_embeddings", None),
        "trainable_parameters_updated": 0,
        "finetuned": False,
    }


class Reader:
    """Thin frozen-reader wrapper; the only generation path used by V3-B."""

    def __init__(self, model_path, device="cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tcdscr.llm.reader_prompt import format_reader_chat
        self.torch = torch
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device,
            trust_remote_code=False)
        self.model.eval()
        self._format = format_reader_chat

    @property
    def device(self):
        return self.model.device

    def generate(self, user_prompt):
        text = self._format(self.tokenizer, user_prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    done.add((row["sample_id"], row["arm"]))
    return done


def load_prompts(prompts_dir, dataset):
    out = {}
    path = os.path.join(prompts_dir, f"{dataset}.jsonl")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[(row["sample_id"], row["arm"])] = row
    return out


def inference_order(arm_order):
    return ("static", "ms") if arm_order == "static_first" else ("ms",
                                                                 "static")


def token_limit_exceeded(total_tokens, max_position_embeddings):
    """§38: a prompt over the model context is flagged, never truncated."""
    if max_position_embeddings is None:
        return False
    return int(total_tokens) > int(max_position_embeddings)


def run_arm(reader, prompt_row, gold):
    """One generation with at most one deterministic formatting retry (§18)."""
    prompt = prompt_row["prompt"]
    t0 = time.time()
    raw = reader.generate(prompt)
    latency = time.time() - t0
    parsed, errors = parse_reader_output(raw)
    retry_used = False
    raw_retry = None
    if parsed is None:
        retry_used = True
        retry_prompt = prompt + "\n\n" + RETRY_SUFFIX
        t1 = time.time()
        raw_retry = reader.generate(retry_prompt)
        latency += time.time() - t1
        parsed, errors = parse_reader_output(raw_retry)
    status = "ok" if parsed is not None and not retry_used else (
        "retry_ok" if parsed is not None else "PARSE_FAILURE")
    return {"raw_output": raw, "raw_output_retry": raw_retry,
            "retry_used": retry_used,
            "retry_prompt_hash": sha256_text(prompt + "\n\n" + RETRY_SUFFIX)
            if retry_used else None,
            "parsed": parsed, "errors": errors, "parse_status": status,
            "latency_s": round(latency, 3)}


def parsed_row(manifest_row, arm, prompt_row, result):
    parsed = result["parsed"]
    allowed = list(prompt_row["evidence_ids"].keys())
    row = {
        "sample_id": manifest_row["sample_id"],
        "dataset": manifest_row["dataset"],
        "event_id": manifest_row["event_id"],
        "cutoff": manifest_row["cutoff"],
        "fold": manifest_row["fold"],
        "arm": arm,
        "gold": manifest_row["gold"],
        "gold_label": GOLD_TO_LABEL[manifest_row["gold"]],
        "parse_failure": parsed is None,
        "retry_used": result["retry_used"],
        "reason_invalid_evidence_reference_rate": None,
    }
    if parsed is None:
        row.update({"parsed_label": None, "correct": None, "confidence": None,
                    "evidence_ids": [], "valid_evidence_ids": [],
                    "invalid_evidence_ids": [], "reason": None,
                    "n_cited": 0, "reason_invalid_refs": []})
        return row
    stats = citation_stats(parsed["evidence_ids"], allowed)
    invalid_refs = reason_invalid_refs(parsed["reason"], allowed)
    row.update({
        "parsed_label": parsed["label"],
        "correct": int(parsed["label"] == row["gold_label"]),
        "confidence": parsed["confidence"],
        "evidence_ids": parsed["evidence_ids"],
        "valid_evidence_ids": stats["valid"],
        "invalid_evidence_ids": stats["invalid"],
        "n_cited": stats["n_cited"],
        "reason": parsed["reason"],
        "reason_invalid_refs": invalid_refs,
    })
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke runs: first N samples per dataset")
    ap.add_argument("--dataset", default=None, choices=("pheme", "maweibo"))
    args = ap.parse_args(argv)

    root = args.out
    prompts_dir = os.path.join(root, "prompts")
    with open(os.path.join(root, "sampling_manifest.json"),
              encoding="utf-8") as fh:
        manifest = json.load(fh)
    model_path = args.model_path or manifest["model_path"]
    os.makedirs(os.path.join(root, "raw_generations"), exist_ok=True)
    os.makedirs(os.path.join(root, "parsed"), exist_ok=True)

    datasets = (args.dataset,) if args.dataset else DATASETS
    samples = [s for s in manifest["samples"] if s["dataset"] in datasets]
    if args.limit is not None:
        kept = []
        for ds in datasets:
            kept.extend([s for s in samples if s["dataset"] == ds][:args.limit])
        samples = kept

    reader = Reader(model_path, device=args.device)
    model_manifest = build_model_manifest(model_path, reader.tokenizer,
                                          reader.model)
    model_manifest["n_planned_samples"] = len(samples)
    model_manifest["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime())

    processed = 0
    for ds in datasets:
        ds_samples = [s for s in samples if s["dataset"] == ds]
        if not ds_samples:
            continue
        prompts = load_prompts(prompts_dir, ds)
        raw_path = os.path.join(root, "raw_generations", f"{ds}.jsonl")
        parsed_path = os.path.join(root, "parsed", f"{ds}.jsonl")
        done = load_done(raw_path)
        with open(raw_path, "a", encoding="utf-8") as raw_fh, \
                open(parsed_path, "a", encoding="utf-8") as par_fh:
            for s in ds_samples:
                sid = s["sample_id"]
                arms = inference_order(s["arm_order"])
                for arm in arms:
                    if (sid, arm) in done:
                        continue
                    prompt_row = prompts[(sid, arm)]
                    expected_social = prompt_row["social_tokens"]
                    expected_total = prompt_row["total_input_tokens"]
                    social_now = (count_tokens(reader.tokenizer,
                                               _social_block(prompt_row))
                                  if prompt_row["n_evidence"] else 0)
                    total_now = count_chat_tokens(reader.tokenizer,
                                                  prompt_row["prompt"])
                    if social_now != expected_social \
                            or total_now != expected_total:
                        raise RuntimeError(
                            f"{sid}/{arm}: frozen prompt token accounting "
                            "does not reproduce")
                    max_len = model_manifest.get("max_position_embeddings")
                    if token_limit_exceeded(total_now, max_len):
                        # §38: never truncate; flag and stop for this sample.
                        result = {"raw_output": None,
                                  "raw_output_retry": None,
                                  "retry_used": False,
                                  "retry_prompt_hash": None,
                                  "parsed": None,
                                  "errors": ["CONTEXT_OVERFLOW"],
                                  "parse_status": "CONTEXT_OVERFLOW",
                                  "latency_s": 0.0}
                    else:
                        result = run_arm(reader, prompt_row, s["gold"])
                    raw_fh.write(json.dumps({
                        "sample_id": sid, "dataset": ds,
                        "event_id": s["event_id"], "cutoff": s["cutoff"],
                        "fold": s["fold"], "arm": arm,
                        "pressure_bin": s["pressure_bin"],
                        "fallback_to_static": s["fallback_to_static"],
                        "prompt_hash": sha256_text(prompt_row["prompt"]),
                        "retry_prompt_hash": result["retry_prompt_hash"],
                        "input_token_count": total_now,
                        "social_context_token_count": social_now,
                        "n_evidence": prompt_row["n_evidence"],
                        "raw_output": result["raw_output"],
                        "raw_output_retry": result["raw_output_retry"],
                        "parse_status": result["parse_status"],
                        "retry_used": result["retry_used"],
                        "parse_errors": result["errors"],
                        "latency_s": result["latency_s"],
                    }) + "\n")
                    raw_fh.flush()
                    row = parsed_row(s, arm, prompt_row, result)
                    par_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    par_fh.flush()
                    processed += 1
                    print(json.dumps({"sample_id": sid, "arm": arm,
                                      "status": row["parse_failure"] and
                                      "PARSE_FAILURE" or "ok",
                                      "n": processed}), flush=True)

    model_manifest["n_generations"] = processed
    model_manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime())
    with open(os.path.join(root, "run_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(model_manifest, fh, indent=1, ensure_ascii=False)
    print(json.dumps({"generations": processed,
                      "model_path": model_path}, indent=1))
    return 0


def _social_block(prompt_row):
    """The evidence block as frozen in the prompt file: everything after the
    ``Observed Social Evidence up to <cutoff>:`` header line and before the
    blank line preceding ``Task:``."""
    prompt = prompt_row["prompt"]
    header = "Observed Social Evidence up to "
    start = prompt.index("\n", prompt.index(header)) + 1
    end = prompt.index("\n\nTask:")
    return prompt[start:end]


if __name__ == "__main__":
    sys.exit(main())
