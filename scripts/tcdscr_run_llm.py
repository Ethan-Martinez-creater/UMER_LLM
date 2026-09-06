#!/usr/bin/env python
"""STEP 12 entry point: batch frozen-LLM inference over a prompt file.

Input JSONL rows must carry: prompt, dataset, event_id, cutoff, budget,
selector_checkpoint_hash. Output rows add: raw_output, label, cache_key.
The same §24 key is never inferred twice.

Usage: python tcdscr_run_llm.py --dataset pheme --input prompts.jsonl
       --output predictions.jsonl [--limit N]
"""
import argparse
import json

from tcdscr_common import PROJECT_DIR  # noqa: F401  (sys.path side effect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("pheme", "maweibo"))
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selector-checkpoint-hash", default="unfrozen_smoke")
    args = ap.parse_args()

    from tcdscr.config.schema import config_from_env
    from tcdscr.llm.cache import LLMResponseCache, build_cache_key
    from tcdscr.llm.parser import parse_label
    from tcdscr.llm.qwen_wrapper import GENERATION_CONFIG, QwenRumorLLM
    cfg = config_from_env(args.dataset)

    rows = []
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit is not None:
        rows = rows[:args.limit]

    cache = LLMResponseCache(f"{cfg.cache_dir}/llm_responses.jsonl")
    llm = QwenRumorLLM(cfg.qwen_model_path)

    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            key = build_cache_key(
                model_id=cfg.qwen_model_path, model_revision="local",
                prompt=row["prompt"],
                generation_config=GENERATION_CONFIG,
                dataset=row.get("dataset", args.dataset),
                event_id=row["event_id"], cutoff=row["cutoff"],
                selector_checkpoint_hash=row.get(
                    "selector_checkpoint_hash",
                    args.selector_checkpoint_hash),
                budget=row.get("budget"))
            raw = cache.get(key)
            cache_hit = raw is not None
            if raw is None:
                raw = llm.generate(row["prompt"])
                cache.put(key, raw)
            out.write(json.dumps({
                **row, "raw_output": raw,
                "label": parse_label(raw),
                "cache_key": key, "cache_hit": cache_hit}) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
