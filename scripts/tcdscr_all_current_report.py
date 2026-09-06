#!/usr/bin/env python
"""Generate results/tcdscr/code_delta_fix/all_current_context_report.json
(final patch §5): chat-template token accounting evidence for All-current.

Runs on the server workspace where the frozen Qwen3-8B tokenizer/config
exist. No model weights are loaded, no generation happens.
"""
import json
import os
import sys

from tcdscr_common import PROJECT_DIR

QWEN_PATH = "/data/jyz/next/model/qwen3-8b"
OUT = "/data/jyz/next/llm/results/tcdscr/code_delta_fix"


def main():
    from transformers import AutoConfig, AutoTokenizer

    sys.path.insert(0, str(PROJECT_DIR))
    from tcdscr.context.evidence_unit import build_evidence_units, render_evidence
    from tcdscr.context.packer import pack_context
    from tcdscr.context.token_budget import EvidenceBudgetSelector
    from tcdscr.data.snapshot_builder import build_snapshot
    from tcdscr.evaluation.baselines import resolve_context_length, select_all_current
    from tcdscr.llm.qwen_wrapper import (GENERATION_CONFIG, count_chat_input_tokens,
                                         format_chat_prompt)
    from tcdscr.tests.conftest import make_event

    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH,
                                              trust_remote_code=False)
    config = AutoConfig.from_pretrained(QWEN_PATH)
    context_length = resolve_context_length(config, tokenizer)
    max_new_tokens = GENERATION_CONFIG["max_new_tokens"]

    mismatch_failures = 0
    probe_prompts = [
        "SOURCE CLAIM\nclaim\n\nTASK\nReturn exactly one label.",
        pack_context("源帖内容", build_snapshot(make_event([
            ("n0", None, 1000, "源帖内容", 0),
            ("n1", "n0", 1100, "回复", 1)]), 60), [])["prompt"],
    ]
    for p in probe_prompts:
        formatted = format_chat_prompt(tokenizer, p)
        n = count_chat_input_tokens(tokenizer, p)
        if n != len(tokenizer(formatted, add_special_tokens=True)["input_ids"]):
            mismatch_failures += 1
        if formatted.count("<|im_start|>user") != 1:
            mismatch_failures += 1
        # thinking disabled = the template pre-fills an EMPTY think block
        assistant_part = formatted.split("<|im_start|>assistant")[1]
        if assistant_part.strip() != "<think>\n\n</think>":
            mismatch_failures += 1

    # synthetic near-limit stress: 120 whole ~800-token pairs vs 40960
    filler = "词语" * 400
    nodes = [("n0", None, 1000, "源帖内容", 0)]
    for i in range(1, 121):
        nodes.append((f"n{i}", "n0", 1000 + i, filler, i))
    snap = build_snapshot(make_event(nodes), 360)
    units = build_evidence_units(snap)
    budget = EvidenceBudgetSelector(tokenizer, budget=2048)
    result = select_all_current(
        snap, units, budget, context_length=context_length,
        max_new_tokens=max_new_tokens,
        prompt_builder=lambda s, acc: pack_context("源帖内容", s, acc)["prompt"],
        chat_counter=lambda p: count_chat_input_tokens(tokenizer, p))

    overflow_failures = 0
    if result["input_tokens"] + max_new_tokens > context_length:
        overflow_failures += 1
    partial_pair_failures = 0
    packed = pack_context("源帖内容", snap, result["selected_units"])["prompt"]
    for i, u in enumerate(result["selected_units"]):
        if render_evidence(u, i + 1) not in packed:
            partial_pair_failures += 1

    report = {
        "model_context_length": context_length,
        "max_new_tokens": max_new_tokens,
        "token_counting_mode": "qwen_chat_template",
        "counter_implementation": (
            "count_chat_input_tokens: apply_chat_template(single user "
            "message, add_generation_prompt=True, enable_thinking=False) "
            "then tokenize — shared with QwenRumorLLM._chat_text"),
        "tests": {
            "overflow_failures": overflow_failures,
            "partial_pair_failures": partial_pair_failures,
            "chat_token_mismatch_failures": mismatch_failures,
        },
        "synthetic_near_limit": {
            "input_tokens": result["input_tokens"],
            "evidence_tokens": result["evidence_tokens"],
            "truncated": result["truncated"],
            "units_before_truncation": result["units_before_truncation"],
            "units_after_truncation": result["units_after_truncation"],
            "tokens_dropped": result["tokens_dropped"],
            "input_plus_reserve": result["input_tokens"] + max_new_tokens,
        },
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "all_current_context_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
