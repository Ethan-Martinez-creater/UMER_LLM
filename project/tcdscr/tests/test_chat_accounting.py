"""Unit tests: chat-template token accounting for All-current (final patch
§1–§4).

The Qwen tokenizer / config tests run only where the frozen model files
exist (the server workspace); the stub-based tests run anywhere.
"""
import os

import pytest

from ..context.evidence_unit import build_evidence_units, render_evidence
from ..context.packer import pack_context
from ..context.token_budget import EvidenceBudgetSelector
from ..data.snapshot_builder import build_snapshot
from ..evaluation.baselines import select_all_current
from .conftest import make_event

QWEN_PATH = "/data/jyz/next/model/qwen3-8b"
_HAS_QWEN = os.path.isdir(QWEN_PATH)


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _small_units_snapshot(n_units=3, words=8):
    nodes = [("n0", None, 1000, "source claim", 0)]
    for i in range(1, n_units + 1):
        text = " ".join(f"w{i}x{j}" for j in range(words))
        nodes.append((f"n{i}", "n0", 1000 + i, text, i))
    snap = build_snapshot(make_event(nodes), 60)
    return snap, build_evidence_units(snap)


def test_all_current_limit_uses_chat_formatted_tokens():
    """A unit that fits under a raw-prompt count must still be rejected when
    the chat template pushes the real input over the limit — proving the
    limit decision uses chat-formatted tokens, not raw ones."""
    snap, units = _small_units_snapshot(n_units=1, words=40)
    budget = EvidenceBudgetSelector(_Tok(), budget=2048)
    template_overhead = 50  # stub for the chat template wrapper tokens

    def counter(prompt):
        return len(prompt.split()) + template_overhead

    # raw counting would accept the unit (raw+8 = 102 <= 110), but the
    # chat-formatted count (raw + 50 template tokens) pushes it past the
    # same limit — the walk must therefore stop with zero units
    result = select_all_current(
        snap, units, budget, context_length=110, max_new_tokens=8,
        prompt_builder=lambda s, acc: pack_context(
            "source claim", s, acc)["prompt"],
        chat_counter=counter)
    assert result["units_after_truncation"] == 0
    assert result["truncated"] is True
    assert result["input_tokens"] + 8 <= 110

    # with enough head-room the same unit is accepted
    result2 = select_all_current(
        snap, units, budget, context_length=160, max_new_tokens=8,
        prompt_builder=lambda s, acc: pack_context(
            "source claim", s, acc)["prompt"],
        chat_counter=counter)
    assert result2["units_after_truncation"] == 1
    assert result2["input_tokens"] + 8 <= 160


def test_all_current_evidence_tokens_exclude_source_and_instruction():
    snap, units = _small_units_snapshot(n_units=2, words=5)
    budget = EvidenceBudgetSelector(_Tok(), budget=2048)
    result = select_all_current(
        snap, units, budget, context_length=40960, max_new_tokens=8,
        prompt_builder=lambda s, acc: pack_context(
            "a fairly long source claim text", s, acc)["prompt"],
        chat_counter=lambda p: len(p.split()))
    expected = sum(budget.count_tokens(render_evidence(u, i + 1))
                   for i, u in enumerate(result["selected_units"]))
    assert result["evidence_tokens"] == expected
    # source + snapshot metadata + instructions are NOT evidence
    assert result["evidence_tokens"] < result["input_tokens"]
    prompt = pack_context("a fairly long source claim text", snap,
                          result["selected_units"])["prompt"]
    non_evidence = len(prompt.split()) - expected
    assert non_evidence > 0


def test_all_current_near_limit_never_overflows():
    """Approach 40960 tokens with whole 500-token pairs: the walk must stop
    before overflowing, never split a pair, and keep input + 8 <= context."""
    if not _HAS_QWEN:
        pytest.skip("frozen Qwen tokenizer not available on this host")
    from transformers import AutoConfig, AutoTokenizer

    from ..evaluation.baselines import resolve_context_length
    from ..llm.qwen_wrapper import count_chat_input_tokens

    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH,
                                              trust_remote_code=False)
    config = AutoConfig.from_pretrained(QWEN_PATH)
    context_length = resolve_context_length(config, tokenizer)
    assert context_length == 40960

    filler = "词语" * 400  # ~800 CJK tokens per rendered pair
    nodes = [("n0", None, 1000, "源帖内容", 0)]
    for i in range(1, 121):
        nodes.append((f"n{i}", "n0", 1000 + i, filler, i))
    snap = build_snapshot(make_event(nodes), 360)
    units = build_evidence_units(snap)

    budget = EvidenceBudgetSelector(tokenizer, budget=2048)
    result = select_all_current(
        snap, units, budget, context_length=context_length,
        max_new_tokens=8,
        prompt_builder=lambda s, acc: pack_context("源帖内容", s, acc)["prompt"],
        chat_counter=lambda p: count_chat_input_tokens(tokenizer, p))
    assert result["input_tokens"] + 8 <= context_length
    assert result["truncated"] is True
    assert 0 < result["units_after_truncation"] < 120
    # partial pair = 0: accepted units render back exactly whole
    packed = pack_context("源帖内容", snap,
                          result["selected_units"])["prompt"]
    for i, u in enumerate(result["selected_units"]):
        assert render_evidence(u, i + 1) in packed


def test_chat_token_count_matches_qwen_wrapper():
    """count_chat_input_tokens must equal what QwenRumorLLM feeds the model:
    same template flags, single user message, generation prompt appended."""
    if not _HAS_QWEN:
        pytest.skip("frozen Qwen tokenizer not available on this host")
    from transformers import AutoTokenizer

    from ..llm.qwen_wrapper import format_chat_prompt
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH,
                                              trust_remote_code=False)

    prompt = "SOURCE CLAIM\nclaim text\n\nTASK\nReturn exactly one label."
    formatted = format_chat_prompt(tokenizer, prompt)
    # single user message, generation prompt appended; thinking disabled
    # shows up as an EMPTY <think></think> block pre-filled by the template
    assert formatted.count("<|im_start|>user") == 1
    assert "<|im_start|>assistant" in formatted
    assistant_part = formatted.split("<|im_start|>assistant")[1]
    assert assistant_part.strip() == "<think>\n\n</think>"

    from ..llm.qwen_wrapper import count_chat_input_tokens
    n = count_chat_input_tokens(tokenizer, prompt)
    assert n == len(tokenizer(formatted, add_special_tokens=True)["input_ids"])
    # determinism: repeated calls agree
    assert count_chat_input_tokens(tokenizer, prompt) == n
