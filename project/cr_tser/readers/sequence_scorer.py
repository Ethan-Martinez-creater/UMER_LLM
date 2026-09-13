"""Teacher-forced A/B sequence scoring (plan §9).

The reader's utility signal is the summed teacher-forced log probability of the
two candidate continuations, normalised into ``p_rumor`` / ``p_nonrumor``.
Generated confidence is deliberately not used (plan §9).

The math is pure and model-free so the verifier and the tests can check it
without loading a reader.
"""
from __future__ import annotations

import math

from ..config.pilot_config import CANDIDATES


def apply_chat(tokenizer, messages, thinking: bool = False) -> str:
    """Single chat formatter shared by all readers (thinking disabled)."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def normalize_ab(score_a: float, score_b: float) -> dict:
    """Stable softmax over the two candidate scores (plan §9)."""
    m = max(score_a, score_b)
    ea = math.exp(score_a - m)
    eb = math.exp(score_b - m)
    denom = ea + eb
    return {
        "p_rumor": ea / denom,
        "p_nonrumor": eb / denom,
        "prediction": "A" if score_a >= score_b else "B",
        "score_A": score_a,
        "score_B": score_b,
    }


def ab_scores(scores: dict) -> dict:
    """``{"A": logp, "B": logp}`` -> the full §9 score record."""
    for c in CANDIDATES:
        if c not in scores:
            raise ValueError(f"missing candidate {c!r} in {scores!r}")
    return normalize_ab(float(scores["A"]), float(scores["B"]))


def gold_probability(out: dict, gold_label: int) -> float:
    """``p_r(y|C)`` for gold 1=RUMOR(A) / 0=NON_RUMOR(B) (plan §10)."""
    if gold_label not in (0, 1):
        raise ValueError("gold label must be 0/1")
    return out["p_rumor"] if gold_label == 1 else out["p_nonrumor"]


def is_correct(out: dict, gold_label: int) -> bool:
    predicted = 1 if out["prediction"] == "A" else 0
    return predicted == gold_label


# --------------------------------------------------------------------------
# Teacher-forced scoring against a real causal LM
# --------------------------------------------------------------------------
def tokenize_prompt(tokenizer, chat_text: str):
    """Token ids of an already chat-formatted prompt.

    ``apply_chat_template(..., tokenize=False)`` returns text that already
    contains the template's special tokens; encoding it with the default
    ``add_special_tokens=True`` would prepend a second BOS/EOS. The single
    prompt tokenizer here always disables that so the ids match exactly what
    the model receives.
    """
    return tokenizer(chat_text, add_special_tokens=False)["input_ids"]


def tokenize_continuation(tokenizer, candidate: str):
    """Token ids of the bare continuation (never special-token wrapped)."""
    return tokenizer(candidate, add_special_tokens=False)["input_ids"]


def continuation_boundary(tokenizer, chat_text: str, candidate: str):
    """Verify the concatenation identity used by teacher forcing.

    Returns ``{"prompt_tokens", "candidate_ids", "joined_ids",
    "boundary_ok"}``. ``boundary_ok`` is True when
    ``tokenize(prompt + candidate)`` equals ``tokenize(prompt) +
    tokenize(candidate)``, i.e. the continuation starts exactly at the
    prompt boundary with no re-tokenization or extra special tokens.
    """
    prompt_ids = tokenize_prompt(tokenizer, chat_text)
    cand_ids = tokenize_continuation(tokenizer, candidate)
    joined = tokenizer(chat_text + candidate, add_special_tokens=False)[
        "input_ids"]
    return {
        "prompt_tokens": len(prompt_ids),
        "candidate_ids": list(cand_ids),
        "joined_tail": list(joined[len(prompt_ids):]),
        "joined_len": len(joined),
        "boundary_ok": list(joined) == list(prompt_ids) + list(cand_ids),
    }


def ab_token_report(tokenizer, user_prompt: str, candidates=CANDIDATES) -> dict:
    """Explicit A/B tokenization record for the P0 sanity artifact (§30).

    Records the prompt token count, each candidate's token ids and the
    boundary check, so the plan's requirement that scoring is teacher-forced
    (not generated text, not confidence) is auditable from the artifact.
    """
    from .base_reader import build_messages
    chat = apply_chat(tokenizer, build_messages(user_prompt))
    prompt_ids = tokenize_prompt(tokenizer, chat)
    report = {
        "prompt_tokens": len(prompt_ids),
        "prompt_token_ids_head": list(prompt_ids[:16]),
        "prompt_token_ids_tail": list(prompt_ids[-8:]),
        "candidates": {},
    }
    for candidate in candidates:
        info = continuation_boundary(tokenizer, chat, candidate)
        info["score_mode"] = "teacher_forced_logprob_sum"
        report["candidates"][candidate] = info
    report["all_boundaries_ok"] = all(
        c["boundary_ok"] for c in report["candidates"].values())
    return report


def sequence_logprob(model, tokenizer, chat_text: str, candidate: str,
                     device=None) -> float:
    """Sum of log p(candidate_k | prompt, candidate_<k) for one candidate.

    Only the candidate tokens are scored; the prompt is teacher-forced context.
    Both the prompt and the continuation are encoded with
    ``add_special_tokens=False`` because the chat template already carries the
    special tokens — this is what prevents a duplicated BOS/EOS.
    """
    import torch

    device = device or next(model.parameters()).device
    prompt_ids = torch.tensor([tokenize_prompt(tokenizer, chat_text)],
                              dtype=torch.long, device=device)
    cont = tokenize_continuation(tokenizer, candidate)
    if not cont:
        raise ValueError(f"candidate {candidate!r} tokenises to nothing")
    cont_ids = torch.tensor([cont], dtype=torch.long, device=device)
    full = torch.cat([prompt_ids, cont_ids], dim=1)
    with torch.no_grad():
        logits = model(full).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    prompt_len = prompt_ids.shape[1]
    total = 0.0
    for k, tok in enumerate(cont):
        pos = prompt_len + k - 1
        total += float(logprobs[0, pos, tok])
    return total


def candidate_logprobs_hf(model, tokenizer, user_prompt, candidates=CANDIDATES,
                          thinking: bool = False, device=None) -> dict:
    """Convenience wrapper: chat-format once, score every candidate."""
    from .base_reader import build_messages
    chat = apply_chat(tokenizer, build_messages(user_prompt), thinking=thinking)
    return {c: sequence_logprob(model, tokenizer, chat, c, device=device)
            for c in candidates}
