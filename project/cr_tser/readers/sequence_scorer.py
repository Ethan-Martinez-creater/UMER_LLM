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
def sequence_logprob(model, tokenizer, chat_text: str, candidate: str,
                     device=None) -> float:
    """Sum of log p(candidate_k | prompt, candidate_<k) for one candidate.

    Only the candidate tokens are scored; the prompt is teacher-forced context.
    A candidate is encoded without special tokens so the continuation is
    exactly the model's next tokens.
    """
    import torch

    device = device or next(model.parameters()).device
    prompt_ids = tokenizer(chat_text, return_tensors="pt")["input_ids"].to(device)
    cont = tokenizer(candidate, add_special_tokens=False)["input_ids"]
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
