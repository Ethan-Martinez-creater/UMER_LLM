"""Teacher-forced A/B sequence scoring (plan §9).

The reader's utility signal is the summed teacher-forced log probability of the
two candidate continuations, normalised into ``p_rumor`` / ``p_nonrumor``.
Generated confidence is deliberately not used (plan §9).

The math is pure and model-free so the verifier and the tests can check it
without loading a reader.
"""
from __future__ import annotations

import hashlib
import math

from ..config.pilot_config import CANDIDATES


def _render_chat(tokenizer, messages, thinking: bool = False,
                 tokenize: bool = False):
    """The single chat-template call site (frozen formatter, plan §9).

    ``apply_chat`` renders text; the amendment-A1 boundary audit asks the same
    call for tokenizer-native ids. Both go through here so the frozen
    formatter — including its ``enable_thinking`` fallback — cannot drift
    between the text path and the native path.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=True,
            enable_thinking=thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=True)


def apply_chat(tokenizer, messages, thinking: bool = False) -> str:
    """Single chat formatter shared by all readers (thinking disabled)."""
    return _render_chat(tokenizer, messages, thinking=thinking, tokenize=False)


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


# --------------------------------------------------------------------------
# v2r1 autoregressive boundary audit (amendment A1)
#
# The legacy ``continuation_boundary`` check re-tokenizes
# ``prompt + candidate`` as one string. A SentencePiece tokenizer may merge the
# token straddling that seam, which fails the concatenation identity even
# though autoregressive inference never re-tokenizes the already-tokenized
# prompt prefix. Amendment A1 therefore keeps the legacy check as an
# **informational** field (``text_retokenization_stable``) and makes the formal
# v2r1 gate the tokenizer-native prompt identity below:
#
#   native_prompt_ids == scorer_prompt_ids  and  valid A/B candidate ids.
#
# The scoring mathematics, the prompt text, the chat template, the candidate
# definitions and the token budget are untouched by this amendment (A1 §2).
# --------------------------------------------------------------------------
def _ids_hash(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def native_prompt_ids(tokenizer, messages, thinking: bool = False) -> list:
    """Tokenizer-native prompt ids (amendment A1 §2.1).

    ``apply_chat_template(..., tokenize=True, add_generation_prompt=True)`` is
    the token sequence the model is actually conditioned on when it generates
    the next token. The v2r1 gate requires it to equal the ids the teacher-
    forced scorer builds from the rendered chat text.
    """
    ids = _render_chat(tokenizer, messages, thinking=thinking, tokenize=True)
    if hasattr(ids, "get"):  # a return_dict tokenizer hands back a BatchEncoding
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(i) for i in ids]


def normalize_candidate_text(text) -> str:
    """The explicit, tested normalization used to check candidate decoding.

    Whitespace is collapsed and the result upper-cased, so a tokenizer that
    renders ``"A"`` as ``" A"`` or ``"a"`` still recovers the intended label
    while anything else (empty text, extra words, a control token) does not.
    """
    return " ".join(str(text).split()).upper()


def candidate_token_audit(tokenizer, candidate: str) -> dict:
    """Validate one A/B candidate's continuation ids (amendment A1 §2.4).

    A candidate is valid only when it is non-empty, contains no special or
    control token, and decodes back to the intended label under
    :func:`normalize_candidate_text`.
    """
    ids = [int(i) for i in tokenize_continuation(tokenizer, candidate)]
    specials = {int(i) for i in (getattr(tokenizer, "all_special_ids", None)
                                 or [])}
    if hasattr(tokenizer, "decode"):
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
    else:  # a test double without decode cannot claim a decode round-trip
        decoded = ""
    present = sorted(set(ids) & specials)
    expected = normalize_candidate_text(candidate)
    got = normalize_candidate_text(decoded)
    audit = {
        "candidate": candidate,
        "candidate_ids": ids,
        "n_tokens": len(ids),
        "non_empty": bool(ids),
        "decoded": decoded,
        "decoded_normalized": got,
        "expected_normalized": expected,
        "decode_matches_label": got == expected,
        "special_ids_present": present,
        "has_unexpected_special_token": bool(present),
    }
    audit["valid"] = bool(audit["non_empty"]
                          and not audit["has_unexpected_special_token"]
                          and audit["decode_matches_label"])
    return audit


def ab_boundary_audit(tokenizer, user_prompt: str, candidates=CANDIDATES,
                      thinking: bool = False) -> dict:
    """v2r1 boundary audit record (amendment A1 §2, §3).

    Formal gate: ``autoregressive_boundary_ok`` — the tokenizer-native prompt
    ids equal the scorer's prompt ids **and** both candidates have valid
    continuation ids.

    Informational: ``text_retokenization_stable`` — the legacy concatenation
    identity. It is recorded in full, including each candidate's joined tail,
    so a tokenizer that fails it (Mistral's ``[/INST]`` seam) stays visible and
    is never silently hidden.
    """
    from .base_reader import build_messages
    messages = build_messages(user_prompt)
    chat = _render_chat(tokenizer, messages, thinking=thinking, tokenize=False)
    scorer_ids = [int(i) for i in tokenize_prompt(tokenizer, chat)]
    native_ids = native_prompt_ids(tokenizer, messages, thinking=thinking)

    cand_audit = {c: candidate_token_audit(tokenizer, c) for c in candidates}
    for candidate in candidates:
        legacy = continuation_boundary(tokenizer, chat, candidate)
        cand_audit[candidate]["text_retokenization_stable"] = bool(
            legacy["boundary_ok"])
        cand_audit[candidate]["joined_tail"] = legacy["joined_tail"]
        cand_audit[candidate]["joined_len"] = legacy["joined_len"]

    native_match = native_ids == scorer_ids
    candidates_valid = all(a["valid"] for a in cand_audit.values())
    retok_stable = all(a["text_retokenization_stable"]
                       for a in cand_audit.values())
    return {
        "audit": "v2r1_a1_autoregressive_boundary",
        "score_mode": "teacher_forced_logprob_sum",
        "prompt_tokens": len(scorer_ids),
        "native_prompt_tokens": len(native_ids),
        "native_prompt_ids_hash": _ids_hash(native_ids),
        "scorer_prompt_ids_hash": _ids_hash(scorer_ids),
        "native_prompt_matches_scorer": native_match,
        "native_prompt_ids_head": native_ids[:16],
        "native_prompt_ids_tail": native_ids[-8:],
        "scorer_prompt_ids_head": scorer_ids[:16],
        "scorer_prompt_ids_tail": scorer_ids[-8:],
        "candidates": cand_audit,
        "candidate_ids_valid": candidates_valid,
        "no_empty_continuation": all(a["non_empty"]
                                     for a in cand_audit.values()),
        "no_special_tokens_in_candidates": all(
            not a["has_unexpected_special_token"]
            for a in cand_audit.values()),
        # informational, never hidden (A1 §3)
        "text_retokenization_stable": retok_stable,
        "legacy_all_boundaries_ok": retok_stable,
        # the formal v2r1 gate
        "autoregressive_boundary_ok": bool(native_match and candidates_valid),
    }


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
