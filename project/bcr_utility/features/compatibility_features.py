"""E3 LIGHT-TOUCH reader-forward compatibility features (M1 plan §17).

Generated **only** after a recorded ZERO-TOUCH failure. Per evidence key and
reader:

* ``source_only_margin`` / ``source_only_entropy`` — teacher-forced A/B
  scoring of the source-only prompt (same construction as the frozen P0
  probe context), defined per (event, cutoff, reader);
* ``source_nll`` — mean per-token NLL of the raw source text, per
  (event, reader);
* ``evidence_nll_per_token`` — mean per-token NLL of the rendered evidence
  unit with no context;
* ``conditional_evidence_nll_per_token`` — the same unit conditioned on the
  raw source text;
* ``nll_gap = evidence_nll_per_token - conditional_evidence_nll_per_token``.

NLLs are raw-LM teacher-forced log-likelihoods (no chat template): they
measure reader/text compatibility, not task behaviour, and they never touch
any utility label.
"""
from __future__ import annotations

import math

from ..config import protocol as P
from ..probes.fingerprint import response_metrics


class CompatibilityRefused(RuntimeError):
    """Raised when an E3 row cannot be built exactly as specified."""


def mean_token_nll(model, tokenizer, context: str, continuation: str,
                   device=None) -> float:
    """Mean per-token NLL of ``continuation`` given ``context`` (raw LM).

    The context is encoded with special tokens (so an empty context still
    carries the model's BOS); the continuation never adds its own specials.
    """
    import torch
    device = device or next(model.parameters()).device
    ctx = tokenizer(context, add_special_tokens=True)["input_ids"]
    cont = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not cont:
        raise CompatibilityRefused("continuation tokenises to nothing")
    ids = torch.tensor([ctx + cont], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    start = len(ctx)
    total = 0.0
    for k, tok in enumerate(cont):
        total += float(logprobs[0, start + k - 1, tok])
    return -total / len(cont)


def e3_feature_dict(source_margin: float, source_entropy: float,
                    source_nll: float, evidence_nll: float,
                    conditional_nll: float) -> dict:
    """The six frozen E3 scalars of one (evidence key, reader) pair."""
    for name, value in (("source_margin", source_margin),
                        ("source_entropy", source_entropy),
                        ("source_nll", source_nll),
                        ("evidence_nll", evidence_nll),
                        ("conditional_nll", conditional_nll)):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise CompatibilityRefused(f"{name}={value!r} is not finite")
    features = {
        "source_only_margin": float(source_margin),
        "source_only_entropy": float(source_entropy),
        "source_nll": float(source_nll),
        "evidence_nll_per_token": float(evidence_nll),
        "conditional_evidence_nll_per_token": float(conditional_nll),
        "nll_gap": float(evidence_nll - conditional_nll),
    }
    if tuple(features) != P.E3_FEATURE_NAMES:
        raise CompatibilityRefused("E3 feature order drift")
    return features


def source_only_metrics(score_a: float, score_b: float, p_rumor: float) -> dict:
    """``(margin, entropy)`` of the source-only A/B response."""
    m = response_metrics(score_a, score_b, p_rumor)
    return m["signed_margin"], m["entropy"]


def validate_e3_rows(rows, expected_keys, readers=P.READER_KEYS) -> dict:
    seen = {(r["key"], r["reader"]) for r in rows}
    problems = []
    expected = {(k, r) for k in expected_keys for r in readers}
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        problems.append(f"{len(missing)} missing: {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} unexpected: {extra[:3]}")
    for row in rows:
        if tuple(row["e3"]) != P.E3_FEATURE_NAMES:
            problems.append(f"{row['key']}: feature order drift")
            break
        for name, value in row["e3"].items():
            if not math.isfinite(value):
                problems.append(f"{row['key']}: {name} not finite")
                break
    if problems:
        raise CompatibilityRefused("; ".join(problems[:8]))
    return {"rows": len(rows), "per_reader": len(rows) // len(readers),
            "ok": True}
