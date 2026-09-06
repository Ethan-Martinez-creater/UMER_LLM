"""Frozen baseline selectors (plan §25, §26).

Eight evaluation arms share the same evidence-unit pipeline; only the score
differs. The structural budget is frozen as
score_i = z(normDegree_i) - 0.25 * z(normDepth_i) with z=0 when std=0 and
earlier-timestamp-first tie-breaking. Randomness is seeded per
(dataset, event, cutoff) so runs are reproducible.
"""
from __future__ import annotations

import hashlib
import random

import torch

BASELINE_NAMES = (
    "source_only", "all_current", "random_budget", "recent_budget",
    "semantic_budget", "structural_budget", "static_selector",
    "tcdscr_dynamic",
)

CONTEXT_LENGTH_PLACEHOLDER_LIMIT = 10 ** 6


class ContextLengthUnresolved(RuntimeError):
    """Raised when no sane context length can be determined (delta-fix §19:
    never guess 32768/131072 — STOP instead)."""


def resolve_context_length(model_config=None, tokenizer=None) -> int:
    """Model context length from config, with sanity checks.

    Prefers ``model_config.max_position_embeddings``, falls back to
    ``tokenizer.model_max_length``; tokenizer placeholder values (the famous
    10^33 sentinel) are rejected. Uses the smaller of the sane candidates so
    all-current prompts can never overflow. No guessing.
    """
    candidates = []
    if model_config is not None:
        value = getattr(model_config, "max_position_embeddings", None)
        if isinstance(value, int) and 1024 <= value <= \
                CONTEXT_LENGTH_PLACEHOLDER_LIMIT:
            candidates.append(value)
    if tokenizer is not None:
        value = getattr(tokenizer, "model_max_length", None)
        if isinstance(value, int) and 1024 <= value <= \
                CONTEXT_LENGTH_PLACEHOLDER_LIMIT:
            candidates.append(value)
    if not candidates:
        raise ContextLengthUnresolved(
            "cannot determine model context length from model config or "
            "tokenizer; refusing to guess (CONTEXT_LENGTH_UNRESOLVED)")
    return min(candidates)


def _seeded_rng(dataset, event_id, cutoff) -> random.Random:
    digest = hashlib.sha256(
        f"{dataset}:{event_id}:{cutoff}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _z(values):
    n = len(values)
    if n == 0:
        return [0.0] * n
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    if std == 0:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def rank_candidates(name, snapshot, units, ctx):
    """Return scores aligned to ``units`` for a baseline arm.

    ``ctx`` keys: sem (N,384 tensor incl. source row 0 aligned to
    snapshot node order), source_pos, selector_scores (N,), dynamic_scores
    (N,). all_current / source_only return None scores (handled below).
    """
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}")
    if name in ("source_only", "all_current"):
        return None
    n_units = len(units)
    if name == "random_budget":
        rng = _seeded_rng(ctx["dataset"], snapshot["event_id"],
                          snapshot["cutoff_minutes"])
        return [rng.random() for _ in range(n_units)]
    if name == "recent_budget":
        # descending score = larger elapsed first = most recent reply first
        # (delta-fix §13/§14); equal timestamps keep snapshot order via the
        # (score desc, order asc) sort in select_evidence
        return [u["elapsed_seconds"] for u in units]
    if name == "semantic_budget":
        src_idx = ctx["source_pos"]
        src_vec = ctx["sem"][src_idx]
        rows = []
        index = {nid: i for i, nid in enumerate(snapshot["node_ids"])}
        for u in units:
            rows.append(index[u["node_id"]])
        sem = ctx["sem"][rows]
        cos = sem @ src_vec
        return cos.tolist()
    if name == "structural_budget":
        index = {nid: i for i, nid in enumerate(snapshot["node_ids"])}
        deg, dep = [], []
        for u in units:
            i = index[u["node_id"]]
            deg.append(ctx["struct3"][i][0])
            dep.append(ctx["struct3"][i][1])
        z_deg, z_dep = _z(deg), _z(dep)
        return [z_deg[k] - 0.25 * z_dep[k] for k in range(n_units)]
    if name == "static_selector":
        index = {nid: i for i, nid in enumerate(snapshot["node_ids"])}
        return [float(ctx["selector_scores"][index[u["node_id"]]])
                for u in units]
    if name == "tcdscr_dynamic":
        index = {nid: i for i, nid in enumerate(snapshot["node_ids"])}
        return [float(ctx["dynamic_scores"][index[u["node_id"]]])
                for u in units]
    raise ValueError(f"unhandled baseline {name!r}")


def select_all_current(snapshot, units, budget_selector, context_length,
                       max_new_tokens, prompt_builder):
    """All-current under the model context limit (delta-fix §17–§22).

    Definition: the full social evidence the model can actually read at this
    snapshot. Units are added whole in deterministic snapshot order (§4.1
    timestamp ascending, original_order ties) until including the next pair
    would push the FINAL prompt token count + max_new_tokens past the
    context length — then stop. Pairs are atomic; nothing is truncated.
    """
    accepted = []
    for unit in units:
        candidate = accepted + [unit]
        prompt = prompt_builder(snapshot, candidate)
        total = budget_selector.count_tokens(prompt)
        if total + max_new_tokens > context_length:
            break
        accepted = candidate
    final_prompt = prompt_builder(snapshot, accepted)
    input_tokens = budget_selector.count_tokens(final_prompt)
    if input_tokens + max_new_tokens > context_length:
        raise RuntimeError(
            "all_current prompt exceeds the model context limit")
    # tokens contributed by the excluded pairs (rendered in their own order)
    dropped = 0
    for i, unit in enumerate(units[len(accepted):], start=len(accepted) + 1):
        from ..context.evidence_unit import render_evidence
        dropped += budget_selector.count_tokens(render_evidence(unit, i))
    return {
        "selected_units": accepted,
        "selected_node_ids": [u["node_id"] for u in accepted],
        "evidence_tokens": input_tokens,
        "truncated": len(accepted) < len(units),
        "units_before_truncation": len(units),
        "units_after_truncation": len(accepted),
        "tokens_dropped": dropped,
        "context_length": context_length,
        "input_tokens": input_tokens,
    }


def select_evidence(name, snapshot, units, ctx, budget_selector):
    """Return (selected_units, evidence_tokens, scores) for one arm."""
    scores = rank_candidates(name, snapshot, units, ctx)
    if name == "source_only":
        return [], 0, scores
    if name == "all_current":
        # context guard is mandatory: the caller must supply the resolved
        # context length, a prompt builder over accepted units, and the
        # generation reserve (delta-fix §17–§22)
        for key in ("context_length", "max_new_tokens", "prompt_builder"):
            if key not in ctx:
                raise ValueError(
                    f"all_current requires ctx[{key!r}] — unbounded prompts "
                    "are forbidden")
        result = select_all_current(
            snapshot, units, budget_selector, ctx["context_length"],
            ctx["max_new_tokens"], ctx["prompt_builder"])
        return result["selected_units"], result["evidence_tokens"], scores
    order = sorted(range(len(units)),
                   key=lambda i: (-scores[i], units[i]["order"]))
    accepted, total = [], 0
    for idx in order:
        from ..context.evidence_unit import render_evidence
        candidate = render_evidence(units[idx], len(accepted) + 1)
        cost = budget_selector.count_tokens(candidate)
        if total + cost > budget_selector.budget:
            continue
        accepted.append(units[idx])
        total += cost
    return accepted, total, scores
