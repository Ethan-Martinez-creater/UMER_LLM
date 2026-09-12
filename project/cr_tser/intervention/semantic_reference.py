"""SRC — Semantic Reference Context (plan §7).

The intervention reference context is written *before* any reader is called and
depends on nothing but:

1. the visible non-source replies of the snapshot,
2. the frozen 384D multilingual MiniLM embeddings,
3. the canonical Qwen tokenizer budget ``B_ref = 1024``.

It is therefore reader-independent, gold-independent and Proxy-independent —
the property the whole intervention design relies on. No Proxy, MF-TSR or
MS-TSR component is used (plan §7).

Ordering rule (plan §7): rank by ``cos(e_reply, e_source)`` descending, break
ties by earlier timestamp, then by the adapter's stable snapshot order.
Packing is greedy over complete units; a unit is never split.

``src_rank_percentile`` is exposed for the atomic-utility feature vector and
``selected_node_ids`` stays in snapshot order because prompt presentation is
chronological (plan §21).
"""
from __future__ import annotations

import math

from ..config.pilot_config import BUDGET_REF
from .evidence_units import render_units_for_budget, unit_token_cost

SEPARATOR = "\n\n"


def _dot(a, b) -> float:
    if hasattr(a, "tolist"):
        a = a.tolist()
    if hasattr(b, "tolist"):
        b = b.tolist()
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(a) -> float:
    if hasattr(a, "tolist"):
        a = a.tolist()
    return math.sqrt(sum(float(x) * float(x) for x in a))


def cosine(a, b) -> float:
    """Cosine similarity that works for torch tensors and plain lists."""
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def rank_units(units, source_embedding, reply_embeddings):
    """``(order, relevance)`` where order ranks unit indices by SRC relevance."""
    relevance = {}
    for u in units:
        emb = reply_embeddings.get(u["node_id"])
        relevance[u["node_id"]] = (
            cosine(emb, source_embedding) if emb is not None else 0.0)
    order = sorted(range(len(units)),
                   key=lambda i: (-relevance[units[i]["node_id"]],
                                  units[i]["timestamp"],
                                  units[i]["snapshot_order"]))
    return order, relevance


def _greedy_pack(units, order, tokenizer, budget):
    costs = {u["node_id"]: unit_token_cost(tokenizer, u) for u in units}
    sep_cost = 0
    try:
        from tcdscr.llm.reader_prompt import count_tokens
        sep_cost = count_tokens(tokenizer, SEPARATOR)
    except Exception:
        sep_cost = 0
    chosen = []
    total = 0
    for i in order:
        cost = costs[units[i]["node_id"]]
        add = cost + (sep_cost if chosen else 0)
        if total + add <= budget:
            chosen.append(i)
            total += add
    # Strict final check on the actually rendered block: the incremental
    # estimate can only over-count, but the plan's "never exceed" rule is
    # enforced against the real render (plan §34).
    while chosen:
        ids = [units[i]["node_id"] for i in chosen]
        block = render_units_for_budget(units, ids)
        from tcdscr.llm.reader_prompt import count_tokens
        real = count_tokens(tokenizer, block)
        if real <= budget:
            return chosen, real, costs
        chosen.pop()
    return [], 0, costs


def build_src(units, source_embedding, reply_embeddings, tokenizer,
              budget: int = BUDGET_REF) -> dict:
    """Build the frozen reference context for one snapshot (plan §7)."""
    if budget != BUDGET_REF:
        raise ValueError(f"B_ref is frozen to {BUDGET_REF}, got {budget}")
    order, relevance = rank_units(units, source_embedding, reply_embeddings)
    chosen, total_tokens, costs = _greedy_pack(units, order, tokenizer, budget)
    chosen_set = set(chosen)
    selected_in_order = [units[i]["node_id"] for i in sorted(chosen_set)]
    ranked_ids = [units[i]["node_id"] for i in order]
    n = len(order)
    rank_percentile = {}
    for rank, i in enumerate(order):
        rank_percentile[units[i]["node_id"]] = \
            (rank / (n - 1)) if n > 1 else 0.0
    return {
        "selected_node_ids": selected_in_order,
        "ranked_node_ids": ranked_ids,
        "n_visible_replies": len(units),
        "n_selected": len(selected_in_order),
        "total_tokens": total_tokens,
        "budget_ref": budget,
        "utilization": total_tokens / budget if budget else 0.0,
        "unit_token_costs": costs,
        "rank_percentile": rank_percentile,
        "relevance": relevance,
    }


def src_token_cost(tokenizer, src: dict, units) -> int:
    """Recompute the rendered SRC token count (verifier helper)."""
    from tcdscr.llm.reader_prompt import count_tokens
    block = render_units_for_budget(units, src["selected_node_ids"])
    return count_tokens(tokenizer, block)
