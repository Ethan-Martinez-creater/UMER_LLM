"""Robust selection arms and 50%-budget packing (plan §21, §22).

The pilot only reranks units that are already in ``C_ref``; nothing is
retrieved from outside the reference context. Every arm packs **complete**
Reply–Parent units under

    B_pilot = floor(0.50 * Tokens(C_ref))

and always continues packing even when scores are negative, so the
compression-budget comparison stays controlled (plan §21). Prompt
presentation is chronological, so an arm returns its ids in snapshot order
even though packing follows score order.
"""
from __future__ import annotations

import hashlib
import random

from ..config.pilot_config import (PILOT_BUDGET_FRACTION, PRIMARY_ARM,
                                   SELECTION_ARMS)
from ..intervention.evidence_units import render_units_for_budget

SEPARATOR = "\n\n"


def robust_score(pred_a: dict, pred_b: dict, node_id: str) -> float:
    """``s_i^robust = min(u_hat_a, u_hat_b)`` (plan §21)."""
    return min(float(pred_a[node_id]), float(pred_b[node_id]))


def density(score: float, token_cost: int) -> float:
    """``d_i = score / max(TokenCost_i, 1)`` (plan §21)."""
    return float(score) / float(max(int(token_cost), 1))


def pilot_target_tokens(src: dict) -> int:
    """``B_pilot = floor(0.50 * Tokens(C_ref))`` (plan §21)."""
    return int(PILOT_BUDGET_FRACTION * int(src["total_tokens"]))


def rank_by_density(node_ids, scores: dict, token_costs: dict, seed=None,
                    random_rank: bool = False):
    """Node ids ordered for packing.

    Ties break on node id so the order is deterministic; ``random_rank`` uses
    the frozen seed for the S1 random baseline.
    """
    ids = list(node_ids)
    if random_rank:
        rng = random.Random(seed)
        rng.shuffle(ids)
        return ids
    return sorted(ids, key=lambda n: (-density(scores[n], token_costs[n]), n))


def pack_within_budget(units, ranked_ids, token_costs, target_tokens,
                       tokenizer) -> dict:
    """Greedy complete-unit packing under ``target_tokens`` (plan §21).

    Scan the ranking; add a complete unit when the running total stays within
    the target; otherwise skip it and keep scanning. Never splits a unit and
    never exceeds the target.
    """
    from tcdscr.llm.reader_prompt import count_tokens
    try:
        sep_cost = count_tokens(tokenizer, SEPARATOR)
    except Exception:
        sep_cost = 0
    chosen, total = [], 0
    for nid in ranked_ids:
        cost = int(token_costs[nid]) + (sep_cost if chosen else 0)
        if total + cost <= target_tokens:
            chosen.append(nid)
            total += cost
    # The running total over-counts slightly (ids/separators); verify against
    # the real render and trim from the weakest pick if it ever exceeds.
    while chosen:
        block = render_units_for_budget(units, chosen)
        real = count_tokens(tokenizer, block)
        if real <= target_tokens:
            return {"selected_node_ids": chosen, "total_tokens": real}
        chosen.pop()
    return {"selected_node_ids": [], "total_tokens": 0}


def _arm_scores(arm: str, src: dict, predictions: dict, reader_keys):
    """``(scores, node_ids)`` for one arm — plan §22 definitions."""
    selected = list(src["selected_node_ids"])
    costs = src["unit_token_costs"]
    if arm == "S0_src_full":
        return None, selected
    if arm in ("S1_random_tm", "S2_semantic_tm"):
        # both are ranked below: random via S1, semantic via SRC relevance
        return None, selected
    if arm == "S3a_single_a":
        return predictions[reader_keys[0]], selected
    if arm == "S3b_single_b":
        return predictions[reader_keys[1]], selected
    if arm == "S4_shared":
        return predictions["shared"], selected
    if arm == PRIMARY_ARM:
        scores = {nid: robust_score(predictions[reader_keys[0]],
                                    predictions[reader_keys[1]], nid)
                  for nid in selected}
        return scores, selected
    if arm == "S6_legacy_utility_tm":
        if "legacy" not in predictions:
            raise KeyError("S6 requires legacy utility scores (PHEME only)")
        return predictions["legacy"], selected
    raise ValueError(f"unknown selection arm {arm!r}")


def build_arm(arm: str, units, src: dict, predictions: dict, tokenizer,
              reader_keys, seed) -> dict:
    """Resolve one selection arm into a frozen evidence subset (plan §22)."""
    costs = src["unit_token_costs"]
    target = pilot_target_tokens(src)
    if arm == "S0_src_full":
        block = render_units_for_budget(units, src["selected_node_ids"])
        from tcdscr.llm.reader_prompt import count_tokens
        return {"arm": arm, "selected_node_ids": list(src["selected_node_ids"]),
                "total_tokens": count_tokens(tokenizer, block),
                "target_tokens": int(src["total_tokens"])}
    if arm == "S1_random_tm":
        ranked = rank_by_density(src["selected_node_ids"], {}, costs, seed=seed,
                                 random_rank=True)
    elif arm == "S2_semantic_tm":
        ranked = [nid for nid in src["ranked_node_ids"]
                  if nid in set(src["selected_node_ids"])]
    else:
        scores, ids = _arm_scores(arm, src, predictions, reader_keys)
        ranked = rank_by_density(ids, scores, costs)
    packed = pack_within_budget(units, ranked, costs, target, tokenizer)
    selected = [nid for nid in src["selected_node_ids"]
                if nid in set(packed["selected_node_ids"])]
    return {"arm": arm, "selected_node_ids": selected,
            "total_tokens": packed["total_tokens"], "target_tokens": target}


def arm_content_hash(arm_result: dict) -> str:
    """Stable hash of a frozen arm subset (plan §31/§33 freeze step)."""
    payload = "|".join(arm_result["selected_node_ids"])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def all_arms(units, src, predictions, tokenizer, reader_keys, seed,
             include_legacy: bool = False) -> dict:
    """Build every applicable arm; S6 only when legacy scores exist."""
    out = {}
    for arm in SELECTION_ARMS:
        if arm == "S6_legacy_utility_tm" and not include_legacy:
            continue
        out[arm] = build_arm(arm, units, src, predictions, tokenizer,
                             reader_keys, seed)
        out[arm]["content_hash"] = arm_content_hash(out[arm])
    return out
