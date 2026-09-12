"""Matched control construction for structured interventions (plan §11).

I3 matches I2 (parent–child pair) and I5 matches I4 (subtree). Matching uses
**only pre-reader features** — graph relations, depth, canonical token cost and
the frozen seed 7319. Reader outcomes can never influence control matching
(plan §11).

Where the plan says "if possible" the matcher degrades in a fixed priority
order and records which level it reached, so the analysis can report control
quality honestly instead of silently pretending to a perfect match.
"""
from __future__ import annotations

import random

from ..config.pilot_config import (MATCH_DEPTH_TOLERANCE, MATCH_TOKEN_TOLERANCE,
                                   PARTITION_SEED)

MATCH_EXACT = "exact_token_depth"
MATCH_TOKEN_ONLY = "token_only"
MATCH_STRUCTURE_ONLY = "structure_only"
MATCH_NONE = "no_match"


def _pos(snapshot):
    return {nid: i for i, nid in enumerate(snapshot["node_ids"])}


def parent_map(snapshot):
    """``{node_id: parent_id or None}`` from resolved snapshot edges."""
    out = {}
    for child, parent in snapshot["edge_index"]:
        out[snapshot["node_ids"][child]] = snapshot["node_ids"][parent]
    return out


def ancestors_of(snapshot, node_id):
    """All ancestor ids of ``node_id`` inside G_t (nearest first)."""
    pmap = parent_map(snapshot)
    chain = []
    cur = pmap.get(node_id)
    guard = 0
    while cur is not None and guard <= len(snapshot["node_ids"]):
        chain.append(cur)
        cur = pmap.get(cur)
        guard += 1
    return chain


def is_ancestor(snapshot, a: str, b: str) -> bool:
    """True when ``a`` is a strict ancestor of ``b`` inside G_t."""
    return a in ancestors_of(snapshot, b)


def is_parent_child(snapshot, a: str, b: str) -> bool:
    pmap = parent_map(snapshot)
    return pmap.get(a) == b or pmap.get(b) == a


def depth_by_id(snapshot) -> dict:
    return {nid: d for nid, d in zip(snapshot["node_ids"], snapshot["depths"])}


def lca(snapshot, node_ids):
    """Lowest common ancestor id of ``node_ids`` (None when empty)."""
    if not node_ids:
        return None
    chains = []
    for nid in node_ids:
        chain = [nid] + ancestors_of(snapshot, nid)
        chains.append(list(reversed(chain)))
    prefix = chains[0]
    for chain in chains[1:]:
        k = 0
        while k < len(prefix) and k < len(chain) and prefix[k] == chain[k]:
            k += 1
        prefix = prefix[:k]
        if not prefix:
            return None
    return prefix[-1]


def in_one_ancestor_subtree(snapshot, node_ids) -> bool:
    """True when all ids share an ancestor below the source (plan §11 I5)."""
    if len(node_ids) <= 1:
        return True
    root = snapshot["source_id"]
    common = lca(snapshot, node_ids)
    return common is not None and common != root


def _shuffle(items, seed):
    items = list(items)
    random.Random(seed).shuffle(items)
    return items


def _token_close(cost, target, tolerance=MATCH_TOKEN_TOLERANCE) -> bool:
    if target <= 0:
        return cost <= 0
    return abs(cost - target) / target <= tolerance


def match_nonadjacent_pair(snapshot, candidate_ids, target_pair,
                           token_costs, seed=PARTITION_SEED) -> dict:
    """I3: a non-adjacent pair matched to the I2 pair (plan §11)."""
    target = tuple(sorted(target_pair))
    target_cost = sum(token_costs[n] for n in target)
    depth = depth_by_id(snapshot)
    target_depth = sum(depth.get(n, 0) for n in target) / len(target)

    hard, token_depth, token_only = [], [], []
    for a in candidate_ids:
        for b in candidate_ids:
            if a >= b:
                continue
            if (a, b) == target:
                continue
            if is_parent_child(snapshot, a, b):
                continue
            if is_ancestor(snapshot, a, b) or is_ancestor(snapshot, b, a):
                continue
            cost = token_costs[a] + token_costs[b]
            dmean = (depth.get(a, 0) + depth.get(b, 0)) / 2
            hard.append((a, b))
            if _token_close(cost, target_cost):
                token_only.append((a, b))
                if abs(dmean - target_depth) <= MATCH_DEPTH_TOLERANCE:
                    token_depth.append((a, b))
    for level, pool in ((MATCH_EXACT, token_depth),
                        (MATCH_TOKEN_ONLY, token_only),
                        (MATCH_STRUCTURE_ONLY, hard)):
        if pool:
            pool = _shuffle(sorted(pool), seed)
            return {"control_ids": list(pool[0]), "match_level": level,
                    "target_ids": list(target),
                    "target_cost": target_cost}
    return {"control_ids": [], "match_level": MATCH_NONE,
            "target_ids": list(target), "target_cost": target_cost}


def match_disconnected_group(snapshot, candidate_ids, target_ids,
                             token_costs, seed=PARTITION_SEED,
                             tries: int = 4000) -> dict:
    """I5: k units not all in one ancestor subtree, cost-matched (plan §11)."""
    target_ids = list(target_ids)
    k = len(target_ids)
    target_cost = sum(token_costs[n] for n in target_ids)
    pool = [n for n in candidate_ids if n not in set(target_ids)]
    if k == 0 or len(pool) < k:
        return {"control_ids": [], "match_level": MATCH_NONE,
                "target_ids": target_ids, "target_cost": target_cost}

    rng = random.Random(seed)
    best = None
    for _ in range(tries):
        pick = rng.sample(pool, k)
        if in_one_ancestor_subtree(snapshot, pick):
            continue
        cost = sum(token_costs[n] for n in pick)
        err = (abs(cost - target_cost) / target_cost) if target_cost > 0 else 0.0
        if best is None or err < best[0]:
            best = (err, sorted(pick), _token_close(cost, target_cost))
            if err == 0.0:
                break
    if best is None:
        return {"control_ids": [], "match_level": MATCH_NONE,
                "target_ids": target_ids, "target_cost": target_cost}
    level = MATCH_EXACT if best[2] else MATCH_STRUCTURE_ONLY
    return {"control_ids": best[1], "match_level": level,
            "target_ids": target_ids, "target_cost": target_cost}
