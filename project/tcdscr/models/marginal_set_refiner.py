"""MF-TSR: Marginal-Fidelity Temporal Set Refinement (Dynamic V2 §10-§20).

Replaces the failed point-wise dynamic score with a deterministic local
set search over the evidence SET.  Every candidate move is scored by the
reduction it brings to the forward-KL teacher distortion

    D(S) = KL(stopgrad(p_full) || q(S))          (set_fidelity.py)

relative to the current set:

    relative_gain = (D_cur - D_new) / max(D_cur, 1e-6)

and only moves with relative_gain >= epsilon are executed.  Legal moves at
each step are all REMOVEs (drop one selected evidence), all ADDs that fit
the 1024-token budget, and all budget-feasible SWAPs; incoming ADD/SWAP
candidates are pruned to the frozen Static-Utility top-64 unselected units
(§23).  The loop runs at most 20 steps (§19); if the refined set ends with
higher distortion than the Static set it falls back to Static (§20).

The only hyper-parameter is epsilon (grid {0, 0.01, 0.05}, chosen per
fold on validation).  Novelty/persistence bonuses NEVER enter this module
(§22); temporal continuity appears only in the tie-break.  The encoder runs
once per snapshot in the caller; this module only calls the small frozen
proxy MLP, batched (§24).  No trainable parameter, no gradient.

Frozen constants (execution protocol §10, §19, §23):

    BUDGET_TOKENS        = 1024   (primary budget, never searched)
    MAX_REFINEMENT_STEPS = 20     (hard cap on the local search)
    PRUNE_TOP            = 64     (incoming candidate pool size)
"""
from __future__ import annotations

import torch

from .set_fidelity import batch_set_distortions
from .temporal_survival import memory_warm_start

BUDGET_TOKENS = 1024
MAX_REFINEMENT_STEPS = 20
PRUNE_TOP = 64
EPSILON_GRID = (0.0, 0.01, 0.05)


def static_pack(costs, u, order, budget=BUDGET_TOKENS):
    """Corrected E2 Static baseline: descending utility greedy with the
    skip-and-continue packing semantics of EvidenceBudgetSelector.select.
    Returns (selected_positions, total_tokens)."""
    idx = sorted(range(len(costs)), key=lambda i: (-u[i], order[i]))
    sel, total = [], 0
    for i in idx:
        if total + costs[i] > budget:
            continue
        sel.append(i)
        total += costs[i]
    return sel, total


def incoming_pool(selected, u, order, prune_top=PRUNE_TOP):
    """Frozen candidate pruning (§23): unselected nodes ranked by current
    Static Utility descending (ties: snapshot order), top ``prune_top``
    kept as ADD/SWAP-in candidates.  Returns (pool, pool_size_before)."""
    rest = [i for i in range(len(u)) if i not in selected]
    ranked = sorted(rest, key=lambda i: (-u[i], order[i]))
    return ranked[:prune_top], len(rest)


def relative_gain(d_cur, d_new):
    """Execution-protocol §16."""
    return (d_cur - d_new) / max(d_cur, 1e-6)


def _mask_rows(sets, m, device):
    """(K, m) 0/1 membership masks.

    Vectorised scatter: identical values to a per-row assignment loop but
    ~900x faster on GPU, which matters because each refinement step builds
    up to ~64x|S| SWAP rows on 1000-node snapshots.
    """
    counts = torch.tensor([len(s) for s in sets], dtype=torch.long,
                          device=device)
    total = int(counts.sum().item())
    if total == 0:
        return torch.zeros(len(sets), m, device=device)
    rows = torch.repeat_interleave(
        torch.arange(len(sets), device=device), counts)
    cols = torch.tensor([i for s in sets for i in sorted(s)],
                        dtype=torch.long, device=device)
    masks = torch.zeros(len(sets), m, device=device)
    masks[rows, cols] = 1.0
    return masks


def _enumerate_moves(selected, tokens, pool, costs, budget):
    """All legal REMOVE / ADD / SWAP moves as (type, i, j, new_set)."""
    cands = []
    for j in sorted(selected):
        cands.append(("REMOVE", None, j, selected - {j}))
    for i in sorted(pool):
        if tokens + costs[i] <= budget:
            cands.append(("ADD", i, None, selected | {i}))
        for j in sorted(selected):
            if tokens - costs[j] + costs[i] <= budget:
                cands.append(("SWAP", i, j, (selected - {j}) | {i}))
    return cands


def refine_set(node_repr, h_source, p_full_probs, u, costs, order, node_ids,
               memory_positions, proxy, budget=BUDGET_TOKENS, epsilon=0.0,
               max_steps=MAX_REFINEMENT_STEPS, prune_top=PRUNE_TOP):
    """Run MF-TSR on one snapshot.

    ``node_repr``: (M, 768) frozen candidate representations (device
    tensor); ``u``/``costs``/``order``/``node_ids`` aligned python lists in
    candidate order; ``memory_positions``: positions of M_previous that are
    still present in this snapshot.  Fully deterministic: identical inputs
    give identical outputs.  Returns a result dict (positions are
    candidate-space indices; moves carry node ids).
    """
    m = len(costs)
    device = node_repr.device
    memory_present = set(memory_positions)
    with torch.no_grad():
        # --- initialisations (execution protocol §10-§12) --------------
        static_sel, static_tokens = static_pack(costs, u, order, budget)
        mem_sel, mem_tokens = memory_warm_start(memory_positions, costs, u,
                                                order, budget)
        d_both = batch_set_distortions(
            p_full_probs, h_source, node_repr,
            _mask_rows([sorted(static_sel), sorted(mem_sel)], m, device),
            proxy)
        d_static, d_mem = float(d_both[0]), float(d_both[1])
        if abs(d_static - d_mem) <= 1e-6:
            init_sel, init_source = mem_sel, "memory"   # temporal continuity
        elif d_static < d_mem:
            init_sel, init_source = static_sel, "static"
        else:
            init_sel, init_source = mem_sel, "memory"

        selected = set(init_sel)
        tokens = sum(costs[i] for i in selected)
        d_cur = d_mem if init_source == "memory" else d_static

        moves, pool_sizes = [], []
        exhausted = False
        for step in range(max_steps):
            pool, pool_before = incoming_pool(selected, u, order, prune_top)
            pool_sizes.append((pool_before, len(pool)))
            cands = _enumerate_moves(selected, tokens, pool, costs, budget)
            if not cands:
                break
            d_new = batch_set_distortions(
                p_full_probs, h_source, node_repr,
                _mask_rows([sorted(c[3]) for c in cands], m, device), proxy)
            gains = [relative_gain(d_cur, float(d)) for d in d_new]
            best = max(gains)
            if best < epsilon:
                break
            tied = [(ci, c) for ci, (c, g) in enumerate(zip(cands, gains))
                    if best - g <= 1e-6]

            def _tie_key(item):
                _, (typ, i, j, new) = item
                kept = -len(new & memory_present)
                tok_after = (tokens + costs[i] if typ == "ADD" else
                             tokens - costs[j] if typ == "REMOVE" else
                             tokens - costs[j] + costs[i])
                util_after = -sum(u[x] for x in new)
                stable = ((order[i],) if typ == "ADD" else
                          (order[j],) if typ == "REMOVE" else
                          (order[i], order[j]))
                return (kept, tok_after, util_after, stable)

            ci, (typ, i, j, new) = min(tied, key=_tie_key)
            dist_before = d_cur
            dist_after = float(d_new[ci])
            if typ == "REMOVE":
                tokens -= costs[j]
            elif typ == "ADD":
                tokens += costs[i]
            else:
                tokens += costs[i] - costs[j]
            selected = new
            d_cur = dist_after
            moves.append({
                "step": step, "move_type": typ,
                "added_node_id": node_ids[i] if i is not None else None,
                "removed_node_id": node_ids[j] if j is not None else None,
                "distortion_before": dist_before,
                "distortion_after": dist_after,
                "relative_gain": gains[ci],
            })
        else:
            exhausted = True

        # max_step_hit: the loop was cut by the cap while a move with
        # relative_gain >= epsilon was still available (§19).
        if exhausted:
            pool, _ = incoming_pool(selected, u, order, prune_top)
            cands = _enumerate_moves(selected, tokens, pool, costs, budget)
            if cands:
                d_new = batch_set_distortions(
                    p_full_probs, h_source, node_repr,
                    _mask_rows([sorted(c[3]) for c in cands], m, device),
                    proxy)
                max_step_hit = any(
                    relative_gain(d_cur, float(d)) >= epsilon
                    for d in d_new)
            else:
                max_step_hit = False
        else:
            max_step_hit = False

        # --- static fallback (execution protocol §20) -------------------
        if d_cur > d_static + 1e-8:
            selected, tokens, d_cur = set(static_sel), static_tokens, d_static
            fallback = True
        else:
            fallback = False

    final = sorted(selected)
    return {
        "selected_positions": final,
        "selected_node_ids": [node_ids[i] for i in final],
        "evidence_tokens": tokens,
        "distortion": d_cur,
        "static_positions": sorted(static_sel),
        "static_node_ids": [node_ids[i] for i in sorted(static_sel)],
        "static_tokens": static_tokens,
        "static_distortion": d_static,
        "warm_start_positions": sorted(mem_sel),
        "warm_start_tokens": mem_tokens,
        "warm_start_distortion": d_mem,
        "init_source": init_source,
        "accepted_moves": moves,
        "fallback_to_static": fallback,
        "max_step_hit": max_step_hit,
        "pool_sizes": pool_sizes,
    }
