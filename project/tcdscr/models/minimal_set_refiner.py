"""MS-TSR: Minimal-Sufficient Temporal Set Refinement (Dynamic V3 §9-§15).

The V2 failure showed that minimising KL to the full-encoder teacher is
satisfied by *deleting* evidence until the proxy mimics the teacher, which
does not preserve the reader decision.  V3 inverts the problem:

    min_S Cost(S)   s.t.  Sufficient_t(S; alpha) and Cost(S) <= 1024

so the search starts from the EMPTY set and grows only until the Static
reader's decision is preserved with a retained margin
(sufficiency.py §8); afterwards redundant evidence is removed backwards.
No SWAP moves exist in the primary method (§13), novelty/persistence
bonuses do not exist, and temporal memory enters only as candidate-pool
augmentation plus tie-break preference (§14).

Frozen constants: budget 1024, alpha in {0.80, 0.90, 0.95, 1.00},
candidate pool = S_static ∪ visible M_prev ∪ top-64 static utility (§10).
"""
from __future__ import annotations

import torch

from .marginal_set_refiner import BUDGET_TOKENS, static_pack  # noqa: F401
from .set_fidelity import set_representation
from .sufficiency import (ALPHA_GRID, batch_decision_margins,
                          batch_proxy_logits, dual_view_agreement,
                          proxy_logits, reader_efficiency)

TOP_K_POOL = 64


def _mask_rows(sets, m, device):
    """(K, m) 0/1 membership masks, vectorised (see marginal_set_refiner)."""
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


def candidate_pool(static_positions, memory_positions, u, order,
                   top_k=TOP_K_POOL):
    """P_t of design §10: Static set ∪ visible previous memory ∪ top-64.

    Returns (sorted pool positions, pool_size_before_pruning) where the
    before-size counts the current candidates that were neither in the
    Static set nor in memory (i.e. the part that pruning ranked).
    """
    pool = set(static_positions) | set(memory_positions)
    rest = [i for i in range(len(u)) if i not in pool]
    ranked = sorted(rest, key=lambda i: (-u[i], order[i]))
    before = len(rest)
    pool |= set(ranked[:top_k])
    return sorted(pool), before


def refine_minimal_set(node_repr, h_source, p_full_probs, u, costs, order,
                       node_ids, static_positions, memory_positions, proxy,
                       budget=BUDGET_TOKENS, alpha=0.95, top_k=TOP_K_POOL):
    """Run MS-TSR on one snapshot.

    ``node_repr``: (M, 768) frozen candidate representations; ``u`` /
    ``costs`` / ``order`` / ``node_ids`` aligned lists in candidate order;
    ``static_positions``: the corrected-E2 Static pack; ``memory_positions``:
    positions of M_{t-1} still visible in this snapshot.  Pure function of
    its inputs (fully deterministic).  Returns a result dict.
    """
    m = len(costs)
    device = node_repr.device
    memory_set = set(memory_positions)
    with torch.no_grad():
        static_tokens = sum(costs[i] for i in static_positions)
        s_logits = proxy_logits(
            proxy, h_source,
            node_repr[static_positions] if static_positions else
            torch.empty(0, node_repr.shape[1], device=device))
        ref_label = int(s_logits.view(-1).argmax())
        static_margin = float(s_logits[ref_label] - s_logits[1 - ref_label])
        q_static = torch.softmax(s_logits.float(), dim=-1)
        agree = dual_view_agreement(p_full_probs, q_static)

        def _info(positions):
            """(logits, margin, sufficient, c1) for one candidate set."""
            lg = proxy_logits(
                proxy, h_source,
                node_repr[positions] if positions else
                torch.empty(0, node_repr.shape[1], device=device))
            mg = float(lg[ref_label] - lg[1 - ref_label])
            c1 = int(lg.view(-1).argmax()) == ref_label
            return lg, mg, (c1 and mg >= alpha * static_margin), c1

        base = {
            "static_positions": sorted(static_positions),
            "static_node_ids": [node_ids[i] for i in sorted(static_positions)],
            "static_tokens": static_tokens,
            "static_logits": [float(x) for x in s_logits.view(-1)],
            "static_margin": static_margin,
            "ref_label": ref_label,
            "dual_view_agree": bool(agree),
            "full_prediction": int(p_full_probs.view(-1).argmax()),
        }

        if not agree:
            # §6: two independent views disagree -> never compress.
            base.update({
                "selected_positions": sorted(static_positions),
                "selected_node_ids": [node_ids[i]
                                      for i in sorted(static_positions)],
                "evidence_tokens": static_tokens,
                "ms_logits": [float(x) for x in s_logits.view(-1)],
                "ms_margin": static_margin,
                "margin_retention": 1.0,
                "compression_attempted": False,
                "sufficiency_reached": True,
                "fallback_to_static": True,
                "fallback_reason": "dual_view_disagreement",
                "pool_positions": sorted(static_positions),
                "pool_size": len(static_positions),
                "pool_size_before_prune": 0,
                "accepted_adds": [], "accepted_removes": [],
                "add_count": 0, "remove_count": 0})
            return base

        pool, pool_before = candidate_pool(static_positions,
                                           memory_positions, u, order, top_k)
        selected, tokens = set(), 0
        _init_logits, cur_margin, sufficient, _c1 = _info([])
        adds, removes = [], []
        step = 0
        # ---- forward construction (§11): grow from the empty set --------
        while not sufficient:
            cands = [i for i in pool if i not in selected
                     and tokens + costs[i] <= budget]
            if not cands:
                break
            masks = _mask_rows([selected | {i} for i in cands], m, device)
            batch_lg = batch_proxy_logits(proxy, h_source, node_repr, masks)
            margins = batch_decision_margins(batch_lg, ref_label).tolist()
            gains = [reader_efficiency(mg - cur_margin, costs[i])
                     for mg, i in zip(margins, cands)]
            best = max(gains)
            tied = [i for i, g in zip(cands, gains) if best - g <= 1e-9]

            def _add_key(i):
                return (0 if i in memory_set else 1, -u[i], costs[i],
                        order[i])
            pick = min(tied, key=_add_key)
            k = cands.index(pick)
            selected.add(pick)
            tokens += costs[pick]
            adds.append({"step": step, "node_id": node_ids[pick],
                         "delta_margin": float(margins[k]) - cur_margin,
                         "margin_after": float(margins[k]),
                         "token_cost": costs[pick],
                         "gain": float(gains[k]),
                         "in_previous_memory": pick in memory_set})
            step += 1
            _lg, cur_margin, sufficient, _c1 = _info(sorted(selected))
        sufficiency_reached = bool(sufficient)
        fallback = False
        reason = None
        if not sufficiency_reached:
            # no budget-feasible ADD can reach sufficiency -> keep Static
            fallback = True
            reason = "budget_exhausted_before_sufficiency"
            selected = set(static_positions)
            tokens = static_tokens
            _lg, cur_margin, sufficient, _c1 = _info(sorted(selected))

        # ---- backward redundancy removal (§12) --------------------------
        if not fallback:
            while selected:
                removable = []
                for j in sorted(selected):
                    cand = selected - {j}
                    _lg, mg, ok, _c1 = _info(sorted(cand))
                    if ok:
                        removable.append((j, mg))
                if not removable:
                    break

                def _rm_key(item):
                    j, _mg = item
                    return (-costs[j], 0 if j not in memory_set else 1,
                            u[j], order[j])
                j, mg = min(removable, key=_rm_key)
                selected = selected - {j}
                tokens -= costs[j]
                cur_margin = mg
                removes.append({"node_id": node_ids[j],
                                "token_saved": costs[j],
                                "margin_after": mg,
                                "in_previous_memory": j in memory_set})
            _lg, cur_margin, sufficient, _c1 = _info(sorted(selected))

        final = sorted(selected)
        final_logits = proxy_logits(
            proxy, h_source,
            node_repr[final] if final else
            torch.empty(0, node_repr.shape[1], device=device))
        base.update({
            "selected_positions": final,
            "selected_node_ids": [node_ids[i] for i in final],
            "evidence_tokens": tokens,
            "ms_logits": [float(x) for x in final_logits.view(-1)],
            "ms_margin": cur_margin,
            "margin_retention": (cur_margin / static_margin
                                 if static_margin > 1e-12 else None),
            "compression_attempted": True,
            "sufficiency_reached": sufficiency_reached,
            "fallback_to_static": fallback,
            "fallback_reason": reason,
            "pool_positions": pool,
            "pool_size": len(pool),
            "pool_size_before_prune": pool_before,
            "accepted_adds": adds,
            "accepted_removes": removes,
            "add_count": len(adds),
            "remove_count": len(removes)})
        return base
