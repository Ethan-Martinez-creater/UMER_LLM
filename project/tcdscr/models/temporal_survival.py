"""MF-TSR temporal memory: conditional evidence survival (Dynamic V2 §11).

The previous MF-TSR set M_previous is a *warm start* only: no persistence
bonus is ever added to a score.  A historical evidence survives into the
current snapshot iff its node still exists there; when the recomputed token
costs push the survivor set over the budget, survivors are dropped in
ascending current Static Utility (fully deterministic, never random), and
the remaining budget is then filled with unselected current evidence in
descending Static Utility until the next whole Reply-Parent Pair no longer
fits (stop, not skip-and-continue: the warm start must stay a prefix of the
descending-utility fill so temporal continuity is meaningful).

Memory stores node positions/ids only; representations, utilities and token
costs are always recomputed from the CURRENT snapshot (design §21 of the
execution protocol).  No trainable parameter lives here.
"""
from __future__ import annotations


def feasible_memory_set(memory_positions, costs, u, order, budget):
    """Survivors of M_previous inside the current snapshot, budget-feasible.

    ``memory_positions`` lists current-snapshot candidate positions of the
    previous final set (membership already resolved by the caller).  While
    the total cost exceeds ``budget`` the survivor with the lowest current
    Static Utility is removed; ties break on smaller snapshot order first
    (deterministic).
    """
    sel = list(memory_positions)
    total = sum(costs[i] for i in sel)
    while total > budget and sel:
        drop = min(sel, key=lambda i: (u[i], order[i]))
        sel.remove(drop)
        total -= costs[drop]
    return sel, total


def memory_warm_start(memory_positions, costs, u, order, budget):
    """S_memory_init of execution-protocol §11.

    1. keep only memory evidence still present in the snapshot (caller
       resolved), trim to budget via ``feasible_memory_set``;
    2. fill the remaining budget with the unselected current evidence in
       descending Static Utility (ties: earlier snapshot order first),
       stopping at the first whole Reply-Parent Pair that does not fit.

    Returns (selected_positions, total_tokens).
    """
    sel, total = feasible_memory_set(memory_positions, costs, u, order,
                                     budget)
    chosen = set(sel)
    rest = [i for i in range(len(costs)) if i not in chosen]
    rest.sort(key=lambda i: (-u[i], order[i]))
    for i in rest:
        if total + costs[i] > budget:
            break
        sel.append(i)
        total += costs[i]
    return sel, total
