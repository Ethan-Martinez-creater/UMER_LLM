"""Intervention family generator (plan §11).

Produces, for one snapshot:

* **I0** — the SRC base context (no removal);
* **I1** — one atomic Reply–Parent removal per selected unit (capped at the
  semantically strongest 20 when the SRC holds more);
* **I2** — the highest-combined-relevance parent–child pair among selected
  units;
* **I3** — its matched non-adjacent control;
* **I4** — the selected subtree whose root has the largest selected-descendant
  count, capped to the earliest five selected units;
* **I5** — its matched disconnected control.

Every structured group that the snapshot cannot supply is emitted with an
explicit ``status`` (``NO_PARENT_CHILD_PAIR`` / ``NO_SUBTREE_INTERVENTION`` /
``NO_MATCHED_CONTROL``) instead of being dropped, so downstream accounting can
see the difference between "measured zero" and "not available" (plan §11).
"""
from __future__ import annotations

from collections import defaultdict

from ..config.pilot_config import ATOMIC_CAP, PARTITION_SEED, SUBTREE_SIZE_CAP
from .interaction_controls import (match_disconnected_group,
                                   match_nonadjacent_pair, parent_map)

STATUS_OK = "OK"
NO_PARENT_CHILD_PAIR = "NO_PARENT_CHILD_PAIR"
NO_SUBTREE_INTERVENTION = "NO_SUBTREE_INTERVENTION"
NO_MATCHED_CONTROL = "NO_MATCHED_CONTROL"


def _children_map(snapshot):
    children = defaultdict(list)
    for child, parent in snapshot["edge_index"]:
        children[snapshot["node_ids"][parent]].append(snapshot["node_ids"][child])
    return children


def descendants_of(snapshot, root_id) -> set:
    """All ids in ``root_id``'s G_t subtree, including ``root_id``."""
    children = _children_map(snapshot)
    out = set()
    stack = [root_id]
    while stack:
        node = stack.pop()
        if node in out:
            continue
        out.add(node)
        stack.extend(children.get(node, []))
    return out


def _selected_ranked(src):
    """SRC-selected node ids in semantic relevance order (plan §11 I1/I2)."""
    selected = set(src["selected_node_ids"])
    return [nid for nid in src["ranked_node_ids"] if nid in selected]


def generate_interventions(snapshot, units, src, tokenizer,
                           seed: int = PARTITION_SEED) -> list:
    """All interventions for one snapshot, in fixed I0..I5 order."""
    del tokenizer  # token costs are already carried by the SRC
    costs = src["unit_token_costs"]
    relevance = src["relevance"]
    selected = list(src["selected_node_ids"])
    ranked = _selected_ranked(src)
    by_id = {u["node_id"]: u for u in units}
    pmap = parent_map(snapshot)
    out = [{
        "intervention_id": "I0",
        "type": "I0_base",
        "remove_node_ids": [],
        "status": STATUS_OK,
        "meta": {},
    }]

    # ---- I1 atomic (capped) ----
    capped = ranked[:ATOMIC_CAP]
    for nid in capped:
        out.append({
            "intervention_id": f"I1:{nid}",
            "type": "I1_atomic",
            "remove_node_ids": [nid],
            "status": STATUS_OK,
            "meta": {"atomic_cap_applied": len(ranked) > ATOMIC_CAP,
                     "src_selected_count": len(ranked),
                     "relevance": relevance.get(nid, 0.0),
                     "token_cost": costs.get(nid, 0)},
        })

    # ---- I2 parent–child pair, highest combined relevance ----
    pairs = []
    selected_set = set(selected)
    for nid in selected:
        p = pmap.get(nid)
        if p is not None and p in selected_set:
            pairs.append((p, nid))
    if pairs:
        best = max(pairs, key=lambda pr: (relevance.get(pr[0], 0.0)
                                          + relevance.get(pr[1], 0.0),
                                          -by_id[pr[0]]["snapshot_order"],
                                          -by_id[pr[1]]["snapshot_order"]))
        i2_ids = sorted(best, key=lambda n: by_id[n]["snapshot_order"])
        out.append({
            "intervention_id": "I2",
            "type": "I2_parent_child",
            "remove_node_ids": i2_ids,
            "status": STATUS_OK,
            "meta": {"combined_relevance": relevance.get(best[0], 0.0)
                     + relevance.get(best[1], 0.0),
                     "token_cost": sum(costs.get(n, 0) for n in i2_ids)},
        })
        i3 = match_nonadjacent_pair(snapshot, selected, i2_ids, costs, seed)
        out.append({
            "intervention_id": "I3",
            "type": "I3_matched_nonadjacent",
            "remove_node_ids": i3["control_ids"],
            "status": STATUS_OK if i3["control_ids"] else NO_MATCHED_CONTROL,
            "meta": {"match_level": i3["match_level"],
                     "target_ids": i3["target_ids"],
                     "target_cost": i3["target_cost"]},
        })
    else:
        out.append({
            "intervention_id": "I2", "type": "I2_parent_child",
            "remove_node_ids": [], "status": NO_PARENT_CHILD_PAIR, "meta": {},
        })
        out.append({
            "intervention_id": "I3", "type": "I3_matched_nonadjacent",
            "remove_node_ids": [], "status": NO_MATCHED_CONTROL,
            "meta": {"reason": NO_PARENT_CHILD_PAIR},
        })

    # ---- I4 selected subtree, largest selected-descendant count ----
    candidates = []
    for nid in selected:
        sub = descendants_of(snapshot, nid)
        selected_in_sub = [x for x in selected if x in sub]
        desc_selected = [x for x in selected_in_sub if x != nid]
        if len(selected_in_sub) >= 2:
            candidates.append((nid, selected_in_sub, len(desc_selected)))
    if candidates:
        # largest selected-descendant count; ties use earlier timestamp
        root, group, _ = max(
            candidates,
            key=lambda c: (c[2], -by_id[c[0]]["timestamp"],
                           -by_id[c[0]]["snapshot_order"]))
        ordered = sorted(group, key=lambda n: (by_id[n]["timestamp"],
                                               by_id[n]["snapshot_order"]))
        i4_ids = ordered[:SUBTREE_SIZE_CAP]
        out.append({
            "intervention_id": "I4",
            "type": "I4_subtree",
            "remove_node_ids": i4_ids,
            "status": STATUS_OK,
            "meta": {"subtree_root": root,
                     "selected_in_subtree": len(group),
                     "capped": len(group) > SUBTREE_SIZE_CAP,
                     "token_cost": sum(costs.get(n, 0) for n in i4_ids)},
        })
        i5 = match_disconnected_group(snapshot, selected, i4_ids, costs, seed)
        out.append({
            "intervention_id": "I5",
            "type": "I5_matched_disconnected",
            "remove_node_ids": i5["control_ids"],
            "status": STATUS_OK if i5["control_ids"] else NO_MATCHED_CONTROL,
            "meta": {"match_level": i5["match_level"],
                     "target_ids": i5["target_ids"],
                     "target_cost": i5["target_cost"]},
        })
    else:
        out.append({
            "intervention_id": "I4", "type": "I4_subtree",
            "remove_node_ids": [], "status": NO_SUBTREE_INTERVENTION, "meta": {},
        })
        out.append({
            "intervention_id": "I5", "type": "I5_matched_disconnected",
            "remove_node_ids": [], "status": NO_MATCHED_CONTROL,
            "meta": {"reason": NO_SUBTREE_INTERVENTION},
        })
    return out


def intervention_summary(interventions) -> dict:
    """Counts by status for the snapshot audit."""
    counts = defaultdict(int)
    for iv in interventions:
        counts[iv["status"]] += 1
    return {"n_interventions": len(interventions), "by_status": dict(counts)}
