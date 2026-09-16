"""Probe context construction — P0/P1/P2/P3 (plan §6.2).

Four contexts are built per probe snapshot, **reader-independently** and
without looking at any utility outcome:

``P0``  source only
``P1``  source + the highest semantic-relevance valid evidence unit
``P2``  source + the lowest semantic-relevance unit among the SRC-selected ones
``P3``  source + the full SRC under the frozen 1024 canonical-token budget

``P1``/``P2`` are the two ends of the frozen SRC relevance ordering, so they
are selected from what the frozen CR-TSER ``build_src`` produced — this module
never recomputes relevance, it only reads ``ranked_node_ids`` /
``selected_node_ids`` / ``relevance``.

M0 implements and tests this selection; it does not call a reader.
"""
from __future__ import annotations

from ..config import protocol as P

try:  # reused read-only from the frozen CR-TSER implementation
    from cr_tser.intervention.evidence_units import render_units_for_budget
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "BCR reuses cr_tser.intervention.evidence_units read-only") from exc


class ProbeContextRefused(RuntimeError):
    """Raised when a probe context cannot be constructed as specified."""


def _unit_index(units) -> dict:
    return {u["node_id"]: u for u in units}


def highest_relevance_unit(src: dict) -> str:
    """Highest semantic-relevance valid evidence unit (``P1``)."""
    ranked = list(src.get("ranked_node_ids") or [])
    if not ranked:
        raise ProbeContextRefused("snapshot has no ranked evidence unit")
    return ranked[0]


def lowest_relevance_selected_unit(src: dict) -> str:
    """Lowest-relevance unit **among the SRC-selected ones** (``P2``)."""
    selected = list(src.get("selected_node_ids") or [])
    if not selected:
        raise ProbeContextRefused("snapshot has no SRC-selected unit")
    rank_position = {nid: i for i, nid in enumerate(src.get("ranked_node_ids")
                                                   or [])}
    missing = [nid for nid in selected if nid not in rank_position]
    if missing:
        raise ProbeContextRefused(
            f"SRC-selected units missing from the ranking: {missing[:3]}")
    return max(selected, key=lambda nid: (rank_position[nid], nid))


def context_node_ids(units, src: dict, context: str) -> list:
    """Node ids of one probe context, in the order the plan prescribes."""
    if context not in P.PROBE_CONTEXTS:
        raise ProbeContextRefused(f"unknown probe context {context!r}")
    if context == "P0":
        return []
    if context == "P1":
        return [highest_relevance_unit(src)]
    if context == "P2":
        return [lowest_relevance_selected_unit(src)]
    return list(src.get("selected_node_ids") or [])


def build_probe_contexts(units, src: dict) -> dict:
    """All four contexts of one probe snapshot (no reader call)."""
    index = _unit_index(units)
    contexts = {}
    for context in P.PROBE_CONTEXTS:
        node_ids = context_node_ids(units, src, context)
        unknown = [nid for nid in node_ids if nid not in index]
        if unknown:
            raise ProbeContextRefused(
                f"{context}: node ids outside the snapshot: {unknown[:3]}")
        contexts[context] = {
            "context": context,
            "requirement": P.PROBE_CONTEXT_REQUIREMENTS[context],
            "node_ids": list(node_ids),
            "n_units": len(node_ids),
            "evidence_block": render_units_for_budget(units, node_ids),
        }
    for context in ("P1", "P2", "P3"):
        if contexts[context]["n_units"] < 1:
            raise ProbeContextRefused(
                f"{context}: empty context; the probe manifest requires at "
                "least one evidence unit")
    if contexts["P0"]["n_units"] != 0:
        raise ProbeContextRefused("P0 must be source-only")
    if contexts["P3"]["node_ids"] != list(src.get("selected_node_ids") or []):
        raise ProbeContextRefused("P3 must be the full SRC in snapshot order")
    return contexts


def context_plan() -> dict:
    """The frozen context contract, for reports and the verifier."""
    return {
        "contexts": list(P.PROBE_CONTEXTS),
        "requirements": dict(P.PROBE_CONTEXT_REQUIREMENTS),
        "selection_is_reader_independent": True,
        "selection_uses_utility_outcomes": False,
        "evidence_source": "frozen CR-TSER build_src (relevance ranking + SRC)",
    }
