"""Reply–Parent evidence units (plan §3.1.C, §7).

One ``Reply + Parent`` pair per visible non-source reply. The reply node is
the graph identity of the unit; the parent text is ``None`` when the snapshot
has no resolved parent edge for that node and is rendered ``[UNAVAILABLE]``
downstream — never invented.

Units are produced in snapshot order, which is timestamp ascending with ties
broken by the adapter's deterministic order (plan §3.1.B). Rendering reuses the
frozen TC-DSCR reader prompt so the token accounting cannot drift from the text
the readers actually see (plan §29).
"""
from __future__ import annotations

from tcdscr.llm import reader_prompt

__all__ = ["build_evidence_units", "unit_token_cost", "canonical_token_costs",
           "render_units_for_budget", "unit_index_map"]


def build_evidence_units(snapshot: dict) -> list:
    """``[unit, ...]`` in snapshot order (plan §3.1.C, §7)."""
    node_ids = snapshot["node_ids"]
    texts = snapshot["texts"]
    parent_ids = snapshot["parent_ids"]
    ts = snapshot["timestamps"]
    elapsed = snapshot["elapsed_seconds"]
    source_id = snapshot["source_id"]
    pos = {nid: i for i, nid in enumerate(node_ids)}
    units = []
    for i, nid in enumerate(node_ids):
        if nid == source_id:
            continue
        parent_id = parent_ids[i]
        parent_text = None
        if parent_id is not None and parent_id in pos:
            parent_text = texts[pos[parent_id]]
        units.append({
            "node_id": nid,
            "reply_text": texts[i],
            "parent_text": parent_text,
            "parent_id": parent_id,
            "timestamp": ts[i],
            "elapsed_seconds": elapsed[i],
            "snapshot_order": i,
            "depth": snapshot["depths"][i],
        })
    return units


def unit_token_cost(tokenizer, unit: dict) -> int:
    """Canonical social-evidence tokens of one complete unit (plan §7).

    Uses the frozen rendering with a fixed id so the cost measures content
    only; the ``E<i>`` id assigned inside a specific arm is counted by the
    arm's own final render check.
    """
    return reader_prompt.count_tokens(
        tokenizer, reader_prompt.render_evidence(unit, 0))


def unit_index_map(units) -> dict:
    """``{node_id: position}`` following the unit list order."""
    return {u["node_id"]: i for i, u in enumerate(units)}


def canonical_token_costs(tokenizer, units) -> dict:
    """``{node_id: canonical token cost}`` for every unit (plan §7, §14)."""
    return {u["node_id"]: unit_token_cost(tokenizer, u) for u in units}


def render_units_for_budget(units, selected_node_ids) -> str:
    """Render exactly the selected units, preserving given order (plan §21).

    The plan fixes chronological presentation, so callers pass ids in snapshot
    order; this function never re-sorts.
    """
    wanted = set(selected_node_ids)
    ordered = [u for u in units if u["node_id"] in wanted]
    return reader_prompt.render_evidence_block(ordered)
