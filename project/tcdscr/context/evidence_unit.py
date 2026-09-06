"""Evidence unit: the minimal Reply-Parent Pair (plan §20).

Every reply node in the snapshot becomes one unit; the source is always
presented separately and never appears inside a pair. Resolved parents show
their text; anything unresolved (missing/external parent, temporally invalid
parent, child-before-parent) shows ``[UNAVAILABLE]``. No full path, no
summarization, no stance annotation, no LLM relevance annotation.
"""
from __future__ import annotations

UNAVAILABLE = "[UNAVAILABLE]"


def build_evidence_units(snapshot: dict) -> list:
    """One unit per reply node, in deterministic snapshot order."""
    source_id = snapshot["source_id"]
    texts = dict(zip(snapshot["node_ids"], snapshot["texts"]))
    depths = snapshot["depths"]
    elapsed = snapshot["elapsed_seconds"]
    units = []
    for i, nid in enumerate(snapshot["node_ids"]):
        if nid == source_id:
            continue
        parent_id = snapshot["parent_ids"][i]
        units.append({
            "node_id": nid,
            "reply_text": texts[nid],
            "parent_id": parent_id,
            "parent_text": texts[parent_id] if parent_id is not None else None,
            "elapsed_seconds": elapsed[i],
            "depth": depths[i],
            "order": i,
        })
    return units


def format_time(seconds: int) -> str:
    return f"{int(seconds)}s"


def render_evidence(unit: dict, index: int) -> str:
    """Render one unit as [Ei]; unresolved parents print [UNAVAILABLE]."""
    parent_text = unit["parent_text"]
    if parent_id_is_unavailable(unit):
        parent_text = UNAVAILABLE
    lines = [
        f"[E{index}]",
        f"time = {format_time(unit['elapsed_seconds'])}",
        f"depth = {unit['depth']}",
        "parent:",
        parent_text,
        "reply:",
        unit["reply_text"],
    ]
    return "\n".join(lines)


def parent_id_is_unavailable(unit: dict) -> bool:
    return unit["parent_id"] is None or unit["parent_text"] is None


def selected_unit_embeddings(units, sem_matrix, node_ids):
    """Semantic rows for a subset of units, aligned to ``units`` order."""
    index = {nid: i for i, nid in enumerate(node_ids)}
    rows = [index[u["node_id"]] for u in units]
    return sem_matrix[rows]
