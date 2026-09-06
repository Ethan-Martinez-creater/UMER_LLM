"""Context packer (plan §22).

Assembles the frozen prompt: source claim, current-snapshot stats (within the
snapshot only — no future counts, no final depth), and the selected evidence
units numbered [E1]..[Ek] in acceptance order. Text passed to the LLM is the
raw social text (no additional normalization); the frozen §7 cleaning applies
to the semantic pipeline only. This decision is recorded in the implementation
notes and known_issues.
"""
from __future__ import annotations

from ..config.schema import MAX_NODES
from .evidence_unit import render_evidence
from .prompts import build_prompt
from .token_budget import format_cutoff


def snapshot_stats(snapshot: dict) -> dict:
    n = len(snapshot["node_ids"])
    observed_replies = 0 if snapshot["cutoff_minutes"] == "SOURCE_ONLY" \
        else n - 1
    max_depth = max(snapshot["depths"]) if snapshot["depths"] else 0
    max_depth = max(max_depth, 0)
    if n > MAX_NODES:
        raise ValueError("snapshot exceeded the 1021-node cap")
    return {"observed_replies": observed_replies, "max_depth": max_depth}


def pack_context(source_text: str, snapshot: dict, selected_units,
                 template_lang: str = "en") -> dict:
    evidence_block = "\n\n".join(
        render_evidence(u, i + 1) for i, u in enumerate(selected_units))
    stats = snapshot_stats(snapshot)
    prompt = build_prompt(
        template_lang=template_lang,
        source_text=source_text,
        elapsed_label=format_cutoff(snapshot["cutoff_minutes"]),
        observed_replies=stats["observed_replies"],
        max_depth=stats["max_depth"],
        evidence_block=evidence_block,
    )
    return {
        "prompt": prompt,
        "evidence_block": evidence_block,
        "selected_node_ids": [u["node_id"] for u in selected_units],
        "num_evidence_units": len(selected_units),
        "observed_replies": stats["observed_replies"],
        "max_depth": stats["max_depth"],
    }
