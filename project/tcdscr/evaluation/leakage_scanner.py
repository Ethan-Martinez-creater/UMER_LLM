"""Leakage scanner (plan §33/§29).

Static checks over a rendered prompt against its source event:
  - future text: a node whose timestamp is beyond the snapshot's max may not
    contribute NEW text to the prompt. A future node's text is counted as a
    hit only when it is NOT a substring of any causally-included node text —
    forward texts commonly quote the source (reposts, template replies), and
    that content is already visible through a legal path.
  - gold leakage: the prompt must not contain gold-label markers.

Whitespace-normalized substring matching keeps the check deterministic.
Fragments shorter than MIN_FRAG_LEN characters carry no identifiable
information (they substring-match task words like "claim") and are exempt.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
MIN_FRAG_LEN = 4


def _norm(text: str) -> str:
    return _WS.sub(" ", text or "").strip().lower()


def scan_prompt(prompt: str, event: dict, snapshot: dict) -> dict:
    prompt_norm = _norm(prompt)
    max_ts = max(snapshot["timestamps"]) if snapshot["timestamps"] else 0
    included_norm = [_norm(t) for t in snapshot["texts"] if t and t.strip()]

    future_hits = []
    for node in event["nodes"]:
        text = node["text"]
        if not text or not text.strip():
            continue
        if node["timestamp"] > max_ts:
            frag = _norm(text)
            if len(frag) >= MIN_FRAG_LEN and frag in prompt_norm \
                    and not any(frag in inc for inc in included_norm):
                future_hits.append({
                    "node_id": node["node_id"],
                    "timestamp": node["timestamp"],
                    "covered_by_included_text": False,
                })

    gold_hits = []
    if _norm("gold") in prompt_norm:
        gold_hits.append("gold keyword present")
    for marker in (f"label: {event['label']}", f"answer: {event['label']}",
                   f"label={event['label']}"):
        if _norm(marker) in prompt_norm:
            gold_hits.append(marker)

    return {
        "pass": not future_hits and not gold_hits,
        "future_text_hits": future_hits,
        "gold_hits": gold_hits,
        "checked_nodes": len(event["nodes"]),
        "snapshot_max_timestamp": max_ts,
    }


def scan_predictions(predictions) -> dict:
    """Aggregate over prediction rows that embed a ``leakage`` field."""
    total = len(predictions)
    failing = [p for p in predictions
               if p.get("leakage") is not None and not p["leakage"]["pass"]]
    return {"checked": total, "failed": len(failing),
            "pass": len(failing) == 0}
