"""Ma-Weibo raw-event adapter (plan §13).

Reads::

    <raw_dir>/<eid>.json          — list of post dicts (mid/id, parent, t, ...)
    <label_file>                  — lines "eid:<int> label:<int>"

Historical field behavior preserved: node text prefers ``original_text`` and
falls back to ``text`` only when the field is absent (this is exactly the
behavior verified by the D6 parity run, original_text coverage 99.9996%).
Parent resolution 99.9992%, cycle events 0, multi-root events 0 — the adapter
raises on multi-root because that frozen audit result changing means the
input changed.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterator

from .normalized_schema import SchemaError, validate_event


def parse_labels(label_file: str) -> dict:
    """Parse Weibo.txt into {eid:int -> label:int}."""
    labels = {}
    with open(label_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"eid:(\d+)\s+label:(\d+)", line)
            if m:
                labels[m.group(1)] = int(m.group(2))
    return labels


def _node_text(post: dict):
    """Return (text, source) where source is 'original_text' or 'fallback'.

    Historical behavior preserved: original_text is preferred; text is used
    only when the field is absent (D6-verified behavior). The source is
    reported per event so the fallback is audited, never silent.
    """
    if "original_text" in post:
        return str(post.get("original_text") or ""), "original_text"
    return str(post.get("text") or ""), "fallback"


def load_event(eid: str, label: int, path: str) -> dict:
    """Parse one Ma-Weibo event JSON file into the normalized schema."""
    with open(path, encoding="utf-8") as fh:
        posts = json.load(fh)

    roots = []
    raw_nodes = []
    text_counts = {"total_nodes": 0, "original_text_used": 0,
                   "text_fallback_count": 0, "empty_text_count": 0}
    for order, p in enumerate(posts):
        pid = p.get("mid", p.get("id"))
        if pid is None or p.get("t") is None:
            raise SchemaError(f"event {eid}: post without mid/id or t")
        parent = p.get("parent", None)
        parent_id = str(parent) if parent is not None else None
        text, text_source = _node_text(p)
        text_counts["total_nodes"] += 1
        text_counts["original_text_used" if text_source == "original_text"
                    else "text_fallback_count"] += 1
        if not text.strip():
            text_counts["empty_text_count"] += 1
        raw_nodes.append({
            "node_id": str(pid),
            "parent_id": parent_id,
            "timestamp": int(p["t"]),
            "text": text,
            "original_order": order,
        })
        if parent is None:
            roots.append(str(pid))

    if not roots:
        raise SchemaError(f"event {eid}: no root post")
    if len(roots) > 1:
        raise SchemaError(
            f"event {eid}: multi-root event (frozen audit says 0 exist)")
    source_id = roots[0]

    known = {n["node_id"] for n in raw_nodes}
    source_ts = next(n["timestamp"] for n in raw_nodes
                     if n["node_id"] == source_id)
    nodes = []
    for n in raw_nodes:
        status = "VALID"
        if n["node_id"] != source_id:
            if n["parent_id"] is None:
                status = "MISSING_PARENT"
            elif n["parent_id"] not in known:
                status = "EXTERNAL_PARENT"
        if n["node_id"] != source_id and n["timestamp"] < source_ts:
            status = "TEMPORAL_INVALID_NODE"
        nodes.append({**n, "status": status})

    event = {
        "event_id": eid,
        "label": label,
        "source_id": source_id,
        "source_timestamp": source_ts,
        "nodes": nodes,
        "text_source_counts": text_counts,
    }
    return validate_event(event)


def aggregate_text_fallback(events) -> dict:
    """Dataset-level fallback audit (delta-fix §34/§35)."""
    events = list(events)
    total = {"total_nodes": 0, "original_text_used": 0,
             "text_fallback_count": 0, "empty_text_count": 0}
    events_with_fallback = 0
    for ev in events:
        counts = ev.get("text_source_counts")
        if counts is None:
            raise ValueError(
                f"event {ev.get('event_id')!r} has no text_source_counts; "
                "re-generate with the audited adapter")
        for key in total:
            total[key] += counts[key]
        if counts["text_fallback_count"]:
            events_with_fallback += 1
    n_nodes = total["total_nodes"]
    return {
        **total,
        "events": len(events),
        "events_with_fallback": events_with_fallback,
        "fallback_rate": (total["text_fallback_count"] / max(n_nodes, 1)),
    }


def iter_events(raw_dir: str, label_file: str) -> Iterator[dict]:
    """Yield every Ma-Weibo event in sorted eid order."""
    labels = parse_labels(label_file)
    for f in sorted(os.listdir(raw_dir)):
        if not f.endswith(".json"):
            continue
        eid = f[:-5]
        if eid not in labels:
            continue
        yield load_event(eid, labels[eid], os.path.join(raw_dir, f))


def event_ids(raw_dir: str, label_file: str):
    """Deterministic sorted list of (eid, label) available on disk."""
    labels = parse_labels(label_file)
    out = []
    for f in sorted(os.listdir(raw_dir)):
        if f.endswith(".json"):
            eid = f[:-5]
            if eid in labels:
                out.append((eid, labels[eid]))
    return out
