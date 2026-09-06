"""PHEME raw-thread adapter (plan §13).

Reads the all-rnr-annotated-threads layout::

    <raw_dir>/<topic>/<rumours|non-rumours>/<thread_id>/source-tweets/*.json
    <raw_dir>/<topic>/<rumours|non-rumours>/<thread_id>/reactions/*.json

Adapters normalize fields only — text cleaning and feature computation happen
later in the pipeline. Timestamp parse failures raise immediately because the
frozen V2 data audit proved 100% timestamp coverage; a parse failure means the
input changed and must stop the pipeline rather than be silently dropped.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterator

from .normalized_schema import SchemaError, validate_event


def parse_twitter_time(s: str) -> int:
    """Parse Twitter created_at into a unix second integer."""
    dt = datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
    return int(dt.timestamp())


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _iter_thread_dirs(raw_dir: str) -> Iterator[tuple]:
    for topic in sorted(d for d in os.listdir(raw_dir)
                        if os.path.isdir(os.path.join(raw_dir, d))):
        for class_dir, label in (("rumours", 1), ("non-rumours", 0)):
            cpath = os.path.join(raw_dir, topic, class_dir)
            if not os.path.isdir(cpath):
                continue
            for thread_dir in sorted(os.listdir(cpath)):
                folder = os.path.join(cpath, thread_dir)
                if os.path.isdir(folder):
                    yield topic, label, folder


def _first_source_file(src_dir):
    for f in sorted(os.listdir(src_dir)):
        if f.endswith(".json") and not f.startswith("._"):
            return os.path.join(src_dir, f)
    return None


def load_event(topic: str, label: int, folder: str) -> dict:
    """Parse one PHEME thread folder into the normalized event schema."""
    src_file = _first_source_file(os.path.join(folder, "source-tweets"))
    if src_file is None:
        raise SchemaError(f"no source tweet in {folder}")
    src = _load_json(src_file)
    source_id = str(src["id_str"])
    source_ts = parse_twitter_time(src["created_at"])

    nodes = [{
        "node_id": source_id,
        "parent_id": None,
        "timestamp": source_ts,
        "text": str(src.get("text", "")),
        "original_order": 0,
        "status": "VALID",
    }]

    reactions_dir = os.path.join(folder, "reactions")
    files = []
    if os.path.isdir(reactions_dir):
        files = sorted(f for f in os.listdir(reactions_dir)
                       if f.endswith(".json") and not f.startswith("._"))
    for order, f in enumerate(files, start=1):
        r = _load_json(os.path.join(reactions_dir, f))
        # 100% reply timestamp coverage is a frozen data fact; fail loudly.
        reply_ts = parse_twitter_time(r["created_at"])
        reply_id = str(r["id_str"])
        parent_raw = r.get("in_reply_to_status_id_str")
        parent_id = str(parent_raw) if parent_raw else None
        status = "VALID"
        if parent_id is None:
            status = "MISSING_PARENT"
        nodes.append({
            "node_id": reply_id,
            "parent_id": parent_id,
            "timestamp": reply_ts,
            "text": str(r.get("text", "")),
            "original_order": order,
            "status": status,
        })

    # Second pass: EXTERNAL_PARENT for replies pointing outside the event,
    # TEMPORAL_INVALID_NODE for nodes earlier than the source timestamp.
    known = {n["node_id"] for n in nodes}
    for n in nodes[1:]:
        if n["parent_id"] is not None and n["parent_id"] not in known:
            n["status"] = "EXTERNAL_PARENT"
        if n["timestamp"] < source_ts:
            n["status"] = "TEMPORAL_INVALID_NODE"

    event = {
        "event_id": source_id,
        "label": label,
        "source_id": source_id,
        "source_timestamp": source_ts,
        "nodes": nodes,
        "topic": topic,
    }
    return validate_event(event)


def iter_events(raw_dir: str) -> Iterator[dict]:
    """Yield every PHEME event in deterministic (topic, thread) order."""
    for topic, label, folder in _iter_thread_dirs(raw_dir):
        yield load_event(topic, label, folder)


def event_ids(raw_dir: str):
    """Deterministic list of (event_id, topic, label, folder) without parsing."""
    out = []
    for topic, label, folder in _iter_thread_dirs(raw_dir):
        src_file = _first_source_file(os.path.join(folder, "source-tweets"))
        if src_file is None:
            continue
        out.append((os.path.splitext(os.path.basename(src_file))[0],
                    topic, label, folder))
    return out
