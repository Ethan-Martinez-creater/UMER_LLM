"""Weibo22 raw-field adapter and P0 feasibility audit (plan §4.1, §30).

The plan requires **true per-node timestamps** for Weibo22. It explicitly
forbids pseudo-time derived from row/node order (§4.1: "No pseudo-time based on
row/node order is allowed").

The publicly released KPG/TD-RvNN files that this project has access to carry
only: event id, binary label, topic, a parent-index chain and vol_5000
vocabulary indices. They carry **no source text, no reply text and no absolute
timestamp of any kind**. The adapter therefore does not invent values: it
audits the release and raises :class:`Weibo22TemporalUnavailable` when the
fields required by the plan are absent.

A second, explicit entry point (:func:`load_normalized_event_files`) reads a
per-event normalized JSONL export *if* a timestamp-bearing release is supplied
later. That path is the one the rest of the pipeline consumes; the raw KPG path
can never silently reach it.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Iterator

from ..config.pilot_config import (CUTOFFS_MIN, VERDICT_READY,
                                   VERDICT_UNAVAILABLE)

LABEL_TOKENS = {"true": 1, "false": 0, "1": 1, "0": 0}


class Weibo22TemporalUnavailable(RuntimeError):
    """Raised when Weibo22 cannot supply true temporal fields (plan §4.1)."""

    verdict = VERDICT_UNAVAILABLE

    def __init__(self, message: str, audit: dict | None = None):
        super().__init__(message)
        self.audit = audit or {}


# --------------------------------------------------------------------------
# Raw KPG / TD-RvNN release parsing (structure only)
# --------------------------------------------------------------------------
def parse_label_file(path: str) -> dict:
    """Parse ``Weibo_label_All.txt``: ``label <TAB> topic <TAB> weibo_id``."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"label row {lineno} has {len(parts)} columns, expected 3")
            raw_label, topic, wid = parts[0].strip(), parts[1], parts[2].strip()
            if raw_label not in LABEL_TOKENS:
                raise ValueError(
                    f"label row {lineno}: unknown label {raw_label!r}")
            if wid in out:
                raise ValueError(f"duplicate weibo_id {wid!r} in label file")
            out[wid] = {"label": LABEL_TOKENS[raw_label], "topic": topic}
    return out


def _split_tree_row(line: str):
    """Split one TD-RvNN row into structural tokens and text tokens."""
    parts = line.rstrip("\n").split("\t")
    struct = [p for p in parts[1:] if ":" not in p]
    text = [p for p in parts[1:] if ":" in p]
    return struct, text


def read_tree_file(path: str) -> dict:
    """Parse the TD-RvNN tree file into ``{weibo_id: [row_dict, ...]}``.

    Each row keeps only what the release actually contains: a parent index
    (or ``None`` for the root), an in-file node index, and the structural
    trailing scalars. No timestamp field is synthesized.
    """
    trees: dict = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            struct, text = _split_tree_row(line)
            if not struct:
                raise ValueError(f"tree row {lineno}: no structural token")
            wid = line.split("\t", 1)[0].strip()
            parent_raw = struct[0]
            try:
                parent = None if parent_raw == "None" else int(parent_raw)
            except ValueError as exc:
                raise ValueError(
                    f"tree row {lineno}: bad parent {parent_raw!r}") from exc
            try:
                index = int(struct[1]) if len(struct) > 1 else None
            except ValueError as exc:
                raise ValueError(
                    f"tree row {lineno}: bad node index {struct[1]!r}") from exc
            trailing = []
            for tok in struct[2:]:
                try:
                    trailing.append(int(tok))
                except ValueError:
                    trailing.append(None)
            trees.setdefault(wid, []).append({
                "line": lineno,
                "parent_index": parent,
                "node_index": index,
                "trailing": trailing,
                "n_text_tokens": len(text),
            })
    return trees


# --------------------------------------------------------------------------
# P0 raw-field audit (plan §30)
# --------------------------------------------------------------------------
def _coverage(n_present: int, n_total: int) -> float:
    return (n_present / n_total) if n_total else 0.0


def audit_release(base_dir: str) -> dict:
    """Full plan-§30 raw-field audit of a KPG/TD-RvNN release directory.

    Expected layout (as shipped by the KPG release)::

        <base_dir>/label_all/Weibo_label_All.txt
        <base_dir>/td_rvnn/data.TD_RvNN.vol_5000.txt

    Reports exactly the fields §30 lists, plus a ``verdict``. Fields that the
    release does not contain are reported as 0/None — never guessed.
    """
    label_path = os.path.join(base_dir, "label_all", "Weibo_label_All.txt")
    tree_path = os.path.join(base_dir, "td_rvnn",
                             "data.TD_RvNN.vol_5000.txt")
    missing = [p for p in (label_path, tree_path) if not os.path.exists(p)]
    if missing:
        return {
            "dataset": "Weibo22",
            "verdict": VERDICT_UNAVAILABLE,
            "verdict_reason": "raw release files not found",
            "missing_files": missing,
        }

    labels = parse_label_file(label_path)
    trees = read_tree_file(tree_path)

    label_counts = Counter(v["label"] for v in labels.values())
    topic_counts = Counter(v["topic"] for v in labels.values())

    n_events = len(trees)
    n_rows = sum(len(rows) for rows in trees.values())
    roots = 0
    duplicate_node_index = 0
    unresolvable_parent = 0
    child_before_parent = 0
    negative_ts = 0
    nodes_with_text = 0
    nodes_with_ts = 0
    events_with_reply = 0
    trees_without_label = 0
    viable = {str(c): 0 for c in CUTOFFS_MIN}

    for wid, rows in trees.items():
        if wid not in labels:
            trees_without_label += 1
        index_to_row = {}
        has_index = True
        for row in rows:
            idx = row["node_index"]
            if idx is None:
                has_index = False
                continue
            if idx in index_to_row:
                duplicate_node_index += 1
            index_to_row[idx] = row
            if row["parent_index"] is None:
                roots += 1
        if len(rows) > 1:
            events_with_reply += 1
        for row in rows:
            if row["parent_index"] is not None:
                parent_row = index_to_row.get(row["parent_index"])
                if parent_row is None:
                    unresolvable_parent += 1
            if row["n_text_tokens"] > 0:
                nodes_with_text += 1
            # The release has no timestamp column at all; these stay zero and
            # the verdict below says so explicitly.
            if row.get("timestamp") is not None:
                nodes_with_ts += 1
                if row["timestamp"] < 0:
                    negative_ts += 1
                if row["parent_index"] is not None:
                    parent_row = index_to_row.get(row["parent_index"])
                    if parent_row and parent_row.get("timestamp") is not None \
                            and parent_row["timestamp"] > row["timestamp"]:
                        child_before_parent += 1

    source_text_coverage = 0.0
    reply_text_coverage = 0.0
    timestamp_coverage = _coverage(nodes_with_ts, n_rows)
    parent_coverage = _coverage(
        n_rows - unresolvable_parent - roots, n_rows)

    temporal_ready = (
        timestamp_coverage > 0.0
        and parent_coverage > 0.0
        and (source_text_coverage > 0.0 or reply_text_coverage > 0.0)
    )
    audit = {
        "dataset": "Weibo22",
        "base_dir": base_dir,
        "event_count": n_events,
        "label_distribution": {str(k): v for k, v in sorted(label_counts.items())},
        "topic_distribution": dict(sorted(topic_counts.items())),
        "source_text_coverage": source_text_coverage,
        "reply_text_coverage": reply_text_coverage,
        "timestamp_coverage": timestamp_coverage,
        "parent_coverage": parent_coverage,
        "duplicate_ids": duplicate_node_index,
        "negative_timestamps": negative_ts,
        "child_earlier_than_parent_count": child_before_parent,
        "unresolvable_parent_rate": _coverage(unresolvable_parent, n_rows),
        "events_with_ge1_valid_reply": events_with_reply,
        "events_viable_15m": viable["15"],
        "events_viable_1h": viable["60"],
        "events_viable_6h": viable["360"],
        "release_defects": {
            "raw_text_available": False,
            "absolute_node_timestamp": False,
            "source_absolute_timestamp": False,
            "vocabulary_index_text_only": True,
            "trees_without_label": trees_without_label,
            "has_node_index": has_index if trees else None,
        },
        "verdict": VERDICT_READY if temporal_ready else VERDICT_UNAVAILABLE,
    }
    if not temporal_ready:
        audit["verdict_reason"] = (
            "the released KPG/TD-RvNN files contain no source/reply text and "
            "no absolute source or node timestamp; the plan forbids pseudo-time "
            "from row or node order (plan §4.1), so no snapshot can be built")
    return audit


def write_audit(base_dir: str, out_dir: str) -> dict:
    """Run the §30 audit and persist JSON + Markdown; returns the audit."""
    audit = audit_release(base_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "weibo22_audit.json"), "w",
              encoding="utf-8") as fh:
        json.dump(audit, fh, indent=1, ensure_ascii=False)
    lines = ["# Weibo22 raw-field audit (plan §30)", ""]
    for key in ("event_count", "label_distribution", "source_text_coverage",
                "reply_text_coverage", "timestamp_coverage",
                "parent_coverage", "duplicate_ids", "negative_timestamps",
                "child_earlier_than_parent_count", "unresolvable_parent_rate",
                "events_with_ge1_valid_reply", "events_viable_15m",
                "events_viable_1h", "events_viable_6h", "verdict"):
        lines.append(f"- **{key}**: {audit.get(key)}")
    if audit.get("verdict_reason"):
        lines += ["", f"> {audit['verdict_reason']}"]
    with open(os.path.join(out_dir, "weibo22_audit.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return audit


# --------------------------------------------------------------------------
# Normalized JSONL export path (the only path the pipeline consumes)
# --------------------------------------------------------------------------
def load_normalized_event_files(paths, limit=None) -> list:
    """Load normalized per-event JSONL files (plan §3.1.A schema).

    Each file/line is one normalized event dict. This is the entry point used
    once a timestamp-bearing Weibo22 export exists; it performs no invention
    and delegates validation to the shared TC-DSCR schema.
    """
    from tcdscr.data.normalized_schema import validate_event

    if isinstance(paths, str):
        paths = [paths]
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(os.path.join(p, f) for f in os.listdir(p)
                                if f.endswith(".json") or f.endswith(".jsonl")))
        else:
            files.append(p)
    events = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        stripped = text.lstrip()
        if stripped.startswith("["):
            rows = json.loads(stripped)
        else:
            rows = [json.loads(line) for line in text.splitlines()
                    if line.strip()]
        for row in rows:
            events.append(validate_event(row))
            if limit is not None and len(events) >= limit:
                return events
    return events


def load_events(base_dir: str, normalized_paths=None) -> list:
    """Load Weibo22 events for the pilot, or fail closed.

    ``normalized_paths`` is the timestamp-bearing export. When it is absent or
    empty this raises :class:`Weibo22TemporalUnavailable` carrying the §30
    audit, because the raw KPG release cannot satisfy the plan's temporal
    requirement.
    """
    if normalized_paths:
        return load_normalized_event_files(normalized_paths)
    audit = audit_release(base_dir) if base_dir else {
        "verdict": VERDICT_UNAVAILABLE,
        "verdict_reason": "no Weibo22 normalized export supplied",
    }
    raise Weibo22TemporalUnavailable(
        f"Weibo22 temporal fields unavailable ({audit.get('verdict')}): "
        f"{audit.get('verdict_reason', '')}", audit=audit)


def dataset_event_ids(base_dir: str) -> list:
    """Sorted ``[(event_id, label)]`` from the release, structure only."""
    labels = parse_label_file(
        os.path.join(base_dir, "label_all", "Weibo_label_All.txt"))
    return sorted((wid, v["label"]) for wid, v in labels.items())
