"""CR-TSER Ma-Weibo dataset bridge (amendment V2 §5–§8, §24).

The raw parser is **not** reimplemented here. This module reuses the
TC-DSCR-audited ``tcdscr.data.maweibo_adapter`` — which already reads
``post["t"]`` as the only timestamp, prefers ``original_text`` with a ``text``
fallback, and raises on a missing root or a multi-root event — and adds only
the CR-TSER V2 eligibility contract (amendment §7):

* ``VALID`` — may enter snapshots, evidence units and interventions;
* ``EMPTY_TEXT`` — kept for audit/topology accounting, but may never form a
  textual evidence unit;
* ``MISSING_PARENT`` / ``EXTERNAL_PARENT`` — kept for audit, never attached to
  the source and never used for a Reply–Parent unit;
* ``TEMPORAL_INVALID_NODE`` — may not enter a causal snapshot;
* duplicate node ids make the whole event invalid and are never re-numbered.

Nothing here repairs data: timestamps come only from the raw ``t`` field,
parents are never re-attached, missing text is never synthesised, and no graph
relation is regenerated.
"""
from __future__ import annotations

import os
from collections import Counter

from tcdscr.data import maweibo_adapter as audited

from ..config.pilot_config import CUTOFFS_MIN
from .snapshot_bridge import (count_parent_cycles,  # noqa: F401 (re-export)
                              valid_reply_parent_units)

#: The amendment V2 node status vocabulary (§7).
V2_NODE_STATUSES = ("VALID", "EMPTY_TEXT", "MISSING_PARENT",
                    "EXTERNAL_PARENT", "TEMPORAL_INVALID_NODE",
                    "DUPLICATE_ID")

#: Only this status may form a textual Reply–Parent evidence unit (§7).
V2_ELIGIBLE_STATUS = "VALID"


class MaWeiboEventInvalid(RuntimeError):
    """Raised when one raw event cannot satisfy the CR-TSER V2 contract."""


def _apply_v2_statuses(event: dict) -> dict:
    """Add the V2 statuses in place; raw values are never modified.

    The audited adapter has already assigned ``MISSING_PARENT`` /
    ``EXTERNAL_PARENT`` / ``TEMPORAL_INVALID_NODE``. This function only adds
    ``EMPTY_TEXT`` and fails closed on duplicate ids.
    """
    nodes = event["nodes"]
    counts = Counter(node["node_id"] for node in nodes)
    duplicates = sorted(nid for nid, n in counts.items() if n > 1)
    if duplicates:
        raise MaWeiboEventInvalid(
            f"event {event['event_id']}: duplicate node ids "
            f"{duplicates[:5]} (never re-numbered; event invalid)")
    for node in nodes:
        if node["status"] == "VALID" and not str(node.get("text") or "").strip():
            node["status"] = "EMPTY_TEXT"
        if node["status"] not in V2_NODE_STATUSES:
            raise MaWeiboEventInvalid(
                f"event {event['event_id']}: unknown V2 status "
                f"{node['status']!r}")
    return event


def load_normalized_event(eid: str, label: int, path: str) -> dict:
    """One Ma-Weibo event through the audited adapter + V2 eligibility.

    ``timestamp`` continues to come only from the raw ``t`` field and text from
    ``original_text`` (``text`` fallback only when the field is absent).
    """
    return _apply_v2_statuses(audited.load_event(eid, label, path))


def iter_events(raw_dir: str, label_file: str):
    """Yield ``(event, error)`` for every labelled raw file, sorted by id.

    Invalid events are yielded as ``(None, {...})`` so the audit can count them
    explicitly instead of losing them.
    """
    for eid, label in audited.event_ids(raw_dir, label_file):
        path = os.path.join(raw_dir, f"{eid}.json")
        try:
            yield load_normalized_event(eid, label, path), None
        except Exception as exc:  # noqa: BLE001 - every failure is audit data
            yield None, {"event_id": eid, "label": label, "path": path,
                         "error": f"{type(exc).__name__}: {exc}"}


def load_events(raw_dir: str, label_file: str, limit=None) -> list:
    """The V2-eligible event pool (used by the pilot pipeline)."""
    events = []
    for event, error in iter_events(raw_dir, label_file):
        if error is not None:
            continue
        events.append(event)
        if limit is not None and len(events) >= limit:
            break
    return events


# --------------------------------------------------------------------------
# Eligibility / viability (amendment §7, §8)
# --------------------------------------------------------------------------
def viable_cutoffs(event: dict) -> dict:
    """``{cutoff_str: bool}`` — cutoffs holding a valid Reply–Parent unit."""
    source = next((n for n in event["nodes"]
                   if n["node_id"] == event["source_id"]), None)
    if source is None or source["status"] != V2_ELIGIBLE_STATUS:
        return {str(c): False for c in CUTOFFS_MIN}
    units = valid_reply_parent_units(event)
    t0 = event["source_timestamp"]
    out = {}
    for cutoff in CUTOFFS_MIN:
        limit = t0 + int(cutoff) * 60
        out[str(cutoff)] = any(reply_ts <= limit and parent_ts <= limit
                               for reply_ts, parent_ts in units)
    return out


def event_viable(event: dict) -> bool:
    """True when the event can produce a valid unit (amendment §8).

    Requires **no cycle among the valid nodes** and at least one frozen cutoff
    holding a valid Reply–Parent unit.
    """
    if count_parent_cycles(event) > 0:
        return False
    return any(viable_cutoffs(event).values())


#: Backwards-compatible alias for the generic graph helper.
count_cycles = count_parent_cycles


# --------------------------------------------------------------------------
# P0-A field audit (amendment §9)
# --------------------------------------------------------------------------
def audit_maweibo(raw_dir: str, label_file: str, limit=None) -> dict:
    """Report every field amendment §9 lists for the composite source.

    The historical TC-DSCR audit numbers are *not* assumed to still hold: they
    are recomputed here from the current source of record.
    """
    registry = audited.event_ids(raw_dir, label_file) if (
        raw_dir and os.path.isdir(raw_dir) and label_file
        and os.path.exists(label_file)) else []
    label_dist = Counter(label for _eid, label in registry)
    raw_json_files = sorted(f for f in os.listdir(raw_dir)
                            if f.endswith(".json")) if (
        raw_dir and os.path.isdir(raw_dir)) else []

    status_counts = Counter()
    reply_status = Counter()
    n_events_ok = 0
    n_nodes = 0
    n_reply_nodes = 0
    n_reply_with_text = 0
    n_nodes_with_ts = 0
    n_source_text_ok = 0
    n_source_ts_ok = 0
    n_valid_unit_events = 0
    cycle_events = 0
    viable = {str(c): 0 for c in CUTOFFS_MIN}
    n_viable_total = 0
    invalid_events = 0
    multi_root_events = 0
    duplicate_id_events = 0
    error_kinds = Counter()

    for event, error in iter_events(raw_dir, label_file) if registry else []:
        if error is not None:
            invalid_events += 1
            kind = error["error"].split(":", 1)[0]
            error_kinds[kind] += 1
            message = error["error"]
            if "multi-root" in message:
                multi_root_events += 1
            if "duplicate node_id" in message:
                duplicate_id_events += 1
            continue
        n_events_ok += 1
        nodes = event["nodes"]
        source_id = event["source_id"]
        n_nodes += len(nodes)
        for node in nodes:
            status_counts[node["status"]] += 1
            if node["timestamp"] is not None:
                n_nodes_with_ts += 1
            if node["node_id"] != source_id:
                n_reply_nodes += 1
                reply_status[node["status"]] += 1
                if str(node.get("text") or "").strip():
                    n_reply_with_text += 1
        source = next(n for n in nodes if n["node_id"] == source_id)
        if source["status"] == V2_ELIGIBLE_STATUS:
            n_source_text_ok += 1
        if source["timestamp"] is not None:
            n_source_ts_ok += 1
        if valid_reply_parent_units(event):
            n_valid_unit_events += 1
        cycles = count_parent_cycles(event)
        if cycles > 0:
            cycle_events += 1
            # a cyclic event is never viable (amendment §8)
            event_viable_flags = {str(c): False for c in CUTOFFS_MIN}
        else:
            event_viable_flags = viable_cutoffs(event)
        for key, flag in event_viable_flags.items():
            if flag:
                viable[key] += 1
        if any(event_viable_flags.values()):
            n_viable_total += 1

    def _cov(num, den):
        return (num / den) if den else 0.0

    # reply–parent ratios use the non-source reply count as denominator: the
    # source node has no parent and must not dilute (or inflate) them.
    resolvable_replies = reply_status["VALID"] + reply_status["EMPTY_TEXT"]
    return {
        "dataset": "maweibo",
        "raw_dir": os.path.abspath(raw_dir) if raw_dir else raw_dir,
        "label_file": os.path.abspath(label_file) if label_file else label_file,
        "raw_event_count": len(registry),
        "raw_json_files": len(raw_json_files),
        "parsed_event_count": n_events_ok,
        "invalid_event_count": invalid_events,
        "invalid_event_kinds": dict(sorted(error_kinds.items())),
        "label_distribution": {str(k): v for k, v in sorted(label_dist.items())},
        "source_text_coverage": _cov(n_source_text_ok, n_events_ok),
        "source_timestamp_coverage": _cov(n_source_ts_ok, n_events_ok),
        "reply_text_coverage": _cov(n_reply_with_text, n_reply_nodes),
        "timestamp_coverage": _cov(n_nodes_with_ts, n_nodes),
        "parent_resolution_coverage": _cov(resolvable_replies, n_reply_nodes),
        "reply_node_count": n_reply_nodes,
        "duplicate_ids": duplicate_id_events,
        "cycle_count": cycle_events,
        "multi_root_event_count": multi_root_events,
        "missing_parent_count": reply_status["MISSING_PARENT"],
        "missing_parent_rate": _cov(reply_status["MISSING_PARENT"],
                                    n_reply_nodes),
        "external_parent_count": reply_status["EXTERNAL_PARENT"],
        "external_parent_rate": _cov(reply_status["EXTERNAL_PARENT"],
                                     n_reply_nodes),
        "empty_text_count": status_counts["EMPTY_TEXT"],
        "temporal_invalid_node_count":
            status_counts["TEMPORAL_INVALID_NODE"],
        "node_status_counts": dict(sorted(status_counts.items())),
        "reply_status_counts": dict(sorted(reply_status.items())),
        "events_with_ge1_valid_reply_parent_unit": n_valid_unit_events,
        "events_viable_15m": viable["15"],
        "events_viable_1h": viable["60"],
        "events_viable_6h": viable["360"],
        "total_viable_events": n_viable_total,
    }
