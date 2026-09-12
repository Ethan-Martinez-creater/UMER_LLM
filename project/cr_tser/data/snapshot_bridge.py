"""Causal snapshot bridge (plan §3.1.B, §6).

Reuses the frozen TC-DSCR snapshot builder unchanged so CR-TSER inherits the
verified causal rule

    V_t = {v_i | timestamp_i <= t}

with the dataset's **true** timestamps. The bridge adds the pilot-specific
checks the plan requires: the three frozen cutoffs, a hard causal assertion
that no future node ever enters a snapshot, and the §6 audit rule that
zero-reply snapshots are retained for the audit but never generate
intervention rows.
"""
from __future__ import annotations

from ..config.pilot_config import CUTOFFS_MIN
from .structural_stats import assert_frozen_cutoffs, assert_valid_cutoff

__all__ = ["build_causal_snapshot", "assert_causal", "snapshot_has_replies",
           "audit_snapshot", "cutoffs"]


def cutoffs():
    """The three frozen pilot cutoffs, in plan order (plan §6)."""
    assert_frozen_cutoffs(CUTOFFS_MIN)
    return tuple(CUTOFFS_MIN)


def build_causal_snapshot(event: dict, cutoff_minutes: int) -> dict:
    """Causally visible nodes at ``source_timestamp + cutoff`` (plan §3.1.B)."""
    assert_valid_cutoff(cutoff_minutes)
    from tcdscr.data.snapshot_builder import build_snapshot
    snap = build_snapshot(event, int(cutoff_minutes))
    assert_causal(event, snap, cutoff_minutes)
    return snap


def assert_causal(event: dict, snapshot: dict, cutoff_minutes: int) -> None:
    """Fail if any snapshot node lies beyond the cutoff (plan §3.1.B, §34)."""
    t0 = event["source_timestamp"]
    limit = t0 + int(cutoff_minutes) * 60
    for node_id, ts in zip(snapshot["node_ids"], snapshot["timestamps"]):
        if ts > limit:
            raise AssertionError(
                f"event {event['event_id']!r} cutoff {cutoff_minutes}m: node "
                f"{node_id!r} has timestamp {ts} > {limit} (future leak)")
    if event["source_id"] not in snapshot["node_ids"]:
        raise AssertionError("snapshot lost the source node")


def reply_node_ids(snapshot: dict) -> list:
    """Visible non-source node ids in deterministic snapshot order."""
    return [nid for nid in snapshot["node_ids"]
            if nid != snapshot["source_id"]]


def snapshot_has_replies(snapshot: dict) -> bool:
    return len(reply_node_ids(snapshot)) > 0


def audit_snapshot(event: dict, snapshot: dict) -> dict:
    """Snapshot-level audit row (plan §6: zero-reply snapshots are audited)."""
    replies = reply_node_ids(snapshot)
    return {
        "event_id": event["event_id"],
        "cutoff": snapshot["cutoff_minutes"],
        "num_nodes": len(snapshot["node_ids"]),
        "num_replies": len(replies),
        "zero_reply": len(replies) == 0,
        "cap_hit": bool(snapshot.get("cap_hit")),
        "num_nodes_before_cap": snapshot.get("num_nodes_before_cap"),
        "max_depth": snapshot.get("max_depth"),
        "unreachable_count": snapshot.get("unreachable_count"),
        "edge_count": len(snapshot["edge_index"]),
    }
