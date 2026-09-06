"""Run manifests: cap statistics (§27) and adapter anomaly audit logs."""
from __future__ import annotations

import json
import os


def _p90(values):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * 0.9
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class CapStatistics:
    """Aggregate 1021-node cap statistics over a set of snapshots (§27)."""

    def __init__(self):
        self.events = 0
        self.cap_hits = 0
        self.nodes_before_cap = []
        self.total_nodes_removed = 0

    def add(self, snapshot: dict):
        self.events += 1
        self.cap_hits += 1 if snapshot["cap_hit"] else 0
        self.nodes_before_cap.append(snapshot["num_nodes_before_cap"])
        self.total_nodes_removed += (snapshot["num_nodes_before_cap"]
                                     - snapshot["num_nodes_after_cap"])

    def to_dict(self):
        return {
            "events": self.events,
            "cap_hits": self.cap_hits,
            "cap_hit_rate": self.cap_hits / max(self.events, 1),
            "mean_nodes_before_cap": (
                sum(self.nodes_before_cap) / max(len(self.nodes_before_cap), 1)),
            "p90_nodes_before_cap": _p90(self.nodes_before_cap),
            "max_nodes_before_cap": max(self.nodes_before_cap) if self.nodes_before_cap else 0,
            "total_nodes_removed": self.total_nodes_removed,
        }


class AdapterAuditLog:
    """Per-event anomaly audit: temporal invalids, parent anomalies, depth."""

    def __init__(self):
        self.rows = []

    def add_event(self, event: dict, snapshots: list):
        row = {
            "event_id": event["event_id"],
            "label": event["label"],
            "n_nodes_raw": len(event["nodes"]),
            "temporal_invalid_nodes": sum(
                1 for n in event["nodes"]
                if n["status"] == "TEMPORAL_INVALID_NODE"),
            "missing_parent_nodes": sum(
                1 for n in event["nodes"]
                if n["status"] == "MISSING_PARENT"),
            "external_parent_nodes": sum(
                1 for n in event["nodes"]
                if n["status"] == "EXTERNAL_PARENT"),
            "snapshots": {
                str(snap["cutoff_minutes"]): {
                    "nodes": snap["num_nodes_after_cap"],
                    "edges": len(snap["edge_index"]),
                    "cap_hit": snap["cap_hit"],
                    "unreachable_nodes": snap["unreachable_count"],
                    "depth_overflow_nodes": snap["depth_overflow_count"],
                }
                for snap in snapshots
            },
        }
        self.rows.append(row)
        return row

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row) + "\n")

    def summary(self) -> dict:
        n = len(self.rows)
        return {
            "events": n,
            "temporal_invalid_nodes_total": sum(r["temporal_invalid_nodes"] for r in self.rows),
            "missing_parent_nodes_total": sum(r["missing_parent_nodes"] for r in self.rows),
            "external_parent_nodes_total": sum(r["external_parent_nodes"] for r in self.rows),
            "events_with_unreachable_nodes": sum(
                1 for r in self.rows
                if any(v["unreachable_nodes"] for v in r["snapshots"].values())),
            "events_with_depth_overflow": sum(
                1 for r in self.rows
                if any(v["depth_overflow_nodes"] for v in r["snapshots"].values())),
        }
