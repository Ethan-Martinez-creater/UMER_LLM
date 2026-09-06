"""Normalized raw-event schema shared by both dataset adapters (plan §13).

An adapter produces::

    {
        "event_id": str,
        "label": int,
        "source_id": str,
        "source_timestamp": int,
        "nodes": [ {node_id, parent_id, timestamp, text, original_order, status} ]
    }

Adapters only read raw data, normalize fields, and flag anomalies; they never
compute model features.
"""
from __future__ import annotations

from ..config.schema import ALLOWED_NODE_STATUS


class SchemaError(ValueError):
    """Raised when an adapter output violates the normalized schema."""


def validate_event(event: dict) -> dict:
    """Validate one normalized event dict; returns it unchanged."""
    for key in ("event_id", "label", "source_id", "source_timestamp", "nodes"):
        if key not in event:
            raise SchemaError(f"event missing key {key!r}")
    if not isinstance(event["event_id"], str) or not event["event_id"]:
        raise SchemaError("event_id must be a non-empty string")
    if event["label"] not in (0, 1):
        raise SchemaError(f"label must be 0/1, got {event['label']!r}")
    if not isinstance(event["source_id"], str) or not event["source_id"]:
        raise SchemaError("source_id must be a non-empty string")
    if not isinstance(event["source_timestamp"], int):
        raise SchemaError("source_timestamp must be int")
    if not isinstance(event["nodes"], list) or not event["nodes"]:
        raise SchemaError("nodes must be a non-empty list")
    seen = set()
    source_seen = 0
    for node in event["nodes"]:
        for key in ("node_id", "parent_id", "timestamp", "text",
                    "original_order", "status"):
            if key not in node:
                raise SchemaError(f"node missing key {key!r}: {node}")
        if not isinstance(node["node_id"], str) or not node["node_id"]:
            raise SchemaError("node_id must be a non-empty string")
        if node["node_id"] in seen:
            raise SchemaError(f"duplicate node_id {node['node_id']!r}")
        seen.add(node["node_id"])
        if node["parent_id"] is not None and not isinstance(node["parent_id"], str):
            raise SchemaError("parent_id must be str or None")
        if not isinstance(node["timestamp"], int):
            raise SchemaError("node timestamp must be int")
        if not isinstance(node["text"], str):
            raise SchemaError("node text must be str")
        if not isinstance(node["original_order"], int):
            raise SchemaError("node original_order must be int")
        if node["status"] not in ALLOWED_NODE_STATUS:
            raise SchemaError(
                f"invalid node status {node['status']!r}, "
                f"allowed: {ALLOWED_NODE_STATUS}")
        if node["node_id"] == event["source_id"]:
            source_seen += 1
    if source_seen != 1:
        raise SchemaError(
            f"exactly one node must match source_id, found {source_seen}")
    return event


def node_parent_status(event: dict, node: dict) -> str:
    """Classify a node's parent relation for the adapter status flags.

    The source itself has no parent. A reply whose parent_id is None is
    MISSING_PARENT; one whose parent id does not exist inside the event is
    EXTERNAL_PARENT. This is a pure structural check on the normalized event.
    """
    if node["node_id"] == event["source_id"]:
        return "VALID"
    parent = node["parent_id"]
    if parent is None or parent == "":
        return "MISSING_PARENT"
    if parent not in {n["node_id"] for n in event["nodes"]}:
        return "EXTERNAL_PARENT"
    return "VALID"
