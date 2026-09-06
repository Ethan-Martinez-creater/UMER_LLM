"""Unit tests: PHEME adapter (plan §28 — test_pheme_timestamp)."""
from datetime import datetime, timezone

from ..data.pheme_adapter import iter_events, load_event
from .conftest import write_pheme_thread


def test_pheme_timestamp(pheme_tree):
    events = list(iter_events(pheme_tree))
    assert len(events) == 2
    by_id = {e["event_id"]: e for e in events}
    ev = by_id["1001"]
    expected = int(datetime(2016, 9, 5, 10, 0, 0,
                            tzinfo=timezone.utc).timestamp())
    assert ev["source_timestamp"] == expected
    # replies parse to absolute unix seconds relative to the source clock
    ts = {n["node_id"]: n["timestamp"] for n in ev["nodes"]}
    assert ts["1002"] - expected == 300
    assert ts["1003"] - expected == 600
    # t0 is the source timestamp, not the event minimum
    assert ev["source_id"] == "1001"
    # thread with no reactions still yields a valid single-node event
    solo = by_id["2001"]
    assert len(solo["nodes"]) == 1


def test_pheme_adapter_status_flags(pheme_tree):
    events = list(iter_events(pheme_tree))
    ev = next(e for e in events if e["event_id"] == "1001")
    statuses = {n["node_id"]: n["status"] for n in ev["nodes"]}
    assert statuses["1001"] == "VALID"
    assert statuses["1002"] == "VALID"
    assert statuses["1003"] == "VALID"


def test_pheme_adapter_missing_and_external(pheme_tree):
    # a reaction without in_reply_to -> MISSING_PARENT; one pointing outside
    # the event -> EXTERNAL_PARENT
    root = pheme_tree
    write_pheme_thread(
        root, "ferguson", "555002",
        {"id_str": "3001", "created_at": "Mon Sep 05 14:00:00 +0000 2016",
         "text": "claim c"},
        [{"id_str": "3002", "created_at": "Mon Sep 05 14:01:00 +0000 2016",
          "text": "no parent field"},
         {"id_str": "3003", "created_at": "Mon Sep 05 14:02:00 +0000 2016",
          "in_reply_to_status_id_str": "9999999", "text": "dangling"}])
    ev = load_event("ferguson", 1, f"{root}/ferguson/rumours/555002")
    statuses = {n["node_id"]: n["status"] for n in ev["nodes"]}
    assert statuses["3002"] == "MISSING_PARENT"
    assert statuses["3003"] == "EXTERNAL_PARENT"
