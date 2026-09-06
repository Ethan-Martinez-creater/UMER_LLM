"""Shared synthetic fixtures for TC-DSCR tests.

All tests run on tiny synthetic events/trees — no real dataset is needed for
the unit suite, so the frozen data is never touched by tests.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# entry-point scripts live in <repo>/scripts; make them importable so the
# fold-argument behavior of the training entries can be tested directly
REPO_DIR = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def make_event(nodes, source_ts=1000, label=1, event_id="e1",
               source_id="n0"):
    """``nodes``: list of (node_id, parent_id, timestamp, text, order)."""
    event = {
        "event_id": event_id,
        "label": label,
        "source_id": source_id,
        "source_timestamp": source_ts,
        "nodes": [
            {"node_id": nid, "parent_id": parent, "timestamp": ts,
             "text": text, "original_order": order, "status": "VALID"}
            for nid, parent, ts, text, order in nodes
        ],
    }
    return event


def simple_event():
    """source(t0=1000) <- A(1100) <- B(3600*2+1000); plus C(1500, parent
    missing) and D(2000, external parent)."""
    return make_event([
        ("n0", None, 1000, "source claim", 0),
        ("n1", "n0", 1100, "reply one", 1),
        ("n2", "n1", 1000 + 3600 * 2, "reply two", 2),
        ("n3", None, 1500, "orphan reply", 3),
        ("n4", "missing-node", 2000, "external reply", 4),
    ])


def write_pheme_thread(root, topic, thread_id, source, reactions):
    """Create a minimal all-rnr-annotated-threads layout under root."""
    base = os.path.join(root, topic, "rumours", thread_id)
    os.makedirs(os.path.join(base, "source-tweets"), exist_ok=True)
    os.makedirs(os.path.join(base, "reactions"), exist_ok=True)
    with open(os.path.join(base, "source-tweets", f"{source['id_str']}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(source, fh)
    for r in reactions:
        with open(os.path.join(base, "reactions", f"{r['id_str']}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(r, fh)
    return base


@pytest.fixture
def pheme_tree(tmp_path):
    root = tmp_path / "pheme_raw"
    root.mkdir()
    write_pheme_thread(
        str(root), "charliehebdo", "555000",
        {"id_str": "1001", "created_at": "Mon Sep 05 10:00:00 +0000 2016",
         "text": "breaking claim"},
        [{"id_str": "1002", "created_at": "Mon Sep 05 10:05:00 +0000 2016",
          "in_reply_to_status_id_str": "1001", "text": "doubt it"},
         {"id_str": "1003", "created_at": "Mon Sep 05 10:10:00 +0000 2016",
          "in_reply_to_status_id_str": "1002", "text": "source?"}])
    write_pheme_thread(
        str(root), "sydneysiege", "555001",
        {"id_str": "2001", "created_at": "Mon Sep 05 12:00:00 +0000 2016",
         "text": "normal claim"},
        [])
    return str(root)


@pytest.fixture
def maweibo_tree(tmp_path):
    raw = tmp_path / "maweibo_raw"
    raw.mkdir()
    labels = tmp_path / "Weibo.txt"
    events = {
        "30001": (1, [
            {"mid": "9001", "parent": None, "t": 1400000000,
             "original_text": "<b>源帖</b> http://t.cn/x 断言"},
            {"mid": "9002", "parent": "9001", "t": 1400000600,
             "original_text": "回复一"},
        ]),
        "30002": (0, [
            {"mid": "9101", "parent": None, "t": 1400010000,
             "original_text": "普通帖"},
        ]),
    }
    with open(labels, "w", encoding="utf-8") as fh:
        for eid, (label, _) in events.items():
            fh.write(f"eid:{eid} label:{label}\n")
    for eid, (label, posts) in events.items():
        with open(raw / f"{eid}.json", "w", encoding="utf-8") as fh:
            json.dump(posts, fh, ensure_ascii=False)
    return str(raw), str(labels)
