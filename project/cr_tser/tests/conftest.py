"""Shared synthetic fixtures for CR-TSER tests.

Everything here is tiny and synthetic: the unit suite never reads a real
dataset, a real reader checkpoint or the GPU. The frozen TC-DSCR modules are
imported through the repository's ``project`` directory so the causality and
snapshot behaviour under test is the *real* implementation, not a copy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve()
PROJECT_DIR = HERE.parents[2]          # <repo>/project
REPO_DIR = HERE.parents[3]             # <repo>
SCRIPTS_DIR = REPO_DIR / "scripts"
for path in (PROJECT_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_configure(config):
    """Keep ``tmp_path`` inside the repo.

    The system temp root (``%TEMP%/pytest-of-<user>``) is not reliably
    writable in this environment; a repo-local base temp keeps the suite
    hermetic and portable.
    """
    if not config.option.basetemp:
        config.option.basetemp = str(REPO_DIR / ".pytest_tmp_cr_tser")


class FakeTokenizer:
    """Whitespace tokenizer matching ``reader_prompt.count_tokens``'s contract."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(str(text).split())))}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        return "\n".join(m["content"] for m in messages)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


def make_event(nodes, source_ts=1000, label=1, event_id="e1",
               source_id="n0"):
    """``nodes``: list of (node_id, parent_id, timestamp, text, order)."""
    return {
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


def chain_event(n_replies=6, minutes=(1, 5, 20, 40, 100, 200, 400, 700),
                words=6, event_id="e1", label=1):
    """source -> r1 -> r2 -> ... a deterministic chain with real timestamps."""
    nodes = [("n0", None, 1000, "source claim text here", 0)]
    parent = "n0"
    for i in range(n_replies):
        ts = 1000 + int(minutes[i % len(minutes)]) * 60
        text = " ".join([f"r{i}w{w}" for w in range(words)])
        nodes.append((f"n{i + 1}", parent, ts, text, i + 1))
        parent = f"n{i + 1}"
    return make_event(nodes, event_id=event_id, label=label)


def star_event(n_replies=6, words=6, event_id="e2", label=0):
    """source with every reply attached directly to it."""
    nodes = [("n0", None, 1000, "source claim text here", 0)]
    for i in range(n_replies):
        text = " ".join([f"s{i}w{w}" for w in range(words)])
        nodes.append((f"n{i + 1}", "n0", 1000 + (i + 1) * 60, text, i + 1))
    return make_event(nodes, event_id=event_id, label=label)


def branch_event(words=6, event_id="e3", label=1):
    """source -> a; a -> b and a -> c.

    Supplies a real parent–child edge (a,b)/(a,c) **and** a non-adjacent
    sibling pair (b,c), so I2 and I3 can both exist for the same snapshot.
    """
    return make_event([
        ("n0", None, 1000, "source claim text here", 0),
        ("a", "n0", 1060, " ".join(f"aw{w}" for w in range(words)), 1),
        ("b", "a", 1120, " ".join(f"bw{w}" for w in range(words)), 2),
        ("c", "a", 1180, " ".join(f"cw{w}" for w in range(words)), 3),
    ], event_id=event_id, label=label)


def embeddings_for(snapshot, dim=384, seed=0):
    """Deterministic reply/source embeddings keyed by node id."""
    g = torch.Generator().manual_seed(seed)
    vecs = torch.randn(len(snapshot["node_ids"]), dim, generator=g)
    reply_emb = {nid: vecs[i] for i, nid in enumerate(snapshot["node_ids"])}
    return reply_emb[snapshot["source_id"]], reply_emb


@pytest.fixture
def simple_chain():
    from cr_tser.data.snapshot_bridge import build_causal_snapshot
    event = chain_event()
    return event, build_causal_snapshot(event, 360)


@pytest.fixture
def simple_star():
    from cr_tser.data.snapshot_bridge import build_causal_snapshot
    event = star_event()
    return event, build_causal_snapshot(event, 360)
