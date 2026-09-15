"""V2-P1A manifest hotfix regression tests (synthetic, no real data).

Pins the composite source-fingerprint contract: ``maweibo_composite`` must
expose the same top-level field set as every other source kind, its ``exists``
must be a real conjunction of both halves (never a constant), and a full
Ma-Weibo manifest build must write all five frozen artifacts. The frozen-source
fail-closed behaviour is pinned alongside so the fix cannot relax it.
"""
from __future__ import annotations

import json
import types

import pytest

from ..config.pilot_config import PilotPaths

SPLIT_TOTAL = 170


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _write_raw(root, eid: str, posts) -> str:
    path = root / "raw"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{eid}.json").write_text(json.dumps(posts), encoding="utf-8")
    return str(path)


def _write_labels(root, rows) -> str:
    target = root / "Weibo.txt"
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(target)


def _composite_paths(raw_dir, label_file, out_root=""):
    return PilotPaths(maweibo_raw=raw_dir, maweibo_labels=label_file,
                      semantic_model="unused", canonical_tokenizer="unused",
                      out_root=str(out_root))


def _event(eid: str, label: int):
    """A ``v2_strict``-viable synthetic event: VALID source plus a textual,
    causally ordered Reply–Parent unit inside the 15m cutoff."""
    return {
        "event_id": eid, "label": label, "source_id": "n0",
        "source_timestamp": 1000,
        "nodes": [
            {"node_id": "n0", "parent_id": None, "timestamp": 1000,
             "text": "source body", "original_order": 0, "status": "VALID"},
            {"node_id": "n1", "parent_id": "n0", "timestamp": 1060,
             "text": "a reply", "original_order": 1, "status": "VALID"},
        ],
    }


class _Tok:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(str(text).split())))}


def _art():
    """Minimal snapshot artifact with one removable reply."""
    return {
        "zero_reply": False,
        "units": [{"node_id": "n1"}],
        "src": {"selected_node_ids": ["n1"], "total_tokens": 4},
        "interventions": [{"intervention_id": "I1", "type": "I1_remove_reply",
                           "status": "OK", "remove_node_ids": ["n1"]}],
        "snapshot": {"node_ids": ["n0", "n1"], "timestamps": [1000, 1060],
                     "edge_index": [[0, 1]], "cap_hit": False},
    }


def _synthetic_source(root, n_events=SPLIT_TOTAL):
    """A synthetic Ma-Weibo composite source plus its event pool."""
    rows, events = [], []
    for i in range(n_events):
        eid = f"e{i:03d}"
        label = i % 2
        posts = [{"mid": f"{eid}_s", "parent": None, "t": 1000,
                  "original_text": "source body"},
                 {"mid": f"{eid}_r", "parent": f"{eid}_s", "t": 1060,
                  "original_text": "a reply"}]
        raw = _write_raw(root, eid, posts)
        rows.append(f"eid:{eid} label:{label}")
        events.append(_event(eid, label))
    return raw, _write_labels(root, rows), events


def _stub_build_environment(monkeypatch, events):
    """Stub only what needs a real encoder/tokenizer/GPU; the source
    fingerprint stays the real implementation under test."""
    import cr_tser_common as common

    monkeypatch.setattr(common, "load_dataset_events",
                        lambda dataset, paths: events)
    monkeypatch.setattr(common, "CrSemanticEncoder",
                        lambda model, dataset, device="cpu":
                        types.SimpleNamespace(dataset=dataset))
    monkeypatch.setattr(common, "canonical_tokenizer", lambda path: _Tok())
    monkeypatch.setattr(common, "snapshot_artifacts",
                        lambda event, cutoff, encoder, tokenizer: _art())


# --------------------------------------------------------------------------
# 1. the unified top-level fingerprint contract
# --------------------------------------------------------------------------
def test_composite_fingerprint_exposes_unified_top_level_fields(tmp_path):
    from ..data.source_manifest import fingerprint_path, source_fingerprint

    raw, labels, _events = _synthetic_source(tmp_path, n_events=1)
    frozen = source_fingerprint("maweibo",
                                _composite_paths(raw, labels))
    single = fingerprint_path(labels)          # the single-source field set
    contract = set(single) - {"path"}
    assert contract <= set(frozen)
    assert frozen["exists"] is True
    assert frozen["path"] == raw
    assert frozen["sha256"] == frozen["combined_source_sha256"]
    # the composite keeps its own halves and its binding hash
    assert frozen["raw_json"]["exists"] is True
    assert frozen["label_file"]["exists"] is True
    assert frozen["combined_source_sha256"]


def test_composite_exists_requires_both_halves(tmp_path):
    import os

    from ..data.source_manifest import source_fingerprint

    raw, labels, _events = _synthetic_source(tmp_path, n_events=1)
    assert source_fingerprint("maweibo",
                              _composite_paths(raw, labels))["exists"] is True

    os.remove(labels)
    missing_labels = source_fingerprint("maweibo",
                                        _composite_paths(raw, labels))
    assert missing_labels["exists"] is False
    assert missing_labels["label_file"]["exists"] is False

    _raw2, labels2, _events = _synthetic_source(tmp_path / "other", n_events=1)
    gone_raw = str(tmp_path / "other" / "absent_raw")
    missing_raw = source_fingerprint("maweibo",
                                     _composite_paths(gone_raw, labels2))
    assert missing_raw["exists"] is False
    assert missing_raw["raw_json"]["exists"] is False


# --------------------------------------------------------------------------
# 2. a real Ma-Weibo manifest build (the KeyError this hotfix fixes)
# --------------------------------------------------------------------------
def test_maweibo_formal_manifest_build_writes_all_artifacts(tmp_path,
                                                            monkeypatch):
    import cr_tser_build_manifests as manifests
    from ..data.pilot_split import viable_event_ids

    raw, labels, events = _synthetic_source(tmp_path)
    # the synthetic pool must really pass the V2 strict viability gate, or the
    # manifest would be built from a pool the pilot can never sample
    assert len(viable_event_ids(events, eligibility="v2_strict")) == len(events)
    _stub_build_environment(monkeypatch, events)

    out_root = str(tmp_path / "out")
    result = manifests.build_manifests(
        "maweibo", _composite_paths(raw, labels, out_root), out_root)

    manifests_dir = tmp_path / "out" / "manifests" / "maweibo"
    expected = ("source.json", "event_split.json", "snapshot_manifest.jsonl",
                "intervention_manifest.jsonl", "hashes.json")
    for name in expected:
        assert (manifests_dir / name).is_file(), name

    source = json.loads((manifests_dir / "source.json")
                        .read_text(encoding="utf-8"))
    assert source["kind"] == "maweibo_composite"
    assert source["exists"] is True

    split = json.loads((manifests_dir / "event_split.json")
                       .read_text(encoding="utf-8"))
    assert split["source"]["exists"] is True
    assert split["source"]["sha256"] == source["combined_source_sha256"]
    assert split["source"]["path"] == raw
    assert split["split_seed"] == 7319
    assert {k: len(split[k]) for k in
            ("foundation_train", "utility_train", "utility_dev",
             "utility_eval")} == {"foundation_train": 80, "utility_train": 50,
                                  "utility_dev": 15, "utility_eval": 25}

    hashes = json.loads((manifests_dir / "hashes.json")
                        .read_text(encoding="utf-8"))
    assert hashes["source"] == source
    assert hashes["n_snapshots"] == result["n_snapshots"] > 0
    assert hashes["n_interventions"] == result["n_interventions"] > 0


# --------------------------------------------------------------------------
# 3. composite replacement still fails closed
# --------------------------------------------------------------------------
def test_composite_replacement_fails_closed(tmp_path, monkeypatch):
    import cr_tser_build_manifests as manifests
    from ..data.source_manifest import (SourceIdentityError,
                                        assert_same_source,
                                        source_fingerprint)

    raw, labels, events = _synthetic_source(tmp_path)
    frozen = source_fingerprint("maweibo", _composite_paths(raw, labels))
    assert_same_source(frozen, dict(frozen), "test")   # identical passes

    with open(labels, "a", encoding="utf-8") as fh:
        fh.write("eid:e999 label:1\n")
    replaced = source_fingerprint("maweibo", _composite_paths(raw, labels))
    with pytest.raises(SourceIdentityError):
        assert_same_source(frozen, replaced, "test")

    # the same guard still stops a manifest rebuild against a swapped source
    _stub_build_environment(monkeypatch, events)
    out_root = str(tmp_path / "out")
    manifests.build_manifests("maweibo", _composite_paths(raw, labels, out_root),
                              out_root)
    with open(labels, "w", encoding="utf-8") as fh:
        fh.write("eid:e000 label:0\n")
    with pytest.raises(SourceIdentityError):
        manifests.build_manifests("maweibo",
                                  _composite_paths(raw, labels, out_root),
                                  out_root, force=True)
