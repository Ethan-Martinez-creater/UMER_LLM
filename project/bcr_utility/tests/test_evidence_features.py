"""M1 E0 evidence-feature tests (synthetic, model-free)."""
from __future__ import annotations

import pytest

from ..config import protocol as P
from ..features import evidence_features as ef


def _snapshot():
    return {
        "event_id": "e1",
        "label": 1,
        "cutoff_minutes": 15,
        "source_id": "n0",
        "node_ids": ["n0", "n1", "n2"],
        "texts": ["source text", "reply one text", "reply two"],
        "timestamps": [1000, 1060, 1120],
        "elapsed_seconds": [0, 60, 120],
        "parent_ids": [None, "n0", "n1"],
        "edge_index": [[1, 0], [2, 1]],
        "depths": [0, 1, 2],
        "statuses": ["VALID", "VALID", "VALID"],
    }


def _artifacts():
    units = [
        {"node_id": "n1", "reply_text": "reply one text",
         "parent_text": "source text", "parent_id": "n0", "timestamp": 1060,
         "elapsed_seconds": 60, "snapshot_order": 1, "depth": 1},
        {"node_id": "n2", "reply_text": "reply two",
         "parent_text": "reply one text", "parent_id": "n1",
         "timestamp": 1120, "elapsed_seconds": 120, "snapshot_order": 2,
         "depth": 2},
    ]
    src = {
        "unit_token_costs": {"n1": 11, "n2": 7},
        "relevance": {"n1": 0.8, "n2": 0.3},
        "rank_percentile": {"n1": 0.0, "n2": 1.0},
    }
    reply_emb = {"n0": [1.0, 0.0], "n1": [1.0, 0.0], "n2": [0.0, 1.0]}
    return {"units": units, "src": src, "reply_emb": reply_emb,
            "snapshot": _snapshot()}


def _entries():
    return [
        {"key": "maweibo|e1|15|n1", "event_id": "e1", "cutoff": 15,
         "node_id": "n1"},
        {"key": "maweibo|e1|15|n2", "event_id": "e1", "cutoff": 15,
         "node_id": "n2"},
    ]


def test_e0_features_exact_cosines_and_scalars():
    art = _artifacts()
    unit = art["units"][0]
    feats = ef.e0_features(unit, art["reply_emb"]["n0"],
                           art["reply_emb"]["n0"], art["reply_emb"]["n1"],
                           art["src"], 15)
    assert feats["cos_reply_source"] == pytest.approx(1.0)
    assert feats["cos_parent_source"] == pytest.approx(1.0)
    assert feats["cos_reply_parent"] == pytest.approx(1.0)
    assert feats["src_relevance"] == pytest.approx(0.8)
    assert feats["src_rank_percentile"] == pytest.approx(0.0)
    assert feats["canonical_token_cost"] == pytest.approx(11.0)
    assert feats["reply_chars"] == float(len("reply one text"))
    assert feats["parent_chars"] == float(len("source text"))
    assert feats["combined_chars"] == pytest.approx(
        len("reply one text") + len("source text"))
    assert feats["elapsed_seconds"] == pytest.approx(60.0)
    assert feats["cutoff_minutes"] == pytest.approx(15.0)
    assert tuple(feats) == P.E0_FEATURE_NAMES


def test_e0_features_missing_parent_yields_zero_cosines():
    art = _artifacts()
    unit = dict(art["units"][0])
    unit["parent_text"] = None
    unit["parent_id"] = None
    feats = ef.e0_features(unit, art["reply_emb"]["n0"], None,
                           art["reply_emb"]["n1"], art["src"], 15)
    assert feats["cos_parent_source"] == 0.0
    assert feats["cos_reply_parent"] == 0.0
    assert feats["parent_chars"] == 0.0


def test_rows_for_snapshot_builds_all_entries():
    rows = ef.rows_for_snapshot(_entries(), _snapshot(), _artifacts())
    assert [r["key"] for r in rows] == [e["key"] for e in _entries()]
    assert rows[0]["parent_available"] == 1
    assert tuple(rows[0]["e0"]) == P.E0_FEATURE_NAMES
    assert tuple(rows[0]["struct"]) == P.B1_STRUCT_NAMES
    audit = ef.validate_e0_rows(rows, [e["key"] for e in _entries()])
    assert audit["ok"] is True


def test_rows_for_snapshot_refuses_unknown_node():
    entries = [{"key": "maweibo|e1|15|nX", "event_id": "e1", "cutoff": 15,
                "node_id": "nX"}]
    with pytest.raises(ef.FeatureExtractionRefused, match="no matching"):
        ef.rows_for_snapshot(entries, _snapshot(), _artifacts())


def test_validate_refuses_missing_keys():
    rows = ef.rows_for_snapshot(_entries(), _snapshot(), _artifacts())
    with pytest.raises(ef.FeatureExtractionRefused, match="missing"):
        ef.validate_e0_rows(rows[:1], [e["key"] for e in _entries()])
