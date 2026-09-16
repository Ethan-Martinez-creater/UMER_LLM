"""Probe context construction, P0/P1/P2/P3 (plan §6.2)."""
from __future__ import annotations

import pytest

from bcr_utility.config import protocol as P
from bcr_utility.probes import probe_contexts as pc


def make_units(n=4):
    return [{
        "node_id": f"n{i}",
        "reply_text": f"reply {i}",
        "parent_text": f"parent {i}",
        "parent_id": "n0" if i else None,
        "timestamp": 1000 + i,
        "elapsed_seconds": 60 * i,
        "snapshot_order": i,
        "depth": 1,
    } for i in range(n)]


def make_src(ranked, selected):
    return {
        "ranked_node_ids": list(ranked),
        "selected_node_ids": list(selected),
        "n_visible_replies": len(ranked),
        "n_selected": len(selected),
        "total_tokens": 100,
        "budget_ref": 1024,
        "rank_percentile": {nid: i / max(len(ranked) - 1, 1)
                            for i, nid in enumerate(ranked)},
        "relevance": {nid: 1.0 - 0.1 * i for i, nid in enumerate(ranked)},
        "unit_token_costs": {nid: 5 for nid in ranked},
    }


def test_p0_is_source_only():
    units, src = make_units(), make_src(["n3", "n1", "n2"], ["n1", "n3"])
    contexts = pc.build_probe_contexts(units, src)
    assert contexts["P0"]["node_ids"] == []
    assert contexts["P0"]["n_units"] == 0
    assert "Reply: " not in contexts["P0"]["evidence_block"]


def test_p1_is_the_highest_relevance_unit():
    units, src = make_units(), make_src(["n3", "n1", "n2"], ["n1", "n3"])
    assert pc.context_node_ids(units, src, "P1") == ["n3"]


def test_p2_is_the_lowest_relevance_src_selected_unit():
    units, src = make_units(), make_src(["n3", "n1", "n2"], ["n1", "n3"])
    assert pc.context_node_ids(units, src, "P2") == ["n1"]
    # n2 is ranked lower but is not SRC-selected, so it can never be P2.
    assert "n2" not in pc.context_node_ids(units, src, "P2")


def test_p3_is_the_full_src_in_snapshot_order():
    units = make_units()
    src = make_src(["n3", "n1", "n2"], ["n1", "n2", "n3"])
    assert pc.context_node_ids(units, src, "P3") == ["n1", "n2", "n3"]
    contexts = pc.build_probe_contexts(units, src)
    assert contexts["P3"]["node_ids"] == src["selected_node_ids"]


def test_every_context_is_reader_independent():
    plan = pc.context_plan()
    assert plan["selection_is_reader_independent"] is True
    assert plan["selection_uses_utility_outcomes"] is False
    assert tuple(plan["contexts"]) == P.PROBE_CONTEXTS
    for context in P.PROBE_CONTEXTS:
        assert plan["requirements"][context] == \
            P.PROBE_CONTEXT_REQUIREMENTS[context]


def test_empty_src_is_refused():
    units = make_units()
    with pytest.raises(pc.ProbeContextRefused):
        pc.build_probe_contexts(units, make_src(["n1"], []))


def test_empty_ranking_is_refused():
    units = make_units()
    with pytest.raises(pc.ProbeContextRefused):
        pc.highest_relevance_unit(make_src([], []))


def test_unknown_context_is_refused():
    units, src = make_units(), make_src(["n1"], ["n1"])
    with pytest.raises(pc.ProbeContextRefused):
        pc.context_node_ids(units, src, "P9")


def test_node_outside_the_snapshot_is_refused():
    units = make_units()
    src = make_src(["n9", "n1"], ["n9"])
    with pytest.raises(pc.ProbeContextRefused):
        pc.build_probe_contexts(units, src)


def test_selected_unit_missing_from_ranking_is_refused():
    units = make_units()
    src = make_src(["n1"], ["n1", "n2"])
    with pytest.raises(pc.ProbeContextRefused):
        pc.lowest_relevance_selected_unit(src)


def test_contexts_render_the_selected_evidence_block():
    units, src = make_units(), make_src(["n3", "n1"], ["n1", "n3"])
    contexts = pc.build_probe_contexts(units, src)
    for context in ("P1", "P2", "P3"):
        assert "Reply: " in contexts[context]["evidence_block"]
    assert contexts["P1"]["requirement"] == \
        P.PROBE_CONTEXT_REQUIREMENTS["P1"]
