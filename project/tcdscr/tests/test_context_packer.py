"""Unit tests: evidence units + token budget + packer (plan §28 Context)."""
from ..context.evidence_unit import (UNAVAILABLE, build_evidence_units,
                                      render_evidence)
from ..context.packer import pack_context
from ..context.token_budget import EvidenceBudgetSelector, format_cutoff
from ..data.snapshot_builder import build_snapshot
from .conftest import make_event, simple_event


class _Tok:
    """Token stub: one token per whitespace-split piece."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _snapshot():
    return build_snapshot(simple_event(), 60)


def test_reply_parent_pair():
    snap = _snapshot()
    units = build_evidence_units(snap)
    unit = next(u for u in units if u["node_id"] == "n1")
    rendered = render_evidence(unit, 1)
    assert rendered.startswith("[E1]")
    assert "parent:" in rendered and "reply:" in rendered
    assert "source claim" in rendered   # parent text shown
    assert "reply one" in rendered      # reply text shown
    assert "time = " in rendered and "depth = " in rendered


def test_unavailable_parent():
    snap = _snapshot()
    units = build_evidence_units(snap)
    unit = next(u for u in units if u["node_id"] == "n3")  # missing parent
    rendered = render_evidence(unit, 2)
    assert UNAVAILABLE in rendered
    assert "orphan reply" in rendered


def test_budget_no_partial_pair():
    # six long units (~93 stub tokens each) against the frozen 512 budget:
    # five fit whole, the sixth must be skipped — never truncated
    nodes = [("n0", None, 1000, "source", 0)]
    for i in range(1, 7):
        text = " ".join(f"w{i}x{j}" for j in range(80))
        nodes.append((f"n{i}", "n0", 1000 + i, text, i))
    event = make_event(nodes)
    snap = build_snapshot(event, 60)
    units = build_evidence_units(snap)
    selector = EvidenceBudgetSelector(_Tok(), budget=512)
    scores = [float(len(units) - i) for i in range(len(units))]
    accepted, tokens = selector.select(units, scores)
    assert 0 < len(accepted) < len(units)
    assert tokens <= 512
    # whole pairs only: the token count equals the sum of the rendered
    # accepted units exactly (indices renumbered in acceptance order)
    assert tokens == sum(selector.count_tokens(render_evidence(u, i + 1))
                         for i, u in enumerate(accepted))
    # deterministic: same inputs -> same acceptance
    again, tokens2 = selector.select(units, scores)
    assert [u["node_id"] for u in again] == [u["node_id"] for u in accepted]
    assert tokens2 == tokens


def test_budget_grid_frozen():
    import pytest
    with pytest.raises(ValueError):
        EvidenceBudgetSelector(_Tok(), budget=40)
    with pytest.raises(ValueError):
        EvidenceBudgetSelector(_Tok(), budget=777)


def test_prompt_no_gold():
    snap = _snapshot()
    units = build_evidence_units(snap)
    packed = pack_context("source claim", snap, units)
    prompt = packed["prompt"]
    low = prompt.lower()
    assert "gold" not in low
    assert f"label: {snap['label']}" not in low
    assert packed["observed_replies"] == len(snap["node_ids"]) - 1


def test_prompt_no_future():
    from ..evaluation.leakage_scanner import scan_prompt
    event = simple_event()
    snap = build_snapshot(event, 60)
    units = build_evidence_units(snap)
    packed = pack_context("source claim", snap, units)
    report = scan_prompt(packed["prompt"], event, snap)
    assert report["pass"], report


def test_prompt_frozen_template_fields():
    snap = _snapshot()
    packed = pack_context("source claim", snap, [])
    assert packed["prompt"].startswith("SOURCE CLAIM\nsource claim")
    assert "elapsed_time = 1h" in packed["prompt"]
    # 60-minute snapshot holds n0, n1, n3, n4 -> 3 observed replies
    assert "observed_replies = 3" in packed["prompt"]
    assert "TASK" in packed["prompt"]
    assert "Return exactly one label" in packed["prompt"]
    assert packed["prompt"].strip().endswith("Return exactly one label.")


def test_format_cutoff_labels():
    assert format_cutoff("SOURCE_ONLY") == "SOURCE_ONLY"
    assert format_cutoff(5) == "5m"
    assert format_cutoff(60) == "1h"
    assert format_cutoff(1440) == "24h"


def _raw_order_event():
    """Adversarial ordering: raw node order differs from timestamp order and
    the source is NOT the first raw node (delta-fix §12)."""
    return make_event([
        ("reply_A", "src", 1000 + 900, "early reply text A", 0),
        ("src", None, 1000, "the real source claim", 1),
        ("reply_B", "src", 1000 + 300, "later raw reply text B", 2),
    ], source_id="src")


def test_source_text_correct_when_raw_order_differs():
    from ..data.snapshot_builder import build_source_only
    event = _raw_order_event()
    # snapshot order is timestamp-sorted: src(1000), reply_B(1300), reply_A
    snap = build_snapshot(event, 60)
    src_pos = snap["node_ids"].index(snap["source_id"])
    source_text = snap["texts"][src_pos]
    assert source_text == "the real source claim"
    # the old cross-index pattern would have returned reply_A's text — the
    # source-only snapshot must also carry the true source text
    solo = build_source_only(event)
    assert solo["texts"][0] == "the real source claim"


def test_prompt_source_bound_by_source_id():
    event = _raw_order_event()
    snap = build_snapshot(event, 60)
    src_pos = snap["node_ids"].index(snap["source_id"])
    packed = pack_context(snap["texts"][src_pos], snap, [])
    # SOURCE CLAIM section must contain exactly the source_id's text
    claim_block = packed["prompt"].split("SOURCE CLAIM\n")[1].split(
        "\n\nCURRENT SNAPSHOT")[0]
    assert claim_block == "the real source claim"
    assert "early reply text A" not in claim_block
    assert "later raw reply text B" not in claim_block
