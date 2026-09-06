"""Unit tests: baseline selectors (delta-fix §13–§23)."""
import pytest

from ..context.evidence_unit import build_evidence_units
from ..context.packer import pack_context
from ..context.token_budget import EvidenceBudgetSelector
from ..data.snapshot_builder import build_snapshot
from ..evaluation.baselines import (ContextLengthUnresolved,
                                    resolve_context_length, rank_candidates,
                                    select_all_current, select_evidence)
from .conftest import make_event


class _Tok:
    """Token stub: one token per whitespace-split piece."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _spread_event():
    """Three replies at 10s / 100s / 1000s after the source (delta-fix §16)."""
    return make_event([
        ("n0", None, 1000, "source claim", 0),
        ("a", "n0", 1010, "reply A", 1),
        ("b", "n0", 1100, "reply B", 2),
        ("c", "n0", 2000, "reply C", 3),
    ])


def test_recent_budget_prefers_latest():
    snap = build_snapshot(_spread_event(), 60)
    units = build_evidence_units(snap)
    scores = rank_candidates("recent_budget", snap, units,
                             {"dataset": "pheme"})
    order = [units[i]["node_id"] for i in
             sorted(range(len(units)),
                    key=lambda i: (-scores[i], units[i]["order"]))]
    # elapsed 1000 / 100 / 10 -> C, B, A: most recent first
    assert order == ["c", "b", "a"]


def _long_units_snapshot(n_units=6, words=60):
    nodes = [("n0", None, 1000, "source claim", 0)]
    for i in range(1, n_units + 1):
        text = " ".join(f"w{i}x{j}" for j in range(words))
        nodes.append((f"n{i}", "n0", 1000 + i, text, i))
    snap = build_snapshot(make_event(nodes), 60)
    return snap, build_evidence_units(snap)


def test_all_current_respects_context_limit():
    snap, units = _long_units_snapshot()
    budget = EvidenceBudgetSelector(_Tok(), budget=2048)
    # stub context: base prompt (~30 tokens) + whole 68-token pairs; 200
    # tokens with 8 reserved fits two whole pairs and rejects the third
    result = select_all_current(
        snap, units, budget, context_length=200, max_new_tokens=8,
        prompt_builder=lambda s, acc: pack_context(
            "source claim", s, acc)["prompt"])
    assert result["input_tokens"] + 8 <= 200
    assert 0 < result["units_after_truncation"] < result["units_before_truncation"]
    assert result["truncated"] is True
    # 4 of the 6 pairs were dropped whole, contributing their token cost
    dropped_pairs = (result["units_before_truncation"]
                     - result["units_after_truncation"])
    assert dropped_pairs == 4
    assert result["tokens_dropped"] > 0


def test_all_current_never_partial_pair():
    snap, units = _long_units_snapshot()
    budget = EvidenceBudgetSelector(_Tok(), budget=2048)
    result = select_all_current(
        snap, units, budget, context_length=200, max_new_tokens=8,
        prompt_builder=lambda s, acc: pack_context(
            "source claim", s, acc)["prompt"])
    # every accepted unit appears whole inside the final prompt
    packed = pack_context("source claim", snap,
                          result["selected_units"])["prompt"]
    for i, u in enumerate(result["selected_units"]):
        from ..context.evidence_unit import render_evidence
        assert render_evidence(u, i + 1) in packed


def test_all_current_preserves_snapshot_order():
    snap, units = _long_units_snapshot()
    budget = EvidenceBudgetSelector(_Tok(), budget=2048)
    result = select_all_current(
        snap, units, budget, context_length=200, max_new_tokens=8,
        prompt_builder=lambda s, acc: pack_context(
            "source claim", s, acc)["prompt"])
    accepted_ids = [u["node_id"] for u in result["selected_units"]]
    chronological = [nid for nid in snap["node_ids"] if nid in accepted_ids]
    assert accepted_ids == chronological


def test_all_current_requires_context_guard():
    snap, units = _long_units_snapshot(n_units=2)
    budget = EvidenceBudgetSelector(_Tok(), budget=2048)
    with pytest.raises(ValueError):
        select_evidence("all_current", snap, units,
                        {"dataset": "pheme"}, budget)


def test_all_current_fits_when_small():
    snap, units = _long_units_snapshot(n_units=1, words=3)
    budget = EvidenceBudgetSelector(_Tok(), budget=2048)
    result = select_all_current(
        snap, units, budget, context_length=4096, max_new_tokens=8,
        prompt_builder=lambda s, acc: pack_context(
            "source claim", s, acc)["prompt"])
    assert result["truncated"] is False
    assert result["units_after_truncation"] == 1
    assert result["tokens_dropped"] == 0


def test_resolve_context_length_prefers_model_config():
    class Cfg:
        max_position_embeddings = 40960

    class Tok:
        model_max_length = 10 ** 33  # the famous placeholder sentinel

    assert resolve_context_length(Cfg(), Tok()) == 40960


def test_resolve_context_length_rejects_placeholder():
    class Tok:
        model_max_length = 10 ** 33

    with pytest.raises(ContextLengthUnresolved):
        resolve_context_length(None, Tok())


def test_resolve_context_length_uses_sane_tokenizer_value():
    class Tok:
        model_max_length = 131072

    assert resolve_context_length(None, Tok()) == 131072
