"""MS-TSR (Dynamic V3-A) unit tests.

Pins the sufficiency semantics (decision preservation + margin retention),
the Dual-View Consensus Gate, the forward reader-margin-gain/token
construction, backward redundancy removal and its tie-breaks, the budget
invariant, the Static fallback paths, the memory-as-tie-break (never a
bonus) rule, the frozen alpha grid, the ADD+REMOVE-only design and the
validation-only protocol.

Controllable oracle: candidate rows are one-hot, so the set representation
of any candidate set decodes exactly to that set (zero vector for the empty
set), and ``_TableProxy`` returns the table logits for the decoded set.
Static packing, budget and utility ordering are therefore fully pinned and
each expected move can be derived analytically.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ..models.marginal_set_refiner import static_pack  # noqa: E402
from ..models.minimal_set_refiner import (candidate_pool,  # noqa: E402
                                          refine_minimal_set)
from ..models.sufficiency import (ALPHA_GRID, decision_margin,  # noqa: E402
                                  dual_view_agreement, reader_efficiency,
                                  sufficiency_conditions)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class _TableProxy:
    """logits of a candidate set decoded from its set representation."""

    def __init__(self, table, default=(-50.0, 0.0)):
        self.table = {frozenset(k): v for k, v in table.items()}
        self.default = default

    def _for_z(self, z):
        key = frozenset(int(i) for i in (z > 0).nonzero(as_tuple=True)[0])
        return torch.tensor(self.table.get(key, self.default),
                            dtype=torch.float32)

    def classify(self, h_source, z_sel):
        if z_sel.dim() == 2:
            return torch.stack([self._for_z(z) for z in z_sel])
        return self._for_z(z_sel)


def _onehot(m, dim=768):
    r = torch.zeros(m, dim)
    for i in range(m):
        r[i, i] = 1.0
    return r


def _run(table, u, costs, memory=(), alpha=0.9, budget=1024,
         p_full=(0.9, 0.1), default=(-50.0, 0.0), top_k=64):
    m = len(costs)
    order = list(range(1, m + 1))
    node_ids = [f"n{i}" for i in range(m)]
    static, _tok = static_pack(costs, u, order, budget)
    proxy = _TableProxy(table, default)
    return refine_minimal_set(_onehot(m), torch.zeros(768),
                              torch.tensor(p_full), u, costs, order,
                              node_ids, static, list(memory), proxy,
                              budget=budget, alpha=alpha, top_k=top_k)


# ------------------------------------------------------- §7-§8 primitives

def test_decision_margin_and_conditions():
    lg = torch.tensor([3.0, 1.0])
    assert decision_margin(lg, 0) == pytest.approx(2.0)
    assert decision_margin(lg, 1) == pytest.approx(-2.0)
    ok, m, c1 = sufficiency_conditions(lg, 0, static_margin=2.0, alpha=0.9)
    assert ok and c1 and m == pytest.approx(2.0)
    ok2, _m2, _c2 = sufficiency_conditions(lg, 0, static_margin=2.5,
                                           alpha=0.9)
    assert not ok2                     # 2.0 < 0.9 * 2.5
    bad = torch.tensor([0.5, 2.0])
    ok3, _m3, c3 = sufficiency_conditions(bad, 0, static_margin=1.0,
                                          alpha=0.8)
    assert not c3 and not ok3          # decision not preserved


def test_dual_view_gate_and_efficiency():
    assert dual_view_agreement(torch.tensor([0.8, 0.2]),
                               torch.tensor([0.7, 0.3]))
    assert not dual_view_agreement(torch.tensor([0.2, 0.8]),
                                   torch.tensor([0.7, 0.3]))
    assert reader_efficiency(0.4, 8) == pytest.approx(0.05)
    assert reader_efficiency(1.0, 0) == pytest.approx(1.0)


# --------------------------------------------------------- §10 pool

def test_candidate_pool_union_and_pruning():
    u = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    order = list(range(1, 8))
    pool, before = candidate_pool([0], [3], u, order, top_k=2)
    assert before == 5        # 1,2,4,5,6 were neither Static nor memory
    assert pool == [0, 1, 2, 3]
    pool2, before2 = candidate_pool([0], [], u, order, top_k=100)
    assert set(pool2) == set(range(7)) and before2 == 6


# ------------------------------------------------- §11 forward ADD search

def test_forward_add_uses_margin_gain_per_token():
    # Static {0} (budget 10) margin 2.0 -> at alpha 0.9 we need >= 1.8.
    # From the empty set (margin 0.5) n2 has the best gain/token.
    table = {(): [0.5, 0.0], (0,): [2.0, 0.0], (1,): [1.0, 0.0],
             (2,): [2.5, 0.0], (3,): [0.6, 0.0], (2, 0): [2.6, 0.0],
             (2, 1): [2.6, 0.0], (2, 3): [2.6, 0.0]}
    res = _run(table, u=[0.9, 0.5, 0.4, 0.3], costs=[10, 2, 5, 1],
               alpha=0.9, budget=10)
    assert res["dual_view_agree"] is True
    assert res["compression_attempted"] is True
    assert [mv["node_id"] for mv in res["accepted_adds"]] == ["n2"]
    mv = res["accepted_adds"][0]
    assert mv["token_cost"] == 5
    assert mv["gain"] == pytest.approx((2.5 - 0.5) / 5)
    assert res["selected_node_ids"] == ["n2"]
    assert res["evidence_tokens"] == 5
    assert res["ms_margin"] == pytest.approx(2.5)
    assert res["sufficiency_reached"] is True
    assert res["fallback_to_static"] is False


def test_empty_set_can_be_sufficient():
    """The formal objective minimises cost: check empty first, then grow."""
    table = {(): [2.0, 0.0], (0,): [2.0, 0.0]}
    res = _run(table, u=[0.9], costs=[10], alpha=0.9)
    assert res["selected_node_ids"] == []
    assert res["evidence_tokens"] == 0
    assert res["add_count"] == 0 and res["remove_count"] == 0
    assert res["sufficiency_reached"] is True


def test_alpha_monotonicity_requires_more_margin():
    # Static {0,1} margin 2.1. alpha=0.90 -> 1.89 is reachable with n1 alone
    # (margin 1.9, 1 token); alpha=1.00 needs 2.1, i.e. the Static set.
    table = {(): [0.5, 0.0], (0,): [2.0, 0.0], (1,): [1.9, 0.0],
             (0, 1): [2.1, 0.0]}
    low = _run(table, u=[0.9, 0.5], costs=[10, 1], alpha=0.90)
    high = _run(table, u=[0.9, 0.5], costs=[10, 1], alpha=1.00)
    assert low["static_margin"] == pytest.approx(2.1)
    assert low["evidence_tokens"] == 1
    assert low["ms_margin"] == pytest.approx(1.9)
    assert high["evidence_tokens"] == 11
    assert high["ms_margin"] == pytest.approx(2.1)
    assert low["ms_margin"] >= 0.90 * low["static_margin"] - 1e-6
    assert high["ms_margin"] >= 1.00 * high["static_margin"] - 1e-6


# ---------------------------------------------- §12 backward REMOVE pass

def test_backward_removal_drops_redundant_evidence():
    # forward path: n1 (margin 1.0, gain 1.0) then n2 (1.9, gain 0.45);
    # afterwards n2 alone is sufficient (1.85) so n1 is redundant.
    table = {(): [0.0, 0.0], (0,): [2.0, 0.0], (1,): [1.0, 0.0],
             (2,): [1.85, 0.0], (1, 2): [1.9, 0.0], (0, 1): [2.1, 0.0],
             (0, 2): [2.2, 0.0]}
    res = _run(table, u=[0.9, 0.5, 0.4], costs=[20, 1, 2], alpha=0.9,
               budget=20)
    assert [mv["node_id"] for mv in res["accepted_adds"]] == ["n1", "n2"]
    assert [mv["node_id"] for mv in res["accepted_removes"]] == ["n1"]
    assert res["selected_node_ids"] == ["n2"]
    assert res["evidence_tokens"] == 2


def test_backward_removal_prefers_max_token_saving():
    # forward path: n1 (0.9, gain 0.90) -> n2 (1.5, gain 0.30) -> n3 (2.5,
    # gain 0.20). Removal candidates at S={n1,n2,n3}: drop n1 -> {n2,n3}
    # (1.9, saves 1 token) and drop n2 -> {n1,n3} (1.85, saves 2 tokens);
    # the larger saving must win.
    table = {(): [0.0, 0.0], (0,): [2.0, 0.0], (1,): [0.9, 0.0],
             (1, 2): [1.5, 0.0], (1, 3): [1.85, 0.0], (1, 2, 3): [2.5, 0.0],
             (2, 3): [1.9, 0.0]}
    res = _run(table, u=[0.9, 0.5, 0.4, 0.3], costs=[20, 1, 2, 5],
               alpha=0.9, budget=20)
    assert [mv["node_id"] for mv in res["accepted_adds"]] == \
        ["n1", "n2", "n3"]
    assert [mv["node_id"] for mv in res["accepted_removes"]] == ["n2"]
    assert sorted(res["selected_node_ids"]) == ["n1", "n3"]
    assert res["evidence_tokens"] == 6


# -------------------------------------------------- §9 budget / §6 fallback

def test_budget_never_exceeded_and_fallback_on_exhaustion():
    # Static {0} (10 tokens, margin 2.0) is sufficient only through n0, but
    # greedy first buys n1 (gain 1.5/6 > 2.0/10) and then n2, filling the
    # 15-token budget before n0 can fit -> no feasible ADD reaches
    # sufficiency -> Static fallback.
    table = {(): [0.0, 0.0], (0,): [2.0, 0.0], (1,): [1.5, 0.0],
             (2,): [0.8, 0.0], (1, 2): [1.6, 0.0]}
    res = _run(table, u=[0.9, 0.5, 0.4], costs=[10, 6, 6], alpha=0.9,
               budget=15)
    assert res["evidence_tokens"] <= 15
    assert res["fallback_to_static"] is True
    assert res["fallback_reason"] == "budget_exhausted_before_sufficiency"
    assert set(res["selected_node_ids"]) == set(res["static_node_ids"])


def test_dual_view_disagreement_keeps_static():
    table = {(): [2.0, 0.0], (0,): [2.0, 0.0]}
    res = _run(table, u=[0.9], costs=[10], alpha=0.9, p_full=(0.1, 0.9))
    assert res["dual_view_agree"] is False
    assert res["compression_attempted"] is False
    assert res["fallback_to_static"] is True
    assert res["fallback_reason"] == "dual_view_disagreement"
    assert res["selected_node_ids"] == res["static_node_ids"]
    assert res["evidence_tokens"] == res["static_tokens"]


def test_sufficiency_never_increases_tokens():
    table = {(): [0.0, 0.0], (0,): [2.0, 0.0], (1,): [0.9, 0.0],
             (2,): [0.95, 0.0], (0, 1): [2.1, 0.0], (0, 2): [2.1, 0.0]}
    res = _run(table, u=[0.9, 0.5, 0.4], costs=[10, 1, 1], alpha=0.9)
    assert res["evidence_tokens"] <= res["static_tokens"]


# ------------------------------------------- §11/§14 tie-break semantics

def test_memory_is_tie_break_not_bonus():
    # two equal-gain ADDs; the memory node wins the tie although its u is low
    table = {(): [0.0, 0.0], (0,): [2.0, 0.0], (1,): [0.9, 0.0],
             (2,): [0.9, 0.0], (0, 2): [1.85, 0.0]}
    res = _run(table, u=[0.9, 0.9, 0.1], costs=[10, 2, 2], alpha=0.9,
               budget=10, memory=[2])
    first = res["accepted_adds"][0]
    assert first["node_id"] == "n2"
    assert first["in_previous_memory"] is True
    assert first["delta_margin"] == pytest.approx(0.9)


def test_memory_never_beats_a_higher_gain():
    table = {(): [0.0, 0.0], (0,): [2.0, 0.0], (1,): [1.5, 0.0],
             (2,): [0.2, 0.0], (0, 1): [1.9, 0.0], (1, 2): [1.6, 0.0],
             (0, 1, 2): [2.0, 0.0]}
    res = _run(table, u=[0.9, 0.4, 0.3], costs=[10, 2, 2], alpha=0.9,
               budget=10, memory=[2])
    first = res["accepted_adds"][0]
    assert first["node_id"] == "n1"      # higher gain wins
    assert first["in_previous_memory"] is False


# ------------------------------------------------------- frozen grid/guards

def test_alpha_grid_is_frozen():
    assert ALPHA_GRID == (0.80, 0.90, 0.95, 1.00)
    src = (SCRIPTS_DIR / "tcdscr_run_dynamic_v3.py").read_text(
        encoding="utf-8")
    assert "from tcdscr.models.sufficiency import ALPHA_GRID" in src
    assert "ALPHA_GRID = " not in src      # never redefined locally
    assert "for alpha in ALPHA_GRID" in src


def test_no_swap_and_no_teacher_fidelity_in_primary_v3():
    for fname in ("sufficiency.py", "minimal_set_refiner.py"):
        src = (MODELS_DIR / fname).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "SWAP":
                pytest.fail(f"{fname}: SWAP literal in primary V3")
            if isinstance(node, (ast.Name, ast.Attribute)):
                nm = node.id if isinstance(node, ast.Name) else node.attr
                assert "distortion" not in nm, (fname, nm)
    params = set(inspect.signature(refine_minimal_set).parameters)
    assert {"alpha", "budget", "top_k"} <= params
    assert not any("novelty" in p for p in params)


def test_validation_never_reads_test_or_calls_qwen():
    # the verifier legitimately rebuilds the split to audit disjointness
    # from the test ids; it never reads test labels or predictions.
    for fname in ("tcdscr_run_dynamic_v3.py",
                  "tcdscr_summarize_dynamic_v3.py"):
        src = (SCRIPTS_DIR / fname).read_text(encoding="utf-8")
        body = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
        assert '["test"]' not in body and "['test']" not in body, fname
    src = (SCRIPTS_DIR / "tcdscr_run_dynamic_v3.py").read_text(
        encoding="utf-8")
    assert "AutoTokenizer" in src
    assert "AutoModel" not in src and "generate(" not in src
