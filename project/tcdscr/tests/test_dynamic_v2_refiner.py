"""Dynamic V2 (MF-TSR) unit tests (execution protocol §37).

Pins the set-fidelity objective (forward KL, zero-vector empty set, no
gold label), the warm-start survival semantics, the exact REMOVE/ADD/SWAP
relative gains, the budget/atomicity invariants, static fallback, the
memory-preserving tie-break, the 20-step cap, the absence of any
novelty/persistence term from the objective, fold-local epsilon selection
and the validation-only protocol (incl. the no-candidate protocol fix).
"""
from __future__ import annotations

import ast
import inspect
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tcdscr.models.marginal_set_refiner as msr  # noqa: E402
from ..models.marginal_set_refiner import (incoming_pool, refine_set,  # noqa: E402
                                           relative_gain, static_pack)
from ..models.set_fidelity import (batch_set_distortions,  # noqa: E402
                                   set_distortion, set_representation)
from ..models.temporal_survival import (feasible_memory_set,  # noqa: E402
                                        memory_warm_start)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _install_table(monkeypatch, fn):
    """Replace batch distortion with a table-driven function of the set."""
    def fake(p_full, h_source, node_repr, masks, proxy):
        out = []
        for k in range(masks.shape[0]):
            s = tuple(int(i) for i in masks[k].nonzero(as_tuple=True)[0])
            out.append(fn(s))
        return torch.tensor(out, dtype=torch.float32)
    monkeypatch.setattr(msr, "batch_set_distortions", fake)
    return fake


def _linear(v, A=100.0):
    return lambda s: A + sum(v[i] for i in s)


def _dmap(table, default=300.0):
    def fn(s):
        return table.get(tuple(s), default)
    return fn


# --------------------------------------------------------------- §7-§9

def test_empty_set_proxy_uses_zero_vector():
    z = set_representation(torch.empty(0, 768))
    assert torch.equal(z, torch.zeros(768))

    class _Spy:
        def classify(self, h_source, z_sel):
            self.last = z_sel
            return torch.tensor([0.0, 0.0])
    spy = _Spy()
    set_distortion(torch.tensor([0.5, 0.5]), torch.randn(768),
                   torch.empty(0, 768), spy)
    assert torch.equal(spy.last, torch.zeros(768))


def test_set_distortion_uses_forward_kl():
    from ..models.selector_proxy import SelectorProxy
    torch.manual_seed(0)
    proxy = SelectorProxy().eval()
    h = torch.randn(768)
    sel = torch.randn(3, 768)
    p = torch.tensor([0.7, 0.3])
    with torch.no_grad():
        logits = proxy.classify(h, sel.mean(0))
    log_q = F.log_softmax(logits, dim=-1)
    q = log_q.exp()
    log_p = p.clamp_min(1e-12).log() - torch.log(
        p.clamp_min(1e-12).sum())
    forward = float((p * (log_p - log_q)).sum())
    reverse = float((q * (log_q - log_p)).sum())
    d = set_distortion(p, h, sel, proxy)
    assert d == pytest.approx(forward, abs=1e-5)
    if abs(forward - reverse) > 1e-8:
        assert d != pytest.approx(reverse, abs=1e-8)


def test_set_distortion_no_gold_label():
    for fn in (set_distortion, batch_set_distortions, set_representation):
        params = set(inspect.signature(fn).parameters)
        assert not any(n == "y" or "gold" in n or "label" in n
                       for n in params), params
    src = (MODELS_DIR / "set_fidelity.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert not any(a.arg == "y" or "gold" in a.arg
                           or "label" in a.arg for a in node.args.args), \
                node.name


# --------------------------------------------------------- §11 survival

def test_memory_warm_start_only_current_nodes():
    import tcdscr_run_dynamic_v2 as runner
    # historical ids not present in the snapshot drop out silently
    assert runner.memory_positions_for(
        ["a", "gone", "b"], ["a", "b", "c"]) == [0, 1]
    # over budget: drop the lowest current Static Utility survivor
    sel, total = feasible_memory_set([0, 1], [10, 10], [0.9, 0.2], [1, 2],
                                     15)
    assert sel == [0] and total == 10
    # fill: descending utility, stop at the first pair that does not fit
    sel, total = memory_warm_start([0], [10, 8, 8], [0.9, 0.8, 0.1],
                                   [1, 2, 3], 18)
    assert sel == [0, 1] and total == 18
    sel, total = memory_warm_start([1], [10, 8, 8], [0.9, 0.8, 0.1],
                                   [1, 2, 3], 18)
    assert sel == [1, 0] and total == 18


# ------------------------------------------------------ §13-§17 moves

def test_relative_gain_formula():
    assert relative_gain(2.0, 1.5) == pytest.approx(0.25)
    assert relative_gain(0.0, 0.0) == 0.0
    assert relative_gain(1e-9, 0.0) == pytest.approx(1e-9 / 1e-6)
    assert relative_gain(100.0, 120.0) == pytest.approx(-0.2)


def test_remove_gain_exact(monkeypatch):
    _install_table(monkeypatch, _dmap({(): 100.0, (0,): 105.0}))
    out = refine_set(torch.zeros(1, 768), torch.zeros(768),
                     torch.tensor([0.5, 0.5]), [1.0], [10], [1], ["n1"],
                     [], object(), budget=10, epsilon=0.0)
    assert len(out["accepted_moves"]) == 1
    mv = out["accepted_moves"][0]
    assert mv["move_type"] == "REMOVE"
    assert mv["removed_node_id"] == "n1"
    assert mv["added_node_id"] is None
    assert mv["distortion_before"] == pytest.approx(105.0)
    assert mv["distortion_after"] == pytest.approx(100.0)
    assert mv["relative_gain"] == pytest.approx(5.0 / 105.0)


def test_add_gain_exact(monkeypatch):
    # Static packing is locally maximal under greedy skip-and-continue, so
    # ADD has to be reached from a memory warm start that stopped early:
    # warm {n1} keeps budget slack, and n2 strictly improves fidelity.
    _install_table(monkeypatch, _dmap({
        (1,): 95.0, (1, 2): 70.0, (0, 2): 100.0, (0,): 120.0,
        (0, 1): 130.0, (): 140.0, (2,): 110.0}))
    out = refine_set(torch.zeros(3, 768), torch.zeros(768),
                     torch.tensor([0.5, 0.5]), [0.9, 0.8, 0.1],
                     [10, 10, 6], [1, 2, 3], ["a", "b", "c"],
                     [1], object(), budget=16, epsilon=0.01)
    mv = out["accepted_moves"][0]
    assert mv["move_type"] == "ADD"
    assert mv["added_node_id"] == "c"
    assert mv["removed_node_id"] is None
    assert mv["distortion_before"] == pytest.approx(95.0)
    assert mv["distortion_after"] == pytest.approx(70.0)
    assert mv["relative_gain"] == pytest.approx(25.0 / 95.0)
    assert set(out["selected_positions"]) == {1, 2}


def test_swap_gain_exact(monkeypatch):
    _install_table(monkeypatch, _linear([-1.0, 5.0, -8.0]))
    out = refine_set(torch.zeros(3, 768), torch.zeros(768),
                     torch.tensor([0.5, 0.5]), [0.9, 0.8, 0.1],
                     [10, 10, 10], [1, 2, 3], ["a", "b", "c"],
                     [], object(), budget=20, epsilon=0.01)
    mv = out["accepted_moves"][0]
    assert mv["move_type"] == "SWAP"
    assert mv["added_node_id"] == "c"
    assert mv["removed_node_id"] == "b"
    assert mv["relative_gain"] == pytest.approx(13.0 / 104.0)
    assert not out["fallback_to_static"]
    assert set(out["selected_positions"]) == {0, 2}


# ------------------------------------------------------- §10 §19 §20

def test_refiner_never_exceeds_budget(monkeypatch):
    rng = random.Random(3090)
    for trial in range(30):
        m = rng.randint(1, 12)
        budget = rng.choice([20, 26, 40])
        costs = [rng.randint(1, 15) for _ in range(m)]
        u = [rng.random() for _ in range(m)]
        v = [rng.uniform(-5, 5) for _ in range(m)]
        _install_table(monkeypatch, _linear(v))
        memory = [i for i in range(m) if rng.random() < 0.3]
        eps = rng.choice([0.0, 0.01, 0.05])
        out = refine_set(torch.zeros(m, 768), torch.zeros(768),
                         torch.tensor([0.5, 0.5]), u, costs,
                         list(range(1, m + 1)),
                         [f"n{i}" for i in range(m)], memory, object(),
                         budget=budget, epsilon=eps)
        assert out["evidence_tokens"] <= budget, (trial, out)
        assert out["evidence_tokens"] == sum(costs[i]
                                             for i in out[
                                                 "selected_positions"])
        assert out["static_tokens"] <= budget
        assert len(out["accepted_moves"]) <= 20
        if out["accepted_moves"]:
            assert all(mv["relative_gain"] >= eps - 1e-6
                       for mv in out["accepted_moves"])


def test_refiner_atomic_reply_parent_pair(monkeypatch):
    _install_table(monkeypatch, _linear([-1.0, -1.0]))
    out = refine_set(torch.zeros(2, 768), torch.zeros(768),
                     torch.tensor([0.5, 0.5]), [0.9, 0.1], [10, 10],
                     [1, 2], ["a", "b"], [], object(), budget=15,
                     epsilon=0.0)
    # whole units only: tokens are an exact sum of per-unit costs
    assert out["evidence_tokens"] == sum(
        [10, 10][i] for i in out["selected_positions"])
    assert len(out["selected_node_ids"]) * 10 == out["evidence_tokens"]


def test_refiner_static_fallback(monkeypatch):
    # memory init ties Static within 1e-6 (float32 representable at the
    # 1.0 scale) but ends 5e-7 WORSE than static with every move blocked
    # by epsilon -> §20 fallback must restore the static set.
    _install_table(monkeypatch, _dmap({
        (0,): 1.0, (1,): 1.0000005, (): 2.0, (0, 1): 2.0}))
    out = refine_set(torch.zeros(2, 768), torch.zeros(768),
                     torch.tensor([0.5, 0.5]), [0.9, 0.1], [1, 1],
                     [1, 2], ["a", "b"], [1], object(), budget=1,
                     epsilon=0.01)
    assert out["init_source"] == "memory"
    assert out["fallback_to_static"] is True
    assert out["selected_positions"] == [0]
    assert out["distortion"] == pytest.approx(1.0)


def test_refiner_tie_prefers_previous_memory(monkeypatch):
    _install_table(monkeypatch, _dmap({
        (0, 2): 100.0, (0,): 110.0, (2, 3): 90.0, (0, 3): 90.0,
        (0, 1): 300.0, (1,): 200.0}))
    out = refine_set(torch.zeros(4, 768), torch.zeros(768),
                     torch.tensor([0.5, 0.5]), [0.9, 0.8, 0.1, 0.05],
                     [10, 12, 4, 4], [1, 2, 3, 4],
                     ["m", "x", "c", "d"], [0], object(), budget=14,
                     epsilon=1e-3)
    assert out["init_source"] == "static"
    mv = out["accepted_moves"][0]
    assert mv["move_type"] == "SWAP"
    # gain ties (0.10 both): keep the memory evidence "m" -> remove "c"
    assert mv["removed_node_id"] == "c"
    assert mv["added_node_id"] == "d"
    assert "m" in out["selected_node_ids"]


def test_refiner_max_steps_20(monkeypatch):
    _install_table(monkeypatch, lambda s: 100.0)
    out = refine_set(torch.zeros(25, 768), torch.zeros(768),
                     torch.tensor([0.5, 0.5]),
                     [0.9 - 0.01 * i for i in range(25)], [1] * 25,
                     list(range(1, 26)), [f"n{i}" for i in range(25)],
                     [], object(), budget=10, epsilon=0.0)
    assert len(out["accepted_moves"]) == 20
    assert out["max_step_hit"] is True
    assert [mv["step"] for mv in out["accepted_moves"]] == list(range(20))


# ------------------------------------------------------- §22 §23 §26

def test_novelty_not_used_in_objective():
    for fname in ("set_fidelity.py", "temporal_survival.py",
                  "marginal_set_refiner.py"):
        src = (MODELS_DIR / fname).read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name,)):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.alias):
                names.add(node.name)
        for banned in ("novelty", "persistence", "recency"):
            assert not any(banned in n.lower() for n in names), (fname,
                                                                 banned)


def test_mask_rows_vectorized_matches_naive():
    """The vectorised scatter must be value-identical to per-row fill."""
    from ..models.marginal_set_refiner import _mask_rows
    sets = [set(), {0, 3}, {1, 2, 4}, {5}, set(range(6))]
    fast = _mask_rows(sets, 6, "cpu")
    naive = torch.zeros(len(sets), 6)
    for k, s in enumerate(sets):
        if s:
            naive[k, sorted(s)] = 1.0
    assert torch.equal(fast, naive)
    assert _mask_rows([], 6, "cpu").shape == (0, 6)


def test_candidate_pruning_top64(monkeypatch):
    _install_table(monkeypatch, lambda s: 100.0)
    m = 200
    out = refine_set(torch.zeros(m, 768), torch.zeros(768),
                     torch.tensor([0.5, 0.5]),
                     [1.0 - i / m for i in range(m)], [10] * m,
                     list(range(1, m + 1)), [f"n{i}" for i in range(m)],
                     [], object(), budget=20, epsilon=0.05)
    # pool after pruning is capped at the frozen 64
    assert all(after <= 64 for _, after in out["pool_sizes"])
    before, after = out["pool_sizes"][0]
    assert before == m - 2 and after == 64
    pool, size = incoming_pool({0, 1}, [1.0 - i / m for i in range(m)],
                               list(range(1, m + 1)))
    assert size == m - 2 and len(pool) == 64
    assert pool[0] == 2  # descending utility among unselected


def test_static_pack_matches_budget_selector():
    costs = [7, 9, 4, 6]
    u = [0.5, 0.9, 0.1, 0.7]
    order = [1, 2, 3, 4]
    sel, total = static_pack(costs, u, order, budget=15)
    # (-u, order): idx1(9) idx3(+6=15) idx0(22 skip) idx2(19 skip)
    assert sel == [1, 3] and total == 15


# --------------------------------------------------- §26 §36 protocol

def test_epsilon_fold_local():
    import tcdscr_summarize_dynamic_v2 as summ

    def seed_summary(mf, flip, moves):
        # same shape as tcdscr_run_dynamic_v2.summarize_run_rows output
        return {"mean_primary_macro_f1": {"static": mf, "mf_tsr": mf},
                "flip_rate": {"static": flip, "mf_tsr": flip},
                "set_mechanism": {"mean_moves_per_snapshot": moves}}
    # fold A: epsilon 0.05 best; fold B: epsilon 0.0 best -> independent
    fold_a = {"0.0": [seed_summary(0.80, 0.02, 5.0)] * 3,
              "0.01": [seed_summary(0.81, 0.015, 2.0)] * 3,
              "0.05": [seed_summary(0.83, 0.01, 0.5)] * 3}
    fold_b = {"0.0": [seed_summary(0.90, 0.03, 4.0)] * 3,
              "0.01": [seed_summary(0.89, 0.03, 2.0)] * 3,
              "0.05": [seed_summary(0.88, 0.04, 0.5)] * 3}
    assert summ.pick_fold_epsilon(fold_a) == "0.05"
    assert summ.pick_fold_epsilon(fold_b) == "0.0"
    # exact tie breaks to lower flip, then fewer moves, then smaller eps
    fold_tie = {"0.0": [seed_summary(0.85, 0.03, 1.0)] * 3,
                "0.01": [seed_summary(0.85, 0.03, 1.0)] * 3,
                "0.05": [seed_summary(0.85, 0.01, 9.0)] * 3}
    assert summ.pick_fold_epsilon(fold_tie) == "0.05"


def test_validation_never_reads_test():
    for fname in ("tcdscr_run_dynamic_v2.py",
                  "tcdscr_dynamic_v2_protocol.py",
                  "tcdscr_summarize_dynamic_v2.py"):
        src = (SCRIPTS_DIR / fname).read_text(encoding="utf-8")
        body = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
        assert '["test"]' not in body, fname
        assert "['test']" not in body, fname
        assert "validation" in body.lower(), fname
    for fname in ("tcdscr_run_dynamic_v2.py",
                  "tcdscr_dynamic_v2_protocol.py"):
        body = (SCRIPTS_DIR / fname).read_text(encoding="utf-8")
        assert "test_split_read" in body, fname


def test_no_candidate_snapshot_not_skipped():
    """E2 arms and the V2 trajectory must keep candidate_count == 0 rows."""
    import tcdscr_run_dynamic_v2 as runner
    import tcdscr_run_e2 as e2
    from ..data.snapshot_builder import build_snapshot
    from .conftest import make_event

    # --- V2 trajectory level ----------------------------------------
    er = {"event_id": "ev", "cutoff": 5, "label": 1,
          "cand_node_ids": [], "u": [], "costs": [], "order": [],
          "node_repr": torch.empty(0, 768), "h_source": torch.zeros(768),
          "p_full": torch.tensor([0.4, 0.6]),
          "sem_c": torch.empty(0, 384), "num_nodes": 1}

    class _FixedProxy:
        def classify(self, h_source, z_sel):
            if z_sel.dim() == 2:
                return torch.tensor([0.1, 0.9]).expand(z_sel.shape[0], 2)
            return torch.tensor([0.1, 0.9])
    rows = runner.run_event_trajectory_v2([er], 0.0, _FixedProxy(), "cpu")
    assert len(rows) == 1
    r = rows[0]
    assert r["candidate_count"] == 0
    assert r["static_prediction"] == 1 and r["mf_tsr_prediction"] == 1
    assert r["static_selected_node_ids"] == []
    assert r["mf_tsr_selected_node_ids"] == []
    assert r["memory_current_ids"] == [] and r["memory_previous_ids"] == []
    assert r["static_evidence_tokens"] == 0
    assert r["mf_tsr_evidence_tokens"] == 0

    # --- corrected E2 evaluate_arms_on_items level -------------------
    class _Enc:
        def __call__(self, node_feat, struct, num_nodes):
            b, nmax, _ = node_feat.shape
            return (torch.zeros(b, nmax, 768), torch.zeros(b, 768),
                    torch.tensor([[0.2, 0.8]]).expand(b, 2).contiguous())

    class _Sel:
        def __call__(self, *a):
            return torch.zeros(a[0].shape[0])

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(max(len(text.split()), 1)))}

    event = make_event([
        ("n0", None, 1000, "source claim", 0),
        ("n1", "n0", 1000 + 7200, "reply one", 1),
    ])
    store_rows = {nid: torch.zeros(384) for nid in ("n0", "n1")}
    items = [e2.build_light_item(event, build_snapshot(event, 5),
                                 store_rows),
             e2.build_light_item(event, build_snapshot(event, 360),
                                 store_rows)]
    assert items[0]["num_nodes"] == 1  # source-only at 5m
    from ..context.token_budget import EvidenceBudgetSelector
    metrics, rows, _diag = e2.evaluate_arms_on_items(
        _Enc(), _Sel(), _FixedProxy(), items,
        EvidenceBudgetSelector(_Tok(), 1024), "cpu", "pheme", 2000)
    assert metrics["5"]["static"]["n"] == 1
    assert metrics["5"]["random"]["n"] == 1
    assert metrics["5"]["semantic"]["n"] == 1
    no_cand = [x for x in rows if x["cutoff"] == "5"]
    assert len(no_cand) == 3
    assert all(x["selected_node_ids"] == [] and x["evidence_tokens"] == 0
               and x["pred"] == 1 for x in no_cand)
