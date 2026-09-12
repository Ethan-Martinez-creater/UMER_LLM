"""Dynamic V2 (MF-TSR) integration tests (execution protocol §38).

Case A: memory evidence still has high marginal fidelity -> survival
Case B: old evidence stopped contributing            -> REMOVE
Case C: new evidence clearly improves fidelity       -> ADD (budget slack)
Case D: budget full, new evidence more valuable      -> SWAP
Case E: static set already optimal                   -> no meaningless edit
Case F: no candidates -> empty selection, prediction still generated

All cases drive the real refine_set/trajectory machinery with a
table-driven distortion oracle, so the exact expected move is analytically
pinned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tcdscr.models.marginal_set_refiner as msr  # noqa: E402
from ..models.marginal_set_refiner import refine_set  # noqa: E402


def _install_table(monkeypatch, table, default=300.0):
    def fake(p_full, h_source, node_repr, masks, proxy):
        out = []
        for k in range(masks.shape[0]):
            s = tuple(int(i) for i in masks[k].nonzero(as_tuple=True)[0])
            out.append(table.get(s, default))
        return torch.tensor(out, dtype=torch.float32)
    monkeypatch.setattr(msr, "batch_set_distortions", fake)


def _run(monkeypatch, table, u, costs, memory, budget, epsilon=0.01,
         default=300.0):
    m = len(costs)
    _install_table(monkeypatch, table, default)
    return refine_set(torch.zeros(m, 768), torch.zeros(768),
                      torch.tensor([0.4, 0.6]), u, costs,
                      list(range(1, m + 1)),
                      [f"n{i}" for i in range(m)], memory, object(),
                      budget=budget, epsilon=epsilon)


def test_case_a_memory_evidence_survives(monkeypatch):
    # n1 (memory evidence) contributes: removing it strictly worsens
    # fidelity -> the set is left alone and n1 survives
    out = _run(monkeypatch, {(1,): 90.0, (): 200.0, (0,): 200.0,
                             (0, 1): 200.0},
               u=[0.1, 0.9], costs=[10, 10], memory=[1], budget=10)
    assert out["accepted_moves"] == []
    assert out["selected_node_ids"] == ["n1"]
    assert not out["fallback_to_static"]


def test_case_b_stale_evidence_removed(monkeypatch):
    # memory n1 wins the init comparison but no longer contributes:
    # D({}) < D({n1}) -> REMOVE is the executed move
    out = _run(monkeypatch, {(1,): 90.0, (): 80.0, (0,): 100.0,
                             (0, 1): 120.0},
               u=[0.9, 0.8], costs=[10, 10], memory=[1], budget=10)
    assert out["init_source"] == "memory"
    types = [mv["move_type"] for mv in out["accepted_moves"]]
    assert types and types[0] == "REMOVE"
    assert out["accepted_moves"][0]["removed_node_id"] == "n1"


def test_case_c_new_evidence_added(monkeypatch):
    # warm start stops before the small n2, leaving budget slack, and n2
    # clearly improves fidelity -> ADD
    out = _run(monkeypatch, {(1,): 95.0, (1, 2): 70.0, (0, 2): 100.0,
                             (0,): 120.0, (0, 1): 130.0, (): 140.0,
                             (2,): 110.0},
               u=[0.9, 0.8, 0.1], costs=[10, 10, 6], memory=[1],
               budget=16)
    assert out["init_source"] == "memory"
    assert out["accepted_moves"][0]["move_type"] == "ADD"
    assert out["accepted_moves"][0]["added_node_id"] == "n2"
    assert sorted(out["selected_node_ids"]) == ["n1", "n2"]
    assert out["evidence_tokens"] == 16


def test_case_d_budget_full_swap(monkeypatch):
    # no slack for ADD; swapping stale n1 for fresh n2 improves fidelity
    out = _run(monkeypatch, {(0, 1): 100.0, (0, 2): 60.0,
                             (1, 2): 90.0, (0,): 110.0, (1,): 130.0,
                             (2,): 70.0, (): 200.0},
               u=[0.9, 0.8, 0.7], costs=[10, 10, 10], memory=[],
               budget=20)
    assert out["accepted_moves"][0]["move_type"] == "SWAP"
    assert out["accepted_moves"][0]["added_node_id"] == "n2"
    assert out["accepted_moves"][0]["removed_node_id"] == "n1"
    assert set(out["selected_node_ids"]) == {"n0", "n2"}
    assert out["evidence_tokens"] == 20


def test_case_e_static_set_untouched(monkeypatch):
    # every single-move neighbour is worse -> zero accepted moves
    out = _run(monkeypatch, {(0, 1): 50.0, (0,): 55.0, (1,): 55.0,
                             (): 60.0, (0, 2): 57.0, (1, 2): 57.0},
               u=[0.9, 0.8, 0.1], costs=[10, 10, 10], memory=[],
               budget=20, epsilon=0.01)
    assert out["accepted_moves"] == []
    assert out["selected_node_ids"] == ["n0", "n1"]
    assert not out["fallback_to_static"]


def test_case_f_no_candidates_prediction_still_produced():
    import tcdscr_run_dynamic_v2 as runner
    er = {"event_id": "ev", "cutoff": 5, "label": 0,
          "cand_node_ids": [], "u": [], "costs": [], "order": [],
          "node_repr": torch.empty(0, 768), "h_source": torch.zeros(768),
          "p_full": torch.tensor([0.3, 0.7]),
          "sem_c": torch.empty(0, 384), "num_nodes": 1}

    class _Proxy:
        def classify(self, h_source, z_sel):
            return torch.tensor([0.9, 0.1])
    out_rows = runner.run_event_trajectory_v2([er], 0.05, _Proxy(), "cpu")
    assert len(out_rows) == 1
    r = out_rows[0]
    assert r["candidate_count"] == 0
    assert r["mf_tsr_selected_node_ids"] == []
    assert r["evidence_tokens"] == 0
    assert r["static_prediction"] == 0 and r["mf_tsr_prediction"] == 0
    assert r["static_distortion"] >= 0.0
    assert r["mf_tsr_distortion"] == r["static_distortion"]
    assert r["memory_current_ids"] == []
