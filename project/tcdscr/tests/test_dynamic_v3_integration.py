"""MS-TSR (Dynamic V3-A) integration tests.

Case A: Dual-View disagreement          -> Static preserved, no compression
Case B: forward ADD                     -> fewer tokens, decision preserved
Case C: backward REMOVE                 -> redundant evidence dropped
Case D: budget exhaustion               -> Static fallback
Case E: empty set already sufficient    -> maximal compression
Case F: no-candidate snapshot           -> empty sets, prediction, memory cleared

Plus the §16 fold-local alpha rule, the §17 dataset gate and the §19
margin-stratified reporting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tcdscr_summarize_dynamic_v3 as summ  # noqa: E402
import tcdscr_run_dynamic_v3 as runner  # noqa: E402
from ..models.marginal_set_refiner import static_pack  # noqa: E402
from ..models.minimal_set_refiner import refine_minimal_set  # noqa: E402


class _TableProxy:
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


class _FixedProxy:
    def __init__(self, logits):
        self.logits = logits

    def classify(self, h_source, z_sel):
        n = z_sel.shape[0] if z_sel.dim() == 2 else 1
        return torch.tensor(self.logits, dtype=torch.float32).expand(n, 2)


def _onehot(m, dim=768):
    r = torch.zeros(m, dim)
    for i in range(m):
        r[i, i] = 1.0
    return r


def _case(table, u, costs, budget, alpha=0.9, memory=(), p_full=(0.9, 0.1)):
    m = len(costs)
    order = list(range(1, m + 1))
    static, _ = static_pack(costs, u, order, budget)
    return refine_minimal_set(_onehot(m), torch.zeros(768),
                              torch.tensor(p_full), u, costs, order,
                              [f"n{i}" for i in range(m)], static,
                              list(memory), _TableProxy(table),
                              budget=budget, alpha=alpha)


def test_case_a_disagreement_preserves_static():
    table = {(): [2.0, 0.0], (0,): [2.0, 0.0], (1,): [3.0, 0.0]}
    res = _case(table, u=[0.9, 0.8], costs=[10, 1], budget=10,
                p_full=(0.05, 0.95))
    assert res["dual_view_agree"] is False
    assert res["fallback_to_static"] is True
    assert res["compression_attempted"] is False
    assert res["selected_node_ids"] == res["static_node_ids"]


def test_case_b_forward_add_compresses():
    table = {(): [0.0, 0.0], (0,): [2.0, 0.0], (1,): [1.9, 0.0]}
    res = _case(table, u=[0.9, 0.1], costs=[10, 2], budget=10)
    assert res["static_tokens"] == 10
    assert res["evidence_tokens"] == 2
    assert res["selected_node_ids"] == ["n1"]
    assert res["ms_margin"] >= 0.9 * res["static_margin"]
    assert res["fallback_to_static"] is False


def test_case_c_backward_remove():
    table = {(): [0.0, 0.0], (0,): [2.0, 0.0], (1,): [1.0, 0.0],
             (2,): [1.85, 0.0], (1, 2): [1.9, 0.0], (0, 1): [2.1, 0.0],
             (0, 2): [2.2, 0.0]}
    res = _case(table, u=[0.9, 0.5, 0.4], costs=[20, 1, 2], budget=20)
    assert [mv["node_id"] for mv in res["accepted_removes"]] == ["n1"]
    assert res["selected_node_ids"] == ["n2"]


def test_case_d_budget_exhaustion_falls_back():
    table = {(): [0.0, 0.0], (0,): [2.0, 0.0], (1,): [1.5, 0.0],
             (2,): [0.8, 0.0], (1, 2): [1.6, 0.0]}
    res = _case(table, u=[0.9, 0.5, 0.4], costs=[10, 6, 6], budget=15)
    assert res["fallback_to_static"] is True
    assert set(res["selected_node_ids"]) == set(res["static_node_ids"])
    assert res["evidence_tokens"] == res["static_tokens"]


def test_case_e_empty_set_sufficient():
    table = {(): [2.0, 0.0], (0,): [2.0, 0.0]}
    res = _case(table, u=[0.9], costs=[10], budget=10)
    assert res["selected_node_ids"] == []
    assert res["evidence_tokens"] == 0
    assert res["sufficiency_reached"] is True
    assert res["ms_margin"] >= 0.9 * res["static_margin"]


def test_case_f_no_candidate_snapshot_and_memory_reset():
    """No-candidate snapshot: empty sets, prediction, memory cleared."""
    er0 = {"event_id": "e", "cutoff": 5, "label": 1, "cand_node_ids": [],
           "u": [], "costs": [], "order": [], "node_repr": torch.empty(0, 768),
           "h_source": torch.zeros(768), "p_full": torch.tensor([0.3, 0.7]),
           "sem_c": torch.empty(0, 384), "num_nodes": 1}
    er1 = {"event_id": "e", "cutoff": 15, "label": 1,
           "cand_node_ids": ["n1", "n2"], "u": [0.9, 0.1],
           "costs": [10, 2], "order": [1, 2],
           "node_repr": _onehot(2), "h_source": torch.zeros(768),
           "p_full": torch.tensor([0.9, 0.1]),
           "sem_c": torch.zeros(2, 384), "num_nodes": 3}
    proxy = _TableProxy({(): [0.0, 0.0], (0,): [2.0, 0.0],
                         (1,): [1.9, 0.0]})
    rows = runner.run_event_trajectory_v3([er0, er1], 0.9, proxy, "cpu",
                                          budget=10)
    r0, r1 = rows
    assert r0["candidate_count"] == 0
    assert r0["static_selected_node_ids"] == []
    assert r0["ms_selected_node_ids"] == []
    assert r0["evidence_tokens"] == 0
    assert r0["fallback_reason"] == "no_candidates"
    # the empty-set row still produces a prediction (argmax of the zero-set
    # logits), and the dual-view flag is recorded against p_full
    assert r0["static_prediction"] == r0["ms_prediction"] == 0
    assert r0["dual_view_agree"] is False   # p_full argues class 1 here
    # memory was reset by the no-candidate snapshot
    assert r1["memory_previous_ids"] == []
    assert r1["alpha"] == 0.9
    assert r1["ms_prediction"] == r1["static_prediction"] == 0


# ------------------------------------------- §16/§17 selection and gate

def _seed_summary(mf_static, mf_ms, flip_static, flip_ms, reduction):
    return {"mean_primary_macro_f1": {"static": mf_static, "ms": mf_ms},
            "flip_rate": {"static": flip_static, "ms": flip_ms},
            "compression": {"mean_token_reduction": reduction},
            "set_mechanism": {"mean_moves_per_snapshot": 1.0}}


def test_fold_local_alpha_selection_rule():
    # alpha 1.0 has the largest reduction AND is non-inferior -> chosen
    fold_a = {
        "0.8": [_seed_summary(0.90, 0.880, 0.02, 0.02, 0.70)] * 3,
        "0.9": [_seed_summary(0.90, 0.899, 0.02, 0.02, 0.55)] * 3,
        "0.95": [_seed_summary(0.90, 0.902, 0.02, 0.02, 0.40)] * 3,
        "1.0": [_seed_summary(0.90, 0.901, 0.02, 0.02, 0.30)] * 3}
    best, stats = summ.pick_fold_alpha(fold_a)
    assert stats["0.8"]["eligible"] is False      # delta MF1 = -0.02
    assert stats["0.9"]["eligible"] is True       # delta MF1 = -0.001
    assert stats["0.95"]["eligible"] is True
    # among the eligible alphas the largest token reduction wins: 0.9 (0.55)
    assert best == "0.9"


def test_fold_local_alpha_none_eligible_is_not_supported():
    fold = {"0.9": [_seed_summary(0.90, 0.85, 0.02, 0.02, 0.80)] * 3,
            "1.0": [_seed_summary(0.90, 0.86, 0.02, 0.02, 0.70)] * 3}
    best, stats = summ.pick_fold_alpha(fold)
    assert best is None
    assert not any(v["eligible"] for v in stats.values())


def test_fold_local_alpha_flip_bound_rejects():
    # huge token reduction but +20% flips -> ineligible
    fold = {"0.9": [_seed_summary(0.90, 0.90, 0.02, 0.024, 0.90)] * 3}
    best, stats = summ.pick_fold_alpha(fold)
    assert best is None
    assert stats["0.9"]["eligible"] is False


def test_dataset_gate_and_decision():
    gate, conds = summ.gate_verdict(0.001, 0.45, 0.02)
    assert gate == "MS_TSR_PROXY_PASS" and all(conds.values())
    gate2, conds2 = summ.gate_verdict(-0.01, 0.45, 0.02)
    assert gate2 == "MS_TSR_NOT_SUPPORTED"
    assert conds2["classification_non_inferiority"] is False
    gate3, _ = summ.gate_verdict(0.0, 0.20, 0.01)
    assert gate3 == "MS_TSR_NOT_SUPPORTED"        # < 30% reduction
    gate4, _ = summ.gate_verdict(0.0, 0.50, 0.09)
    assert gate4 == "MS_TSR_NOT_SUPPORTED"        # > 5% flip increase
    assert summ.dataset_decision(
        {"pheme": {"gate": "MS_TSR_PROXY_PASS"},
         "maweibo": {"gate": "MS_TSR_PROXY_PASS"}})[1] == \
        "START_V3_B_READER_TRANSFER_PILOT"
    assert summ.dataset_decision(
        {"pheme": {"gate": "MS_TSR_PROXY_PASS"},
         "maweibo": {"gate": "MS_TSR_NOT_SUPPORTED"}})[1] == \
        "STOP_FOR_RESEARCH_REVIEW"
    assert summ.dataset_decision(
        {"pheme": {"gate": "MS_TSR_NOT_SUPPORTED"},
         "maweibo": {"gate": "MS_TSR_NOT_SUPPORTED"}})[1] == \
        "MS_TSR_NOT_SUPPORTED_REDESIGN"


def test_margin_stratified_bins():
    rows = []
    for i, (m, gold, st, ms) in enumerate([
            (0.1, 1, 1, 1), (0.2, 1, 1, 0), (0.5, 0, 0, 0),
            (0.6, 0, 0, 0), (0.9, 1, 1, 1), (1.2, 1, 1, 1)]):
        rows.append({
            "static_margin": m, "gold": gold,
            "static_prediction": st, "ms_prediction": ms,
            "static_evidence_tokens": 100, "ms_evidence_tokens": 40,
            "fallback_to_static": False, "dual_view_agree": True})
    strata = summ.margin_strata(rows)
    assert set(strata) == {"low_confidence", "medium", "high_confidence"}
    assert strata["low_confidence"]["n_snapshots"] == 2
    assert strata["low_confidence"]["delta_macro_f1"] < 0
    assert strata["high_confidence"]["delta_macro_f1"] == pytest.approx(0.0)
    for v in strata.values():
        assert v["mean_token_reduction"] == pytest.approx(0.6)


# --------------------------------------- verifier objective audit (gate-only)

def test_verifier_audit_accepts_gate_only_p_full():
    import tcdscr_verify_dynamic_v3 as verify
    issues = []
    verify._model_audit(issues)
    assert issues == []


@pytest.mark.parametrize("expr,caught", [
    ("(p_full_probs.view(-1) * u).sum()", True),
    ("p_full_probs.detach()", True),
    ("torch.softmax(p_full_probs, -1)", True),
    ("dual_view_agreement(p_full_probs, q_static)", False),
    ("int(p_full_probs.view(-1).argmax())", False),
])
def test_verifier_audit_flags_non_gate_p_full(expr, caught):
    import ast
    import tcdscr_verify_dynamic_v3 as verify
    tree = ast.parse("def f(p_full_probs, u, q_static, torch):\n"
                     "    return " + expr + "\n")
    assert bool(verify._p_full_objective_violations(tree)) is caught


def test_verifier_refreshes_report_verifier_line():
    import shutil
    import tcdscr_verify_dynamic_v3 as verify
    root = Path(__file__).resolve().parent / "_report_refresh_tmp"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    try:
        path = root / "MS_TSR_VALIDATION_REPORT.md"
        path.write_text("# R\n\n## Verifier\nissues = not run\n\n"
                        "## Recommendation\nOverall = X\n", encoding="utf-8")
        verify._refresh_report(str(root), {
            "n_issues": 0, "runs": 30, "rows": 319680,
            "no_candidate_rows": 41568})
        text = path.read_text(encoding="utf-8")
        assert ("issues = 0 (runs=30, rows=319680, "
                "no_candidate_rows=41568)") in text
        assert "not run" not in text
        assert "Overall = X" in text
    finally:
        shutil.rmtree(root, ignore_errors=True)
