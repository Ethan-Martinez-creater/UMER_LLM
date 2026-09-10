"""Formal E3 contract + temporal-leakage tests (E3 order §12, §39).

Pins the frozen Dynamic Evidence Memory mechanics:
- 5m dynamic ranking must equal static ranking (novelty constant, no
  persistence) — mismatch is an implementation error;
- novelty = 1 - max true cosine over M_{t-1}, 1.0 for empty memory;
- persistence is decided exclusively by M_{t-1};
- memory updates to the current cutoff's selected ids only;
- candidates always span the whole current snapshot (an old unselected node
  may enter memory later);
- memory never contains future nodes;
- dynamic score formula is exactly u + lambda_n*novelty + lambda_p*persistence;
- static and dynamic share the same budget;
- trajectory cutoffs execute in strict ascending order;
- the runner never reads the test split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tcdscr_run_e3 as e3  # noqa: E402

from ..models.dynamic_memory import (dynamic_scores, memory_embeddings,  # noqa: E402
                                     novelty_scores, persistence_flags,
                                     select_with_costs)


def _embedding_pool(n=12, dim=384, seed=7):
    torch.manual_seed(seed)
    embs = torch.randn(n, dim)
    return torch.nn.functional.normalize(embs, dim=-1)


def _make_units(node_ids):
    return [{"node_id": nid, "order": i, "reply_text": "t", "parent_id": None,
             "parent_text": None, "elapsed_seconds": 1, "depth": 1}
            for i, nid in enumerate(node_ids)]


def _make_event_rows(cutoff_nodes, u_map, costs=None, emb_pool=None,
                     n_repr_dim=768):
    """Synthetic ordered cutoffs; node 0 is always the source."""
    emb_pool = emb_pool if emb_pool is not None else _embedding_pool()
    rows = []
    for k, nids in enumerate(cutoff_nodes):
        src_pos = nids.index("n0")
        cand = [i for i, nid in enumerate(nids) if nid != "n0"]
        cand_ids = [nids[i] for i in cand]
        units = _make_units(cand_ids)
        torch.manual_seed(100 + k)
        node_repr = torch.randn(len(nids), n_repr_dim)
        units_index = {u["node_id"]: cand[i] for i, u in enumerate(units)}
        idx_of = {nid: i for i, nid in enumerate(nids)}
        sem_all = emb_pool[[idx_of[nid] for nid in nids]]
        u = torch.tensor([u_map.get(nid, 0.0) for nid in cand_ids])
        rows.append({
            "cutoff": (5, 15, 30)[k],
            "label": 1,
            "event_id": "ev1",
            "src_pos": src_pos,
            "num_nodes": len(nids),
            "node_repr": node_repr,
            "sem_all": sem_all,
            "node_ids": nids,
            "cand_node_ids": cand_ids,
            "u": u,
            "sem_c": sem_all[[idx_of[nid] for nid in cand_ids]],
            "units": units,
            "units_index": units_index,
            "costs": costs if costs is not None
            else [10] * len(units),
        })
    return rows


def _classify_fixed(h_source, sel_repr):
    return torch.tensor([1.0, 0.0])


def test_dynamic_first_cutoff_equals_static():
    rows = _make_event_rows(
        [["n0", "n1", "n2", "n3"], ["n0", "n1", "n2", "n3", "n4"]],
        {"n1": 3.0, "n2": 1.0, "n3": 2.0, "n4": 0.5})
    out, mismatch = e3.run_event_trajectory(rows, (1.0, 0.25, 20),
                                            _classify_fixed)
    assert mismatch == 0
    first = out[0]
    assert first["static_selected_node_ids"] == \
        first["dynamic_selected_node_ids"]


def test_novelty_empty_memory_is_one():
    embs = _embedding_pool(4)
    out = novelty_scores(embs, None)
    assert torch.allclose(out, torch.ones(4))
    out2 = novelty_scores(embs, torch.empty(0, 384))
    assert torch.allclose(out2, torch.ones(4))


def test_novelty_uses_true_cosine():
    m = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    same = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    orth = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    opp = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
    scaled = torch.tensor([[3.0, 0.0, 0.0, 0.0]])  # same direction, 3x norm
    n = novelty_scores(torch.cat([same, orth, opp, scaled]), m)
    assert abs(float(n[0])) < 1e-6          # cos=1 -> novelty 0
    assert abs(float(n[1]) - 1.0) < 1e-6    # cos=0 -> novelty 1
    assert abs(float(n[2]) - 2.0) < 1e-6    # cos=-1 -> novelty 2
    # normalization matters: a scaled vector has dot=3 but cosine=1
    assert abs(float(n[3])) < 1e-6


def test_persistence_only_previous_memory():
    flags = persistence_flags(["n1", "n2", "n3"], ["n1"])
    assert flags.tolist() == [1.0, 0.0, 0.0]
    assert persistence_flags(["n1", "n2"], []).tolist() == [0.0, 0.0]
    assert persistence_flags(["n1", "n2"], None).tolist() == [0.0, 0.0]
    # seen-but-never-selected nodes are NOT persistent
    assert persistence_flags(["n1", "n2"], ["n1"]).tolist() == [1.0, 0.0]


def test_memory_update_uses_current_selected_ids():
    rows = _make_event_rows(
        [["n0", "n1", "n2", "n3"], ["n0", "n1", "n2", "n3", "n4"]],
        {"n1": 3.0, "n2": 1.0, "n3": 2.0, "n4": 0.5})
    out, _ = e3.run_event_trajectory(rows, (0.0, 0.0, 20), _classify_fixed)
    for r in out:
        assert r["memory_current_ids"] == r["dynamic_selected_node_ids"]
    assert out[1]["memory_previous_ids"] == out[0]["memory_current_ids"]


def test_unselected_old_node_can_be_selected_later():
    # cutoff1: n1 has a huge cost (skipped), n2 fits -> only n2 selected
    rows = _make_event_rows(
        [["n0", "n1", "n2"], ["n0", "n1", "n2", "n3"]],
        {"n1": 10.0, "n2": 5.0, "n3": 3.0})
    for r in rows:
        r["costs"] = [30, 5] if len(r["units"]) == 2 else [12, 5, 8]
    out, _ = e3.run_event_trajectory(rows, (0.0, 0.0, 15), _classify_fixed)
    assert out[0]["dynamic_selected_node_ids"] == ["n2"]   # n1 did not fit
    assert "n1" not in out[1]["memory_previous_ids"]
    # later, with more budget room, the previously-unselected n1 is selectable
    assert "n1" in out[1]["dynamic_selected_node_ids"]


def test_memory_never_contains_future_node():
    rows = _make_event_rows(
        [["n0", "n1", "n2"], ["n0", "n1", "n2", "n3"],
         ["n0", "n1", "n2", "n3", "n4"]],
        {"n1": 3.0, "n2": 2.0, "n3": 1.0, "n4": 0.5})
    out, _ = e3.run_event_trajectory(rows, (0.0, 0.0, 40), _classify_fixed)
    seen = set()
    for r in out:
        assert set(r["memory_previous_ids"]) <= seen
        assert set(r["dynamic_selected_node_ids"]) <= \
            set(r["memory_previous_ids"] + r["dynamic_selected_node_ids"])
        seen |= set(r["dynamic_selected_node_ids"])


def test_dynamic_score_formula_exact():
    u = torch.tensor([1.0, -2.0, 0.5])
    nov = torch.tensor([0.2, 0.7, 1.0])
    per = torch.tensor([1.0, 0.0, 0.0])
    d = dynamic_scores(u, nov, per, 0.5, 0.25)
    expected = u + 0.5 * nov + 0.25 * per
    assert torch.equal(d, expected)


def test_static_dynamic_same_budget():
    rows = _make_event_rows(
        [["n0", "n1", "n2", "n3"], ["n0", "n1", "n2", "n3", "n4"]],
        {"n1": 3.0, "n2": 1.0, "n3": 2.0, "n4": 0.5})
    for r in rows:
        r["costs"] = [40, 40, 30] if len(r["units"]) == 3 else [40, 40, 30, 5]
    budget = 70
    out, _ = e3.run_event_trajectory(rows, (0.25, 0.1, budget),
                                     _classify_fixed)
    for r in out:
        assert r["static_evidence_tokens"] <= budget
        assert r["dynamic_evidence_tokens"] <= budget


def test_trajectory_cutoffs_monotonic():
    rows = _make_event_rows(
        [["n0", "n1"], ["n0", "n1", "n2"], ["n0", "n1", "n2", "n3"]],
        {"n1": 1.0, "n2": 2.0, "n3": 3.0})
    out, _ = e3.run_event_trajectory(rows, (0.0, 0.0, 40), _classify_fixed)
    cutoffs = [int(r["cutoff"]) for r in out]
    assert cutoffs == sorted(cutoffs)


def test_e3_never_reads_test_split():
    src = (SCRIPTS_DIR / "tcdscr_run_e3.py").read_text(encoding="utf-8")
    assert 'events["test"]' not in src
    assert "evaluate_test" not in src
    assert "test_metrics.json" not in src
    assert "test_split_read" in src  # manifest flag exists and is False
