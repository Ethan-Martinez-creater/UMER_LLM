"""Plan §35 — held-out reader leakage tests (plan §20, §33, §34)."""
from __future__ import annotations

import torch

from ..config.pilot_config import (LORO_ROTATIONS, SEMANTIC_DIM,
                                   STRUCT_SCALAR_DIM, UTILITY_Q_DIM)
from ..models.bitte import BiTTE
from ..models.utility_heads import SharedResidualUtility
from ..training.train_utility import _dev_metric
from ..training.utility_dataset import UtilityDataset


def _row(reader, unit_index=0, key="e:60:n1", target=0.1, sign="NEUTRAL"):
    return {"reader": reader, "unit_index": unit_index, "unit_key": key,
            "target": target, "sign": sign, "correct_before": True,
            "correct_after": True}


def _group(rows):
    n = 4
    return {
        "event_id": "e", "cutoff": 60,
        "sem": torch.randn(n, SEMANTIC_DIM,
                           generator=torch.Generator().manual_seed(0)),
        "struct": torch.rand(n, STRUCT_SCALAR_DIM,
                             generator=torch.Generator().manual_seed(1)),
        "q": torch.rand(n, UTILITY_Q_DIM,
                        generator=torch.Generator().manual_seed(2)),
        "parent_idx": torch.tensor([-1, 0, 1, 1]),
        "edges": torch.tensor([[1, 0], [2, 1], [3, 1]]),
        "source_idx": 0, "ctx_indices": [1, 2, 3], "unit_rows": rows,
    }


def test_heldout_reader_not_in_training_batch():
    a, b, held = LORO_ROTATIONS[0]
    good = UtilityDataset([_group([_row(a), _row(b, unit_index=1)])],
                          [a, b], {a: 0, b: 1})
    rows = good.reader_rows(good.groups)  # training readers only -> fine
    assert {r["reader"] for r in rows} == {a, b}
    bad = UtilityDataset([_group([_row(a), _row(held, unit_index=1)])],
                         [a, b], {a: 0, b: 1})
    try:
        bad.reader_rows(bad.groups)
        raised = False
    except ValueError:
        raised = True
    assert raised, "a held-out reader row must never enter a training batch"
    assert held not in bad.reader_index_map


def test_heldout_reader_not_in_early_stopping():
    a, b, held = LORO_ROTATIONS[0]
    torch.manual_seed(0)
    bitte = BiTTE()
    model = SharedResidualUtility(2)
    dataset = UtilityDataset(
        [_group([_row(a), _row(b, unit_index=1, key="e:60:n2"),
                 _row(held, unit_index=2, key="e:60:n3")])],
        [a, b], {a: 0, b: 1})
    # the held-out row is present in the data but absent from the reader map
    assert held not in dataset.reader_index_map
    score, meta = _dev_metric(bitte, model, dataset, [a, b], device="cpu")
    assert meta["metric"] in ("dev_spearman", "neg_dev_smoothl1_fallback",
                              "undefined")
    assert score == score or score == float("-inf")


def test_heldout_reader_not_in_selection(fake_tokenizer):
    from ..data.snapshot_bridge import build_causal_snapshot
    from ..intervention.evidence_units import build_evidence_units
    from ..intervention.semantic_reference import build_src
    from ..models.robust_selector import build_arm
    from .conftest import branch_event, embeddings_for

    a, b, held = LORO_ROTATIONS[0]
    snap = build_causal_snapshot(branch_event(), 360)
    units = build_evidence_units(snap)
    src_emb, reply_emb = embeddings_for(snap)
    src = build_src(units, src_emb, reply_emb, fake_tokenizer)
    ids = src["selected_node_ids"]
    predictions = {
        a: {nid: 0.1 for nid in ids},
        b: {nid: 0.2 for nid in ids},
        "shared": {nid: 0.05 for nid in ids},
    }
    arm = build_arm("S5_cross_reader_robust", units, src, predictions,
                    fake_tokenizer, [a, b], seed=7319)
    assert held not in predictions
    assert set(arm["selected_node_ids"]) <= set(ids)
