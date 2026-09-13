"""Plan §35 — held-out reader leakage tests (plan §20, §33, §34).

The review round moved isolation to the **entrance**: the rotation builder
filters the label cache to the two training readers, and ``UtilityDataset``
refuses to be constructed with a held-out reader present.
"""
from __future__ import annotations

import pytest
import torch

from ..config.pilot_config import (LORO_ROTATIONS, SEMANTIC_DIM,
                                   STRUCT_SCALAR_DIM, UTILITY_Q_DIM)
from ..models.bitte import BiTTE
from ..models.utility_heads import SharedResidualUtility
from ..training.train_utility import _dev_metric, _batch_forward
from ..training.utility_dataset import (UtilityDataset, filter_rows_for_readers,
                                        assert_only_readers)


def _row(reader, unit_index=0, key="e:60:n1", target=0.1, sign="NEUTRAL"):
    return {"reader": reader, "unit_index": unit_index, "unit_key": key,
            "target": target, "sign": sign, "correct_before": True,
            "correct_after": True}


def _group(rows):
    n = 4
    return {
        "dataset": "pheme", "event_id": "e", "cutoff": 60,
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
    all_rows = [_row(a), _row(b, unit_index=1, key="e:60:n2"),
                _row(held, unit_index=2, key="e:60:n3")]
    kept, dropped = filter_rows_for_readers(all_rows, [a, b])
    assert {r["reader"] for r in kept} == {a, b}
    assert len(dropped) == 1 and dropped[0]["reader"] == held

    good = UtilityDataset([_group(kept)], [a, b], {a: 0, b: 1})
    rows = good.reader_rows(good.groups)
    assert {r["reader"] for r in rows} == {a, b}
    # the held-out reader never appears anywhere in the dataset object
    assert all(r["reader"] != held for g in good.groups
               for r in g["unit_rows"])
    # an unfiltered group is refused at construction (fail closed)
    with pytest.raises(ValueError):
        UtilityDataset([_group(all_rows)], [a, b], {a: 0, b: 1})
    with pytest.raises(ValueError):
        assert_only_readers(all_rows, [a, b])


def test_heldout_reader_not_in_early_stopping():
    a, b, held = LORO_ROTATIONS[0]
    all_rows = [_row(a), _row(b, unit_index=1, key="e:60:n2"),
                _row(held, unit_index=2, key="e:60:n3")]
    kept, _dropped = filter_rows_for_readers(all_rows, [a, b])
    torch.manual_seed(0)
    bitte = BiTTE()
    model = SharedResidualUtility(2)
    dataset = UtilityDataset([_group(kept)], [a, b], {a: 0, b: 1})
    assert held not in dataset.reader_index_map
    score, meta = _dev_metric(bitte, model, dataset, [a, b], device="cpu")
    assert meta["metric"] in ("dev_spearman", "neg_dev_smoothl1_fallback",
                              "undefined")
    assert score == score or score == float("-inf")


def test_batch_forward_never_touches_heldout_reader():
    """``_batch_forward`` must resolve every row through the training map."""
    a, b, held = LORO_ROTATIONS[0]
    all_rows = [_row(a), _row(b, unit_index=1, key="e:60:n2"),
                _row(held, unit_index=2, key="e:60:n3")]
    kept, _dropped = filter_rows_for_readers(all_rows, [a, b])
    torch.manual_seed(0)
    bitte = BiTTE()
    model = SharedResidualUtility(2)
    group = _group(kept)
    packed = _batch_forward(bitte, model, [group], {a: 0, b: 1}, "cpu")
    assert packed is not None
    _z, r, _u, _s, keys = packed
    # only the two training reader indices appear
    assert set(r.tolist()) == {0, 1}
    assert all(held not in key for key in keys)
    assert held not in {a: 0, b: 1}


def test_heldout_reader_not_in_selection(fake_tokenizer):
    from ..data.snapshot_bridge import build_causal_snapshot
    from ..intervention.evidence_units import build_evidence_units, evidence_key
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
    # canonical key round-trip is available for artifact consumers
    key = evidence_key("pheme", snap["event_id"], 360, ids[0])
    assert key.split("|")[0] == "pheme"
