"""Unit tests: Formal E1 training plumbing (delta on §36/§37).

The GPU collate path builds the 1021D adjacency signature with the historical
formula; these tests pin its numerical identity to the frozen CPU reference
so the input definition (384D + 1024D) is provably unchanged.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

tcdscr_run_e1 = pytest.importorskip(
    "tcdscr_run_e1", reason="E1 entry script not on sys.path")

from ..data.snapshot_builder import build_snapshot  # noqa: E402
from ..data.structural_features import build_snapshot_features  # noqa: E402
from .conftest import make_event  # noqa: E402


def _event():
    nodes = [("n0", None, 1000, "source claim", 0)]
    for i in range(1, 40):
        parent = "n0" if i % 3 else f"n{i - 1}"
        nodes.append((f"n{i}", parent, 1000 + i, f"reply {i}", i))
    return make_event(nodes)


def test_gpu_collate_matches_frozen_struct_definition():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the E1 collate path")
    event = _event()
    snap = build_snapshot(event, 60)
    gen = torch.Generator().manual_seed(3)
    sem = torch.rand(len(snap["node_ids"]), 384, generator=gen)

    # frozen CPU reference
    ref = build_snapshot_features(snap, sem)
    src_pos = ref["node_ids"].index(ref["source_id"])
    cand = [i for i in range(len(ref["node_ids"])) if i != src_pos]

    # E1 light item + GPU collate
    store = {nid: sem[i] for i, nid in enumerate(snap["node_ids"])}
    item = tcdscr_run_e1.light_features(event, snap, store)
    node_feat, struct, num_nodes, _edges, labels = \
        tcdscr_run_e1.collate_light_semantic([item], "cuda")

    assert torch.allclose(node_feat[0, :item["num_nodes"]].cpu(),
                          ref["node_feat"], atol=1e-6)
    assert torch.allclose(struct[0, :item["num_nodes"]].cpu(),
                          ref["struct_feat"], atol=1e-6), \
        "GPU-built struct must equal the frozen CPU definition"
    assert int(num_nodes[0]) == item["num_nodes"]
    assert int(labels[0]) == event["label"]


def test_e1_primary_cutoffs_frozen():
    assert tcdscr_run_e1.PRIMARY_CUTOFFS == (5, 15, 30, 60, 180, 360)
    assert tcdscr_run_e1.EVAL_CUTOFFS == (
        "SOURCE_ONLY", 5, 15, 30, 60, 180, 360, 1440)
    # SOURCE_ONLY is not a training snapshot: it appears only in EVAL_CUTOFFS
    assert "SOURCE_ONLY" not in tcdscr_run_e1.PRIMARY_CUTOFFS


def test_e1_hparams_follow_umer_recipe():
    assert tcdscr_run_e1.HPARAMS == {
        "batch_size": 32,
        "max_epochs": 60,
        "patience": 7,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "label_smoothing": 0.1,
    }
