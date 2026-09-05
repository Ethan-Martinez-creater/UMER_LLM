import sys
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization_rounds.round_013_joint_trifusion.model import (
    JointTriFusionModel,
    sample_view_subset,
)


def test_zero_selects_all_views_without_copying_values():
    logits = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    selected, indices = sample_view_subset(logits, 0)
    assert torch.equal(selected, logits)
    assert torch.equal(indices, torch.tensor([[0, 1, 2, 3]]).expand(3, -1))


def test_two_views_are_distinct_per_event_and_reproducible():
    logits = torch.arange(40, dtype=torch.float32).reshape(5, 4, 2)
    torch.manual_seed(2000)
    selected_a, indices_a = sample_view_subset(logits, 2)
    torch.manual_seed(2000)
    selected_b, indices_b = sample_view_subset(logits, 2)
    assert torch.equal(indices_a, indices_b)
    assert torch.equal(selected_a, selected_b)
    assert selected_a.shape == (5, 2, 2)
    assert torch.all(indices_a[:, 0] != indices_a[:, 1])
    expected = logits.gather(1, indices_a.unsqueeze(-1).expand(-1, -1, 2))
    assert torch.equal(selected_a, expected)


@pytest.mark.parametrize("count", [-1, 5])
def test_invalid_selected_count_is_rejected(count):
    with pytest.raises(ValueError):
        sample_view_subset(torch.zeros(2, 4, 2), count)


def test_invalid_rank_is_rejected():
    with pytest.raises(ValueError):
        sample_view_subset(torch.zeros(4, 2), 2)


def test_text_ce_uses_only_selected_classification_views():
    labels = torch.tensor([0, 1])
    selected = torch.tensor([[[4.0, -4.0]], [[-4.0, 4.0]]])
    all_views = torch.zeros(2, 4, 2)
    outputs = {
        "logits": torch.zeros(2, 2),
        "graph_logits": torch.zeros(2, 2),
        "text_logits": selected[:, 0],
        "text_view_logits": all_views,
        "supervised_text_view_logits": selected,
    }
    losses = JointTriFusionModel.loss(outputs, labels)
    expected = torch.nn.functional.cross_entropy(selected[:, 0], labels)
    assert torch.allclose(losses["text"], expected)


class DummyGraph(torch.nn.Module):
    def forward(self, node_feats, struct_feats, num_nodes):
        pooled = node_feats[:, :, 0].mean(dim=1)
        return torch.stack([-pooled, pooled], dim=1)


class DummyText(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        score = input_ids[:, 0].float()
        return SimpleNamespace(logits=torch.stack([-score, score], dim=1))


def test_joint_forward_encodes_four_but_classifies_two_views():
    torch.manual_seed(2000)
    memory_features = torch.randn(3, 8)
    model = JointTriFusionModel(
        DummyGraph(), DummyText(), memory_features, torch.tensor([0, 1, 0]),
        retrieval_text_dim=2, retrieval_k=2,
        fusion_initialization="random", fusion_type="linear",
    )
    model.train()
    outputs = model(
        node_feats=torch.randn(2, 3, 2),
        struct_feats=torch.zeros(2, 1),
        num_nodes=torch.tensor([3, 2]),
        text_inputs={"input_ids": torch.arange(8).reshape(8, 1)},
        memory_positions=torch.tensor([-1, -1]),
        view_count=4,
        classification_view_count=2,
    )
    assert outputs["text_view_logits"].shape == (2, 4, 2)
    assert outputs["supervised_text_view_logits"].shape == (2, 2, 2)
    assert outputs["selected_view_indices"].shape == (2, 2)
    assert torch.all(
        outputs["selected_view_indices"][:, 0]
        != outputs["selected_view_indices"][:, 1]
    )
    assert outputs["logits"].shape == (2, 2)
