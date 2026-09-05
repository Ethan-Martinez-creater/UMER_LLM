import torch
import torch.nn as nn
from types import SimpleNamespace

from optimization_rounds.round_013_joint_trifusion.model import JointTriFusionModel
from rumor_detection.models.original_model import OriginalRumorDetector


class DummyTextModel(nn.Module):
    def forward(self, **kwargs):
        batch = kwargs["input_ids"].size(0)
        return SimpleNamespace(logits=torch.zeros(batch, 2))


def test_optional_node_return_preserves_default_logits_exactly():
    torch.manual_seed(7)
    model = OriginalRumorDetector(
        num_classes=2,
        hidden_dim=8,
        text_feat_dim=4,
        extra_feat_dim=11,
        struct_feat_dim=8,
        dropout_rate=0.0,
    ).eval()
    node = torch.randn(2, 3, 15)
    struct = torch.randn(2, 3, 8)
    counts = torch.tensor([3, 2])
    with torch.no_grad():
        default = model(node, struct, counts)
        logits, event_features, node_features = model(
            node,
            struct,
            counts,
            return_features=True,
            return_node_features=True,
        )
    assert torch.equal(default, logits)
    assert event_features.shape == (2, 24)
    assert node_features.shape == (2, 3, 24)


def test_node_cognitive_loss_masks_groups_and_backpropagates():
    predictions = torch.randn(2, 3, 14, requires_grad=True)
    outputs = {
        "logits": torch.randn(2, 2, requires_grad=True),
        "graph_logits": torch.randn(2, 2, requires_grad=True),
        "text_logits": torch.randn(2, 2, requires_grad=True),
        "text_view_logits": torch.randn(2, 1, 2, requires_grad=True),
        "supervised_text_view_logits": torch.randn(2, 1, 2, requires_grad=True),
        "node_cognitive_predictions": predictions,
    }
    shape = (2, 3)
    targets = {
        "stance": torch.zeros(shape, dtype=torch.long),
        "stance_mask": torch.tensor(
            [[True, True, False], [False, False, False]]
        ),
        "role": torch.ones(shape, dtype=torch.long),
        "role_mask": torch.tensor(
            [[True, False, False], [True, False, False]]
        ),
        "salience": torch.full(shape, 0.5),
        "salience_mask": torch.tensor(
            [[True, True, False], [True, False, False]]
        ),
    }
    losses = JointTriFusionModel.loss(
        outputs,
        torch.tensor([0, 1]),
        node_cognitive_targets=targets,
        node_cognitive_weight=0.2,
    )
    assert losses["node_cognitive"].item() > 0.0
    losses["total"].backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_joint_node_head_is_training_optional_and_logit_neutral():
    torch.manual_seed(11)
    graph = OriginalRumorDetector(
        num_classes=2,
        hidden_dim=8,
        text_feat_dim=4,
        extra_feat_dim=11,
        struct_feat_dim=8,
        dropout_rate=0.0,
    )
    model = JointTriFusionModel(
        graph,
        DummyTextModel(),
        torch.randn(4, 16),
        torch.tensor([0, 1, 0, 1]),
        retrieval_text_dim=4,
        retrieval_k=2,
        fusion_initialization="random",
        node_cognitive_target_dim=14,
    ).eval()
    node = torch.randn(2, 3, 15)
    struct = torch.randn(2, 3, 8)
    counts = torch.tensor([3, 2])
    text_inputs = {"input_ids": torch.ones(2, 4, dtype=torch.long)}
    with torch.no_grad():
        inference = model(node, struct, counts, text_inputs)
        training_aux = model(
            node,
            struct,
            counts,
            text_inputs,
            compute_node_cognitive=True,
        )
    assert inference["node_cognitive_predictions"] is None
    assert training_aux["node_cognitive_predictions"].shape == (2, 3, 14)
    assert torch.equal(inference["logits"], training_aux["logits"])
