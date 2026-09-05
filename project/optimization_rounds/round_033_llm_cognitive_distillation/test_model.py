import torch
from types import SimpleNamespace

from optimization_rounds.round_013_joint_trifusion.model import JointTriFusionModel
from rumor_detection.models.original_model import OriginalRumorDetector


class TinyTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 2)

    def forward(self, input_ids, **_):
        pooled = input_ids.float().mean(dim=1, keepdim=True)
        return SimpleNamespace(logits=self.projection(pooled))


def test_return_features_preserves_graph_logits():
    torch.manual_seed(7)
    model = OriginalRumorDetector(
        hidden_dim=8, text_feat_dim=4, extra_feat_dim=11,
        struct_feat_dim=7, dropout_rate=0.0,
    ).eval()
    node = torch.randn(2, 3, 15)
    struct = torch.randn(2, 3, 7)
    counts = torch.tensor([3, 2])
    plain = model(node, struct, counts)
    logits, features = model(node, struct, counts, return_features=True)
    assert torch.equal(plain, logits)
    assert features.shape == (2, model.d_model)


def test_masked_cognitive_loss_ignores_inactive_targets():
    outputs = {
        "logits": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "graph_logits": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "text_logits": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "text_view_logits": torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]]),
        "cognitive_predictions": torch.tensor([[0.0, 1.0], [0.5, 0.5]]),
    }
    labels = torch.tensor([1, 0])
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    losses = JointTriFusionModel.loss(
        outputs, labels, cognitive_targets=targets, cognitive_mask=mask,
        cognitive_weight=0.2,
    )
    assert losses["cognitive"].item() == 1.0
    baseline = JointTriFusionModel.loss(outputs, labels)
    assert torch.allclose(losses["total"], baseline["total"] + 0.2)


def test_joint_cognitive_head_backpropagates_into_graph_encoder():
    torch.manual_seed(11)
    graph = OriginalRumorDetector(
        hidden_dim=8, text_feat_dim=4, extra_feat_dim=11,
        struct_feat_dim=7, dropout_rate=0.0,
    )
    model = JointTriFusionModel(
        graph,
        TinyTextModel(),
        memory_features=torch.randn(4, 16),
        memory_labels=torch.tensor([0, 1, 0, 1]),
        retrieval_text_dim=4,
        cognitive_target_dim=18,
        fusion_initialization="random",
    )
    outputs = model(
        node_feats=torch.randn(2, 3, 15),
        struct_feats=torch.randn(2, 3, 7),
        num_nodes=torch.tensor([3, 2]),
        text_inputs={"input_ids": torch.tensor([[1, 2], [3, 4]])},
        memory_positions=torch.tensor([-1, -1]),
    )
    assert outputs["cognitive_predictions"].shape == (2, 18)
    losses = model.loss(
        outputs,
        torch.tensor([0, 1]),
        cognitive_targets=torch.rand(2, 18),
        cognitive_mask=torch.ones(2, 18),
        cognitive_weight=0.5,
    )
    losses["total"].backward()
    assert model.cognitive_head[1].weight.grad is not None
    assert graph.fusion_proj[0].weight.grad is not None
