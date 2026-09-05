from types import SimpleNamespace

import torch

from optimization_rounds.round_013_joint_trifusion.model import JointTriFusionModel
from rumor_detection.models.original_model import OriginalRumorDetector


class ScalarTextModel(torch.nn.Module):
    def forward(self, input_ids, **_):
        value = input_ids.float().mean(dim=1)
        return SimpleNamespace(logits=torch.stack([-value, value], dim=1))


def test_teacher_view_is_excluded_from_classification_average():
    torch.manual_seed(13)
    graph = OriginalRumorDetector(
        hidden_dim=8, text_feat_dim=4, extra_feat_dim=11,
        struct_feat_dim=7, dropout_rate=0.0,
    )
    model = JointTriFusionModel(
        graph,
        ScalarTextModel(),
        memory_features=torch.randn(3, 16),
        memory_labels=torch.tensor([0, 1, 0]),
        retrieval_text_dim=4,
        fusion_initialization="random",
    ).eval()
    output = model(
        node_feats=torch.randn(1, 2, 15),
        struct_feats=torch.randn(1, 2, 7),
        num_nodes=torch.tensor([2]),
        text_inputs={"input_ids": torch.tensor([[1], [3], [100]])},
        memory_positions=torch.tensor([-1]),
        view_count=3,
        classification_view_count=0,
        teacher_view_count=1,
    )
    assert torch.equal(
        output["text_logits"], torch.tensor([[-2.0, 2.0]])
    )
    assert output["supervised_text_view_logits"].shape == (1, 2, 2)
    assert torch.equal(
        output["teacher_text_view_logits"], torch.tensor([[[-100.0, 100.0]]])
    )
