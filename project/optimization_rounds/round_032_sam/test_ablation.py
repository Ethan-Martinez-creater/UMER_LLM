from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from optimization_rounds.round_013_joint_trifusion.model import (
    JointTriFusionModel,
)


class DummyGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, node_feats, struct_feats, num_nodes):
        self.calls += 1
        return node_feats.new_tensor([[0.2, -0.2], [-0.1, 0.1]])


class DummyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, **text_inputs):
        self.calls += 1
        return SimpleNamespace(
            logits=torch.tensor([
                [0.3, -0.3], [0.1, -0.1],
                [-0.2, 0.2], [-0.4, 0.4],
            ])
        )


class AblationTests(unittest.TestCase):
    def build(self, ablation):
        graph, text = DummyGraph(), DummyText()
        memory = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        model = JointTriFusionModel(
            graph,
            text,
            memory,
            labels,
            retrieval_text_dim=2,
            retrieval_k=2,
            retrieval_temperature=0.5,
            fusion_initialization="random",
            ablate_evidence=ablation,
        )
        return model, graph, text

    def forward(self, model):
        return model(
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 2),
            torch.tensor([3, 2]),
            {},
            memory_positions=torch.tensor([-1, -1]),
            view_count=2,
        )

    def test_graph_ablation_skips_graph_and_zeroes_graph_evidence(self):
        model, graph, text = self.build("graph")
        output = self.forward(model)
        self.assertEqual(graph.calls, 0)
        self.assertEqual(text.calls, 1)
        self.assertTrue(torch.equal(
            output["graph_logits"], torch.zeros_like(output["graph_logits"])
        ))

    def test_text_ablation_skips_text_and_zeroes_all_text_views(self):
        model, graph, text = self.build("text")
        output = self.forward(model)
        self.assertEqual(graph.calls, 1)
        self.assertEqual(text.calls, 0)
        self.assertTrue(torch.equal(
            output["text_view_logits"],
            torch.zeros_like(output["text_view_logits"]),
        ))

    def test_memory_ablation_zeroes_retrieval_only(self):
        model, graph, text = self.build("memory")
        output = self.forward(model)
        self.assertEqual(graph.calls, 1)
        self.assertEqual(text.calls, 1)
        self.assertTrue(torch.equal(
            output["retrieval_logits"],
            torch.zeros_like(output["retrieval_logits"]),
        ))

    def test_unknown_ablation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build("unknown")


if __name__ == "__main__":
    unittest.main()
