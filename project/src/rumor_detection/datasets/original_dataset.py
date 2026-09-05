"""Graph-feature dataset used exclusively by OriginalRumorDetector."""

from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset


class OriginalRumorDataset(Dataset):
    def __init__(self, graph_dir, label_df, augment=False, aug_config=None):
        self.graph_dir = graph_dir
        self.augment = bool(augment)
        self.aug_config = aug_config or {}
        labels = label_df.copy()
        labels["event_id"] = labels["event_id"].astype(str)
        self.samples = []
        for _, row in labels.iterrows():
            event_id = str(row["event_id"])
            graph_path = os.path.join(graph_dir, f"{event_id}.pt")
            if not os.path.isfile(graph_path):
                raise FileNotFoundError(f"Graph feature not found: {graph_path}")
            self.samples.append({
                "event_id": event_id,
                "graph_path": graph_path,
                "label": int(row["label"]),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            graph = torch.load(
                sample["graph_path"], map_location="cpu", weights_only=True
            )
        except TypeError:
            graph = torch.load(sample["graph_path"], map_location="cpu")
        node_feats = graph["node_feat"]
        struct_feats = graph["struct_feat"]
        num_nodes = graph["num_nodes"]
        if isinstance(num_nodes, torch.Tensor):
            num_nodes = int(num_nodes.item())
        if self.augment and bool(self.aug_config.get("graph_enabled", True)):
            node_feats = self._zero_node_features(node_feats, num_nodes)
        return (
            node_feats,
            struct_feats,
            num_nodes,
            torch.tensor(sample["label"], dtype=torch.long),
        )

    def _zero_node_features(self, node_feats, num_nodes):
        if num_nodes <= 1:
            return node_feats
        dropout = float(self.aug_config.get("node_dropout", 0.05))
        drop_count = max(1, int(num_nodes * dropout))
        indices = torch.randperm(num_nodes)[:drop_count]
        augmented = node_feats.clone()
        augmented[indices] = 0.0
        return augmented
