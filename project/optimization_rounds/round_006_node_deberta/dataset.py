"""Graph dataset that replaces MiniLM node text vectors with DeBERTa vectors."""

from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset


def replace_text_features(node_feat, deberta_feat, original_text_dim=384):
    """Preserve non-text columns while replacing the original text prefix."""
    if node_feat.ndim != 2 or deberta_feat.ndim != 2:
        raise ValueError("node_feat and deberta_feat must be matrices")
    if node_feat.size(0) != deberta_feat.size(0):
        raise ValueError("DeBERTa feature count must match graph node count")
    if node_feat.size(1) < int(original_text_dim):
        raise ValueError("node_feat is narrower than original_text_dim")
    return torch.cat(
        [deberta_feat.to(dtype=node_feat.dtype), node_feat[:, original_text_dim:]],
        dim=1,
    )


def prepend_text_features(node_feat, deberta_feat):
    """Keep the complete original node vector after a DeBERTa prefix."""
    if node_feat.ndim != 2 or deberta_feat.ndim != 2:
        raise ValueError("node_feat and deberta_feat must be matrices")
    if node_feat.size(0) != deberta_feat.size(0):
        raise ValueError("DeBERTa feature count must match graph node count")
    return torch.cat(
        [deberta_feat.to(dtype=node_feat.dtype), node_feat], dim=1
    )


class NodeDebertaGraphDataset(Dataset):
    def __init__(self, graph_dir, label_df, node_cache_path=None, augment=False,
                 aug_config=None, node_cache=None, keep_original_text=False):
        self.graph_dir = str(graph_dir)
        self.augment = bool(augment)
        self.aug_config = aug_config or {}
        self.keep_original_text = bool(keep_original_text)
        if node_cache is None:
            if node_cache_path is None:
                raise ValueError("node_cache_path or node_cache is required")
            node_cache = torch.load(
                node_cache_path, map_location="cpu", weights_only=True
            )
        self.node_cache = node_cache
        labels = label_df.copy()
        labels["event_id"] = labels["event_id"].astype(str)
        self.samples = []
        for _, row in labels.iterrows():
            event_id = str(row["event_id"])
            graph_path = os.path.join(self.graph_dir, f"{event_id}.pt")
            if not os.path.isfile(graph_path):
                raise FileNotFoundError(f"Graph feature not found: {graph_path}")
            if event_id not in self.node_cache:
                raise KeyError(f"Node cache missing event {event_id}")
            self.samples.append({
                "event_id": event_id,
                "graph_path": graph_path,
                "label": int(row["label"]),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        graph = torch.load(
            sample["graph_path"], map_location="cpu", weights_only=True
        )
        if self.keep_original_text:
            node_feat = prepend_text_features(
                graph["node_feat"], self.node_cache[sample["event_id"]]
            )
        else:
            node_feat = replace_text_features(
                graph["node_feat"], self.node_cache[sample["event_id"]]
            )
        struct_feat = graph["struct_feat"]
        num_nodes = int(graph["num_nodes"])
        if self.augment and bool(self.aug_config.get("graph_enabled", True)):
            dropout = float(self.aug_config.get("node_dropout", 0.05))
            if num_nodes > 1 and dropout > 0:
                count = max(1, int(num_nodes * dropout))
                dropped = torch.randperm(num_nodes)[:count]
                node_feat = node_feat.clone()
                node_feat[dropped] = 0.0
        return (
            node_feat,
            struct_feat,
            num_nodes,
            torch.tensor(sample["label"], dtype=torch.long),
        )
