"""Collation for variable-size original-model graph tensors."""

import torch
from torch.nn.utils.rnn import pad_sequence


def original_graph_collate(batch):
    node_feats, struct_feats, num_nodes, labels = zip(*batch)
    return (
        pad_sequence(node_feats, batch_first=True, padding_value=0.0),
        pad_sequence(struct_feats, batch_first=True, padding_value=0.0),
        torch.tensor(num_nodes, dtype=torch.long),
        torch.stack(labels, dim=0),
    )
