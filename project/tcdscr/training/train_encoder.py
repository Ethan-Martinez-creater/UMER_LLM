"""Causal encoder training loop (plan §16/§36; smoke = 1 epoch).

Each epoch samples one dynamic snapshot per train event (sampler.py), runs a
forward pass, and applies cross-entropy on p_full. SOURCE_ONLY snapshots are
excluded by construction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .sampler import SnapshotSampler


def train_encoder(encoder, per_event_features, labels, epochs=1, lr=1e-4,
                  weight_decay=0.01, seed=2000, device="cpu", log=print):
    """``per_event_features``: {event_id: {cutoff: feature_dict}}.

    ``labels``: {event_id: 0/1}. Returns history list of per-epoch losses.
    """
    encoder.to(device)
    encoder.train()
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sampler = SnapshotSampler(seed)
    history = []
    for epoch in range(epochs):
        batch = sampler.sample_batch(per_event_features)
        y = torch.tensor([labels[f["event_id"]] for f in batch],
                         dtype=torch.long, device=device)
        node_feat, struct_feat, num_nodes, _ = _stack(batch)
        node_feat = node_feat.to(device)
        struct_feat = struct_feat.to(device)
        num_nodes = num_nodes.to(device)
        logits = encoder(node_feat, struct_feat, num_nodes)[2]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss))
        log(f"encoder epoch {epoch + 1}/{epochs} loss={float(loss):.4f}")
    encoder.eval()
    return history


def _stack(features):
    from ..models.causal_social_encoder import collate_snapshots
    return collate_snapshots(features)
