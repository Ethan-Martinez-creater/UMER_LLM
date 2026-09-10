"""Static selector + proxy training (plan §17/§18; smoke = 1 epoch).

The encoder is frozen: node/event representations and p_full come from a
no-grad forward pass. Trainable parameters are the utility scorer and the
proxy classifier, optimized with L = L_cls + 1.0 L_fid + 0.05 L_div.
"""
from __future__ import annotations

import torch

from ..models.selector_proxy import proxy_loss
from .sampler import SnapshotSampler


def _repr_batch(encoder, features, device):
    node_feat, struct_feat, num_nodes, _ = _collate([features])
    with torch.no_grad():
        node_repr, event_repr, logits = encoder(
            node_feat.to(device), struct_feat.to(device),
            num_nodes.to(device))
    n = features["node_feat"].size(0)
    return node_repr[0, :n], event_repr[0], logits[0]


def _collate(features):
    from ..models.causal_social_encoder import collate_snapshots
    return collate_snapshots(features)


def train_selector(encoder, selector, proxy, per_event_features, labels,
                   epochs=1, lr=1e-4, weight_decay=0.01, seed=2000,
                   device="cpu", log=print):
    for m in (encoder,):
        m.to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    selector.to(device)
    proxy.to(device)
    selector.train()
    proxy.train()
    opt = torch.optim.AdamW(
        list(selector.parameters()) + list(proxy.parameters()),
        lr=lr, weight_decay=weight_decay)
    sampler = SnapshotSampler(seed + 1)
    history = []
    for epoch in range(epochs):
        batch_events = sorted(per_event_features)
        losses, parts = [], {"l_cls": 0.0, "l_fid": 0.0, "l_div": 0.0}
        for eid in batch_events:
            feats = sampler.sample_one(per_event_features[eid])
            y = torch.tensor([labels[eid]], dtype=torch.long, device=device)
            node_repr, event_repr, logits_full = _repr_batch(
                encoder, feats, device)
            n = node_repr.size(0)
            sem = feats["node_feat"].to(device)
            struct3 = feats["struct_feat"][:, -3:].to(device)
            # candidates exclude the source (always row 0? no — locate it)
            src_pos = feats["node_ids"].index(_source_id(feats))
            cand = [i for i in range(n) if i != src_pos]
            if not cand:
                continue
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            mask[cand] = True
            h_source = node_repr[src_pos]
            u = selector(node_repr[mask], event_repr, sem[mask],
                         sem[src_pos], struct3[mask])
            alpha, _z_sel, p_sel = proxy(node_repr[mask], h_source, u)
            total, components = proxy_loss(
                p_sel, y[0], logits_full, alpha, sem[mask])
            opt.zero_grad()
            total.backward()
            opt.step()
            losses.append(float(total))
            for k in parts:
                parts[k] += components[k]
        history.append({"loss": sum(losses) / max(len(losses), 1),
                        "events": len(losses), **parts})
        log(f"selector epoch {epoch + 1}/{epochs} "
            f"loss={history[-1]['loss']:.4f}")
    selector.eval()
    proxy.eval()
    return history


def _source_id(features):
    """Snapshot features always keep the source; find its node_id."""
    # snapshot construction guarantees the source is present; the caller
    # passes it via features["node_ids"] and features["source_id"] is stored
    # by the smoke pipeline before collation.
    return features["source_id"]
