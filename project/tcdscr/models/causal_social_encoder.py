"""Causal social encoder (plan §16) — the only model component reused from
UMER, derived from ``OriginalGraphBranch`` with node_feat_dim=384 and
struct_feat_dim=1024. The full UMER fusion (NodeTokenSetEncoder, retrieval,
DeBERTa four-view, tri-fusion) is deliberately NOT reused.

Outputs node contextual embeddings h_i (768), an event embedding g_t (768),
and full-classifier logits p_full:
    node_repr, event_repr, logits = encoder(node_feat, struct_feat, num_nodes)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config.schema import (NODE_REPR_DIM, SEMANTIC_DIM, STRUCT_FEAT_DIM)


class OriginalGraphBranch(nn.Module):
    """One-layer Transformer over content-plus-structure node tokens.

    Structurally identical to the UMER branch of the same name (d_model 768,
    nhead 8, ffn 1536, pre-norm, CLS token, mean+CLS readout) but typed for
    the TC-DSCR inputs: 384D semantic node features and 1024D struct features.
    """

    def __init__(self, node_feat_dim=SEMANTIC_DIM, struct_feat_dim=STRUCT_FEAT_DIM,
                 hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.struct_feat_dim = struct_feat_dim
        self.d_model = hidden_dim * 3
        self.node_projection = nn.Sequential(
            nn.Linear(node_feat_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.struct_projection = _StructureFeatureEncoder(
            struct_feat_dim=struct_feat_dim,
            hidden_dim=hidden_dim,
            d_model=self.d_model,
            dropout_rate=dropout_rate,
        )
        self.token_norm = nn.LayerNorm(self.d_model)
        self.token_dropout = nn.Dropout(dropout_rate)
        self.s_cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=hidden_dim * 6,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.struct_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=1, enable_nested_tensor=False)
        except TypeError:
            self.struct_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=1)
        self.readout_projection = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, node_feats, struct_feats, num_nodes,
                return_node_features=False):
        device = node_feats.device
        batch_size, max_nodes, _ = node_feats.shape
        if node_feats.size(-1) < self.node_feat_dim:
            raise ValueError(
                f"expected at least {self.node_feat_dim} node features, "
                f"got {node_feats.size(-1)}")
        node_feats = node_feats[..., :self.node_feat_dim]
        valid_nodes = (
            torch.arange(max_nodes, device=device)[None, :]
            < num_nodes.clamp(min=0, max=max_nodes)[:, None])
        base_tokens = self.node_projection(node_feats)
        base_tokens = base_tokens + self.struct_projection(struct_feats)
        node_tokens = self.token_dropout(self.token_norm(base_tokens))
        cls_token = self.s_cls_token.expand(batch_size, 1, -1)
        sequence = torch.cat([cls_token, node_tokens], dim=1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        padding_mask = torch.cat([cls_mask, ~valid_nodes], dim=1)
        output = self.struct_transformer(
            sequence, src_key_padding_mask=padding_mask)
        cls_out = output[:, 0, :]
        node_out = output[:, 1:, :]
        valid_float = valid_nodes.unsqueeze(-1).to(node_out.dtype)
        denom = valid_float.sum(dim=1).clamp_min(1.0)
        mean_pool = (node_out * valid_float).sum(dim=1) / denom
        event_features = self.readout_projection(
            torch.cat([cls_out, mean_pool], dim=-1))
        if return_node_features:
            return event_features, node_out
        return event_features


class _StructureFeatureEncoder(nn.Module):
    """1021D adjacency + 3D summary projection — same shape as UMER's."""

    def __init__(self, struct_feat_dim=1024, hidden_dim=256, d_model=768,
                 dropout_rate=0.3):
        super().__init__()
        self.struct_feat_dim = struct_feat_dim
        self.summary_dim = 3 if struct_feat_dim >= 4 else 0
        self.adj_dim = struct_feat_dim - self.summary_dim
        self.adj_projection = nn.Sequential(
            nn.Linear(self.adj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        ) if self.adj_dim > 0 else None
        self.summary_projection = nn.Sequential(
            nn.Linear(self.summary_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        ) if self.summary_dim > 0 else None
        in_dim = hidden_dim * int(self.adj_projection is not None)
        in_dim += hidden_dim * int(self.summary_projection is not None)
        self.output_projection = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, struct_feats):
        parts = []
        if self.adj_projection is not None:
            parts.append(self.adj_projection(struct_feats[..., :self.adj_dim]))
        if self.summary_projection is not None:
            parts.append(
                self.summary_projection(struct_feats[..., self.adj_dim:]))
        return self.output_projection(torch.cat(parts, dim=-1))


class CausalSocialEncoder(nn.Module):
    """graph branch + classifier; exposes h_i, g_t and p_full."""

    def __init__(self, node_feat_dim=SEMANTIC_DIM,
                 struct_feat_dim=STRUCT_FEAT_DIM, hidden_dim=256,
                 num_classes=2):
        super().__init__()
        self.graph_branch = OriginalGraphBranch(
            node_feat_dim=node_feat_dim,
            struct_feat_dim=struct_feat_dim,
            hidden_dim=hidden_dim,
            dropout_rate=0.3,
        )
        # Same classifier shape as the UMER detector so that "classifier
        # (if dimensions compatible)" can be copied under §16.1.
        self.classifier = nn.Sequential(
            nn.LayerNorm(NODE_REPR_DIM),
            nn.Dropout(0.35 * 0.8),
            nn.Linear(NODE_REPR_DIM, num_classes),
        )

    def forward(self, node_feat, struct_feat, num_nodes,
                return_node_features=False):
        """node_feat (B,N,384), struct_feat (B,N,1024), num_nodes (B,).

        Unbatched inputs (2-D node_feat) are promoted to a batch of one.
        Returns (node_repr, event_repr, logits); node_repr has shape
        (B,N,768) with invalid (padding) rows filled with zeros.
        """
        squeeze = node_feat.ndim == 2
        if squeeze:
            node_feat = node_feat.unsqueeze(0)
            struct_feat = struct_feat.unsqueeze(0)
            num_nodes = num_nodes.unsqueeze(0) \
                if torch.is_tensor(num_nodes) else torch.tensor([num_nodes])
        num_nodes = num_nodes.to(node_feat.device, dtype=torch.long)
        event_repr, node_repr = self.graph_branch(
            node_feat, struct_feat, num_nodes, return_node_features=True)
        logits = self.classifier(event_repr)
        # zero out padding rows so downstream pooling never sees garbage
        valid = torch.arange(node_repr.size(1), device=node_repr.device)[None, :] \
            < num_nodes[:, None]
        node_repr = node_repr * valid.unsqueeze(-1).to(node_repr.dtype)
        if squeeze:
            return node_repr[0], event_repr[0], logits[0]
        return node_repr, event_repr, logits


def load_umer_init(encoder: CausalSocialEncoder, ckpt_path: str) -> dict:
    """UMER checkpoint init (plan §16.1).

    Copies: node projection (old W_old[:, 0:384] -> new 384D projection),
    StructureFeatureEncoder, TransformerEncoder, CLS token, readout, and the
    classifier (dimensions compatible). Never copies retrieval, DeBERTa
    four-view, tri-fusion (node_token_branch/fusion_proj), or anything 11D
    dependent.
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = blob["model_state_dict"] if "model_state_dict" in blob else blob

    mapping = {}
    for key, value in sd.items():
        if key.startswith("graph_model.graph_branch."):
            new_key = "graph_branch." + key[len("graph_model.graph_branch."):]
            mapping[new_key] = value
        elif key.startswith("graph_model.classifier."):
            mapping["classifier." + key[len("graph_model.classifier."):]] = value

    own = encoder.state_dict()
    copied, skipped = [], []
    for key, value in mapping.items():
        if key not in own:
            skipped.append(key)
            continue
        if key == "graph_branch.node_projection.0.weight":
            if value.shape[1] < SEMANTIC_DIM:
                raise ValueError(
                    f"UMER node projection has {value.shape[1]} cols, "
                    f"expected >= {SEMANTIC_DIM} to slice [:, :384]")
            value = value[:, :SEMANTIC_DIM].clone()
        if own[key].shape != value.shape:
            raise ValueError(
                f"shape mismatch for {key}: own {tuple(own[key].shape)} "
                f"vs ckpt {tuple(value.shape)}")
        own[key] = value
        copied.append(key)

    encoder.load_state_dict(own)
    return {"copied": sorted(copied), "skipped": sorted(skipped),
            "source_checkpoint": ckpt_path}


def collate_snapshots(features_list):
    """Pad a list of Module-3 feature dicts into a training batch."""
    batch_size = len(features_list)
    max_nodes = max(f["node_feat"].size(0) for f in features_list)
    sem_dim = features_list[0]["node_feat"].size(1)
    struct_dim = features_list[0]["struct_feat"].size(1)
    node_feat = torch.zeros(batch_size, max_nodes, sem_dim)
    struct_feat = torch.zeros(batch_size, max_nodes, struct_dim)
    num_nodes = torch.tensor([f["node_feat"].size(0) for f in features_list],
                             dtype=torch.long)
    edge_index_cpu = []
    for i, f in enumerate(features_list):
        n = f["node_feat"].size(0)
        node_feat[i, :n] = f["node_feat"]
        struct_feat[i, :n] = f["struct_feat"]
        if f["edge_index"].numel() > 0:
            edge_index_cpu.append(f["edge_index"] + i * max_nodes)
    edge_index = torch.cat(edge_index_cpu, dim=1) if edge_index_cpu \
        else torch.empty((2, 0), dtype=torch.long)
    return node_feat, struct_feat, num_nodes, edge_index
