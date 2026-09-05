"""The original Node-token Set plus Transformer rumor detector."""

from __future__ import annotations

import torch
import torch.nn as nn


NODE_TOKEN_DIM = 14


def _valid_node_mask(num_nodes, max_nodes, device):
    lengths = num_nodes.to(device=device, dtype=torch.long).clamp(
        min=1, max=max_nodes
    )
    positions = torch.arange(max_nodes, device=device).unsqueeze(0)
    return positions < lengths.unsqueeze(1), lengths


def extract_node_tokens(node_feats, struct_feats):
    """Extract the 11 non-text and 3 propagation features used by the Set branch."""
    if node_feats.ndim != 3 or struct_feats.ndim != 3:
        raise ValueError("node_feats and struct_feats must both be 3D tensors")
    if node_feats.shape[:2] != struct_feats.shape[:2]:
        raise ValueError("node_feats and struct_feats must share batch/node axes")
    if node_feats.size(-1) < 11 or struct_feats.size(-1) < 3:
        raise ValueError("Insufficient feature dimensions for node tokens")
    if node_feats.size(-1) >= 396:
        extra = node_feats[..., -12:-1]
    else:
        extra = node_feats[..., -11:]
    return torch.cat([extra, struct_feats[..., -3:]], dim=-1)


class NodeTokenSetEncoder(nn.Module):
    """Permutation-invariant encoder for non-text propagation node tokens."""

    def __init__(self, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.register_buffer(
            "feature_mask", torch.ones(NODE_TOKEN_DIM, dtype=torch.float32)
        )
        self.token_mlp = nn.Sequential(
            nn.Linear(NODE_TOKEN_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout_rate * 0.6),
        )

    def forward(self, node_feats, struct_feats, num_nodes):
        tokens = extract_node_tokens(node_feats, struct_feats)
        encoded = self.token_mlp(
            tokens * self.feature_mask.to(dtype=tokens.dtype)
        )
        valid_mask, lengths = _valid_node_mask(
            num_nodes, encoded.size(1), encoded.device
        )
        mask_float = valid_mask.unsqueeze(-1).to(encoded.dtype)
        mean_pool = (encoded * mask_float).sum(dim=1) / lengths.unsqueeze(1)
        max_pool = encoded.masked_fill(~valid_mask.unsqueeze(-1), -torch.inf)
        max_pool = torch.nan_to_num(max_pool.amax(dim=1), neginf=0.0)
        return self.output_proj(torch.cat([mean_pool, max_pool], dim=-1))


class StructureFeatureEncoder(nn.Module):
    """Encode adjacency signatures and three dense structural summaries."""

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
                self.summary_projection(struct_feats[..., self.adj_dim:])
            )
        return self.output_projection(torch.cat(parts, dim=-1))


class OriginalGraphBranch(nn.Module):
    """Original one-layer Transformer over content-plus-structure node tokens."""

    def __init__(self, node_feat_dim=395, struct_feat_dim=1024,
                 hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.struct_feat_dim = struct_feat_dim
        self.graph_encoder_type = "original"
        self.d_model = hidden_dim * 3
        self.node_projection = nn.Sequential(
            nn.Linear(node_feat_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.struct_projection = StructureFeatureEncoder(
            struct_feat_dim=struct_feat_dim,
            hidden_dim=hidden_dim,
            d_model=self.d_model,
            dropout_rate=dropout_rate,
        )
        self.token_norm = nn.LayerNorm(self.d_model)
        self.token_dropout = nn.Dropout(dropout_rate)
        self.s_cls_token = nn.Parameter(
            torch.randn(1, 1, self.d_model) * 0.02
        )
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
                encoder_layer, num_layers=1, enable_nested_tensor=False
            )
        except TypeError:
            self.struct_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=1
            )
        self.use_max_readout = False
        self.readout_projection = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(
        self, node_feats, struct_feats, num_nodes, return_node_features=False
    ):
        device = node_feats.device
        batch_size, max_nodes, _ = node_feats.shape
        if node_feats.size(-1) < self.node_feat_dim:
            raise ValueError(
                f"Expected at least {self.node_feat_dim} node features, "
                f"got {node_feats.size(-1)}"
            )
        node_feats = node_feats[..., :self.node_feat_dim]
        valid_nodes = (
            torch.arange(max_nodes, device=device)[None, :]
            < num_nodes.clamp(min=0, max=max_nodes)[:, None]
        )
        base_tokens = self.node_projection(node_feats)
        base_tokens = base_tokens + self.struct_projection(struct_feats)
        node_tokens = self.token_dropout(self.token_norm(base_tokens))
        cls_token = self.s_cls_token.expand(batch_size, 1, -1)
        sequence = torch.cat([cls_token, node_tokens], dim=1)
        cls_mask = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=device
        )
        padding_mask = torch.cat([cls_mask, ~valid_nodes], dim=1)
        output = self.struct_transformer(
            sequence, src_key_padding_mask=padding_mask
        )
        cls_out = output[:, 0, :]
        node_out = output[:, 1:, :]
        valid_float = valid_nodes.unsqueeze(-1).to(node_out.dtype)
        denom = valid_float.sum(dim=1).clamp_min(1.0)
        mean_pool = (node_out * valid_float).sum(dim=1) / denom
        event_features = self.readout_projection(
            torch.cat([cls_out, mean_pool], dim=-1)
        )
        if return_node_features:
            return event_features, node_out
        return event_features


class OriginalRumorDetector(nn.Module):
    """The only model supported by this project."""

    def __init__(self, num_classes=2, hidden_dim=256, text_feat_dim=384,
                 extra_feat_dim=11, struct_feat_dim=1024,
                 dropout_rate=0.35):
        super().__init__()
        self.d_model = hidden_dim * 3
        self.node_feat_dim = text_feat_dim + extra_feat_dim
        self.node_token_branch = NodeTokenSetEncoder(
            hidden_dim=hidden_dim,
            output_dim=self.d_model,
            dropout_rate=dropout_rate,
        )
        self.graph_dropout = nn.Dropout(dropout_rate * 0.6)
        self.graph_branch = OriginalGraphBranch(
            node_feat_dim=self.node_feat_dim,
            struct_feat_dim=struct_feat_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(self.d_model, num_classes),
        )

    def encode_event(
        self, node_feats, struct_feats, num_nodes, return_node_features=False
    ):
        node_token_feat = self.node_token_branch(
            node_feats, struct_feats, num_nodes
        )
        graph_output = self.graph_branch(
            node_feats,
            struct_feats,
            num_nodes,
            return_node_features=return_node_features,
        )
        if return_node_features:
            graph_feat, node_features = graph_output
        else:
            graph_feat = graph_output
            node_features = None
        graph_feat = self.graph_dropout(graph_feat)
        event_features = self.fusion_proj(
            torch.cat([node_token_feat, graph_feat], dim=-1)
        )
        if return_node_features:
            return event_features, node_features
        return event_features

    def forward(
        self,
        node_feats,
        struct_feats,
        num_nodes,
        return_features=False,
        return_node_features=False,
    ):
        encoded = self.encode_event(
            node_feats,
            struct_feats,
            num_nodes,
            return_node_features=return_node_features,
        )
        if return_node_features:
            fused, node_features = encoded
        else:
            fused = encoded
            node_features = None
        logits = self.classifier(fused)
        if return_features and return_node_features:
            return logits, fused, node_features
        if return_features:
            return logits, fused
        if return_node_features:
            return logits, node_features
        return logits
