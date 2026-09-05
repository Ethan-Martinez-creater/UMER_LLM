"""Original propagation model with 768-dimensional DeBERTa node text features."""

from rumor_detection.models import OriginalRumorDetector


def build_model(cfg):
    return OriginalRumorDetector(
        num_classes=cfg.model.num_classes,
        hidden_dim=cfg.model.hidden_dim,
        text_feat_dim=768,
        extra_feat_dim=cfg.model.extra_feat_dim,
        struct_feat_dim=cfg.model.struct_feat_dim,
        dropout_rate=cfg.model.dropout_rate,
    )
