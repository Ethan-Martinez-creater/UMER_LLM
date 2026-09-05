"""Train the single supported original rumor detector."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "protocol"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rumor_detection.models import OriginalRumorDetector  # noqa: E402
from rumor_detection.training.trainer import Trainer  # noqa: E402
from rumor_detection.utils.device import get_device  # noqa: E402
from rumor_detection.utils.seed import set_seed  # noqa: E402
from strict_kfold_data import create_strict_kfold_dataloaders  # noqa: E402


def build_model(cfg, scheme):
    if scheme == "original":
        return OriginalRumorDetector(
            num_classes=cfg.model.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            text_feat_dim=cfg.model.text_feat_dim,
            extra_feat_dim=cfg.model.extra_feat_dim,
            struct_feat_dim=cfg.model.struct_feat_dim,
            dropout_rate=cfg.model.dropout_rate,
        )
    if scheme == "round_001_temporal_structure_bias":
        from optimization_rounds.round_001_temporal_structure_bias import (
            build_model as build_round_model,
        )
        return build_round_model(cfg)
    if scheme == "round_002_semantic_evolution_scl":
        from optimization_rounds.round_002_semantic_evolution_scl import (
            build_model as build_round_model,
        )
        return build_round_model(cfg)
    if scheme == "round_003_source_semantic_residual":
        from optimization_rounds.round_003_source_semantic_residual import (
            build_model as build_round_model,
        )
        return build_round_model(cfg)
    if scheme == "round_004_graph_guided_text_attention":
        from optimization_rounds.round_004_graph_guided_text_attention import (
            build_model as build_round_model,
        )
        return build_round_model(cfg)
    if scheme == "round_005_focal_ce":
        from optimization_rounds.round_005_focal_ce import (
            build_model as build_round_model,
        )
        return build_round_model(cfg)
    if scheme == "round_011_structure_aux_gate":
        from optimization_rounds.round_011_structure_aux_gate import (
            build_model as build_round_model,
        )
        return build_round_model(cfg)
    raise ValueError(f"Unknown optimization scheme: {scheme}")


def run_fold(cfg, fold_index, training_seed, partition_seed, max_epochs,
             run_dir, scheme="original"):
    set_seed(int(training_seed))
    cfg.seed = int(training_seed)
    graph_dir = Path(cfg.data.output_dir) / "graph_final"
    train_loader, val_loader, test_loader = create_strict_kfold_dataloaders(
        cfg,
        str(graph_dir),
        fold_index=int(fold_index),
        partition_seed=int(partition_seed),
    )
    model = build_model(cfg, scheme)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if scheme == "original" and parameter_count != 8_526_850:
        raise RuntimeError(f"Unexpected model parameter count: {parameter_count}")
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scheme: {scheme}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Model params: {parameter_count:,}")
    print(f"Fold: {fold_index + 1}/5, training seed: {training_seed}")
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        cfg,
        get_device(),
        experiment_dir=str(run_dir),
    )
    trainer.run(max_epochs=int(max_epochs))
    metrics_path = run_dir / "metrics" / "final_test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "scheme": scheme,
        "model": model.__class__.__name__,
        "model_params": parameter_count,
        "fold": int(fold_index + 1),
        "training_seed": int(training_seed),
        "partition_seed": int(partition_seed),
        "metrics": metrics,
    }
