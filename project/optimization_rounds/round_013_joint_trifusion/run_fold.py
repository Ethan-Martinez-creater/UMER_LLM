"""Strict-fold joint training of graph, DeBERTa multi-view, and retrieval memory."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.is_dir() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))
for path in (ROOT, ROOT / "src", ROOT / "protocol", ROOT / "runner"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    get_linear_schedule_with_warmup,
)

from analysis.evaluate_retrieval_calibration import event_semantic_features  # noqa: E402
from optimization_rounds.round_006_node_deberta.screen_fold import (  # noqa: E402
    index_raw_text,
    load_node_ids,
    ordered_labels,
    seed_all,
    strict_indices,
)
from optimization_rounds.round_009_multiview_deberta.run_fold import (  # noqa: E402
    METRIC_NAMES,
    build_event_views,
    classification_metrics,
    select_threshold,
)
from optimization_rounds.round_013_joint_trifusion.model import (  # noqa: E402
    JointTriFusionModel,
)
from optimization_rounds.round_016_rdrop_joint.regularization import (  # noqa: E402
    rdrop_loss,
)
from optimization_rounds.round_021_cross_view_consistency.regularization import (  # noqa: E402
    cross_view_consistency_loss,
)
from optimization_rounds.round_022_checkpoint_averaging.averaging import (  # noqa: E402
    average_state_dicts,
)
from optimization_rounds.round_025_layerwise_lr_decay.optimizer import (  # noqa: E402
    build_text_optimizer_groups,
)
from optimization_rounds.round_033_llm_cognitive_distillation.targets import (  # noqa: E402
    active_event_coverage,
    load_aligned_cognitive_targets,
)
from optimization_rounds.round_034_cognitive_view_distillation.views import (  # noqa: E402
    load_aligned_cognitive_views,
)
from optimization_rounds.round_039_node_cognitive_distillation.node_targets import (  # noqa: E402
    active_event_coverage as node_active_event_coverage,
    load_aligned_node_targets,
    pack_node_targets,
)
from rumor_detection.config.loader import load_config  # noqa: E402
from rumor_detection.datasets.collate import original_graph_collate  # noqa: E402
from rumor_detection.datasets.original_dataset import OriginalRumorDataset  # noqa: E402
from train_original import build_model  # noqa: E402


def load_maweibo_graph_text(graph_dir, raw_dir, event_ids, horizon_hours=240):
    """Rebuild Ma-Weibo graph node IDs from chronological raw JSON posts.

    The Ma-Weibo graph preprocessing retained ``num_nodes`` but did not write
    ``node_ids``. It first filters to the configured horizon, sorts posts by
    timestamp, and truncates to the graph maximum. Repeating those deterministic
    steps recovers the text nodes used by each event graph.
    """
    event_nodes = {}
    raw_text = {}
    needed = set()
    horizon_seconds = int(horizon_hours) * 3600
    for event_id in event_ids:
        graph = torch.load(
            graph_dir / f"{event_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        num_nodes = int(graph["num_nodes"])
        path = raw_dir / f"{event_id}.json"
        with path.open("r", encoding="utf-8") as handle:
            posts = json.load(handle)
        timestamps = [int(post.get("t", 0) or 0) for post in posts]
        min_timestamp = min(timestamps) if timestamps else 0
        indexed_posts = [
            (offset, post)
            for offset, post in enumerate(posts)
            if 0 <= int(post.get("t", 0) or 0) - min_timestamp <= horizon_seconds
        ]
        indexed_posts.sort(
            key=lambda item: (int(item[1].get("t", 0) or 0), item[0])
        )
        selected = [post for _, post in indexed_posts[:num_nodes]]
        if len(selected) != num_nodes:
            raise ValueError(
                f"Ma-Weibo raw/graph node mismatch for event {event_id}: "
                f"eligible_raw={len(indexed_posts)}, graph={num_nodes}"
            )
        node_ids = []
        for post in selected:
            node_id = str(post.get("mid", post.get("id", "")))
            if not node_id:
                raise ValueError(f"empty Ma-Weibo node ID in event {event_id}")
            node_ids.append(node_id)
            needed.add(node_id)
            raw_text[node_id] = str(
                post.get("original_text", post.get("text", "")) or ""
            ).strip()
        event_nodes[event_id] = node_ids
    return event_nodes, needed, raw_text


class JointEventDataset(Dataset):
    def __init__(
        self,
        graph_dataset,
        views,
        indices,
        tokenizer,
        memory_positions,
        all_views=False,
        sampled_view_count=1,
        max_length=256,
        teacher_views=None,
    ):
        self.graph_dataset = graph_dataset
        self.views = views
        self.indices = list(map(int, indices))
        self.tokenizer = tokenizer
        self.memory_positions = memory_positions
        self.all_views = bool(all_views)
        self.sampled_view_count = int(sampled_view_count)
        if self.sampled_view_count < 1:
            raise ValueError("sampled_view_count must be positive")
        self.max_length = int(max_length)
        self.teacher_views = teacher_views
        if self.teacher_views is not None and len(self.teacher_views) != len(self.views):
            raise ValueError("teacher views must align with ordinary event views")
        if self.teacher_views is not None and self.all_views:
            raise ValueError("teacher views are training-only")

    def __len__(self):
        return len(self.indices)

    def encode(self, text):
        return self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding=False
        )

    def __getitem__(self, item):
        index = self.indices[item]
        graph = self.graph_dataset[index]
        if self.teacher_views is not None:
            available = self.views[index]
            if self.sampled_view_count > len(available):
                raise ValueError("sampled_view_count exceeds available views")
            ordinary = random.sample(available, self.sampled_view_count)
            teacher = self.teacher_views[index]
            if teacher is None:
                teacher = random.choice(available)
            encoded = [self.encode(text) for text in ordinary]
            encoded.append(self.encode(teacher))
        elif self.all_views:
            encoded = [self.encode(text) for text in self.views[index]]
        elif self.sampled_view_count > 1:
            available = self.views[index]
            if self.sampled_view_count > len(available):
                raise ValueError("sampled_view_count exceeds available views")
            encoded = [
                self.encode(text)
                for text in random.sample(available, self.sampled_view_count)
            ]
        else:
            encoded = self.encode(random.choice(self.views[index]))
        return graph, encoded, int(self.memory_positions.get(index, -1)), index


def make_collate(tokenizer, all_views=False, teacher_view_count=0):
    def collate(items):
        graphs, encoded, positions, indices = zip(*items)
        node, struct, counts, labels = original_graph_collate(graphs)
        if all_views:
            view_count = len(encoded[0])
            flat = [view for event in encoded for view in event]
        else:
            view_count = 1
            flat = list(encoded)
        text_inputs = tokenizer.pad(flat, padding=True, return_tensors="pt")
        return {
            "node_feats": node,
            "struct_feats": struct,
            "num_nodes": counts,
            "labels": labels,
            "text_inputs": text_inputs,
            "memory_positions": torch.tensor(positions, dtype=torch.long),
            "indices": torch.tensor(indices, dtype=torch.long),
            "view_count": view_count,
            "teacher_view_count": int(teacher_view_count),
        }

    return collate


def move_batch(batch, device):
    return {
        "node_feats": batch["node_feats"].to(device, non_blocking=True),
        "struct_feats": batch["struct_feats"].to(device, non_blocking=True),
        "num_nodes": batch["num_nodes"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "text_inputs": {
            key: value.to(device, non_blocking=True)
            for key, value in batch["text_inputs"].items()
        },
        "memory_positions": batch["memory_positions"].to(
            device, non_blocking=True
        ),
        "indices": batch["indices"].to(device, non_blocking=True),
        "view_count": batch["view_count"],
        "teacher_view_count": batch.get("teacher_view_count", 0),
    }


def evaluate(
    model,
    loader,
    device,
    threshold=None,
    graph_aux_weight=0.15,
    text_aux_weight=0.15,
):
    model.eval()
    labels, probabilities, losses = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(
                    batch["node_feats"],
                    batch["struct_feats"],
                    batch["num_nodes"],
                    batch["text_inputs"],
                    batch["memory_positions"],
                    view_count=batch["view_count"],
                    teacher_view_count=batch["teacher_view_count"],
                )
                loss = model.loss(
                    outputs,
                    batch["labels"],
                    graph_weight=graph_aux_weight,
                    text_weight=text_aux_weight,
                )["total"]
            rumor = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            labels.extend(batch["labels"].cpu().numpy().tolist())
            probabilities.extend(rumor.cpu().numpy().tolist())
            losses.append(float(loss) * len(batch["labels"]))
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if threshold is None:
        threshold, result = select_threshold(labels, probabilities)
    else:
        result = classification_metrics(
            labels, (probabilities >= float(threshold)).astype(np.int64)
        )
    result["loss"] = sum(losses) / max(len(labels), 1)
    return threshold, result


def checkpoint_key(metrics, selection_rule):
    """Return a validation-only ordering key without reading test metrics."""
    if selection_rule == "legacy_four_metric_min":
        return (
            min(metrics[name] for name in METRIC_NAMES),
            metrics["macro_f1"],
            metrics["accuracy"],
        )
    if selection_rule == "four_metric_mean":
        return (
            sum(metrics[name] for name in METRIC_NAMES) / len(METRIC_NAMES),
        )
    if selection_rule == "target_macro_weighted_accuracy":
        return (
            metrics["macro_f1"],
            metrics["weighted_f1"],
            metrics["accuracy"],
        )
    raise ValueError(f"Unknown checkpoint selection rule: {selection_rule}")


def merge_full_outer_train_indices(train_indices, validation_indices, test_indices):
    """Merge only inner train/validation and prove outer test remains disjoint."""
    train = set(map(int, train_indices))
    validation = set(map(int, validation_indices))
    test = set(map(int, test_indices))
    if train & validation:
        raise ValueError("inner train and validation indices overlap")
    if train & test or validation & test:
        raise ValueError("outer test overlaps the refit training pool")
    return sorted(train | validation)


def load_initialized_model(
    cfg,
    initialization,
    graph_checkpoint,
    deberta_model_path,
    deberta_checkpoint,
    memory_features,
    memory_labels,
    device,
    fusion_type="linear",
    ablate_evidence="none",
    cognitive_target_dim=0,
    node_cognitive_target_dim=0,
):
    graph_model = build_model(cfg, "original")
    if initialization == "warm_start":
        if not graph_checkpoint or not deberta_checkpoint:
            raise ValueError("warm_start requires both task checkpoints")
        graph_state = torch.load(
            graph_checkpoint, map_location="cpu", weights_only=False
        )["model_state_dict"]
        graph_model.load_state_dict(graph_state)
    text_model = AutoModelForSequenceClassification.from_pretrained(
        deberta_model_path, num_labels=2, local_files_only=True
    )
    if initialization == "warm_start":
        text_model.load_state_dict(
            torch.load(deberta_checkpoint, map_location="cpu", weights_only=True)
        )
    elif initialization != "clean":
        raise ValueError(f"Unknown initialization: {initialization}")
    return JointTriFusionModel(
        graph_model,
        text_model,
        memory_features,
        memory_labels,
        retrieval_k=31,
        retrieval_temperature=0.05,
        fusion_initialization=(
            "late_fusion" if initialization == "warm_start" else "random"
        ),
        fusion_type=fusion_type,
        ablate_evidence=ablate_evidence,
        cognitive_target_dim=cognitive_target_dim,
        node_cognitive_target_dim=node_cognitive_target_dim,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-index", type=int, required=True, choices=range(5))
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--partition-seed", type=int, default=3090)
    parser.add_argument(
        "--inner-split-strategy",
        choices=("label", "topic_label"),
        default="label",
    )
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument(
        "--refit-full-outer-train",
        action="store_true",
        help="Merge the frozen inner train/validation pool and save one fixed epoch.",
    )
    parser.add_argument(
        "--fixed-stop-epoch",
        type=int,
        default=0,
        help="Validation-selected stage-one epoch used only by full-outer refit.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Stage-one validation threshold reused without refit/test reselection.",
    )
    parser.add_argument(
        "--scheduler-horizon-epochs",
        type=int,
        default=0,
        help="LR-plan horizon; zero preserves the existing --max-epochs behavior.",
    )
    parser.add_argument(
        "--protocol-smoke-no-test",
        action="store_true",
        help="Exercise a smoke-labelled training path without evaluating outer test.",
    )
    parser.add_argument(
        "--initialization", choices=("clean", "warm_start"), default="clean"
    )
    parser.add_argument("--scheme", default="round_014_clean_joint_trifusion")
    parser.add_argument("--graph-checkpoint")
    parser.add_argument("--deberta-checkpoint")
    parser.add_argument("--graph-lr", type=float, default=1e-4)
    parser.add_argument("--text-lr", type=float, default=2e-5)
    parser.add_argument("--text-layerwise-lr-decay", type=float, default=1.0)
    parser.add_argument("--fusion-lr", type=float, default=1e-3)
    parser.add_argument("--rdrop-weight", type=float, default=0.0)
    parser.add_argument(
        "--train-view-count", type=int, default=1, choices=(1, 2, 4)
    )
    parser.add_argument(
        "--classification-view-count", type=int, default=0, choices=(0, 1, 2, 4),
        help="Views used by text CE and fusion during training; 0 uses all encoded views.",
    )
    parser.add_argument("--view-consistency-weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-average-top-k", type=int, default=1, choices=(1, 2, 3)
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=(
            "legacy_four_metric_min",
            "four_metric_mean",
            "target_macro_weighted_accuracy",
        ),
        default="legacy_four_metric_min",
    )
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument(
        "--group-dro-step-size",
        type=float,
        default=0.0,
        help="Enable training-fold topic-label Group DRO when positive.",
    )
    parser.add_argument(
        "--group-dro-min-group-size",
        type=int,
        default=128,
        help="Merge smaller training-only raw groups before Group DRO.",
    )
    parser.add_argument(
        "--sam-rho",
        type=float,
        default=0.0,
        help="Enable effective-batch SAM with this neighborhood radius.",
    )
    parser.add_argument(
        "--fusion-type",
        choices=("linear", "reliability_gate", "conditional_residual"),
        default="linear",
    )
    parser.add_argument(
        "--ablate-evidence",
        choices=("none", "graph", "text", "memory"),
        default="none",
    )
    parser.add_argument("--graph-aux-weight", type=float, default=0.15)
    parser.add_argument("--text-aux-weight", type=float, default=0.15)
    parser.add_argument("--cognitive-targets")
    parser.add_argument("--cognitive-target-schema")
    parser.add_argument("--cognitive-aux-weight", type=float, default=0.0)
    parser.add_argument(
        "--minimum-cognitive-train-coverage", type=float, default=0.95
    )
    parser.add_argument("--node-cognitive-targets")
    parser.add_argument("--node-cognitive-target-schema")
    parser.add_argument("--node-cognitive-aux-weight", type=float, default=0.0)
    parser.add_argument(
        "--minimum-node-cognitive-train-coverage", type=float, default=0.95
    )
    parser.add_argument("--cognitive-training-views")
    parser.add_argument("--cognitive-training-view-schema")
    parser.add_argument("--teacher-view-count", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--minimum-cognitive-view-coverage", type=float, default=0.99
    )
    parser.add_argument(
        "--view-strategy", choices=("standard", "key_reply"), default="standard"
    )
    parser.add_argument(
        "--deberta-model", default="/data/jyz/next/model/deberta-v3-base"
    )
    parser.add_argument(
        "--raw-dir",
        default="/data/jyz/rumor_detection/data/PHEME_extension/all-rnr-annotated-threads",
    )
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--raw-format", choices=("pheme", "ma_weibo"), default="pheme"
    )
    parser.add_argument("--physical-event-batch-size", type=int, default=0)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "round_013_joint_trifusion" / "screen"),
    )
    args = parser.parse_args()
    if args.view_consistency_weight > 0.0 and args.train_view_count < 2:
        parser.error("view consistency requires --train-view-count 2")
    if args.classification_view_count > args.train_view_count:
        parser.error("classification view count cannot exceed train view count")
    if args.rdrop_weight > 0.0 and args.view_consistency_weight > 0.0:
        parser.error("R-Drop and cross-view consistency cannot be enabled together")
    if args.group_dro_step_size < 0.0:
        parser.error("Group DRO step size cannot be negative")
    if args.group_dro_min_group_size < 1:
        parser.error("Group DRO minimum group size must be positive")
    if args.group_dro_step_size > 0.0 and args.raw_format != "pheme":
        parser.error("topic-label Group DRO currently supports PHEME only")
    if args.group_dro_step_size > 0.0 and args.rdrop_weight > 0.0:
        parser.error("Group DRO and R-Drop cannot be enabled together")
    if args.group_dro_step_size > 0.0 and args.refit_full_outer_train:
        parser.error("Group DRO is not defined for full-outer refit")
    if args.sam_rho < 0.0:
        parser.error("SAM rho cannot be negative")
    if args.sam_rho > 0.0 and args.group_dro_step_size > 0.0:
        parser.error("SAM and Group DRO cannot be enabled together")
    if args.sam_rho > 0.0 and args.rdrop_weight > 0.0:
        parser.error("SAM and R-Drop cannot be enabled together")
    if args.sam_rho > 0.0 and args.refit_full_outer_train:
        parser.error("SAM is not defined for full-outer refit")
    if args.graph_aux_weight < 0.0 or args.text_aux_weight < 0.0:
        parser.error("auxiliary loss weights cannot be negative")
    if args.cognitive_aux_weight < 0.0:
        parser.error("cognitive auxiliary loss weight cannot be negative")
    if not 0.0 <= args.minimum_cognitive_train_coverage <= 1.0:
        parser.error("minimum cognitive coverage must be within [0,1]")
    if args.cognitive_aux_weight > 0.0:
        if not args.cognitive_targets or not args.cognitive_target_schema:
            parser.error("cognitive loss requires target and schema files")
        if args.ablate_evidence == "graph":
            parser.error("cognitive loss requires the graph branch")
        if args.rdrop_weight > 0.0 or args.group_dro_step_size > 0.0:
            parser.error("cognitive loss is not combined with R-Drop or Group DRO")
        if args.refit_full_outer_train:
            parser.error("cognitive loss screen is stage-one only")
    elif args.cognitive_targets or args.cognitive_target_schema:
        parser.error("cognitive files require a positive cognitive auxiliary weight")
    if args.node_cognitive_aux_weight < 0.0:
        parser.error("node cognitive auxiliary loss weight cannot be negative")
    if not 0.0 <= args.minimum_node_cognitive_train_coverage <= 1.0:
        parser.error("minimum node cognitive coverage must be within [0,1]")
    if args.node_cognitive_aux_weight > 0.0:
        if not args.node_cognitive_targets or not args.node_cognitive_target_schema:
            parser.error("node cognitive loss requires target and schema files")
        if args.cognitive_aux_weight > 0.0 or args.teacher_view_count > 0:
            parser.error("node, event, and view cognitive screens are separate")
        if args.ablate_evidence == "graph":
            parser.error("node cognitive loss requires the graph branch")
        if args.rdrop_weight > 0.0 or args.group_dro_step_size > 0.0:
            parser.error(
                "node cognitive loss is not combined with R-Drop or Group DRO"
            )
        if args.refit_full_outer_train:
            parser.error("node cognitive loss screen is stage-one only")
    elif args.node_cognitive_targets or args.node_cognitive_target_schema:
        parser.error(
            "node cognitive files require a positive node cognitive auxiliary weight"
        )
    if not 0.0 <= args.minimum_cognitive_view_coverage <= 1.0:
        parser.error("minimum cognitive-view coverage must be within [0,1]")
    if args.teacher_view_count > 0:
        if not args.cognitive_training_views or not args.cognitive_training_view_schema:
            parser.error("teacher view requires cognitive view and schema files")
        if args.view_consistency_weight <= 0.0:
            parser.error("teacher view requires positive view consistency")
        if args.cognitive_aux_weight > 0.0:
            parser.error("cognitive target and cognitive view screens are separate")
        if args.node_cognitive_aux_weight > 0.0:
            parser.error("node cognitive target and cognitive view screens are separate")
        if args.refit_full_outer_train:
            parser.error("cognitive view screen is stage-one only")
    elif args.cognitive_training_views or args.cognitive_training_view_schema:
        parser.error("cognitive view files require --teacher-view-count 1")
    if args.ablate_evidence == "graph" and args.graph_aux_weight != 0.0:
        parser.error("graph ablation requires --graph-aux-weight 0")
    if args.ablate_evidence == "text" and args.text_aux_weight != 0.0:
        parser.error("text ablation requires --text-aux-weight 0")
    if args.ablate_evidence == "text" and args.view_consistency_weight != 0.0:
        parser.error("text ablation requires --view-consistency-weight 0")
    if not 0.0 < args.text_layerwise_lr_decay <= 1.0:
        parser.error("text layerwise LR decay must be within (0, 1]")
    if args.inner_split_strategy == "topic_label" and args.raw_format != "pheme":
        parser.error("topic-label inner split currently supports PHEME only")
    if args.refit_full_outer_train:
        if args.fixed_stop_epoch < 1:
            parser.error("full-outer refit requires --fixed-stop-epoch")
        if args.fixed_threshold is None or not 0.0 <= args.fixed_threshold <= 1.0:
            parser.error("full-outer refit requires --fixed-threshold within [0, 1]")
        if args.checkpoint_average_top_k != 1:
            parser.error("full-outer refit forbids checkpoint averaging")
        if args.max_train_samples > 0:
            parser.error("full-outer refit forbids sample truncation")
    elif args.fixed_stop_epoch or args.fixed_threshold is not None:
        parser.error("fixed epoch/threshold are valid only with full-outer refit")
    if args.protocol_smoke_no_test:
        if "smoke" not in args.scheme.lower():
            parser.error("no-test protocol smoke requires a smoke-labelled scheme")
    scheduler_horizon_epochs = (
        args.scheduler_horizon_epochs
        if args.scheduler_horizon_epochs > 0
        else args.max_epochs
    )
    training_epoch_limit = (
        args.fixed_stop_epoch
        if args.refit_full_outer_train
        else args.max_epochs
    )
    if scheduler_horizon_epochs < training_epoch_limit:
        parser.error("scheduler horizon cannot be shorter than the training limit")
    seed_all(args.training_seed)
    cfg = load_config(str(ROOT / "config" / "base.yaml"))
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg.data.output_dir)
    graph_dir = data_dir / "graph_final"
    event_ids, labels, frame = ordered_labels(data_dir)
    train_idx, val_idx, test_idx = strict_indices(
        labels, args.fold_index, args.partition_seed
    )
    if args.inner_split_strategy == "topic_label":
        from analysis.evaluate_topic_prior_calibration import topic_index
        from optimization_rounds.round_026_topic_stratified_validation.splits import (
            rebuild_topic_label_inner_split,
        )

        topics = topic_index(args.raw_dir)
        train_idx, val_idx = rebuild_topic_label_inner_split(
            labels,
            event_ids,
            train_idx,
            val_idx,
            topics,
            args.partition_seed,
        )
    stage_one_train_size = len(train_idx)
    stage_one_validation_size = len(val_idx)
    if args.refit_full_outer_train:
        train_idx = merge_full_outer_train_indices(
            train_idx, val_idx, test_idx
        )
    memory_positions = {int(index): offset for offset, index in enumerate(train_idx)}
    train_graph = OriginalRumorDataset(
        str(graph_dir), frame, augment=True,
        aug_config=cfg.training.get("augmentation", None),
    )
    eval_graph = OriginalRumorDataset(str(graph_dir), frame, augment=False)
    memory_features = event_semantic_features(eval_graph)[train_idx]
    memory_labels = torch.as_tensor(labels[train_idx], dtype=torch.long)

    if args.raw_format == "ma_weibo":
        event_nodes, needed, raw_text = load_maweibo_graph_text(
            graph_dir, Path(args.raw_dir), event_ids
        )
    else:
        event_nodes, needed = load_node_ids(graph_dir, event_ids)
        raw_text = index_raw_text(Path(args.raw_dir), needed)
    if args.view_strategy == "standard":
        views = [
            build_event_views(event_nodes[event_id], raw_text)
            for event_id in event_ids
        ]
    else:
        from optimization_rounds.round_015_key_reply_joint.views import (
            build_key_reply_views,
        )

        views = []
        for event_id in event_ids:
            graph = torch.load(
                graph_dir / f"{event_id}.pt",
                map_location="cpu",
                weights_only=True,
            )
            views.append(
                build_key_reply_views(
                    event_nodes[event_id],
                    raw_text,
                    graph["node_feat"][: int(graph["num_nodes"])],
                    graph.get("edge_index", torch.empty((2, 0), dtype=torch.long)),
                )
            )
    tokenizer = DebertaV2Tokenizer(
        vocab_file=str(Path(args.deberta_model) / "spm.model")
    )
    selected_train = list(map(int, train_idx))
    if args.max_train_samples > 0:
        selected_train = selected_train[: args.max_train_samples]
    cognitive_training_views = None
    cognitive_view_report = None
    cognitive_view_train_coverage = 0.0
    if args.teacher_view_count > 0:
        cognitive_training_views, cognitive_view_report = (
            load_aligned_cognitive_views(
                event_ids,
                args.cognitive_training_views,
                args.cognitive_training_view_schema,
            )
        )
        available = sum(
            cognitive_training_views[index] is not None for index in selected_train
        )
        cognitive_view_train_coverage = available / max(len(selected_train), 1)
        if cognitive_view_train_coverage < args.minimum_cognitive_view_coverage:
            raise RuntimeError(
                "cognitive training-view coverage below declared minimum: "
                f"{cognitive_view_train_coverage:.6f} < "
                f"{args.minimum_cognitive_view_coverage:.6f}"
            )
    group_definition = None
    group_dro = None
    group_index = None
    if args.group_dro_step_size > 0.0:
        from analysis.evaluate_topic_prior_calibration import topic_index
        from optimization_rounds.round_031_topic_label_group_dro.group_dro import (
            GroupDROState,
            build_training_groups,
        )

        group_definition = build_training_groups(
            event_ids,
            labels,
            selected_train,
            topic_index(args.raw_dir),
            minimum_raw_group_size=args.group_dro_min_group_size,
        )
        group_dro = GroupDROState(
            group_definition.counts,
            step_size=args.group_dro_step_size,
        )
    multi_view_train = args.train_view_count > 1
    default_train_batch_size = 8 // args.train_view_count
    train_batch_size = (
        args.physical_event_batch_size
        if args.physical_event_batch_size > 0
        else default_train_batch_size
    )
    if 64 % train_batch_size != 0:
        parser.error("physical event batch size must divide effective batch 64")
    train_loader = DataLoader(
        JointEventDataset(
            train_graph, views, selected_train, tokenizer, memory_positions,
            all_views=False,
            sampled_view_count=args.train_view_count,
            teacher_views=cognitive_training_views,
        ),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=make_collate(
            tokenizer,
            all_views=(multi_view_train or args.teacher_view_count > 0),
            teacher_view_count=args.teacher_view_count,
        ),
    )
    val_loader = DataLoader(
        JointEventDataset(
            eval_graph, views, val_idx, tokenizer, memory_positions,
            all_views=True,
        ),
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=make_collate(tokenizer, all_views=True),
    )
    test_loader = DataLoader(
        JointEventDataset(
            eval_graph, views, test_idx, tokenizer, memory_positions,
            all_views=True,
        ),
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=make_collate(tokenizer, all_views=True),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cognitive_targets = None
    cognitive_masks = None
    cognitive_report = None
    cognitive_train_coverage = 0.0
    if args.cognitive_aux_weight > 0.0:
        cognitive_targets, cognitive_masks, cognitive_report = (
            load_aligned_cognitive_targets(
                event_ids, args.cognitive_targets, args.cognitive_target_schema
            )
        )
        cognitive_train_coverage = active_event_coverage(
            cognitive_masks, selected_train
        )
        if cognitive_train_coverage < args.minimum_cognitive_train_coverage:
            raise RuntimeError(
                "cognitive target coverage below declared minimum: "
                f"{cognitive_train_coverage:.6f} < "
                f"{args.minimum_cognitive_train_coverage:.6f}"
            )
        cognitive_targets = cognitive_targets.to(device)
        cognitive_masks = cognitive_masks.to(device)
    node_cognitive_targets = None
    node_cognitive_report = None
    node_cognitive_train_coverage = 0.0
    if args.node_cognitive_aux_weight > 0.0:
        node_cognitive_targets, node_cognitive_report = load_aligned_node_targets(
            event_ids,
            args.node_cognitive_targets,
            args.node_cognitive_target_schema,
        )
        node_cognitive_train_coverage = node_active_event_coverage(
            node_cognitive_targets, selected_train
        )
        if (
            node_cognitive_train_coverage
            < args.minimum_node_cognitive_train_coverage
        ):
            raise RuntimeError(
                "node cognitive target coverage below declared minimum: "
                f"{node_cognitive_train_coverage:.6f} < "
                f"{args.minimum_node_cognitive_train_coverage:.6f}"
            )
    if group_definition is not None:
        group_index = group_definition.index_to_group.to(device)
    class_counts = np.bincount(labels[train_idx], minlength=2)
    if args.class_weight_power > 0.0:
        rumor_weight = (class_counts[0] / class_counts[1]) ** args.class_weight_power
        class_weights = torch.tensor(
            [1.0, rumor_weight], dtype=torch.float32, device=device
        )
    else:
        class_weights = None
    model = load_initialized_model(
        cfg,
        args.initialization,
        args.graph_checkpoint,
        args.deberta_model,
        args.deberta_checkpoint,
        memory_features,
        memory_labels,
        device,
        fusion_type=args.fusion_type,
        ablate_evidence=args.ablate_evidence,
        cognitive_target_dim=(
            int(cognitive_report["target_dimension"])
            if cognitive_report is not None
            else 0
        ),
        node_cognitive_target_dim=(
            int(node_cognitive_report["target_dimension"])
            if node_cognitive_report is not None
            else 0
        ),
    )
    optimizer_groups = [
        {"params": model.graph_model.parameters(), "lr": args.graph_lr}
    ]
    optimizer_groups.extend(
        build_text_optimizer_groups(
            model.text_model, args.text_lr, args.text_layerwise_lr_decay
        )
    )
    optimizer_groups.append(
        {"params": model.fusion.parameters(), "lr": args.fusion_lr}
    )
    if model.cognitive_head is not None:
        optimizer_groups.append(
            {"params": model.cognitive_head.parameters(), "lr": args.graph_lr}
        )
    if model.node_cognitive_head is not None:
        optimizer_groups.append(
            {"params": model.node_cognitive_head.parameters(), "lr": args.graph_lr}
        )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    accumulation = 64 // train_batch_size
    updates_per_epoch = (len(train_loader) + accumulation - 1) // accumulation
    total_updates = updates_per_epoch * scheduler_horizon_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * 0.10), total_updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sam_controller = None
    if args.sam_rho > 0.0:
        from optimization_rounds.round_032_sam.sam import SAMController

        sam_controller = SAMController(optimizer, rho=args.sam_rho)
    run_dir = Path(args.output_root) / f"fold_{args.fold_index + 1}_train_{args.training_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / (
        "fixed_refit_model.pt"
        if args.refit_full_outer_train
        else "best_joint_model.pt"
    )
    best_key, best_threshold, best_epoch, stale = None, 0.5, None, 0
    top_checkpoints = []
    averaging_candidates = []
    history = []

    def training_batch_losses(raw_batch, observe_group=True):
        batch = move_batch(raw_batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs_a = model(
                batch["node_feats"], batch["struct_feats"],
                batch["num_nodes"], batch["text_inputs"],
                batch["memory_positions"], view_count=batch["view_count"],
                classification_view_count=args.classification_view_count,
                teacher_view_count=batch["teacher_view_count"],
                compute_node_cognitive=(node_cognitive_targets is not None),
            )
            if args.rdrop_weight > 0.0:
                outputs_b = model(
                    batch["node_feats"], batch["struct_feats"],
                    batch["num_nodes"], batch["text_inputs"],
                    batch["memory_positions"], view_count=batch["view_count"],
                    classification_view_count=args.classification_view_count,
                    teacher_view_count=batch["teacher_view_count"],
                )
                losses = rdrop_loss(
                    model, outputs_a, outputs_b, batch["labels"],
                    weight=args.rdrop_weight, class_weights=class_weights,
                )
            else:
                if group_dro is None:
                    packed_node_targets = (
                        pack_node_targets(
                            node_cognitive_targets,
                            batch["indices"].detach().cpu().tolist(),
                            batch["node_feats"].size(1),
                            device,
                        )
                        if node_cognitive_targets is not None
                        else None
                    )
                    losses = model.loss(
                        outputs_a,
                        batch["labels"],
                        graph_weight=args.graph_aux_weight,
                        text_weight=args.text_aux_weight,
                        class_weights=class_weights,
                        cognitive_targets=(
                            cognitive_targets[batch["indices"]]
                            if cognitive_targets is not None
                            else None
                        ),
                        cognitive_mask=(
                            cognitive_masks[batch["indices"]]
                            if cognitive_masks is not None
                            else None
                        ),
                        cognitive_weight=args.cognitive_aux_weight,
                        node_cognitive_targets=packed_node_targets,
                        node_cognitive_weight=args.node_cognitive_aux_weight,
                    )
                else:
                    from optimization_rounds.round_031_topic_label_group_dro.group_dro import (
                        per_event_supervised_losses,
                    )

                    losses = per_event_supervised_losses(
                        outputs_a,
                        batch["labels"],
                        class_weights=class_weights,
                    )
                    group_ids = group_index[batch["indices"]]
                    if bool(torch.any(group_ids < 0)):
                        raise RuntimeError(
                            "training batch contains an unassigned Group DRO event"
                        )
                    losses["total"] = group_dro.weighted_loss(
                        losses["per_event"], group_ids
                    )
                    if observe_group:
                        group_dro.observe(losses["per_event"], group_ids)
                if "cognitive" not in losses:
                    losses["cognitive"] = losses["total"].new_zeros(())
                if "node_cognitive" not in losses:
                    losses["node_cognitive"] = losses["total"].new_zeros(())
                if args.view_consistency_weight > 0.0:
                    consistency = cross_view_consistency_loss(
                        outputs_a["text_view_logits"]
                    )
                    losses["total"] = losses["total"] + (
                        args.view_consistency_weight * consistency
                    )
                    losses["consistency"] = consistency
                else:
                    losses["consistency"] = losses["total"].new_zeros(())
        return losses

    def clip_model_gradients():
        if hasattr(model, "gradient_clip_parameter_groups"):
            clip_groups = model.gradient_clip_parameter_groups()
            if not clip_groups:
                raise RuntimeError("model returned no gradient clip groups")
            for parameters in clip_groups:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    for epoch in range(1, training_epoch_limit + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "total": 0.0,
            "final": 0.0,
            "graph": 0.0,
            "text": 0.0,
            "cognitive": 0.0,
            "node_cognitive": 0.0,
            "consistency": 0.0,
        }
        sam_grad_norm_total = 0.0
        sam_completed_updates = 0
        sam_skipped_nonfinite = 0
        if sam_controller is None:
            for step, raw_batch in enumerate(train_loader, 1):
                losses = training_batch_losses(raw_batch)
                scaler.scale(losses["total"] / accumulation).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    clip_model_gradients()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if group_dro is not None:
                        group_dro.step()
                for name in totals:
                    totals[name] += float(losses[name].detach())
        else:
            from optimization_rounds.round_032_sam.sam import (
                capture_torch_rng_state,
                effective_batch_windows,
                manually_unscale_gradients_,
                restore_torch_rng_state,
            )

            for raw_window in effective_batch_windows(
                train_loader, accumulation
            ):
                replay_states = []
                for raw_batch in raw_window:
                    replay_states.append(capture_torch_rng_state())
                    losses = training_batch_losses(raw_batch)
                    scaler.scale(
                        losses["total"] / accumulation
                    ).backward()
                    for name in totals:
                        totals[name] += float(losses[name].detach())
                scale = scaler.get_scale()
                first_pass_finite = manually_unscale_gradients_(
                    optimizer, scale
                )
                if not first_pass_finite:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update(new_scale=max(float(scale) / 2.0, 1.0))
                    sam_skipped_nonfinite += 1
                    continue
                sam_grad_norm_total += sam_controller.perturb()
                optimizer.zero_grad(set_to_none=True)
                try:
                    for raw_batch, rng_state in zip(
                        raw_window, replay_states
                    ):
                        restore_torch_rng_state(rng_state)
                        replay_losses = training_batch_losses(
                            raw_batch, observe_group=False
                        )
                        scaler.scale(
                            replay_losses["total"] / accumulation
                        ).backward()
                finally:
                    sam_controller.restore()
                scaler.unscale_(optimizer)
                clip_model_gradients()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                sam_completed_updates += 1
        threshold, validation = evaluate(
            model,
            val_loader,
            device,
            threshold=(
                args.fixed_threshold if args.refit_full_outer_train else None
            ),
            graph_aux_weight=args.graph_aux_weight,
            text_aux_weight=args.text_aux_weight,
        )
        key = checkpoint_key(validation, args.checkpoint_selection)
        monitor_prefix = (
            "refit_train_pool_monitor"
            if args.refit_full_outer_train
            else "val"
        )
        record = {
            "epoch": epoch,
            **{f"train_{name}": value / len(train_loader) for name, value in totals.items()},
            "threshold": threshold,
            **{
                f"{monitor_prefix}_{name}": value
                for name, value in validation.items()
            },
        }
        if group_dro is not None:
            record["group_dro"] = group_dro.snapshot(group_definition.names)
        if sam_controller is not None:
            record["sam"] = {
                "rho": args.sam_rho,
                "completed_updates": sam_completed_updates,
                "skipped_nonfinite_first_passes": sam_skipped_nonfinite,
                "mean_first_pass_gradient_norm": (
                    sam_grad_norm_total / max(sam_completed_updates, 1)
                ),
                "replayed_microbatches": len(train_loader),
            }
        history.append(record)
        print(json.dumps(record), flush=True)
        if args.refit_full_outer_train:
            if epoch == training_epoch_limit:
                best_threshold = float(args.fixed_threshold)
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "threshold": best_threshold,
                        "fold": args.fold_index + 1,
                        "training_seed": args.training_seed,
                        "retrieval_k": model.retrieval_k,
                        "retrieval_temperature": model.retrieval_temperature,
                        "selection_scope": "stage_one_validation_history_only",
                        "refit_full_outer_train": True,
                        "fixed_stop_epoch": training_epoch_limit,
                        "scheduler_horizon_epochs": scheduler_horizon_epochs,
                    },
                    checkpoint_path,
                )
            continue
        if args.checkpoint_average_top_k > 1:
            qualifies = (
                len(top_checkpoints) < args.checkpoint_average_top_k
                or key > top_checkpoints[-1][0]
            )
            if qualifies:
                cpu_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                top_checkpoints.append((key, epoch, cpu_state))
                top_checkpoints.sort(key=lambda item: item[0], reverse=True)
                if len(top_checkpoints) > args.checkpoint_average_top_k:
                    top_checkpoints.pop()
        if best_key is None or key > best_key:
            best_key, best_threshold, best_epoch, stale = key, threshold, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": best_threshold,
                    "fold": args.fold_index + 1,
                    "training_seed": args.training_seed,
                    "retrieval_k": model.retrieval_k,
                    "retrieval_temperature": model.retrieval_temperature,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    selected_checkpoint_kind = (
        "fixed_epoch_full_outer_refit"
        if args.refit_full_outer_train
        else "single"
    )
    selected_source_epochs = [best_epoch]
    if args.checkpoint_average_top_k > 1 and not args.refit_full_outer_train:
        for count in range(2, len(top_checkpoints) + 1):
            source_epochs = [item[1] for item in top_checkpoints[:count]]
            averaged_state = average_state_dicts(
                [item[2] for item in top_checkpoints[:count]]
            )
            model.load_state_dict(averaged_state)
            threshold, validation = evaluate(
                model,
                val_loader,
                device,
                graph_aux_weight=args.graph_aux_weight,
                text_aux_weight=args.text_aux_weight,
            )
            key = checkpoint_key(validation, args.checkpoint_selection)
            candidate = {
                "checkpoint_count": count,
                "source_epochs": source_epochs,
                "threshold": threshold,
                **{f"val_{name}": value for name, value in validation.items()},
            }
            averaging_candidates.append(candidate)
            print(json.dumps({"checkpoint_average": candidate}), flush=True)
            if key > best_key:
                best_key, best_threshold = key, threshold
                selected_checkpoint_kind = f"top_{count}_parameter_average"
                selected_source_epochs = source_epochs
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "threshold": best_threshold,
                        "fold": args.fold_index + 1,
                        "training_seed": args.training_seed,
                        "retrieval_k": model.retrieval_k,
                        "retrieval_temperature": model.retrieval_temperature,
                    },
                    checkpoint_path,
                )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if args.protocol_smoke_no_test:
        test_metrics = None
    else:
        _, test_metrics = evaluate(
            model,
            test_loader,
            device,
            threshold=checkpoint["threshold"],
            graph_aux_weight=args.graph_aux_weight,
            text_aux_weight=args.text_aux_weight,
        )
    result = {
        "scheme": args.scheme,
        "initialization": args.initialization,
        "task_checkpoint_initialization": args.initialization == "warm_start",
        "view_strategy": args.view_strategy,
        "fusion_type": args.fusion_type,
        "ablate_evidence": args.ablate_evidence,
        "graph_aux_weight": args.graph_aux_weight,
        "text_aux_weight": args.text_aux_weight,
        "cognitive_aux_weight": args.cognitive_aux_weight,
        "cognitive_train_coverage": cognitive_train_coverage,
        "cognitive_target_report": cognitive_report,
        "node_cognitive_aux_weight": args.node_cognitive_aux_weight,
        "node_cognitive_train_coverage": node_cognitive_train_coverage,
        "node_cognitive_target_report": node_cognitive_report,
        "teacher_view_count": args.teacher_view_count,
        "cognitive_view_train_coverage": cognitive_view_train_coverage,
        "cognitive_view_report": cognitive_view_report,
        "fold": args.fold_index + 1,
        "training_seed": args.training_seed,
        "partition_seed": args.partition_seed,
        "inner_split_strategy": args.inner_split_strategy,
        "refit_full_outer_train": args.refit_full_outer_train,
        "stage_one_train_size": stage_one_train_size,
        "stage_one_validation_size": stage_one_validation_size,
        "final_training_size": len(train_idx),
        "fixed_stop_epoch": (
            args.fixed_stop_epoch if args.refit_full_outer_train else None
        ),
        "scheduler_horizon_epochs": scheduler_horizon_epochs,
        "threshold_selection_scope": (
            "frozen_stage_one_validation_history"
            if args.refit_full_outer_train
            else "current_inner_validation"
        ),
        "epoch_monitor_scope": (
            "in_sample_original_validation_subset_no_selection"
            if args.refit_full_outer_train
            else "held_out_inner_validation"
        ),
        "single_checkpoint": str(checkpoint_path),
        "joint_gradient_branches": ["graph", "deberta", "fusion"]
        + (
            ["training_only_cognitive_head"]
            if model.cognitive_head is not None
            else []
        )
        + (
            ["training_only_node_cognitive_head"]
            if model.node_cognitive_head is not None
            else []
        ),
        "retrieval_memory_scope": (
            "full_outer_train_buffer_leave_one_out"
            if args.refit_full_outer_train
            else "strict_train_fold_buffer_leave_one_out"
        ),
        "physical_batch_size": train_batch_size,
        "train_view_count": args.train_view_count,
        "classification_view_count": (
            args.classification_view_count or args.train_view_count
        ),
        "sequences_per_step": train_batch_size * (
            args.train_view_count + args.teacher_view_count
        ),
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": 64,
        "learning_rates": {
            "graph": args.graph_lr,
            "deberta": args.text_lr,
            "fusion": args.fusion_lr,
            "cognitive_head": (
                args.graph_lr if model.cognitive_head is not None else None
            ),
            "node_cognitive_head": (
                args.graph_lr if model.node_cognitive_head is not None else None
            ),
        },
        "text_layerwise_lr_decay": args.text_layerwise_lr_decay,
        "rdrop_weight": args.rdrop_weight,
        "view_consistency_weight": args.view_consistency_weight,
        "checkpoint_average_top_k": args.checkpoint_average_top_k,
        "checkpoint_selection": args.checkpoint_selection,
        "selected_checkpoint_kind": selected_checkpoint_kind,
        "selected_source_epochs": selected_source_epochs,
        "averaging_candidates": averaging_candidates,
        "class_weight_power": args.class_weight_power,
        "class_weights": (
            class_weights.detach().cpu().tolist() if class_weights is not None
            else [1.0, 1.0]
        ),
        "group_dro": (
            {
                "step_size": args.group_dro_step_size,
                "update_scope": "effective_event_batch_64",
                "initial_distribution": "empirical_training_group_frequency",
                "loss_scope": "final_plus_0.15_graph_plus_0.15_text",
                "consistency_scope": "ordinary_batch_mean",
                "group_definition": group_definition.as_dict(),
                "final_state": group_dro.snapshot(group_definition.names),
            }
            if group_dro is not None
            else None
        ),
        "sam": (
            {
                "rho": args.sam_rho,
                "base_optimizer": "AdamW",
                "perturbation_scope": "all_trainable_parameters",
                "first_pass_scope": "effective_event_batch_64",
                "second_pass_scope": "same_replayed_effective_event_batch_64",
                "torch_rng_replayed": True,
                "graph_and_text_views_replayed": True,
                "gradient_clip_scope": "second_pass_only",
            }
            if sam_controller is not None
            else None
        ),
        "threshold": checkpoint["threshold"],
        "test_evaluated": not args.protocol_smoke_no_test,
        "metrics": test_metrics,
    }
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    result_filename = (
        "smoke_result.json" if args.protocol_smoke_no_test else "result.json"
    )
    (run_dir / result_filename).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
