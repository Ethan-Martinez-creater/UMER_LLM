from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
for path in (PROJECT, PROJECT / "src", PROJECT / "protocol", PROJECT / "runner"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from optimization_rounds.round_006_node_deberta.screen_fold import (  # noqa: E402
    index_raw_text,
    load_node_ids,
    ordered_labels,
    strict_indices,
)
from optimization_rounds.round_009_multiview_deberta.run_fold import (  # noqa: E402
    build_event_views,
    classification_metrics,
)
from optimization_rounds.round_013_joint_trifusion.run_fold import (  # noqa: E402
    JointEventDataset,
    load_initialized_model,
    make_collate,
    move_batch,
)
from rumor_detection.config.loader import load_config  # noqa: E402
from rumor_detection.datasets.original_dataset import OriginalRumorDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--deberta-model", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition-seed", type=int, default=3090)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_fold_rows(rows, expected_indices, fold, event_ids):
    expected = set(map(int, expected_indices))
    observed = {int(row["dataset_index"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"fold {fold} rows do not exactly match outer test indices")
    if any(int(row["fold"]) != int(fold) for row in rows):
        raise ValueError(f"fold marker mismatch in fold {fold}")
    if any(str(row["event_id"]) != str(event_ids[int(row["dataset_index"])]) for row in rows):
        raise ValueError(f"event/index mismatch in fold {fold}")


@torch.no_grad()
def infer_fold(model, loader, threshold, fold, event_ids, device):
    rows = []
    model.eval()
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
        probabilities = torch.softmax(outputs["logits"].float(), dim=-1)[:, 1]
        graph_probabilities = torch.softmax(outputs["graph_logits"].float(), dim=-1)[:, 1]
        text_probabilities = torch.softmax(outputs["text_logits"].float(), dim=-1)[:, 1]
        view_probabilities = torch.softmax(
            outputs["ordinary_text_view_logits"].float(), dim=-1
        )[..., 1]
        retrieval_scores = outputs["retrieval_logits"].float().reshape(-1)
        for offset, index in enumerate(batch["indices"].detach().cpu().tolist()):
            probability = float(probabilities[offset])
            label = int(batch["labels"][offset])
            rows.append(
                {
                    "event_id": str(event_ids[index]),
                    "dataset_index": int(index),
                    "fold": int(fold),
                    "label": label,
                    "probability": probability,
                    "threshold": float(threshold),
                    "prediction": int(probability >= float(threshold)),
                    "error": int(int(probability >= float(threshold)) != label),
                    "num_nodes": int(batch["num_nodes"][offset]),
                    "graph_probability": float(graph_probabilities[offset]),
                    "text_probability": float(text_probabilities[offset]),
                    "text_view_probability_std": float(
                        view_probabilities[offset].std(unbiased=False)
                    ),
                    "retrieval_score": float(retrieval_scores[offset]),
                    "memory_position": int(batch["memory_positions"][offset]),
                    "view_count": int(batch["view_count"]),
                }
            )
    return rows


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    graph_dir = args.data_dir / "graph_final"
    event_ids, labels, frame = ordered_labels(args.data_dir)
    event_nodes, needed = load_node_ids(graph_dir, event_ids)
    raw_text = index_raw_text(args.raw_dir, needed)
    views = [build_event_views(event_nodes[event_id], raw_text) for event_id in event_ids]
    cfg = load_config(str(PROJECT / "config" / "base.yaml"))
    dataset = OriginalRumorDataset(str(graph_dir), frame, augment=False)
    tokenizer = DebertaV2Tokenizer(vocab_file=str(args.deberta_model / "spm.model"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir = args.output.parent / "oof_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    checkpoints = []
    for fold_index in range(5):
        _, _, test_indices = strict_indices(labels, fold_index, args.partition_seed)
        fold = fold_index + 1
        fold_output = fold_dir / f"fold_{fold}.jsonl"
        if fold_output.is_file():
            rows = read_rows(fold_output)
            validate_fold_rows(rows, test_indices, fold, event_ids)
            all_rows.extend(rows)
            continue
        checkpoint_path = args.checkpoint_root / f"fold_{fold}_train_2000" / "best_joint_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        model = load_initialized_model(
            cfg,
            "clean",
            None,
            args.deberta_model,
            None,
            state["memory_features"].detach().cpu(),
            state["memory_labels"].detach().cpu(),
            device,
            fusion_type="linear",
        )
        model.load_state_dict(state)
        loader = DataLoader(
            JointEventDataset(dataset, views, test_indices, tokenizer, {}, all_views=True),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
            collate_fn=make_collate(tokenizer, all_views=True),
        )
        rows = infer_fold(
            model, loader, float(checkpoint["threshold"]), fold, event_ids, device
        )
        validate_fold_rows(rows, test_indices, fold, event_ids)
        temporary = fold_output.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(fold_output)
        all_rows.extend(rows)
        checkpoints.append(
            {
                "fold": fold,
                "path": str(checkpoint_path),
                "threshold": float(checkpoint["threshold"]),
                "state_key_count": len(state),
            }
        )
        del model, checkpoint, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(all_rows) != len(event_ids) or len({row["dataset_index"] for row in all_rows}) != len(event_ids):
        raise RuntimeError("outer test predictions do not cover each event exactly once")
    all_rows.sort(key=lambda row: int(row["dataset_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    y_true = np.asarray([row["label"] for row in all_rows], dtype=np.int64)
    y_pred = np.asarray([row["prediction"] for row in all_rows], dtype=np.int64)
    summary = {
        "experiment_id": "umer_round043_cognitive_selective_routing_v1",
        "prediction_scope": "exactly one frozen outer-test checkpoint per event",
        "partition_seed": args.partition_seed,
        "event_count": len(all_rows),
        "unique_event_count": len({row["event_id"] for row in all_rows}),
        "fold_counts": {
            str(fold): sum(row["fold"] == fold for row in all_rows) for fold in range(1, 6)
        },
        "all_memory_positions_are_minus_one": all(row["memory_position"] == -1 for row in all_rows),
        "branch_and_view_scores_exported": True,
        "metrics": classification_metrics(y_true, y_pred),
        "checkpoints_loaded_this_run": checkpoints,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
