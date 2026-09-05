"""Strict-fold training with reply-window augmentation and view ensembling."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from optimization_rounds.round_006_node_deberta.screen_fold import (  # noqa: E402
    index_raw_text,
    load_node_ids,
    ordered_labels,
    seed_all,
    strict_indices,
)
from rumor_detection.config.loader import load_config  # noqa: E402


METRIC_NAMES = ("accuracy", "weighted_f1", "macro_f1", "rumor_f1")


def build_event_views(node_ids, raw_text, separator=" [SEP] "):
    """Create deterministic windows that expose different reply subsets."""
    source = raw_text[node_ids[0]] if node_ids else ""
    replies = [raw_text[node_id] for node_id in node_ids[1:]]

    def join(selected):
        parts = [source] + [text for text in selected if text]
        return separator.join(parts)

    return (
        join(replies),
        join(list(reversed(replies))),
        join(replies[::2]),
        join(replies[1::2]),
    )


def classification_metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "weighted_f1": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "rumor_f1": f1_score(
            labels, predictions, pos_label=1, zero_division=0
        ),
    }


def select_threshold(labels, rumor_probabilities):
    best = None
    for threshold in np.linspace(0.25, 0.75, 101):
        predictions = (rumor_probabilities >= threshold).astype(np.int64)
        metrics = classification_metrics(labels, predictions)
        key = (
            min(metrics[name] for name in METRIC_NAMES),
            metrics["macro_f1"],
            metrics["accuracy"],
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    return best[1], best[2]


class RandomViewDataset(Dataset):
    def __init__(self, views, labels, indices, tokenizer, max_length=256):
        self.views = views
        self.labels = labels
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        text = random.choice(self.views[index])
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding=False
        )
        encoded["labels"] = int(self.labels[index])
        return encoded


class AllViewsDataset(Dataset):
    def __init__(self, views, labels, indices, tokenizer, max_length=256):
        self.views = views
        self.labels = labels
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        encoded = [
            self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            for text in self.views[index]
        ]
        return encoded, int(self.labels[index])


def evaluate_multiview(model, loader, tokenizer, device, threshold=None):
    model.eval()
    labels, probabilities, losses = [], [], []
    with torch.no_grad():
        for encoded_views, targets in loader:
            batch_size = len(targets)
            view_count = len(encoded_views[0])
            flat = [view for event in encoded_views for view in event]
            batch = tokenizer.pad(flat, padding=True, return_tensors="pt")
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits
                logits = logits.reshape(batch_size, view_count, 2).mean(dim=1)
                loss = F.cross_entropy(logits, targets.to(device))
            rumor = torch.softmax(logits, dim=-1)[:, 1]
            labels.extend(targets.numpy().tolist())
            probabilities.extend(rumor.cpu().numpy().tolist())
            losses.append(float(loss) * batch_size)
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if threshold is None:
        threshold, metrics = select_threshold(labels, probabilities)
    else:
        predictions = (probabilities >= float(threshold)).astype(np.int64)
        metrics = classification_metrics(labels, predictions)
    metrics["loss"] = sum(losses) / max(len(labels), 1)
    return threshold, metrics


def collate_all_views(items):
    views, labels = zip(*items)
    return list(views), torch.tensor(labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-index", type=int, required=True, choices=range(5))
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--partition-seed", type=int, default=3090)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument(
        "--model-path", default="/data/jyz/next/model/deberta-v3-base"
    )
    parser.add_argument(
        "--raw-dir",
        default=(
            "/data/jyz/rumor_detection/data/PHEME_extension/"
            "all-rnr-annotated-threads"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "round_009_multiview_deberta" / "screen"),
    )
    args = parser.parse_args()
    seed_all(args.training_seed)
    cfg = load_config(str(ROOT / "config" / "base.yaml"))
    data_dir = Path(cfg.data.output_dir)
    event_ids, labels, _ = ordered_labels(data_dir)
    train_idx, val_idx, test_idx = strict_indices(
        labels, args.fold_index, args.partition_seed
    )
    event_nodes, needed = load_node_ids(data_dir / "graph_final", event_ids)
    raw_text = index_raw_text(Path(args.raw_dir), needed)
    views = [
        build_event_views(event_nodes[event_id], raw_text)
        for event_id in event_ids
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=2, local_files_only=True
    ).to(device)

    def train_collate(items):
        return tokenizer.pad(items, padding=True, return_tensors="pt")

    train_loader = DataLoader(
        RandomViewDataset(views, labels, train_idx, tokenizer),
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        AllViewsDataset(views, labels, val_idx, tokenizer),
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_all_views,
    )
    test_loader = DataLoader(
        AllViewsDataset(views, labels, test_idx, tokenizer),
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_all_views,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01
    )
    accumulation = 8
    updates_per_epoch = (len(train_loader) + accumulation - 1) // accumulation
    total_updates = updates_per_epoch * args.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * 0.10), total_updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_dir = (
        Path(args.output_root)
        / f"fold_{args.fold_index + 1}_train_{args.training_seed}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    best_key, best_threshold, stale = None, 0.5, 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach())
        threshold, val_metrics = evaluate_multiview(
            model, val_loader, tokenizer, device
        )
        key = (
            min(val_metrics[name] for name in METRIC_NAMES),
            val_metrics["macro_f1"],
            val_metrics["accuracy"],
        )
        record = {
            "epoch": epoch,
            "train_loss": running / len(train_loader),
            "threshold": threshold,
            **{f"val_{name}": value for name, value in val_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if best_key is None or key > best_key:
            best_key, best_threshold, stale = key, threshold, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
            if stale >= int(cfg.training.patience):
                break
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    _, test_metrics = evaluate_multiview(
        model, test_loader, tokenizer, device, threshold=best_threshold
    )
    result = {
        "scheme": "round_009_multiview_deberta",
        "fold": args.fold_index + 1,
        "training_seed": args.training_seed,
        "partition_seed": args.partition_seed,
        "checkpoint_selection": "validation_min_of_four_metrics",
        "threshold_selection": "validation_only",
        "threshold": best_threshold,
        "physical_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 64,
        "views": ["early", "late_reversed", "even_replies", "odd_replies"],
        "metrics": test_metrics,
    }
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
