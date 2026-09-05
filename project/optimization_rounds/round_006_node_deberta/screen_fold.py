"""Screen one strict fold of the node-level DeBERTa replacement pipeline."""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "protocol"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from optimization_rounds.round_006_node_deberta.dataset import (  # noqa: E402
    NodeDebertaGraphDataset,
)
from optimization_rounds.round_006_node_deberta.model import (  # noqa: E402
    build_model,
)
from rumor_detection.config.loader import load_config  # noqa: E402
from rumor_detection.datasets.collate import original_graph_collate  # noqa: E402
from rumor_detection.training.trainer import Trainer  # noqa: E402
from rumor_detection.utils.seed import set_seed  # noqa: E402


class ThreadDataset(Dataset):
    def __init__(self, texts, labels, indices, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = int(self.labels[index])
        return encoded


class NodeTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=96):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def ordered_labels(data_dir):
    event_ids = [
        value.strip()
        for value in (data_dir / "splits" / "all_event_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if value.strip()
    ]
    labels = pd.read_csv(data_dir / "labels.csv", dtype={"event_id": str})
    label_map = dict(
        zip(labels["event_id"].astype(str), labels["label"].astype(int))
    )
    values = np.asarray([label_map[event_id] for event_id in event_ids])
    frame = pd.DataFrame({"event_id": event_ids, "label": values})
    return event_ids, values, frame


def strict_indices(labels, fold_index, partition_seed=3090):
    indices = np.arange(len(labels))
    splitter = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=int(partition_seed)
    )
    train_val, test = list(splitter.split(indices, labels))[int(fold_index)]
    train, validation = train_test_split(
        train_val,
        test_size=0.10,
        random_state=int(partition_seed),
        shuffle=True,
        stratify=labels[train_val],
    )
    return train.tolist(), validation.tolist(), test.tolist()


def read_tweet(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return str(json.load(handle).get("text", "")).strip()
    except (OSError, ValueError, AttributeError):
        return ""


def load_node_ids(graph_dir, event_ids):
    event_nodes = {}
    needed = set()
    for event_id in event_ids:
        graph = torch.load(
            graph_dir / f"{event_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        node_ids = [str(value) for value in graph.get("node_ids", [])]
        if len(node_ids) != int(graph["num_nodes"]):
            raise ValueError(f"node_ids mismatch for event {event_id}")
        event_nodes[event_id] = node_ids
        needed.update(node_ids)
    return event_nodes, needed


def index_raw_text(raw_dir, needed):
    paths = {}
    patterns = (
        raw_dir / "*" / "*" / "*" / "source-tweets" / "*.json",
        raw_dir / "*" / "*" / "*" / "reactions" / "*.json",
    )
    for pattern in patterns:
        for value in glob.glob(str(pattern)):
            path = Path(value)
            node_id = path.stem
            if node_id in needed and not path.name.startswith("._"):
                paths[node_id] = path
    missing = needed.difference(paths)
    if missing:
        raise FileNotFoundError(f"Missing raw JSON for {len(missing)} nodes")
    return {node_id: read_tweet(path) for node_id, path in paths.items()}


def make_thread_texts(event_ids, event_nodes, raw_text):
    return [
        " [SEP] ".join(raw_text[node_id] for node_id in event_nodes[event_id])
        for event_id in event_ids
    ]


def evaluate_text_model(model, loader, device):
    model.eval()
    losses, labels, predictions = [], [], []
    with torch.no_grad():
        for batch in loader:
            target = batch.pop("labels")
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits
                loss = F.cross_entropy(logits, target.to(device))
            losses.append(float(loss) * len(target))
            labels.extend(target.numpy().tolist())
            predictions.extend(logits.argmax(-1).cpu().numpy().tolist())
    result = classification_metrics(labels, predictions)
    result["loss"] = sum(losses) / max(len(labels), 1)
    return result


def train_text_encoder(
    texts, labels, train_idx, val_idx, tokenizer, model_path, output_dir,
    device, max_epochs=60, patience=7,
):
    def collate(items):
        return tokenizer.pad(items, padding=True, return_tensors="pt")

    def loader(indices, shuffle):
        return DataLoader(
            ThreadDataset(texts, labels, indices, tokenizer),
            batch_size=8,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate,
        )

    train_loader = loader(train_idx, True)
    val_loader = loader(val_idx, False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, local_files_only=True
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01
    )
    accumulation = 8
    updates_per_epoch = (len(train_loader) + accumulation - 1) // accumulation
    total_updates = updates_per_epoch * int(max_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * 0.10),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = output_dir / "best_model.pt"
    best_macro_f1, stale = -1.0, 0
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
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
        validation = evaluate_text_model(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running / len(train_loader),
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["macro_f1"] > best_macro_f1 + 1e-6:
            best_macro_f1, stale = validation["macro_f1"], 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= int(patience):
                break
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return model


def encode_node_cache(
    model, tokenizer, event_ids, event_nodes, raw_text, cache_path, device
):
    ordered_nodes = []
    slices = {}
    for event_id in event_ids:
        start = len(ordered_nodes)
        ordered_nodes.extend(event_nodes[event_id])
        slices[event_id] = (start, len(ordered_nodes))
    texts = [raw_text[node_id] for node_id in ordered_nodes]

    def collate(items):
        return tokenizer.pad(items, padding=True, return_tensors="pt")

    loader = DataLoader(
        NodeTextDataset(texts, tokenizer),
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                hidden = model.deberta(**batch).last_hidden_state
                pooled = model.pooler(hidden)
            chunks.append(pooled.cpu().to(torch.float16))
    all_features = torch.cat(chunks, dim=0)
    cache = {
        event_id: all_features[start:end].contiguous()
        for event_id, (start, end) in slices.items()
    }
    torch.save(cache, cache_path)
    return cache


def train_graph_model(
    cfg, frame, graph_dir, node_cache, train_idx, val_idx, test_idx,
    output_dir, device, max_epochs,
):
    train_base = NodeDebertaGraphDataset(
        graph_dir,
        frame,
        augment=True,
        aug_config=cfg.training.get("augmentation", None),
        node_cache=node_cache,
    )
    eval_base = NodeDebertaGraphDataset(
        graph_dir, frame, augment=False, node_cache=node_cache
    )
    common = {
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.training.num_workers),
        "pin_memory": True,
        "collate_fn": original_graph_collate,
    }
    train_loader = DataLoader(
        Subset(train_base, train_idx), shuffle=True, **common
    )
    val_loader = DataLoader(
        Subset(eval_base, val_idx), shuffle=False, **common
    )
    test_loader = DataLoader(
        Subset(eval_base, test_idx), shuffle=False, **common
    )
    model = build_model(cfg)
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        cfg,
        device,
        experiment_dir=str(output_dir),
    )
    trainer.run(max_epochs=int(max_epochs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-index", type=int, default=0, choices=range(5))
    parser.add_argument("--training-seed", type=int, default=2000)
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
        default=str(ROOT / "results" / "round_006_node_deberta" / "screen"),
    )
    args = parser.parse_args()
    seed_all(args.training_seed)
    set_seed(args.training_seed)
    cfg = load_config(str(ROOT / "config" / "base.yaml"))
    data_dir = Path(cfg.data.output_dir)
    graph_dir = data_dir / "graph_final"
    event_ids, labels, frame = ordered_labels(data_dir)
    train_idx, val_idx, test_idx = strict_indices(
        labels, args.fold_index, args.partition_seed
    )
    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx):
        raise RuntimeError("Overlapping strict partitions")
    output_dir = (
        Path(args.output_root)
        / f"fold_{args.fold_index + 1}_train_{args.training_seed}"
    ).resolve()
    text_dir = output_dir / "text"
    graph_output = output_dir / "graph"
    text_dir.mkdir(parents=True, exist_ok=True)
    graph_output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scheme": "round_006_node_deberta",
        "fold": args.fold_index + 1,
        "training_seed": args.training_seed,
        "partition_seed": args.partition_seed,
        "partition_ratio": "72:8:20",
        "text_physical_batch": 8,
        "text_gradient_accumulation": 8,
        "text_effective_batch": 64,
        "text_checkpoint_selection": "validation_macro_f1",
        "graph_batch": int(cfg.training.batch_size),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"strict split train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)}",
        flush=True,
    )
    event_nodes, needed = load_node_ids(graph_dir, event_ids)
    raw_text = index_raw_text(Path(args.raw_dir), needed)
    thread_texts = make_thread_texts(event_ids, event_nodes, raw_text)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model = train_text_encoder(
        thread_texts,
        labels,
        train_idx,
        val_idx,
        tokenizer,
        args.model_path,
        text_dir,
        device,
        max_epochs=args.max_epochs,
        patience=int(cfg.training.patience),
    )
    cache_path = output_dir / "node_cache.pt"
    node_cache = encode_node_cache(
        text_model,
        tokenizer,
        event_ids,
        event_nodes,
        raw_text,
        cache_path,
        device,
    )
    del text_model
    torch.cuda.empty_cache()
    train_graph_model(
        cfg,
        frame,
        graph_dir,
        node_cache,
        train_idx,
        val_idx,
        test_idx,
        graph_output,
        device,
        args.max_epochs,
    )


if __name__ == "__main__":
    main()
