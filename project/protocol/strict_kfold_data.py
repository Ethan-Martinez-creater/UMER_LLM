"""Strict outer five-fold partitions with an inner validation split."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rumor_detection.datasets.collate import original_graph_collate
from rumor_detection.datasets.original_dataset import OriginalRumorDataset
from rumor_detection.datasets.splits import (
    _dataset_labels,
    load_event_labels,
)


def strict_kfold_indices(dataset, fold_index: int, partition_seed: int = 3090,
                         n_splits: int = 5, validation_fraction: float = 0.10):
    """Return disjoint train/validation/test indices for one outer fold."""
    if not 0 <= int(fold_index) < int(n_splits):
        raise ValueError(f"fold_index must be in [0, {n_splits})")
    labels = np.asarray(_dataset_labels(dataset), dtype=np.int64)
    indices = np.arange(len(labels), dtype=np.int64)
    outer = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(partition_seed),
    )
    train_val_idx, test_idx = list(outer.split(indices, labels))[int(fold_index)]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=float(validation_fraction),
        random_state=int(partition_seed),
        shuffle=True,
        stratify=labels[train_val_idx],
    )
    train_set, val_set, test_set = map(
        set, (train_idx.tolist(), val_idx.tolist(), test_idx.tolist())
    )
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Strict k-fold partition contains overlapping subsets")
    if train_set | val_set | test_set != set(indices.tolist()):
        raise RuntimeError("Strict k-fold partition does not cover the dataset")
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def create_strict_kfold_dataloaders(cfg, graph_dir, fold_index: int,
                                    partition_seed: int = 3090):
    """Create loaders for one shared strict outer fold."""
    splits_dir = os.path.join(cfg.data.output_dir, "splits")
    all_ids_path = os.path.join(splits_dir, "all_event_ids.txt")
    if not os.path.exists(all_ids_path):
        raise FileNotFoundError(f"Split file not found: {all_ids_path}")
    with open(all_ids_path, "r", encoding="utf-8") as handle:
        event_ids = [line.strip() for line in handle if line.strip()]
    label_df = load_event_labels(cfg, event_ids)
    full_dataset = OriginalRumorDataset(
        graph_dir=graph_dir,
        label_df=label_df,
        augment=False,
    )
    train_idx, val_idx, test_idx = strict_kfold_indices(
        full_dataset, fold_index=fold_index, partition_seed=partition_seed
    )
    aug_config = cfg.training.get("augmentation", None)
    if aug_config is not None:
        train_base = copy.deepcopy(full_dataset)
        train_base.augment = True
        train_base.aug_config = aug_config
    else:
        train_base = full_dataset
    train_dataset = Subset(train_base, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    common = {
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.training.num_workers),
        "pin_memory": True,
        "collate_fn": original_graph_collate,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    print(
        "Strict fold "
        f"{fold_index + 1}/5, partition_seed={partition_seed}, "
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader
