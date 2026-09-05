"""Dataset helpers required by the strict PHEME five-fold protocol."""

import os

import pandas as pd


def _dataset_labels(dataset):
    if hasattr(dataset, "samples"):
        return [int(sample["label"]) for sample in dataset.samples]
    if hasattr(dataset, "labels"):
        return [int(label) for label in dataset.labels]
    raise ValueError("Dataset does not expose labels for stratification")


def load_event_labels(cfg, event_ids):
    labels_file = cfg.data.labels_file
    if not os.path.isfile(labels_file):
        raise FileNotFoundError(f"Event label manifest not found: {labels_file}")
    labels = pd.read_csv(labels_file, dtype={"event_id": str})
    required = {"event_id", "label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Label manifest must contain {sorted(required)}")
    labels = labels[["event_id", "label"]].copy()
    labels["event_id"] = labels["event_id"].astype(str)
    labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype(int)
    if labels["event_id"].duplicated().any():
        raise ValueError("Label manifest contains duplicate event IDs")
    label_map = dict(zip(labels["event_id"], labels["label"]))
    missing = [event_id for event_id in event_ids if event_id not in label_map]
    if missing:
        raise ValueError(f"Missing labels for {len(missing)} events")
    return pd.DataFrame([
        {"event_id": event_id, "label": int(label_map[event_id])}
        for event_id in event_ids
    ])
