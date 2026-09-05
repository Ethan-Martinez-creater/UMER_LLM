"""Evaluate train-fold topic-prior logit calibration without test tuning."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "protocol", ROOT / "runner"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rumor_detection.config.loader import load_config  # noqa: E402
from rumor_detection.datasets.collate import original_graph_collate  # noqa: E402
from rumor_detection.datasets.original_dataset import OriginalRumorDataset  # noqa: E402
from rumor_detection.datasets.splits import load_event_labels  # noqa: E402
from strict_kfold_data import strict_kfold_indices  # noqa: E402
from train_original import build_model  # noqa: E402


METRICS = ("accuracy", "weighted_f1", "macro_f1", "rumor_f1")


def metrics(labels, predictions):
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


def topic_index(raw_dir):
    raw_dir = Path(raw_dir).resolve()
    result = {}
    pattern = raw_dir / "*" / "*" / "*" / "source-tweets" / "*.json"
    for value in glob.glob(str(pattern)):
        path = Path(value)
        if not path.name.startswith("._"):
            result[path.stem] = path.relative_to(raw_dir).parts[0]
    return result


def train_topic_deltas(event_ids, labels, train_indices, topics, shrinkage=20.0):
    train_labels = np.asarray([labels[index] for index in train_indices])
    global_prior = (train_labels.sum() + 1.0) / (len(train_labels) + 2.0)
    counts = {}
    for index in train_indices:
        topic = topics[event_ids[index]]
        positive, total = counts.get(topic, (0, 0))
        counts[topic] = (positive + int(labels[index]), total + 1)

    def logit(probability):
        probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        return math.log(probability / (1.0 - probability))

    global_logit = logit(global_prior)
    deltas = {}
    for topic, (positive, total) in counts.items():
        prior = (positive + shrinkage * global_prior) / (total + shrinkage)
        deltas[topic] = logit(prior) - global_logit
    return deltas


def apply_calibration(base_logits, topic_deltas, alpha, bias):
    return np.asarray(base_logits) + float(alpha) * np.asarray(topic_deltas) + float(bias)


def select_calibration(labels, base_logits, topic_deltas):
    best = None
    for alpha in np.linspace(-0.5, 2.0, 26):
        for bias in np.linspace(-1.0, 1.0, 41):
            calibrated = apply_calibration(base_logits, topic_deltas, alpha, bias)
            value = metrics(labels, (calibrated >= 0.0).astype(np.int64))
            key = (
                min(value[name] for name in METRICS),
                value["macro_f1"],
                value["accuracy"],
            )
            if best is None or key > best[0]:
                best = (key, float(alpha), float(bias), value)
    return best[1], best[2], best[3]


def infer(model, checkpoint_path, dataset, indices, cfg, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=64,
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=True,
        collate_fn=original_graph_collate,
    )
    labels, logits = [], []
    with torch.no_grad():
        for node_feat, struct_feat, num_nodes, target in loader:
            output = model(
                node_feat.to(device, non_blocking=True),
                struct_feat.to(device, non_blocking=True),
                num_nodes.to(device, non_blocking=True),
            )
            labels.extend(target.numpy().tolist())
            logits.extend((output[:, 1] - output[:, 0]).cpu().numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(logits), [
        dataset.samples[index]["event_id"] for index in indices
    ]


def mean_metrics(rows, key):
    return {
        name: float(np.mean([row[key][name] for row in rows]))
        for name in METRICS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--scheme", default="original")
    parser.add_argument("--run-subdir", default="")
    parser.add_argument(
        "--checkpoint-name", default="best_val_loss_model.pth"
    )
    parser.add_argument("--seeds", default="2000,3090,5090")
    parser.add_argument("--partition-seed", type=int, default=3090)
    parser.add_argument(
        "--raw-dir",
        default=(
            "/data/jyz/rumor_detection/data/PHEME_extension/"
            "all-rnr-annotated-threads"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cfg = load_config(str(ROOT / "config" / "base.yaml"))
    data_dir = Path(cfg.data.output_dir)
    event_ids = [
        value.strip()
        for value in (data_dir / "splits" / "all_event_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if value.strip()
    ]
    frame = load_event_labels(cfg, event_ids)
    labels = frame["label"].to_numpy(dtype=np.int64)
    dataset = OriginalRumorDataset(
        str(data_dir / "graph_final"), frame, augment=False
    )
    topics = topic_index(args.raw_dir)
    missing = set(event_ids).difference(topics)
    if missing:
        raise FileNotFoundError(f"Missing topics for {len(missing)} events")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows = []
    for seed in seeds:
        for fold in range(5):
            train_idx, val_idx, test_idx = strict_kfold_indices(
                dataset, fold, args.partition_seed
            )
            checkpoint = (
                Path(args.results_root)
                / f"fold_{fold + 1}_train_{seed}"
            )
            if args.run_subdir:
                checkpoint = checkpoint / args.run_subdir
            checkpoint = checkpoint / "checkpoints" / args.checkpoint_name
            model = build_model(cfg, args.scheme)
            val_y, val_logits, val_ids = infer(
                model, checkpoint, dataset, val_idx, cfg, device
            )
            test_y, test_logits, test_ids = infer(
                model, checkpoint, dataset, test_idx, cfg, device
            )
            deltas = train_topic_deltas(
                event_ids, labels, train_idx, topics
            )
            val_topic_delta = np.asarray([deltas[topics[event_id]] for event_id in val_ids])
            test_topic_delta = np.asarray([deltas[topics[event_id]] for event_id in test_ids])
            alpha, bias, validation_metrics = select_calibration(
                val_y, val_logits, val_topic_delta
            )
            base_test = metrics(test_y, (test_logits >= 0).astype(np.int64))
            calibrated_logits = apply_calibration(
                test_logits, test_topic_delta, alpha, bias
            )
            calibrated_test = metrics(
                test_y, (calibrated_logits >= 0).astype(np.int64)
            )
            row = {
                "seed": seed,
                "fold": fold + 1,
                "alpha": alpha,
                "bias": bias,
                "validation_metrics": validation_metrics,
                "base_test": base_test,
                "calibrated_test": calibrated_test,
            }
            fold_rows.append(row)
            print(json.dumps(row), flush=True)
    per_seed = {}
    for seed in seeds:
        rows = [row for row in fold_rows if row["seed"] == seed]
        base = mean_metrics(rows, "base_test")
        calibrated = mean_metrics(rows, "calibrated_test")
        per_seed[str(seed)] = {
            "base_means": base,
            "calibrated_means": calibrated,
            "passes_all": all(calibrated[name] > 0.88 for name in METRICS),
        }
    passing = [
        seed for seed in seeds if per_seed[str(seed)]["passes_all"]
    ]
    summary = {
        "scheme": args.scheme,
        "results_root": str(Path(args.results_root).resolve()),
        "per_seed": per_seed,
        "passing_seeds": passing,
        "two_pass_one_fail_stop": len(passing) == 2 and len(seeds) == 3,
        "folds": fold_rows,
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["per_seed"], indent=2), flush=True)


if __name__ == "__main__":
    main()
