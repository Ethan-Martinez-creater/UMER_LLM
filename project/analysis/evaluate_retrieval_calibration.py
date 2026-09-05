"""Evaluate strict-fold semantic kNN retrieval fused with model logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "protocol", ROOT / "runner"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.evaluate_topic_prior_calibration import (  # noqa: E402
    METRICS,
    infer,
    mean_metrics,
    metrics,
)
from rumor_detection.config.loader import load_config  # noqa: E402
from rumor_detection.datasets.original_dataset import OriginalRumorDataset  # noqa: E402
from rumor_detection.datasets.splits import load_event_labels  # noqa: E402
from strict_kfold_data import strict_kfold_indices  # noqa: E402
from train_original import build_model  # noqa: E402


def event_semantic_features(dataset, text_dim=384):
    rows = []
    for sample in dataset.samples:
        graph = torch.load(
            sample["graph_path"], map_location="cpu", weights_only=True
        )
        count = int(graph["num_nodes"])
        text = graph["node_feat"][:count, :text_dim].float()
        source = text[0]
        mean = text.mean(dim=0)
        maximum = text.max(dim=0).values
        blocks = [source, mean, maximum, mean - source]
        blocks = [torch.nn.functional.normalize(block, dim=0) for block in blocks]
        rows.append(torch.cat(blocks, dim=0))
    return torch.nn.functional.normalize(torch.stack(rows), dim=1)


def top_neighbors(train_features, query_features, max_k=63):
    similarities = query_features @ train_features.T
    return torch.topk(
        similarities, k=min(int(max_k), train_features.size(0)), dim=1
    )


def retrieval_logit(top_values, top_indices, train_labels, k, temperature):
    k = min(int(k), top_values.size(1))
    values = top_values[:, :k]
    indices = top_indices[:, :k]
    neighbor_labels = train_labels[indices].to(values.dtype)
    weights = torch.softmax(values / float(temperature), dim=1)
    probability = (weights * neighbor_labels).sum(dim=1)
    probability = probability.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(probability / (1.0 - probability)).cpu().numpy()


def retrieval_grid(train_features, query_features, train_labels):
    values, indices = top_neighbors(train_features, query_features, max_k=63)
    result = {}
    for k in (3, 5, 9, 15, 31, 63):
        for temperature in (0.05, 0.10, 0.20):
            result[(k, temperature)] = retrieval_logit(
                values, indices, train_labels, k, temperature
            )
    return result


def select_retrieval_fusion(labels, base_logits, retrieval_logits):
    best = None
    for (k, temperature), retrieved in retrieval_logits.items():
        for weight in np.linspace(0.0, 2.0, 9):
            for bias in np.linspace(-1.0, 1.0, 21):
                combined = base_logits + float(weight) * retrieved + float(bias)
                value = metrics(labels, (combined >= 0.0).astype(np.int64))
                key = (
                    min(value[name] for name in METRICS),
                    value["macro_f1"],
                    value["accuracy"],
                )
                if best is None or key > best[0]:
                    best = (
                        key, int(k), float(temperature), float(weight),
                        float(bias), value,
                    )
    return {
        "k": best[1],
        "temperature": best[2],
        "weight": best[3],
        "bias": best[4],
        "metrics": best[5],
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
    labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.long)
    dataset = OriginalRumorDataset(
        str(data_dir / "graph_final"), frame, augment=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = event_semantic_features(dataset).to(device)
    labels_device = labels.to(device)
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
            val_y, val_base, _ = infer(
                model, checkpoint, dataset, val_idx, cfg, device
            )
            test_y, test_base, _ = infer(
                model, checkpoint, dataset, test_idx, cfg, device
            )
            train_tensor = torch.tensor(train_idx, device=device)
            val_tensor = torch.tensor(val_idx, device=device)
            test_tensor = torch.tensor(test_idx, device=device)
            val_grid = retrieval_grid(
                features[train_tensor], features[val_tensor], labels_device[train_tensor]
            )
            choice = select_retrieval_fusion(val_y, val_base, val_grid)
            test_grid = retrieval_grid(
                features[train_tensor], features[test_tensor], labels_device[train_tensor]
            )
            retrieved = test_grid[(choice["k"], choice["temperature"])]
            combined = (
                test_base + choice["weight"] * retrieved + choice["bias"]
            )
            row = {
                "seed": seed,
                "fold": fold + 1,
                "selection": choice,
                "base_test": metrics(
                    test_y, (test_base >= 0.0).astype(np.int64)
                ),
                "retrieval_test": metrics(
                    test_y, (retrieved >= 0.0).astype(np.int64)
                ),
                "fused_test": metrics(
                    test_y, (combined >= 0.0).astype(np.int64)
                ),
            }
            fold_rows.append(row)
            print(json.dumps(row), flush=True)
    per_seed = {}
    for seed in seeds:
        rows = [row for row in fold_rows if row["seed"] == seed]
        fused = mean_metrics(rows, "fused_test")
        per_seed[str(seed)] = {
            "base_means": mean_metrics(rows, "base_test"),
            "retrieval_means": mean_metrics(rows, "retrieval_test"),
            "fused_means": fused,
            "passes_all": all(fused[name] > 0.88 for name in METRICS),
        }
    passing = [
        seed for seed in seeds if per_seed[str(seed)]["passes_all"]
    ]
    summary = {
        "scheme": args.scheme,
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
    print(json.dumps(per_seed, indent=2), flush=True)


if __name__ == "__main__":
    main()
