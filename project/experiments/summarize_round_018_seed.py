"""Summarize one Round 018 seed from an existing fold-1 result and folds 2-5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "weighted_f1", "macro_f1", "rumor_f1")


def summarize(result_paths, target=0.88):
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in result_paths]
    folds = [int(record["fold"]) for record in records]
    if sorted(folds) != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected folds 1-5 exactly once, got {folds}")
    seeds = {int(record["training_seed"]) for record in records}
    schemes = {record["scheme"] for record in records}
    if len(seeds) != 1 or len(schemes) != 1:
        raise ValueError("All folds must use the same seed and scheme")
    ordered = sorted(records, key=lambda record: int(record["fold"]))
    means, stds = {}, {}
    for name in METRICS:
        values = np.asarray([record["metrics"][name] for record in ordered], dtype=float)
        means[name] = float(values.mean())
        stds[name] = float(values.std(ddof=0))
    return {
        "scheme": next(iter(schemes)),
        "training_seed": next(iter(seeds)),
        "folds": [
            {"fold": int(record["fold"]), **{name: record["metrics"][name] for name in METRICS}}
            for record in ordered
        ],
        "mean": means,
        "std": stds,
        "target_strictly_greater_than": float(target),
        "all_mean_metrics_strictly_exceed_target": all(
            means[name] > target for name in METRICS
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold1-result", required=True)
    parser.add_argument("--remaining-root", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(args.fold1_result)] + [
        Path(args.remaining_root) / f"fold_{fold}_train_{args.training_seed}" / "result.json"
        for fold in range(2, 6)
    ]
    result = summarize(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
