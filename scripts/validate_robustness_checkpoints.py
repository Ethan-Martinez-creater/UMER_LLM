from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folds = []
    memory_dimensions = set()
    for fold in range(1, 6):
        path = (
            args.checkpoint_root
            / f"fold_{fold}_train_2000"
            / "best_joint_model.pt"
        )
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        state = checkpoint.get("model_state_dict", {})
        required = {"memory_features", "memory_labels"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"fold {fold} missing state keys: {missing}")
        features = state["memory_features"]
        labels = state["memory_labels"]
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError(f"fold {fold} memory tensors have invalid ranks")
        if features.size(0) != labels.numel():
            raise ValueError(f"fold {fold} memory feature/label count mismatch")
        threshold = float(checkpoint["threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"fold {fold} threshold outside [0,1]")
        memory_dimensions.add(int(features.size(1)))
        folds.append(
            {
                "fold": fold,
                "path": str(path),
                "threshold": threshold,
                "memory_shape": list(features.shape),
                "state_key_count": len(state),
                "contains_new_node_head": any(
                    name.startswith("node_cognitive_head.") for name in state
                ),
            }
        )
    if memory_dimensions != {1536}:
        raise ValueError(f"unexpected retrieval dimensions: {memory_dimensions}")
    report = {
        "checkpoint_count": len(folds),
        "all_compatible": True,
        "expected_retrieval_dimension": 1536,
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
