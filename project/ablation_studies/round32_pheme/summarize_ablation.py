from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("accuracy", "weighted_f1", "macro_f1", "rumor_f1")


def load_mean(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "mean" in payload:
        return {name: float(payload["mean"][name]) for name in METRICS}
    return {name: float(payload[name]["mean"]) for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--no-sam-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = load_mean(args.full_summary)
    rows = {"full": {"mean": full, "source": str(args.full_summary)}}
    sources = {
        "no_graph": args.ablation_root / "no_graph" / "summary.json",
        "no_text": args.ablation_root / "no_text" / "summary.json",
        "no_memory": args.ablation_root / "no_memory" / "summary.json",
        "no_consistency": (
            args.ablation_root / "no_consistency" / "summary.json"
        ),
        "no_sam": args.no_sam_summary,
    }
    for name, source in sources.items():
        mean = load_mean(source)
        rows[name] = {
            "mean": mean,
            "delta_vs_full": {
                metric: mean[metric] - full[metric] for metric in METRICS
            },
            "source": str(source),
        }
    payload = {
        "dataset": "PHEME",
        "partition_seed": 3090,
        "training_seed": 2000,
        "folds": 5,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
