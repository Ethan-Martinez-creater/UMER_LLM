import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization_rounds.round_024_full_consistency_stochastic_pair.compare_twofold import (  # noqa: E402
    build_report,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    baseline_folds = baseline["folds"][:2]
    candidate_folds = []
    for fold_number in (1, 2):
        path = args.candidate_root / f"fold_{fold_number}_train_2000" / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        candidate_folds.append({"fold": fold_number, **result["metrics"]})

    report = build_report(baseline_folds, candidate_folds)
    report["scheme"] = "round_025_layerwise_lr_decay"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
