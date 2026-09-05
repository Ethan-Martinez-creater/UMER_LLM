import argparse
import json
from pathlib import Path

from optimization_rounds.round_024_full_consistency_stochastic_pair.compare_twofold import (
    build_report,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    candidates = []
    for fold_number in (1, 2):
        result = json.loads(
            (
                args.candidate_root
                / f"fold_{fold_number}_train_2000"
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        candidates.append({"fold": fold_number, **result["metrics"]})
    report = build_report(baseline["folds"][:2], candidates)
    report.update(
        {
            "scheme": "umer_round034_cognitive_view_distillation_v1",
            "baseline": "round_032_sam",
            "single_changed_variable": (
                "training-only LLM cognitive view included in consistency loss"
            ),
            "ordinary_training_views": 2,
            "teacher_training_views": 1,
            "teacher_view_used_for_classification_or_fusion": False,
            "teacher_view_used_at_validation_test_or_inference": False,
            "view_consistency_weight": 0.5,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
