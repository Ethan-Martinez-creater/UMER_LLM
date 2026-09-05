import argparse
import json
from pathlib import Path

from optimization_rounds.round_024_full_consistency_stochastic_pair.compare_twofold import (
    build_report,
)


EXPERIMENT_ID = "umer_round039_node_cognitive_distillation_w0_2_v1"


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
            "scheme": EXPERIMENT_ID,
            "baseline": "round_032_sam",
            "single_changed_variable": (
                "training-only node-grounded LLM stance/role/salience loss"
            ),
            "node_cognitive_aux_weight": 0.2,
            "target_schema_id": "umer_node_cognitive_target_v1",
            "teacher": "Qwen3-8B compact label-blind protocol v3",
            "teacher_used_at_validation_or_test": False,
            "teacher_used_at_inference": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

