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
        result_path = (
            args.candidate_root
            / f"fold_{fold_number}_train_2000"
            / "result.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        candidates.append({"fold": fold_number, **result["metrics"]})
    report = build_report(baseline["folds"][:2], candidates)
    report.update(
        {
            "scheme": "umer_round033_llm_cognitive_distillation_w0_5_v1",
            "baseline": "round_032_sam",
            "single_changed_variable": "training-only LLM cognitive auxiliary loss",
            "cognitive_aux_weight": 0.5,
            "target_schema_id": "umer_cognitive_event_target_v1",
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
