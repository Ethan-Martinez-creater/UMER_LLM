import argparse
import json
from pathlib import Path


METRICS = ("accuracy", "weighted_f1", "macro_f1", "rumor_f1")
GATED_METRICS = METRICS[:3]


def mean_metrics(folds):
    return {
        metric: sum(fold[metric] for fold in folds) / len(folds)
        for metric in METRICS
    }


def build_report(baseline_folds, candidate_folds):
    if [fold["fold"] for fold in baseline_folds] != [1, 2]:
        raise ValueError("baseline must contain fold1 and fold2 in order")
    if [fold["fold"] for fold in candidate_folds] != [1, 2]:
        raise ValueError("candidate must contain fold1 and fold2 in order")
    baseline_mean = mean_metrics(baseline_folds)
    candidate_mean = mean_metrics(candidate_folds)
    per_fold_delta = []
    no_fold_regression = True
    for base, candidate in zip(baseline_folds, candidate_folds):
        delta = {
            metric: candidate[metric] - base[metric]
            for metric in METRICS
        }
        per_fold_delta.append({"fold": candidate["fold"], **delta})
        no_fold_regression &= all(delta[metric] >= -0.003 for metric in GATED_METRICS)

    mean_delta = {
        metric: candidate_mean[metric] - baseline_mean[metric]
        for metric in METRICS
    }
    all_three_means_improve = all(
        mean_delta[metric] > 0.0 for metric in GATED_METRICS
    )
    macro_gain_gate = mean_delta["macro_f1"] >= 0.005
    passed = no_fold_regression and all_three_means_improve and macro_gain_gate
    return {
        "scheme": "round_024_full_consistency_stochastic_pair",
        "baseline": "round_021_cross_view_consistency",
        "baseline_folds": baseline_folds,
        "candidate_folds": candidate_folds,
        "baseline_twofold_mean": baseline_mean,
        "candidate_twofold_mean": candidate_mean,
        "per_fold_delta": per_fold_delta,
        "twofold_mean_delta": mean_delta,
        "gates": {
            "per_fold_max_allowed_drop": 0.003,
            "no_gated_metric_regresses_over_limit_in_either_fold": no_fold_regression,
            "all_three_twofold_means_strictly_improve": all_three_means_improve,
            "minimum_macro_f1_twofold_gain": 0.005,
            "macro_f1_gain_gate_passed": macro_gain_gate,
        },
        "eligible_to_run_folds_3_to_5": passed,
    }


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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
