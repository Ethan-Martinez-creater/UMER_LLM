import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization_rounds.round_024_full_consistency_stochastic_pair.compare_twofold import (  # noqa: E402
    build_report,
)


def folds(acc1=0.90, acc2=0.88, weighted1=0.90, weighted2=0.88,
          macro1=0.89, macro2=0.87, rumor1=0.86, rumor2=0.84):
    return [
        {"fold": 1, "accuracy": acc1, "weighted_f1": weighted1,
         "macro_f1": macro1, "rumor_f1": rumor1},
        {"fold": 2, "accuracy": acc2, "weighted_f1": weighted2,
         "macro_f1": macro2, "rumor_f1": rumor2},
    ]


def test_all_gates_must_pass_before_expansion():
    baseline = folds()
    candidate = folds(
        acc1=0.904, acc2=0.884,
        weighted1=0.904, weighted2=0.884,
        macro1=0.896, macro2=0.876,
        rumor1=0.861, rumor2=0.841,
    )
    report = build_report(baseline, candidate)
    assert report["eligible_to_run_folds_3_to_5"] is True
    assert report["twofold_mean_delta"]["macro_f1"] == pytest.approx(0.006)


def test_one_fold_regression_blocks_expansion_even_if_mean_improves():
    baseline = folds()
    candidate = folds(
        acc1=0.896, acc2=0.894,
        weighted1=0.896, weighted2=0.894,
        macro1=0.886, macro2=0.886,
    )
    report = build_report(baseline, candidate)
    assert report["gates"][
        "no_gated_metric_regresses_over_limit_in_either_fold"
    ] is False
    assert report["eligible_to_run_folds_3_to_5"] is False


def test_wrong_fold_order_is_rejected():
    with pytest.raises(ValueError):
        build_report(list(reversed(folds())), folds())
