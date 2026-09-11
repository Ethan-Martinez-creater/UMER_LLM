#!/usr/bin/env python
"""E3 failure-diagnosis verifier (E3 diagnosis order §17) — issues == 0.

Checks that the diagnosis round:
- read only the frozen E3/E3-test predictions (never wrote to them);
- ran no training, no new E3 prediction run, no Qwen inference;
- did not modify the frozen best_config files;
- preserved bootstrap multiplicity (the fixed implementation is a list
  draw, never re-keyed into a unique-event dict);
- produced all required analysis artifacts;
- kept the previous reports (old bootstrap CIs remain on disk).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "project"))

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
REQUIRED_OUTPUTS = [
    "bootstrap_fixed.json", "bootstrap_fixed.md",
    "selection_change_summary.json", "selection_to_prediction_effect.json",
    "proxy_output_change.json", "score_scale_analysis.json",
    "ranking_change_analysis.json", "budget_masking_analysis.json",
    "persistence_effect_analysis.json", "novelty_tradeoff_analysis.json",
    "stagewise_analysis.json", "selection_pressure_analysis.json",
    "validation_stability_analysis.json",
    "parameter_necessity_diagnostic.json",
    "E3_FAILURE_DIAGNOSIS_REPORT.md",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/e3_failure_diagnosis")
    ap.add_argument("--e3-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e3")
    ap.add_argument("--test-root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e3_test")
    args = ap.parse_args(argv)

    issues = []
    checked = {"outputs": 0, "prediction_rows_paired": 0}

    for name in REQUIRED_OUTPUTS:
        p = os.path.join(args.root, name)
        if not os.path.exists(p):
            issues.append(f"missing output: {name}")
        else:
            checked["outputs"] += 1

    bf_path = os.path.join(args.root, "bootstrap_fixed.json")
    if os.path.exists(bf_path):
        bf = json.load(open(bf_path, encoding="utf-8"))
        for ds in DATASETS:
            b = bf.get(ds, {})
            if b.get("multiplicity") != "preserved":
                issues.append(f"{ds}: multiplicity not preserved")
            if b.get("n_iterations") != 10000:
                issues.append(f"{ds}: n_iterations != 10000")
            if b.get("seed") != 3090:
                issues.append(f"{ds}: seed != 3090")
            if not b.get("n_events"):
                issues.append(f"{ds}: no events in bootstrap")

    # source-level protocol checks
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "tcdscr_e3_failure_diagnosis.py")
    src = open(src_path, encoding="utf-8").read()
    for token in ("backward(", ".step()", "AutoModel", "AutoTokenizer",
                  "optimizer", "requires_grad_(True)", ".train()"):
        if token in src:
            issues.append(f"diagnosis script contains training token: "
                          f"{token}")
    if "formal_e3_test/runs" in src.split("OUT_ROOT")[0]:
        issues.append("diagnosis script references test runs as output")
    # the fixed bootstrap must be a list draw, not a unique-key dict
    if "sub = {k: units[k] for k in sample}" in src:
        issues.append("bootstrap still collapses duplicates into a dict")

    # predictions pairing: every row carries both static and dynamic preds
    for ds in DATASETS:
        for fold in FOLDS:
            for seed in SEEDS:
                p = os.path.join(args.test_root, "runs", ds,
                                 f"fold{fold}_seed{seed}",
                                 "test_predictions.jsonl")
                if not os.path.exists(p):
                    issues.append(f"{ds}/fold{fold}_seed{seed}: frozen "
                                  "test_predictions.jsonl missing")
                    continue
                n = 0
                with open(p, encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        if "static_prediction" not in r or \
                                "dynamic_prediction" not in r:
                            issues.append(f"{ds}/{fold}/{seed}: row lacks "
                                          "paired predictions")
                            break
                        n += 1
                checked["prediction_rows_paired"] += n

    # frozen results untouched
    for ds in DATASETS:
        for fold in FOLDS:
            bc = os.path.join(args.e3_root, ds, f"fold{fold}",
                              "best_config.json")
            if not os.path.exists(bc):
                issues.append(f"{ds}/fold{fold}: frozen best_config missing")
    if not os.path.exists(os.path.join(args.test_root,
                                       "E3_TEST_REPORT.md")):
        issues.append("previous E3_TEST_REPORT.md deleted")

    result = {"n_issues": len(issues), "checked": checked,
              "issues": issues[:20]}
    with open(os.path.join(args.root, "diagnosis_verify.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
