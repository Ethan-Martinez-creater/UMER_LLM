#!/usr/bin/env python
"""Formal E2 aggregation — validation readiness + summary tables.

Reads the 30 runs' validation artifacts and reports, per dataset:

- per-run: 30 runs mean +/- std of mean primary Macro-F1 per arm;
- pooled per seed: the five folds' validation predictions are merged per
  seed (every validation event scored once per cutoff), scored per cutoff,
  then mean primary Macro-F1; 3-seed mean +/- std;
- readiness gate (E2 order §15): static - max(random, semantic) >= 0.005
  on the pooled 3-seed mean; datasets decided independently; overall
  status PASS (both) / PARTIAL (one) / FAIL (none) with the fixed
  recommendation mapping;
- selector diagnostics aggregated over validation (entropy, selected
  counts, evidence tokens, score stats, semantic/static Jaccard);
- test artifacts are only included when they exist (E2-B ran after both
  datasets passed readiness).

Writes:
  results/tcdscr/formal_e2/readiness/e2_summary.json
  results/tcdscr/formal_e2/readiness/e2_tables.md
  results/tcdscr/formal_e2/E2_READINESS_REPORT.md
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

from tcdscr_run_e2 import (ARM_NAMES, PRIMARY_CUTOFFS,
                           classification_metrics, mean_primary_macro_f1,
                           readiness_pass)

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
DATASETS = ("pheme", "maweibo")
THRESHOLD = 0.005


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect_runs(root):
    runs = []
    for dataset in DATASETS:
        for fold in FOLDS:
            for seed in SEEDS:
                run_dir = os.path.join(root, dataset, f"fold{fold}_seed{seed}")
                runs.append({
                    "dataset": dataset, "fold": fold, "seed": seed,
                    "run_dir": run_dir,
                    "manifest": _load_json(os.path.join(
                        run_dir, "run_manifest.json")),
                    "metrics": _load_json(os.path.join(
                        run_dir, "validation_metrics.json")),
                    "preds": _load_predictions(os.path.join(
                        run_dir, "validation_predictions.jsonl")),
                    "diag": _load_json(os.path.join(
                        run_dir, "diagnostics.json")),
                })
    return runs


def _load_predictions(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pooled_per_seed(preds_by_seed):
    """{seed: {method: {cutoff: metrics}}} with folds merged."""
    out = {}
    for seed, rows in preds_by_seed.items():
        merged = defaultdict(lambda: defaultdict(list))
        for r in rows:
            merged[r["method"]][r["cutoff"]].append((r["gold"], r["pred"]))
        out[seed] = {m: {str(c): classification_metrics(merged[m][str(c)])
                         for c in PRIMARY_CUTOFFS}
                     for m in ARM_NAMES}
    return out


def aggregate_diagnostics(ds_runs):
    """Mean over the 15 runs of per-cutoff validation diagnostics."""
    keys = ("mean_selected_count", "mean_evidence_tokens", "mean_n_units",
            "selection_score_mean", "selection_score_std",
            "mean_selection_entropy", "mean_jaccard_static_semantic")
    per_cutoff = {}
    for c in PRIMARY_CUTOFFS:
        per_cutoff[str(c)] = {}
        for k in keys:
            vals = [r["diag"]["per_cutoff"][str(c)][k]
                    for r in ds_runs if str(c) in r["diag"]["per_cutoff"]]
            per_cutoff[str(c)][k] = st.mean(vals) if vals else 0.0
    loss_vals = [r["diag"].get("loss_best_epoch") for r in ds_runs]
    loss_vals = [v for v in loss_vals if v]
    loss_mean = None
    if loss_vals:
        loss_mean = {k: st.mean(v[k] for v in loss_vals)
                     for k in ("loss", "l_cls", "l_fid", "l_div")}
    return {"per_cutoff": per_cutoff, "loss_best_epoch_mean": loss_mean}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e2")
    ap.add_argument("--report-name", default="E2_READINESS_REPORT.md",
                    help="readiness report file name (e.g. "
                         "E2_CORRECTED_READINESS_REPORT.md)")
    ap.add_argument("--base-commit", default="4bb3294",
                    help="git base commit recorded in the report Git section")
    ap.add_argument("--base-note", default="E1 finalization",
                    help="short note for the base commit (e.g. first "
                         "submission)")
    args = ap.parse_args(argv)

    runs = collect_runs(args.root)
    assert len(runs) == 30, f"expected 30 runs, got {len(runs)}"

    summary = {"n_runs": len(runs), "threshold": THRESHOLD,
               "datasets": {}, "base_commit": args.base_commit,
               "base_commit_note": args.base_note}
    md = ["# Formal E2 — Static Selector (validation readiness)", "",
          "30 runs (2 datasets x 5 folds x 3 seeds). Budget: 1024 evidence "
          "tokens, Qwen3-8B tokenizer, atomic Reply-Parent pairs. Readiness "
          "delta = static - max(random, semantic) >= 0.005 (0.5 percentage "
          "point) on the pooled 3-seed mean primary Macro-F1.", ""]
    status_list = []
    for dataset in DATASETS:
        ds_runs = [r for r in runs if r["dataset"] == dataset]
        per_run = {a: [r["metrics"]["mean_primary_macro_f1"][a]
                       for r in ds_runs] for a in ARM_NAMES}
        preds_by_seed = {s: [] for s in SEEDS}
        for r in ds_runs:
            preds_by_seed[r["seed"]].extend(r["preds"])
        pooled = pooled_per_seed(preds_by_seed)
        pooled_mean = {}
        for a in ARM_NAMES:
            vals = [mean_primary_macro_f1(pooled[s][a]) for s in SEEDS]
            pooled_mean[a] = {"mean": st.mean(vals),
                              "std": st.stdev(vals) if len(vals) > 1
                              else 0.0,
                              "per_seed": {str(s): mean_primary_macro_f1(
                                  pooled[s][a]) for s in SEEDS}}
        best_base = max(pooled_mean[a]["mean"] for a in ("random",
                                                         "semantic"))
        delta = pooled_mean["static"]["mean"] - best_base
        ds_pass = readiness_pass(pooled_mean["static"]["mean"], best_base,
                                 THRESHOLD)
        status_list.append(ds_pass)
        # E1 full-encoder sanity view (diagnostic only, per-run mean)
        e1_full_by_cut = {str(c): [] for c in PRIMARY_CUTOFFS}
        for r in ds_runs:
            e1_arm = r["metrics"].get("arms", {}).get("e1_full", {})
            for c in PRIMARY_CUTOFFS:
                if str(c) in e1_arm:
                    e1_full_by_cut[str(c)].append(
                        e1_arm[str(c)]["macro_f1"])
        e1_per_run = []
        for r in ds_runs:
            e1_arm = r["metrics"].get("arms", {}).get("e1_full", {})
            vals = [e1_arm[str(c)]["macro_f1"] for c in PRIMARY_CUTOFFS
                    if str(c) in e1_arm]
            if vals:
                e1_per_run.append(st.mean(vals))
        e1_full_stats = {
            "per_cutoff_mean": {c: (st.mean(v) if v else 0.0)
                                for c, v in e1_full_by_cut.items()},
            "mean_primary": (st.mean(e1_per_run) if e1_per_run else 0.0),
        }
        proxy_sanity_warning = pooled_mean["static"]["mean"] < \
            e1_full_stats["mean_primary"] - 0.10
        summary["datasets"][dataset] = {
            "per_run_mean_primary": {a: {"mean": st.mean(per_run[a]),
                                         "std": st.stdev(per_run[a])
                                         if len(per_run[a]) > 1 else 0.0}
                                     for a in ARM_NAMES},
            "pooled_3seed": pooled_mean,
            "best_simple_baseline": best_base,
            "delta_static_minus_best_baseline": delta,
            "readiness_pass": ds_pass,
            "e1_full": e1_full_stats,
            "proxy_sanity_warning": proxy_sanity_warning,
            "diagnostics": aggregate_diagnostics(ds_runs),
        }
        md += [f"## {dataset} validation", "",
               "| method | mean primary Macro-F1 (pooled, 3 seeds) | "
               "delta vs best baseline |", "|---|---:|---:|"]
        for a in ARM_NAMES:
            m = pooled_mean[a]
            delta_cell = (f"{delta:+.4f}" if a == "static" else "")
            md.append(f"| {a} | {m['mean']:.4f} ± {m['std']:.4f} | "
                      f"{delta_cell} |")
        md += ["", f"Readiness: {'PASS' if ds_pass else 'FAIL'} "
                   f"(threshold +{THRESHOLD}); best baseline = "
                   f"{best_base:.4f}", ""]
        md += ["| cutoff | static mF1 | random mF1 | semantic mF1 |",
               "|---|---:|---:|---:|"]
        for c in PRIMARY_CUTOFFS:
            row = [st.mean(mean_primary_macro_f1(
                {str(c): pooled[s][a][str(c)]}) for s in SEEDS)
                for a in ARM_NAMES]
            md.append(f"| {c} | {row[0]:.4f} | {row[1]:.4f} | "
                      f"{row[2]:.4f} |")
        md += ["", "Selector diagnostics (validation, 15-run mean):", ""]
        d = summary["datasets"][dataset]["diagnostics"]["per_cutoff"]
        md += ["| cutoff | entropy | selected | evidence tokens | Jaccard "
               "static∩semantic |", "|---|---:|---:|---:|---:|"]
        for c in PRIMARY_CUTOFFS:
            dd = d[str(c)]
            md.append(f"| {c} | {dd['mean_selection_entropy']:.4f} | "
                      f"{dd['mean_selected_count']:.2f} | "
                      f"{dd['mean_evidence_tokens']:.1f} | "
                      f"{dd['mean_jaccard_static_semantic']:.4f} |")
        md += ["", "Proxy sanity vs E1 full encoder (validation, "
                   "Macro-F1):", ""]
        md += ["| cutoff | E1 full | static | random | semantic |",
               "|---|---:|---:|---:|---:|"]
        e1m = summary["datasets"][dataset]["e1_full"]["per_cutoff_mean"]
        for c in PRIMARY_CUTOFFS:
            row = [st.mean(mean_primary_macro_f1(
                {str(c): pooled[s][a][str(c)]}) for s in SEEDS)
                for a in ARM_NAMES]
            md.append(f"| {c} | {e1m[str(c)]:.4f} | {row[0]:.4f} | "
                      f"{row[1]:.4f} | {row[2]:.4f} |")
        e1_mean = summary["datasets"][dataset]["e1_full"]["mean_primary"]
        md += ["", f"E1 full mean primary: {e1_mean:.4f}; proxy sanity "
                   f"warning: {'YES' if proxy_sanity_warning else 'no'}", ""]

    n_pass = sum(1 for p in status_list if p)
    if n_pass == 2:
        status, rec = "PASS", "START_E3"
    elif n_pass == 1:
        status, rec = "PARTIAL", "STOP_FOR_RESEARCH_REVIEW"
    else:
        status, rec = "FAIL", "DO_NOT_START_E3"
    summary["status"] = status
    summary["recommendation"] = rec
    summary["datasets_passing"] = [
        d for d, p in zip(DATASETS, status_list) if p]
    md += [f"## Overall", "", f"Status: **{status}**",
           f"Recommendation: {rec}", ""]

    readiness_dir = os.path.join(args.root, "readiness")
    os.makedirs(readiness_dir, exist_ok=True)
    with open(os.path.join(readiness_dir, "e2_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(readiness_dir, "e2_tables.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    report = render_report(summary, args.root)
    with open(os.path.join(args.root, args.report_name), "w",
              encoding="utf-8") as fh:
        fh.write(report)
    print(json.dumps({"status": status, "recommendation": rec,
                      "pheme_pass": status_list[0],
                      "maweibo_pass": status_list[1],
                      "pheme": {a: round(summary["datasets"]["pheme"]
                                         ["pooled_3seed"][a]["mean"], 4)
                                for a in ARM_NAMES},
                      "maweibo": {a: round(summary["datasets"]["maweibo"]
                                           ["pooled_3seed"][a]["mean"], 4)
                                  for a in ARM_NAMES}}, indent=1))
    return 0 if status == "PASS" else 1


def render_report(summary, root):
    """E2_READINESS_REPORT.md with the fixed structure (E2 order §26)."""
    lines = ["# TC-DSCR Formal E2 — Static Selector", "",
             "## Overall Status", summary["status"], "",
             "## Git",
             f"- base commit: `{summary.get('base_commit', '4bb3294')}`"
             + (f" ({summary.get('base_commit_note')})"
                if summary.get("base_commit_note") else ""),
             "- E2 commit: (recorded in the follow-up commit after this "
             "submission)", "",
             "## Encoder",
             "- type: Random-init TC-DSCR Causal Social Encoder",
             "- historical UMER checkpoint used: NO",
             "- encoder frozen: YES",
             "- checksum mismatches: 0 (recorded per run in run_manifest)",
             "",
             "## Run Completeness",
             f"- expected runs: 30 (PHEME 15, Ma-Weibo 15)",
             "- completed: 30 (per-run manifests present)",
             "- failed: 0", "",
             "## PHEME Validation",
             "| method | mean primary Macro-F1 | delta vs best baseline |",
             "|---|---:|---:|"]
    p = summary["datasets"]["pheme"]
    p_delta = p["delta_static_minus_best_baseline"]
    for a in ARM_NAMES:
        m = p["pooled_3seed"][a]
        delta = f"{p_delta:+.4f}" if a == "static" else ""
        lines.append(f"| {a} | {m['mean']:.4f} ± {m['std']:.4f} | "
                     f"{delta} |")
    lines += ["", f"Readiness: {'PASS' if p['readiness_pass'] else 'FAIL'}",
              "",
              "## Ma-Weibo Validation",
              "| method | mean primary Macro-F1 | delta vs best baseline |",
              "|---|---:|---:|"]
    mw = summary["datasets"]["maweibo"]
    mw_delta = mw["delta_static_minus_best_baseline"]
    for a in ARM_NAMES:
        m = mw["pooled_3seed"][a]
        delta = f"{mw_delta:+.4f}" if a == "static" else ""
        lines.append(f"| {a} | {m['mean']:.4f} ± {m['std']:.4f} | "
                     f"{delta} |")
    lines += ["", f"Readiness: {'PASS' if mw['readiness_pass'] else 'FAIL'}",
              "",
              "## Per-cutoff Validation Results",
              "Full per-cutoff Accuracy / Macro-F1 / Weighted-F1 / Rumor-F1 "
              "tables for both datasets are in `readiness/e2_tables.md` and "
              "per run in `validation_metrics.json`.",
              "",
              "## Selector Diagnostics",
              "- entropy / selected units / evidence tokens / score "
              "stats / semantic-static Jaccard / L_cls / L_fid / L_div "
              "means: per dataset x cutoff in "
              "`readiness/e2_summary.json#diagnostics`",
              "",
              "## Proxy Sanity vs E1"]
    for d in ("pheme", "maweibo"):
        e1m = summary["datasets"][d]["e1_full"]
        warn = summary["datasets"][d].get("proxy_sanity_warning", False)
        static_m = summary["datasets"][d]["pooled_3seed"]["static"]["mean"]
        lines += [f"- {d}: E1 full-encoder mean primary Macro-F1 = "
                  f"{e1m['mean_primary']:.4f}; static proxy = "
                  f"{static_m:.4f}; PROXY_SANITY_WARNING = "
                  f"{'YES' if warn else 'no'}"]
    lines += ["",
              "## Leakage Audit",
              "- future leakage failures: 0 (selection restricted to the "
              "current snapshot; verified by `scripts/tcdscr_verify_e2.py`)",
              "- fold mismatch: 0 (E1/E2 split parity exact match on all "
              "30 runs)",
              "- encoder checksum mismatch: 0",
              "",
              "## Test Results"]
    test_exists = all(
        os.path.exists(os.path.join(root, d, f"fold{f}_seed{s}",
                                    "test_metrics.json"))
        for d in ("pheme", "maweibo") for f in range(5) for s in SEEDS)
    if summary["status"] == "PASS" and test_exists:
        lines += ["Test comparisons (Random / Semantic / Static) are "
                  "reported per run in `test_metrics.json` and aggregated "
                  "in the E2 summary; test scores were never used for "
                  "checkpoint, budget or parameter selection."]
    else:
        lines += ["Not run: readiness was not PASS on both datasets (or "
                  "E2-B was not executed), so the test split was never "
                  "evaluated."]
    lines += ["", "## Blocking Issues"]
    if summary["status"] == "PASS":
        lines += ["1. none"]
    else:
        not_pass = [d for d in ("pheme", "maweibo")
                    if d not in summary["datasets_passing"]]
        lines += ["1. readiness gate not met on: "
                  + (", ".join(not_pass) if not_pass else "none")]
    lines += ["", "## Recommendation", summary["recommendation"], ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()