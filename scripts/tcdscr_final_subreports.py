#!/usr/bin/env python
"""Per-gap sub-reports for the Final Evidence Closure stage."""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "project")
for _p in (HERE, PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATASETS = ("pheme", "maweibo")
DISPLAY = {"pheme": "PHEME", "maweibo": "Ma-Weibo"}
ARMS = ("STATIC_FULL", "UTILITY_TOKEN_MATCHED", "RANDOM_TOKEN_MATCHED",
        "MS_TSR")
_REPO = os.path.abspath(os.path.join(HERE, os.pardir))
_TR = (os.path.join(_REPO, "results", "tcdscr")
      if os.path.exists(os.path.join(_REPO, "results", "tcdscr"))
      else "/data/jyz/next/llm/results/tcdscr")
ROOT = os.path.join(_TR, "final_evidence")


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def w(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args(argv)
    gap_a = load(os.path.join(args.root, "gap_a", "gap_a_summary.json"), {})
    gap_f = load(os.path.join(args.root, "gap_f", "gap_f_summary.json"), {})
    reader = load(os.path.join(args.root, "reader", "reader_summary.json"), {})
    manifest = load(os.path.join(args.root, "reader",
                                 "reader_sampling_manifest.json"), {})
    mm = load(os.path.join(args.root, "reader", "diagnostics",
                           "reader_model_manifest.json"), {})

    a = ["# Gap A — Static Utility All-Event Held-out Report", "",
         "Protocol: ALL_EVENT outer test split, budget 1024, frozen "
         "corrected-E2 checkpoints, 30 runs (2 datasets x 5 folds x 3 "
         "seeds), paired event bootstrap 10000 / seed 4096, fold-stratified.",
         ""]
    for ds in DATASETS:
        e = gap_a.get("datasets", {}).get(ds)
        a += [f"## {DISPLAY[ds]}", ""]
        if not e:
            a += ["not available", ""]
            continue
        a += [f"- Static: {e['static']:.5f}",
              f"- Random: {e['random']:.5f}",
              f"- Semantic: {e['semantic']:.5f}",
              f"- best simple baseline: {e['best_simple_baseline']}",
              f"- Static - Random: {e['comparisons']['static_minus_random']['point']:+.5f} "
              f"CI [{e['comparisons']['static_minus_random']['ci_low']:+.5f}, "
              f"{e['comparisons']['static_minus_random']['ci_high']:+.5f}]",
              f"- Static - Semantic: {e['comparisons']['static_minus_semantic']['point']:+.5f} "
              f"CI [{e['comparisons']['static_minus_semantic']['ci_low']:+.5f}, "
              f"{e['comparisons']['static_minus_semantic']['ci_high']:+.5f}]",
              f"- **Status: {e['primary']['status']}** "
              f"(meets historical 0.005 threshold: "
              f"{e['primary']['meets_historical_0.005_threshold']})", ""]
        a += ["| cutoff | Static | Random | Semantic |", "|---|---:|---:|---:|"]
        for c, vals in sorted(e["per_cutoff"].items(), key=lambda kv: int(kv[0])):
            a.append(f"| {c}m | {vals['static']:.4f} | {vals['random']:.4f} "
                     f"| {vals['semantic']:.4f} |")
        a += ["", "| fold | Static | Random | Semantic |",
              "|---|---:|---:|---:|"]
        for f, vals in sorted(e["per_fold"].items()):
            a.append(f"| {f} | {vals['static']:.4f} | {vals['random']:.4f} "
                     f"| {vals['semantic']:.4f} |")
        a.append("")
    w(os.path.join(args.root, "gap_a", "GAP_A_HELDOUT_REPORT.md"), a)

    f_lines = ["# Gap F — Dynamic V1 All-Event Held-out Report", "",
               "Protocol: ALL_EVENT outer test split, frozen per-fold "
               "best_config (no test-time search), 30 runs, paired event "
               "bootstrap 10000 / seed 4096, fold-stratified.", ""]
    for ds in DATASETS:
        e = gap_f.get("datasets", {}).get(ds)
        f_lines += [f"## {DISPLAY[ds]}", ""]
        if not e:
            f_lines += ["not available", ""]
            continue
        f_lines += [f"- Static: {e['static_macro_f1']:.5f}",
                    f"- Dynamic: {e['dynamic_macro_f1']:.5f}",
                    f"- Delta: {e['delta_macro_f1']:+.6f} "
                    f"CI [{e['delta_macro_f1_ci']['ci_low']:+.6f}, "
                    f"{e['delta_macro_f1_ci']['ci_high']:+.6f}]",
                    f"- FlipRate delta: {e['delta_flip_rate']:+.6f} "
                    f"CI [{e['delta_flip_rate_ci']['ci_low']:+.6f}, "
                    f"{e['delta_flip_rate_ci']['ci_high']:+.6f}]",
                    f"- **Status: {e['status']}**", ""]
        f_lines += ["| cutoff | Static | Dynamic |", "|---|---:|---:|"]
        for c, vals in sorted(e["per_cutoff"].items(), key=lambda kv: int(kv[0])):
            f_lines.append(f"| {c}m | {vals['static']:.4f} | "
                           f"{vals['dynamic']:.4f} |")
        f_lines += ["", "| fold | Static | Dynamic |", "|---|---:|---:|"]
        for fold, vals in sorted(e["per_fold"].items()):
            f_lines.append(f"| {fold} | {vals['static']:.4f} | "
                           f"{vals['dynamic']:.4f} |")
        f_lines.append("")
    w(os.path.join(args.root, "gap_f", "GAP_F_ALL_EVENT_REPORT.md"),
      f_lines)

    r = ["# Final Fold-Local Frozen-Qwen Reader Report", "",
         f"- model: `{mm.get('model', {}).get('model_path')}` "
         f"(hash match vs V3-B: {mm.get('matches_v3b')})",
         "- prompt: v3b-2, temperature 0, do_sample false",
         f"- sampling seed {manifest.get('sampling_seed')}, context source "
         f"seed {manifest.get('context_source_seed')}, alpha "
         f"{manifest.get('alpha')}, budget {manifest.get('budget')}",
         f"- samples: {manifest.get('n_samples')} (300 per dataset), four "
         f"arms: {', '.join(ARMS)}", ""]
    for ds in DATASETS:
        e = reader.get("datasets", {}).get(ds)
        r += [f"## {DISPLAY[ds]}", ""]
        if not e:
            r += ["not available", ""]
            continue
        r += ["| Arm | Macro-F1 | Accuracy | Rumor-F1 | Social tokens |",
              "|---|---:|---:|---:|---:|"]
        for arm in ARMS:
            d = e["detection"][arm]
            r.append(f"| {arm} | {d['macro_f1']:.4f} | {d['accuracy']:.4f} "
                     f"| {d['rumor_f1']:.4f} | "
                     f"{e['tokens'][arm]['mean_social_tokens']:.2f} |")
        ms = e["comparisons"]["ms_vs_static"]
        ut = e["comparisons"]["ms_vs_utility_tm"]
        r += ["",
              f"- MS vs Static: {ms['point']:+.5f} CI [{ms['ci_low']:+.5f}, "
              f"{ms['ci_high']:+.5f}] -> **{e['gate']}**"
              + (f" / **{e['transfer_label']}**"
                 if e.get("transfer_label") else ""),
              f"- MS vs Utility-TM: {ut['point']:+.5f} CI "
              f"[{ut['ci_low']:+.5f}, {ut['ci_high']:+.5f}] -> "
              f"**{e['ms_vs_utility_tm_interpretation']}**",
              f"- MS social-token reduction vs Static Full: "
              f"{e['tokens']['MS_TSR']['mean_reduction_vs_static']:.4f}",
              f"- unsupported citation rate (MS): "
              f"{e['citation']['MS_TSR']['unsupported_citation_rate']:.4f}",
              f"- token matching never overshoots: "
              f"{e['token_matching']['never_overshoot']}",
              f"- exact McNemar p (MS vs Static): "
              f"{e['paired_outcomes']['ms_vs_static']['mcnemar_exact_p']:.4f}",
              ""]
        r += ["| cutoff | " + " | ".join(ARMS) + " |",
              "|---|" + "---:|" * len(ARMS)]
        for c, vals in sorted(e["per_cutoff"].items(), key=lambda kv: int(kv[0])):
            r.append(f"| {c}m | " + " | ".join(
                f"{vals[a]['macro_f1']:.4f}" for a in ARMS) + " |")
        r.append("")
    w(os.path.join(args.root, "reader", "FINAL_READER_REPORT.md"), r)
    print("sub-reports written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
