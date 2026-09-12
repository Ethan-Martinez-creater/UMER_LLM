#!/usr/bin/env python
"""Generate FINAL_EVIDENCE_REPORT.md and FINAL_EVIDENCE_SUMMARY.json."""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "project")
for _path in (HERE, PROJECT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DATASETS = ("pheme", "maweibo")
DISPLAY = {"pheme": "PHEME", "maweibo": "Ma-Weibo"}
ARMS = ("STATIC_FULL", "UTILITY_TOKEN_MATCHED", "RANDOM_TOKEN_MATCHED",
        "MS_TSR")
_REPO = os.path.abspath(os.path.join(HERE, os.pardir))
_TR = (os.path.join(_REPO, "results", "tcdscr")
      if os.path.exists(os.path.join(_REPO, "results", "tcdscr"))
      else "/data/jyz/next/llm/results/tcdscr")
ROOT = os.path.join(_TR, "final_evidence")
V3B_READER = os.path.join(_TR, "dynamic_v3_reader")
CODE = "/data/jyz/next/llm/tcdscr_code"


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def f4(x):
    return "n/a" if x is None else f"{x:+.4f}"


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--v3b-reader-root", default=V3B_READER)
    ap.add_argument("--code-root", default=CODE)
    args = ap.parse_args(argv)

    gap_a = load(os.path.join(args.root, "gap_a", "gap_a_summary.json"), {})
    gap_f = load(os.path.join(args.root, "gap_f", "gap_f_summary.json"), {})
    reader = load(os.path.join(args.root, "reader", "reader_summary.json"), {})
    manifest = load(os.path.join(args.root, "reader",
                                 "reader_sampling_manifest.json"), {})
    verify = load(os.path.join(args.root, "final_evidence_verify.json"), {})
    model_man = load(os.path.join(args.root, "reader", "diagnostics",
                                  "reader_model_manifest.json"), {})

    # closure status
    def gap_a_closed():
        return bool(gap_a) and all(
            gap_a.get("datasets", {}).get(ds, {}).get("primary")
            for ds in DATASETS)
    def gap_f_closed():
        return bool(gap_f) and all(
            gap_f.get("datasets", {}).get(ds, {}).get("status")
            for ds in DATASETS)
    def reader_closed():
        return bool(reader) and all(
            reader.get("datasets", {}).get(ds, {}).get("detection")
            for ds in DATASETS)
    def gap_c_closed():
        return reader_closed() and all(
            reader["datasets"][ds]["token_matching"]["never_overshoot"]
            for ds in DATASETS
            if "token_matching" in reader["datasets"][ds])

    summary = {
        "stage": "final_evidence_closure",
        "documentation_correction": "DONE",
        "gap_a": gap_a, "gap_f": gap_f, "reader": reader,
        "closure": {"Gap A": "CLOSED" if gap_a_closed() else "NOT CLOSED",
                    "Gap F": "CLOSED" if gap_f_closed() else "NOT CLOSED",
                    "Gap B": "CLOSED" if reader_closed() else "NOT CLOSED",
                    "Gap C": "CLOSED" if gap_c_closed() else "NOT CLOSED"},
        "verify_issues": verify.get("n_issues"),
    }

    violations = []
    if verify.get("n_issues") not in (0, None):
        violations.append("verifier issues > 0")
    if model_man.get("matches_v3b") is False:
        violations.append("model checkpoint mismatch")
    if reader and verify.get("reader", {}).get("prior_reader_overlap", 0):
        violations.append("sample contamination (prior reader events)")
    summary["protocol_violations"] = violations
    summary["final_recommendation"] = (
        "REQUIRES_RESEARCH_REVIEW" if violations
        else "READY_FOR_PAPER_CONSOLIDATION")

    lines = ["# TC-DSCR Final Evidence Closure", ""]
    lines += ["## 1. Frozen Research State", "",
              "- research freeze: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`",
              "- consolidation baseline: `bb063317ccb026c5cb0dee2da5f33747e5ea5972`",
              "- E1 causal encoder = VALID; corrected E2 static utility = VALID "
              "(all-event canonical, validation)",
              "- Dynamic V1 = NOT SUPPORTED historically, previous held-out "
              "result CONDITIONAL_HELD_OUT",
              "- Dynamic V2 / MF-TSR = NOT SUPPORTED AS FINAL METHOD",
              "- Dynamic V3 / MS-TSR = MS_TSR_COMPRESSION_ONLY",
              "- V3-B Qwen reader pilot = VALIDATION PILOT (PHEME "
              "WEAK_TRANSFER, Ma-Weibo TRANSFER_FAIL)",
              "- Contribution D = VALIDATION-PILOT SUPPORTED, FINAL HELD-OUT "
              "EVIDENCE PENDING", "",
              "This round is the FINAL MAIN EXPERIMENT ROUND for the current "
              "TC-DSCR line: frozen-model inference only, no training, no "
              "test-time tuning.", ""]
    lines += ["## 2. Documentation Correction", "",
              "The over-strong claim that identical no-candidate predictions "
              "cannot change the Macro-F1 delta sign was removed from the "
              "consolidation artifacts and replaced with: Dynamic V1 remains "
              "a supported historical negative finding based on its "
              "negligible candidate-conditioned held-out effect, corrected "
              "bootstrap intervals crossing zero, and consistent all-event "
              "validation diagnostics; however, because Macro-F1 is "
              "nonlinear, the final all-event held-out effect is not inferred "
              "from identical no-candidate predictions and must be "
              "established by Gap F.", ""]

    lines += ["## 3. Gap A — Static Held-out Evidence", "",
              "Protocol: ALL_EVENT outer test, budget 1024, frozen corrected-E2 "
              "checkpoints, 30 runs (2 datasets x 5 folds x 3 seeds).", ""]
    for ds in DATASETS:
        e = gap_a.get("datasets", {}).get(ds)
        lines += [f"### {DISPLAY[ds]}", ""]
        if not e:
            lines += ["not available", ""]
            continue
        lines += [f"Static:   {e['static']:.4f}",
                  f"Random:   {e['random']:.4f}",
                  f"Semantic: {e['semantic']:.4f}",
                  "",
                  f"Best simple baseline: {e['best_simple_baseline']}",
                  f"Static - Random:   {f4(e['comparisons']['static_minus_random']['point'])} "
                  f"CI [{f4(e['comparisons']['static_minus_random']['ci_low'])}, "
                  f"{f4(e['comparisons']['static_minus_random']['ci_high'])}]",
                  f"Static - Semantic: {f4(e['comparisons']['static_minus_semantic']['point'])} "
                  f"CI [{f4(e['comparisons']['static_minus_semantic']['ci_low'])}, "
                  f"{f4(e['comparisons']['static_minus_semantic']['ci_high'])}]",
                  f"Primary (Static - best): {f4(e['primary']['point'])} "
                  f"CI [{f4(e['primary']['ci_low'])}, {f4(e['primary']['ci_high'])}]",
                  f"Status: **{e['primary']['status']}** "
                  f"(meets historical 0.005 threshold: "
                  f"{e['primary']['meets_historical_0.005_threshold']})",
                  ""]

    lines += ["## 4. Gap F — Dynamic V1 All-Event Held-out Closure", "",
              "Protocol: ALL_EVENT outer test, frozen per-fold best_config, "
              "30 runs, no test-time search.", ""]
    for ds in DATASETS:
        e = gap_f.get("datasets", {}).get(ds)
        lines += [f"### {DISPLAY[ds]}", ""]
        if not e:
            lines += ["not available", ""]
            continue
        lines += [f"Static:  {e['static_macro_f1']:.4f}",
                  f"Dynamic: {e['dynamic_macro_f1']:.4f}",
                  f"Delta:   {f4(e['delta_macro_f1'])} "
                  f"CI [{f4(e['delta_macro_f1_ci']['ci_low'])}, "
                  f"{f4(e['delta_macro_f1_ci']['ci_high'])}]",
                  f"FlipRate delta: {f4(e['delta_flip_rate'])} "
                  f"CI [{f4(e['delta_flip_rate_ci']['ci_low'])}, "
                  f"{f4(e['delta_flip_rate_ci']['ci_high'])}]",
                  f"Status: **{e['status']}**", ""]

    lines += ["## 5. Final Reader Protocol", "",
              f"- Qwen: `{model_man.get('model', {}).get('model_path')}`",
              f"- model hash match vs V3-B: "
              f"**{model_man.get('matches_v3b')}**",
              f"- prompt: v3b-2; temperature 0; do_sample false",
              f"- sample seed: {manifest.get('sampling_seed')}",
              f"- context source seed: {manifest.get('context_source_seed')}; "
              f"alpha {manifest.get('alpha')}; budget {manifest.get('budget')}",
              "- prior-reader exclusion: every V3-B validation pilot event id",
              f"- samples: {manifest.get('n_samples')} "
              f"(300/dataset unless a deterministic shortage is recorded)",
              f"- arms: {', '.join(ARMS)}", ""]

    for name, key in (("6. Reader Results — PHEME", "pheme"),
                      ("7. Reader Results — Ma-Weibo", "maweibo")):
        lines += [f"## {name}", ""]
        e = reader.get("datasets", {}).get(key)
        if not e:
            lines += ["not available", ""]
            continue
        lines += ["| Arm | Macro-F1 | Accuracy | Rumor-F1 | Social tokens |",
                  "|---|---:|---:|---:|---:|"]
        for a in ARMS:
            d = e["detection"][a]
            tok = e["tokens"][a]["mean_social_tokens"]
            lines.append(f"| {a} | {d['macro_f1']:.4f} | {d['accuracy']:.4f} "
                         f"| {d['rumor_f1']:.4f} | {f3(tok)} |")
        ms = e["comparisons"]["ms_vs_static"]
        ut = e["comparisons"]["ms_vs_utility_tm"]
        lines += ["",
                  f"MS vs Static: delta {f4(ms['point'])} "
                  f"CI [{f4(ms['ci_low'])}, {f4(ms['ci_high'])}]",
                  f"  gate: **{e['gate']}**"
                  + (f" / **{e['transfer_label']}**"
                     if e.get("transfer_label") else ""),
                  f"  MS social-token reduction vs Static Full: "
                  f"{f3(e['tokens']['MS_TSR']['mean_reduction_vs_static'])}",
                  f"  unsupported citation rate (MS): "
                  f"{e['citation']['MS_TSR']['unsupported_citation_rate']:.4f}",
                  "",
                  f"MS vs Utility-TM: delta {f4(ut['point'])} "
                  f"CI [{f4(ut['ci_low'])}, {f4(ut['ci_high'])}]",
                  f"  interpretation: **{e['ms_vs_utility_tm_interpretation']}**",
                  "",
                  f"paired McNemar (MS vs Static) exact p = "
                  f"{e['paired_outcomes']['ms_vs_static']['mcnemar_exact_p']:.4f}",
                  f"paired McNemar (MS vs Utility-TM) exact p = "
                  f"{e['paired_outcomes']['ms_vs_utility_tm']['mcnemar_exact_p']:.4f}",
                  ""]

    lines += ["## 8. Token / Latency", ""]
    for ds in DATASETS:
        e = reader.get("datasets", {}).get(ds)
        if not e:
            continue
        c = e["cost"]
        lines += [f"- {DISPLAY[ds]}: mean latency "
                  f"{f3(c['mean_latency_sec'])} s, median "
                  f"{f3(c['median_latency_sec'])} s, total input tokens "
                  f"{c['total_input_tokens']}, total generated tokens "
                  f"{c['total_generated_tokens']}"]
        m = e["token_matching"]
        lines += [f"  - utility token gap mean {f3(m['utility_token_gap']['mean'])} "
                  f"(P90 {f3(m['utility_token_gap']['p90'])}), random token gap "
                  f"mean {f3(m['random_token_gap']['mean'])} "
                  f"(P90 {f3(m['random_token_gap']['p90'])}); "
                  f"never overshoot: {m['never_overshoot']}"]
    lines += ["", f"- monetary cost: {reader.get('datasets', {}).get('pheme', {}).get('cost', {}).get('monetary_cost_note', 'not reported')}",
              ""]

    lines += ["## 9. Grounding / Citation", ""]
    for ds in DATASETS:
        e = reader.get("datasets", {}).get(ds)
        if not e:
            continue
        for a in ARMS:
            c = e["citation"][a]
            lines.append(f"- {DISPLAY[ds]} {a}: valid {c['valid_citation_rate']:.4f}, "
                         f"unsupported {c['unsupported_citation_rate']:.4f}, "
                         f"no-citation {f3(c['no_citation_rate'])}, "
                         f"mean cited {f3(c['mean_cited_evidence_count'])}")
    lines += ["", "Only evidence-id grounding is measured; natural-language "
              "hallucination is not claimed.", ""]

    lines += ["## 10. Fold-wise Results", ""]
    for ds in DATASETS:
        e = reader.get("datasets", {}).get(ds)
        if not e or "per_fold" not in e:
            continue
        lines += [f"### {DISPLAY[ds]}", "",
                  "| fold | " + " | ".join(ARMS) + " |",
                  "|---|" + "---:|" * len(ARMS)]
        for fold, vals in sorted(e["per_fold"].items()):
            lines.append(f"| {fold} | " + " | ".join(
                f"{vals[a]['macro_f1']:.4f}" for a in ARMS) + " |")
        lines.append("")

    lines += ["## 11. Cutoff-wise Results", ""]
    for ds in DATASETS:
        e = reader.get("datasets", {}).get(ds)
        if not e or "per_cutoff" not in e:
            continue
        lines += [f"### {DISPLAY[ds]}", "",
                  "| cutoff | " + " | ".join(ARMS) + " |",
                  "|---|" + "---:|" * len(ARMS)]
        for c, vals in sorted(e["per_cutoff"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {c}m | " + " | ".join(
                f"{vals[a]['macro_f1']:.4f}" for a in ARMS) + " |")
        lines.append("")

    lines += ["## 12. Statistical Summary", "",
              f"- reader bootstrap: {reader.get('bootstrap', {}).get('iterations')} "
              f"iterations, seed {reader.get('bootstrap', {}).get('seed')}, "
              f"stratified by {reader.get('bootstrap', {}).get('stratified')}, "
              f"multiplicity {reader.get('bootstrap', {}).get('multiplicity')}",
              f"- Gap A bootstrap: {gap_a.get('bootstrap', {})}",
              f"- Gap F bootstrap: {gap_f.get('bootstrap', {})}",
              f"- verifier issues: {verify.get('n_issues')}", ""]

    lines += ["## 13. Evidence Closure Status", ""]
    for k in ("Gap A", "Gap F", "Gap B", "Gap C"):
        lines += [f"{k}:", summary["closure"][k], ""]

    lines += ["## 14. Research Implication", ""]
    lines += ["Based only on the measured results:", ""]
    for ds in DATASETS:
        ga = gap_a.get("datasets", {}).get(ds)
        gf = gap_f.get("datasets", {}).get(ds)
        rd = reader.get("datasets", {}).get(ds)
        if ga:
            lines.append(f"- {DISPLAY[ds]} Gap A: static - best baseline = "
                         f"{f4(ga['primary']['point'])} "
                         f"({ga['primary']['status']}).")
        if gf:
            lines.append(f"- {DISPLAY[ds]} Gap F: dynamic - static = "
                         f"{f4(gf['delta_macro_f1'])} ({gf['status']}).")
        if rd:
            lines.append(f"- {DISPLAY[ds]} reader: MS vs Static = "
                         f"{f4(rd['comparisons']['ms_vs_static']['point'])} "
                         f"({rd['gate']}), MS vs Utility-TM = "
                         f"{f4(rd['comparisons']['ms_vs_utility_tm']['point'])} "
                         f"({rd['ms_vs_utility_tm_interpretation']}).")
    lines += ["", "No new method is proposed here; claim adjustments are left "
              "to the review stage.", ""]

    lines += ["## 15. Final Recommendation", "",
              f"**{summary['final_recommendation']}**", ""]
    if summary["protocol_violations"]:
        lines += ["Triggered by: " + "; ".join(summary["protocol_violations"]),
                  ""]

    with open(os.path.join(args.root, "FINAL_EVIDENCE_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(args.root, "FINAL_EVIDENCE_SUMMARY.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(json.dumps({"closure": summary["closure"],
                      "recommendation": summary["final_recommendation"],
                      "verify_issues": summary["verify_issues"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
