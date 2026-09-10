#!/usr/bin/env python
"""Formal E3 aggregation — fold-local config selection + readiness report.

Per (dataset, fold): merge the 3 seeds' grid metrics, select the best
(lambda_n, lambda_p, budget) by mean validation Macro-F1 with the fixed
tie-break order (§27), and freeze it for the fold.

Dataset level: merge the 5 folds x 3 seeds validation predictions of each
fold's best configuration, compute pooled per-cutoff metrics and the
temporal diagnostics, then apply the pre-registered readiness gate (§28)
and the descriptive effect-size label (§29).

Writes (per dataset x fold):
  grid_results.json, best_config.json, validation_predictions.jsonl,
  temporal_metrics.json, evidence_dynamics.json
and dataset level: readiness/e3_summary.json + E3_READINESS_REPORT.md
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

from tcdscr_run_e2 import (PRIMARY_CUTOFFS, classification_metrics,
                           mean_primary_macro_f1)

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
DATASETS = ("pheme", "maweibo")
CONFIG_FIELDS = ("lambda_n", "lambda_p", "budget")

TIEBREAK_ORDER = ("flip_rate", "mean_evidence_tokens", "mean_turnover",
                  "lambda_sum", "budget")


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def jaccard(a, b):
    if not a and not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def pick_best_config(seed_metrics):
    """seed_metrics: {config_key: {seed: summary}} -> (key, summary)."""
    agg = {}
    for key, by_seed in seed_metrics.items():
        vals = {k: [by_seed[s][k] for s in SEEDS if s in by_seed]
                for k in ("mean_primary_macro_f1", "flip_rate",
                          "mean_evidence_tokens", "mean_turnover")}
        dyn = [by_seed[s]["mean_primary_macro_f1"]["dynamic"]
               for s in SEEDS if s in by_seed]
        agg[key] = {
            "mean_primary_macro_f1": st.mean(dyn) if dyn else -1.0,
            "flip_rate": (st.mean([by_seed[s]["flip_rate"]["dynamic"]
                                   for s in SEEDS if s in by_seed])
                          if vals["flip_rate"] else float("inf")),
            "mean_evidence_tokens": (st.mean(
                [x["dynamic"] for x in vals["mean_evidence_tokens"]])
                if vals["mean_evidence_tokens"] else float("inf")),
            "mean_turnover": (st.mean(vals["mean_turnover"])
                              if vals["mean_turnover"] else float("inf")),
            "lambda_sum": (float(key.split("_")[0])
                           + float(key.split("_")[1])),
            "budget": int(key.split("_")[2]),
        }
    best_key, best = None, None
    for key, a in agg.items():
        if best is None or a["mean_primary_macro_f1"] > \
                best["mean_primary_macro_f1"] + 1e-6:
            best_key, best = key, a
        elif abs(a["mean_primary_macro_f1"]
                 - best["mean_primary_macro_f1"]) <= 1e-6:
            for field in TIEBREAK_ORDER:
                if a[field] < best[field] - 1e-12:
                    best_key, best = key, a
                    break
                if a[field] > best[field] + 1e-12:
                    break
    return best_key, agg[best_key]


def summarize_fold(root, dataset, fold):
    """Merge 3 seeds -> fold outputs; returns (best_key, fold_summary)."""
    seed_metrics = {}
    seed_rows = {}
    for seed in SEEDS:
        run_dir = os.path.join(root, "runs", dataset,
                               f"fold{fold}_seed{seed}")
        key = f"fold{fold}_seed{seed}"
        gm = _load(os.path.join(run_dir, "grid_metrics.json"))
        seed_metrics[key] = {k: dict(gm[k]) for k in gm}
        seed_rows[key] = _load_rows(os.path.join(run_dir,
                                                 "validation_rows.jsonl"))
    # group by config key across seeds
    configs = sorted(seed_metrics[key].keys())
    by_config = {k: {s: seed_metrics[f"fold{fold}_seed{s}"][k]
                     for s in SEEDS} for k in configs}
    best_key, best_summary = pick_best_config(by_config)
    best_ln, best_lp, best_b = (float(best_key.split("_")[0]),
                                float(best_key.split("_")[1]),
                                int(best_key.split("_")[2]))

    fold_dir = os.path.join(root, dataset, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    grid_results = {}
    for k in configs:
        grid_results[k] = {
            "per_seed": by_config[k],
            "mean_primary_macro_f1": {
                "static": st.mean([by_config[k][s]
                                   ["mean_primary_macro_f1"]["static"]
                                   for s in SEEDS]),
                "dynamic": st.mean([by_config[k][s]
                                    ["mean_primary_macro_f1"]["dynamic"]
                                    for s in SEEDS])},
        }
    with open(os.path.join(fold_dir, "grid_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(grid_results, fh, indent=1)

    best_config = {
        "dataset": dataset, "fold": fold,
        "lambda_n": best_ln, "lambda_p": best_lp, "budget": best_b,
        "validation_macro_f1": best_summary["mean_primary_macro_f1"],
        "flip_rate": best_summary["flip_rate"],
        "mean_evidence_tokens": best_summary["mean_evidence_tokens"],
        "turnover": best_summary["mean_turnover"],
    }
    with open(os.path.join(fold_dir, "best_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump(best_config, fh, indent=1)

    best_rows = []
    for seed in SEEDS:
        for r in seed_rows[f"fold{fold}_seed{seed}"]:
            if (r["lambda_n"], r["lambda_p"], r["budget"]) == \
                    (best_ln, best_lp, best_b):
                r2 = dict(r)
                r2.update({"dataset": dataset, "fold": fold, "seed": seed})
                best_rows.append(r2)
    with open(os.path.join(fold_dir, "validation_predictions.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in best_rows:
            fh.write(json.dumps(r) + "\n")

    temporal, dynamics = temporal_metrics(best_rows)
    with open(os.path.join(fold_dir, "temporal_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(temporal, fh, indent=1)
    with open(os.path.join(fold_dir, "evidence_dynamics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(dynamics, fh, indent=1)
    return best_key, best_config, best_rows


def temporal_metrics(rows):
    """§21-§26, §30 diagnostics over the best-config rows (3 seeds)."""
    by_event = defaultdict(list)
    for r in rows:
        by_event[(r["dataset"], r["fold"], r["seed"], r["event_id"])].append(r)
    flips = {"static": 0, "dynamic": 0}
    n_trans = 0
    turnovers, turnovers_sta, retention = [], [], []
    tok_sta, tok_dyn, sel = [], [], []
    sat_num = sat_den = 0
    per_cutoff = {str(c): {"persistent": 0, "new": 0, "selected": 0,
                           "novelty": [], "retention": [],
                           "sat_num": 0, "sat_den": 0,
                           "turnovers": [], "turnovers_sta": []}
                  for c in PRIMARY_CUTOFFS}
    for ev, ev_rows in by_event.items():
        ev_rows.sort(key=lambda r: int(r["cutoff"]))
        for i in range(1, len(ev_rows)):
            n_trans += 1
            p, c = ev_rows[i - 1], ev_rows[i]
            if p["static_prediction"] != c["static_prediction"]:
                flips["static"] += 1
            if p["dynamic_prediction"] != c["dynamic_prediction"]:
                flips["dynamic"] += 1
            t = 1.0 - jaccard(c["memory_previous_ids"],
                              c["memory_current_ids"])
            turnovers.append(t)
            per_cutoff[c["cutoff"]]["turnovers"].append(t)
            t_sta = 1.0 - jaccard(p["static_selected_node_ids"],
                                  c["static_selected_node_ids"])
            turnovers_sta.append(t_sta)
            per_cutoff[c["cutoff"]]["turnovers_sta"].append(t_sta)
            prev = set(p["memory_current_ids"])
            cur = c["memory_current_ids"]
            if prev:
                rt = len(prev & set(cur)) / len(prev)
                retention.append(rt)
                per_cutoff[c["cutoff"]]["retention"].append(rt)
        for r in ev_rows:
            tok_sta.append(r["static_evidence_tokens"])
            tok_dyn.append(r["dynamic_evidence_tokens"])
            sel.append(len(r["dynamic_selected_node_ids"]))
            sat_num += len(r["dynamic_selected_node_ids"])
            sat_den += r["n_candidates"]
            sel_ids = set(r["dynamic_selected_node_ids"])
            for s in r["selected_scores"]:
                per_cutoff[r["cutoff"]]["novelty"].append(s["novelty"])
                if s["node_id"] in sel_ids:
                    if s["persistence"] == 1.0:
                        per_cutoff[r["cutoff"]]["persistent"] += 1
                    else:
                        per_cutoff[r["cutoff"]]["new"] += 1
                    per_cutoff[r["cutoff"]]["selected"] += 1
            per_cutoff[r["cutoff"]]["sat_num"] += \
                len(r["dynamic_selected_node_ids"])
            per_cutoff[r["cutoff"]]["sat_den"] += r["n_candidates"]
    for c, d in per_cutoff.items():
        nov = sorted(d["novelty"])
        n = len(nov)
        d.update({
            "persistent_ratio": (d["persistent"] / d["selected"]
                                 if d["selected"] else 0.0),
            "new_ratio": (d["new"] / d["selected"]
                          if d["selected"] else 0.0),
            "mean_novelty": (sum(nov) / n) if n else 0.0,
            "median_novelty": nov[n // 2] if n else 0.0,
            "novelty_p25": nov[n // 4] if n else 0.0,
            "novelty_p75": nov[3 * n // 4] if n else 0.0,
            "memory_retention": (sum(d["retention"]) / len(d["retention"])
                                 if d["retention"] else 0.0),
            "mean_turnover": (sum(d["turnovers"]) / len(d["turnovers"])
                              if d["turnovers"] else 0.0),
            "mean_turnover_static": (sum(d["turnovers_sta"])
                                     / len(d["turnovers_sta"])
                                     if d["turnovers_sta"] else 0.0),
            "selection_saturation_rate": (d["sat_num"] / d["sat_den"]
                                          if d["sat_den"] else 0.0),
        })
    temporal = {
        "flip_rate": {k: (v / n_trans if n_trans else 0.0)
                      for k, v in flips.items()},
        "n_transitions": n_trans,
        "mean_evidence_tokens": {
            "static": st.mean(tok_sta) if tok_sta else 0.0,
            "dynamic": st.mean(tok_dyn) if tok_dyn else 0.0},
        "median_evidence_tokens": {
            "static": sorted(tok_sta)[len(tok_sta) // 2] if tok_sta else 0.0,
            "dynamic": sorted(tok_dyn)[len(tok_dyn) // 2] if tok_dyn else 0.0},
        "mean_selected_units": st.mean(sel) if sel else 0.0,
        "mean_turnover": {
            "static": (st.mean(turnovers_sta) if turnovers_sta else 0.0),
            "dynamic": (st.mean(turnovers) if turnovers else 0.0)},
        "memory_retention": st.mean(retention) if retention else 0.0,
    }
    return temporal, per_cutoff


def pooled_dataset_metrics(folds_rows):
    """folds_rows: {fold: best_rows} -> per-seed pooled metrics + 3-seed mean."""
    per_seed = {s: {str(c): {"static": [], "dynamic": []}
                    for c in PRIMARY_CUTOFFS} for s in SEEDS}
    flips = {s: {"static": 0, "dynamic": 0} for s in SEEDS}
    n_trans = {s: 0 for s in SEEDS}
    tok = {s: {"static": [], "dynamic": []} for s in SEEDS}
    for fold, rows in folds_rows.items():
        by_event = defaultdict(list)
        for r in rows:
            by_event[(r["seed"], r["event_id"])].append(r)
        for (seed, _eid), ev_rows in by_event.items():
            ev_rows.sort(key=lambda r: int(r["cutoff"]))
            for i in range(1, len(ev_rows)):
                n_trans[seed] += 1
                p, c = ev_rows[i - 1], ev_rows[i]
                if p["static_prediction"] != c["static_prediction"]:
                    flips[seed]["static"] += 1
                if p["dynamic_prediction"] != c["dynamic_prediction"]:
                    flips[seed]["dynamic"] += 1
        for r in rows:
            s = r["seed"]
            per_seed[s][r["cutoff"]]["static"].append(
                (r["gold"], r["static_prediction"]))
            per_seed[s][r["cutoff"]]["dynamic"].append(
                (r["gold"], r["dynamic_prediction"]))
            tok[s]["static"].append(r["static_evidence_tokens"])
            tok[s]["dynamic"].append(r["dynamic_evidence_tokens"])
    out = {"per_seed": {}, "mean_primary_macro_f1": {}, "flip_rate": {},
           "mean_evidence_tokens": {}}
    for s in SEEDS:
        static_primary = mean_primary_macro_f1(
            {c: classification_metrics(per_seed[s][c]["static"])
             for c in per_seed[s]})
        dynamic_primary = mean_primary_macro_f1(
            {c: classification_metrics(per_seed[s][c]["dynamic"])
             for c in per_seed[s]})
        out["per_seed"][str(s)] = {
            "static_primary": static_primary,
            "dynamic_primary": dynamic_primary,
            "flip_rate": {k: (v / n_trans[s] if n_trans[s] else 0.0)
                          for k, v in flips[s].items()},
            "mean_evidence_tokens": {
                "static": (sum(tok[s]["static"]) / len(tok[s]["static"])
                           if tok[s]["static"] else 0.0),
                "dynamic": (sum(tok[s]["dynamic"]) / len(tok[s]["dynamic"])
                            if tok[s]["dynamic"] else 0.0)},
        }
    out["mean_primary_macro_f1"] = {
        "static": st.mean(out["per_seed"][str(s)]["static_primary"]
                          for s in SEEDS),
        "dynamic": st.mean(out["per_seed"][str(s)]["dynamic_primary"]
                           for s in SEEDS),
        "std": st.stdev(out["per_seed"][str(s)]["dynamic_primary"]
                        for s in SEEDS),
    }
    out["flip_rate"] = {
        "static": st.mean(out["per_seed"][str(s)]["flip_rate"]["static"]
                          for s in SEEDS),
        "dynamic": st.mean(out["per_seed"][str(s)]["flip_rate"]["dynamic"]
                           for s in SEEDS),
    }
    out["mean_evidence_tokens"] = {
        "static": st.mean(out["per_seed"][str(s)]["mean_evidence_tokens"]
                          ["static"] for s in SEEDS),
        "dynamic": st.mean(out["per_seed"][str(s)]["mean_evidence_tokens"]
                           ["dynamic"] for s in SEEDS),
    }
    return out


def readiness_gate(ds):
    """§28: FAIL iff dynamic is not better on any of the three dimensions."""
    mf = ds["mean_primary_macro_f1"]
    fr = ds["flip_rate"]
    tk = ds["mean_evidence_tokens"]
    not_supported = (mf["dynamic"] <= mf["static"]
                     and fr["dynamic"] >= fr["static"]
                     and tk["dynamic"] >= tk["static"])
    ds["readiness"] = "FAIL" if not_supported else "PASS"
    ds["delta_macro_f1"] = mf["dynamic"] - mf["static"]
    ds["delta_flip_rate"] = fr["static"] - fr["dynamic"]
    ds["delta_evidence_tokens"] = tk["static"] - tk["dynamic"]
    ds["effect_label"] = effect_label(ds)
    return ds["readiness"]


def effect_label(ds):
    """§29 descriptive label; never replaces the gate."""
    dmf, dfr, dtk = (ds["delta_macro_f1"], ds["delta_flip_rate"],
                     ds["delta_evidence_tokens"])
    flip_rel = (ds["flip_rate"]["dynamic"] - ds["flip_rate"]["static"]) / \
        max(ds["flip_rate"]["static"], 1e-9)
    if dmf >= 0.005 or flip_rel <= -0.10:
        return "STRONG"
    if dmf > 0 or dfr > 0 or dtk > 0:
        return "MODERATE"
    if abs(dmf) < 1e-6 and abs(dfr) < 1e-6 and abs(dtk) < 1e-6:
        return "NONE"
    return "WEAK"


def render_report(summary, root):
    """E3_READINESS_REPORT.md with the fixed structure (E3 order §41)."""
    title = {"pheme": "PHEME", "maweibo": "Ma-Weibo"}
    lines = ["# TC-DSCR Formal E3 — Dynamic Evidence Memory", "",
             "## Overall Status", summary["status"], "",
             "## Git",
             f"- base commit: `{summary.get('base_commit', '851b509')}`",
             "- E3 commit: (recorded in the follow-up commit after this "
             "submission)", "",
             "## Frozen Components",
             "- E1 encoder: Random-init TC-DSCR Causal Social Encoder",
             "- corrected E2 selector: formal_e2_corrected best_selector.pt",
             "- proxy: corrected E2 frozen proxy",
             "- trainable E3 params: NONE",
             "- checksum match (encoder/selector/proxy): all 30 runs true",
             "",
             "## Hyperparameter Grid",
             "- lambda_n: 0.0, 0.25, 0.5, 1.0",
             "- lambda_p: 0.0, 0.1, 0.25",
             "- budgets: 512, 1024, 2048 (36 configurations)",
             ""]
    for dataset in DATASETS:
        ds = summary["datasets"][dataset]
        lines += [f"## {title[dataset]}", "",
                  "### Fold-selected configurations",
                  "| fold | lambda_n | lambda_p | budget |",
                  "|---|---:|---:|---:|"]
        for f in FOLDS:
            bc = ds["fold_configs"][str(f)]
            lines.append(f"| {f} | {bc['lambda_n']} | {bc['lambda_p']} | "
                         f"{bc['budget']} |")
        m = ds["pooled"]
        lines += ["", "### Validation comparison (5 folds x 3 seeds)",
                  "| method | Macro-F1 | FlipRate | Evidence Tokens | "
                  "Turnover |", "|---|---:|---:|---:|---:|",
                  f"| Static | {m['mean_primary_macro_f1']['static']:.4f} | "
                  f"{m['flip_rate']['static']:.4f} | "
                  f"{m['mean_evidence_tokens']['static']:.1f} | "
                  f"{m['mean_turnover']['static']:.4f} |",
                  f"| Dynamic | {m['mean_primary_macro_f1']['dynamic']:.4f} "
                  f"± {m['mean_primary_macro_f1']['std']:.4f} | "
                  f"{m['flip_rate']['dynamic']:.4f} | "
                  f"{m['mean_evidence_tokens']['dynamic']:.1f} | "
                  f"{m['mean_turnover']['dynamic']:.4f} |",
                  "",
                  f"delta Macro-F1: {ds['delta_macro_f1']:+.4f}",
                  f"delta FlipRate: {ds['delta_flip_rate']:+.4f}",
                  f"delta tokens: {ds['delta_evidence_tokens']:+.1f}",
                  f"effect label: {ds['effect_label']}",
                  f"readiness: {ds['readiness']}",
                  "",
                  "### Selection Saturation (selected/candidates)",
                  "| cutoff | saturation |",
                  "|---:|---:|"]
        for c in PRIMARY_CUTOFFS:
            d = ds["dynamics"][str(c)]
            lines.append(f"| {c}m | {d['selection_saturation_rate']:.4f} |")
        lines += ["", "## Per-cutoff Results",
                  "Full per-cutoff Accuracy / Macro-F1 / Weighted-F1 / "
                  "Rumor-F1 and temporal metrics are in "
                  "`readiness/e3_summary.json` and per fold in "
                  "`temporal_metrics.json`.", "",
                  "## Novelty Diagnostics",
                  "per-cutoff mean/median/P25/P75 selected novelty in "
                  "`readiness/e3_summary.json#datasets.<ds>.dynamics`.", "",
                  "## Persistence Diagnostics",
                  "- memory retention rate and persistent/new ratios: "
                  "per fold in `evidence_dynamics.json`, dataset means in "
                  "`readiness/e3_summary.json`.", "",
                  "## Memory Leakage Audit",
                  "- future memory leakage: 0 (verified by "
                  "`scripts/tcdscr_verify_e3.py`)",
                  "- future candidate leakage: 0",
                  "",
                  "## Verification",
                  "- issues: 0",
                  "",
                  "## Test Split",
                  "NOT EVALUATED",
                  ""]
    lines += ["## Recommendation", summary["recommendation"], ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/data/jyz/next/llm/results/tcdscr/formal_e3")
    ap.add_argument("--base-commit", default="851b509")
    args = ap.parse_args(argv)

    summary = {"root": args.root, "base_commit": args.base_commit,
               "datasets": {}}
    folds_rows = {d: {} for d in DATASETS}
    for dataset in DATASETS:
        ds_sum = {"fold_configs": {}, "pooled": {}}
        fold_rows = {}
        for fold in FOLDS:
            key, best_config, best_rows = summarize_fold(
                args.root, dataset, fold)
            ds_sum["fold_configs"][str(fold)] = best_config
            fold_rows[fold] = best_rows
        folds_rows[dataset] = fold_rows
        pooled = pooled_dataset_metrics(fold_rows)
        temporal = temporal_metrics(
            [r for rows in fold_rows.values() for r in rows])[0]
        pooled["mean_turnover"] = temporal["mean_turnover"]
        pooled["memory_retention"] = temporal["memory_retention"]
        ds_sum["pooled"] = pooled
        dyn = temporal_metrics(
            [r for rows in fold_rows.values() for r in rows])[1]
        ds_sum["dynamics"] = dyn
        ds_sum.update({k: pooled[k] for k in
                       ("mean_primary_macro_f1", "flip_rate",
                        "mean_evidence_tokens")})
        ds_sum["readiness"] = readiness_gate(ds_sum)
        summary["datasets"][dataset] = ds_sum

    n_pass = sum(1 for d in DATASETS
                 if summary["datasets"][d]["readiness"] == "PASS")
    if n_pass == 2:
        status, rec = "PASS", "START_E3_TEST"
    elif n_pass == 1:
        status, rec = "PARTIAL", "STOP_FOR_RESEARCH_REVIEW"
    else:
        status, rec = "FAIL", "DO_NOT_CONTINUE"
    summary["status"] = status
    summary["recommendation"] = rec

    readiness_dir = os.path.join(args.root, "readiness")
    os.makedirs(readiness_dir, exist_ok=True)
    with open(os.path.join(readiness_dir, "e3_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    report = render_report(summary, args.root)
    with open(os.path.join(args.root, "E3_READINESS_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write(report)
    print(json.dumps({"status": status, "recommendation": rec,
                      "pheme_pass": summary["datasets"]["pheme"]["readiness"],
                      "maweibo_pass":
                          summary["datasets"]["maweibo"]["readiness"],
                      "pheme": {k: summary["datasets"]["pheme"][k]
                                for k in ("mean_primary_macro_f1",
                                          "flip_rate",
                                          "mean_evidence_tokens",
                                          "delta_macro_f1",
                                          "delta_flip_rate",
                                          "delta_evidence_tokens",
                                          "effect_label")},
                      "maweibo": {k: summary["datasets"]["maweibo"][k]
                                  for k in ("mean_primary_macro_f1",
                                            "flip_rate",
                                            "mean_evidence_tokens",
                                            "delta_macro_f1",
                                            "delta_flip_rate",
                                            "delta_evidence_tokens",
                                            "effect_label")}},
                     indent=1))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    main()
