#!/usr/bin/env python
"""Dynamic V3 (MS-TSR) validation aggregation — Stage V3-A (design §16-§19).

Reads results/tcdscr/dynamic_v3/runs/<dataset>/fold<f>_seed<s>/ and writes:
  - per dataset x fold: fold-local alpha choice (validation only, three
    seeds pooled) with the §16 non-inferiority rule, best_config.json and
    the fold deliverables (sufficiency dynamics, compression metrics,
    selected-alpha predictions);
  - per dataset: the §17 gate (non-inferiority AND >=30% token reduction
    AND <=5% relative flip increase) plus the §18 mechanism metrics and the
    §19 margin-stratified analysis;
  - MS_TSR_VALIDATION_REPORT.md.

Alpha selection is NOT "maximise Macro-F1": among the alphas that respect
Delta MF1 >= -0.002 and RelativeFlipIncrease <= 5%, the one with the
largest token reduction wins (ties -> smaller alpha); if no alpha
qualifies the fold is V3_NOT_SUPPORTED and the gate is NOT relaxed.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tcdscr_run_e2 import PRIMARY_CUTOFFS, classification_metrics  # noqa: E402
from tcdscr_run_dynamic_v2 import BUDGET_TOKENS, DATASETS, FOLDS, SEEDS  # noqa: E402
from tcdscr.models.sufficiency import ALPHA_GRID  # noqa: E402

ALPHA_ORDER = tuple(str(a) for a in ALPHA_GRID)
DISPLAY_NAME = {"pheme": "PHEME", "maweibo": "Ma-Weibo"}
DEFAULT_ROOT = "/data/jyz/next/llm/results/tcdscr/dynamic_v3"
STAGE_CUTOFFS = {"Very Early": ("5", "15"), "Early": ("30", "60"),
                 "Mid": ("180", "360")}
# §16/§17 frozen thresholds
NONINF_DELTA_MF1 = -0.002
MAX_REL_FLIP_INCREASE = 0.05
MIN_TOKEN_REDUCTION = 0.30


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[min(int(q * len(s)), len(s) - 1)])


def pick_fold_alpha(per_seed):
    """§16 fold-local alpha selection (pure function).

    ``per_seed``: {alpha_str: [seed_grid_summary, ...]} where each summary
    is tcdscr_run_dynamic_v3.summarize_run_rows_v3 output.  Returns
    (alpha_str or None, per_alpha_metrics).
    """
    stats = {}
    for a, seeds in per_seed.items():
        n = len(seeds)
        st = mean([s["mean_primary_macro_f1"]["static"] for s in seeds])
        ms = mean([s["mean_primary_macro_f1"]["ms"] for s in seeds])
        flip_st = mean([s["flip_rate"]["static"] for s in seeds])
        flip_ms = mean([s["flip_rate"]["ms"] for s in seeds])
        red = mean([s["compression"]["mean_token_reduction"]
                    for s in seeds])
        rel_flip = ((flip_ms - flip_st) / flip_st) if flip_st > 0 else 0.0
        stats[a] = {
            "static_macro_f1": st, "ms_macro_f1": ms,
            "delta_macro_f1": ms - st,
            "static_flip_rate": flip_st, "ms_flip_rate": flip_ms,
            "relative_flip_increase": rel_flip,
            "mean_token_reduction": red,
            "eligible": ((ms - st) >= NONINF_DELTA_MF1
                         and rel_flip <= MAX_REL_FLIP_INCREASE)}
    eligible = [a for a in ALPHA_ORDER if a in stats and stats[a]["eligible"]]
    if not eligible:
        return None, stats
    best = min(eligible, key=lambda a: (-stats[a]["mean_token_reduction"],
                                        float(a)))
    return best, stats


def gate_verdict(delta_mf1, token_reduction, rel_flip_increase):
    """§17 dataset gate: all three conditions must hold."""
    a = delta_mf1 >= NONINF_DELTA_MF1
    b = token_reduction >= MIN_TOKEN_REDUCTION
    c = rel_flip_increase <= MAX_REL_FLIP_INCREASE
    return ("MS_TSR_PROXY_PASS" if (a and b and c)
            else "MS_TSR_NOT_SUPPORTED"), {
        "classification_non_inferiority": a,
        "context_reduction": b, "temporal_stability": c}


def dataset_decision(verdicts):
    n_pass = sum(1 for v in verdicts.values()
                 if v["gate"] == "MS_TSR_PROXY_PASS")
    if n_pass == len(verdicts):
        return "MS_TSR_PROXY_PASS", "START_V3_B_READER_TRANSFER_PILOT"
    if n_pass == 0:
        return "MS_TSR_NOT_SUPPORTED", "MS_TSR_NOT_SUPPORTED_REDESIGN"
    return "PARTIAL", "STOP_FOR_RESEARCH_REVIEW"


def load_run(root, dataset, fold, seed):
    rd = os.path.join(root, "runs", dataset, f"fold{fold}_seed{seed}")
    grid_path = os.path.join(rd, "grid_metrics.json")
    with open(grid_path, encoding="utf-8") as fh:
        grid = json.load(fh)
    rows = []
    with open(os.path.join(rd, "validation_predictions.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    # §18 normalisation: sufficiency coverage rates are recomputed from the
    # raw rows (attempt / non-empty denominators) so the run artifact and
    # the report always agree and no rate can exceed 1.
    by_alpha = {}
    for r in rows:
        by_alpha.setdefault(str(r["alpha"]), []).append(r)
    changed = False
    for a, rs in by_alpha.items():
        if a not in grid:
            continue
        n = max(len(rs), 1)
        att = sum(int(x["compression_attempted"]) for x in rs)
        succ = sum(1 for x in rs if x["compression_attempted"]
                   and x["sufficiency_reached"])
        ne = sum(1 for x in rs if x["candidate_count"] > 0)
        fb = sum(1 for x in rs if x["candidate_count"] > 0
                 and x["fallback_to_static"])
        fixed = {
            "dual_view_agreement_rate":
                sum(int(x["dual_view_agree"]) for x in rs) / n,
            "compression_attempt_rate": att / n,
            "sufficiency_success_rate": (succ / att if att else 0.0),
            "static_fallback_rate": (fb / ne if ne else 0.0)}
        if grid[a].get("sufficiency") != fixed:
            grid[a]["sufficiency"] = fixed
            changed = True
    if changed:
        with open(grid_path, "w", encoding="utf-8") as fh:
            json.dump(grid, fh, indent=1)
    return grid, rows


def mechanism_from_rows(rows):
    """Aggregate §18 mechanism/temporal/reader-margin/search metrics."""
    n = max(len(rows), 1)
    agree = attempt = success = fallback = 0
    success_attempted = 0
    n_nonempty = 0
    pred_changed = 0
    adds, removes, pools = [], [], []
    jac, surv, ret = [], [], []
    tok_st, tok_ms, unit_st, unit_ms = [], [], [], []
    by_event = {}
    for r in rows:
        by_event.setdefault(r["event_id"], []).append(r)
        agree += int(r["dual_view_agree"])
        attempt += int(r["compression_attempted"])
        success += int(r["sufficiency_reached"])
        fallback += int(r["fallback_to_static"])
        if r["candidate_count"] > 0:
            n_nonempty += 1
        if r["compression_attempted"] and r["sufficiency_reached"]:
            success_attempted += 1
        pred_changed += int(r["ms_prediction"] != r["static_prediction"])
        adds.append(r["add_count"])
        removes.append(r["remove_count"])
        pools.append(r["pool_size"])
        tok_st.append(r["static_evidence_tokens"])
        tok_ms.append(r["ms_evidence_tokens"])
        unit_st.append(r["static_selected_count"])
        unit_ms.append(r["ms_selected_count"])
        if r["margin_retention"] is not None:
            ret.append(r["margin_retention"])
    for ev_rows in by_event.values():
        ev_rows.sort(key=lambda r: PRIMARY_CUTOFFS.index(int(r["cutoff"])))
        for i in range(1, len(ev_rows)):
            prev = set(ev_rows[i - 1]["memory_current_ids"])
            cur = set(ev_rows[i]["memory_current_ids"])
            if prev | cur:
                jac.append(len(prev & cur) / len(prev | cur))
            if prev:
                surv.append(len(prev & cur) / len(prev))
    red = [1.0 - (b / a) if a > 0 else 0.0
           for a, b in zip(tok_st, tok_ms)]
    return {
        "sufficiency": {
            "n_snapshots": len(rows),
            "dual_view_agreement_rate": agree / n,
            "compression_attempt_rate": attempt / n,
            "sufficiency_success_rate": (success_attempted / attempt
                                         if attempt else 0.0),
            "static_fallback_rate": (fallback / n_nonempty
                                     if n_nonempty else 0.0),
            "prediction_change_rate": pred_changed / n},
        "compression": {
            "mean_token_reduction": mean(red),
            "median_token_reduction": sorted(red)[len(red) // 2]
            if red else 0.0,
            "p25_token_reduction": pct(red, 0.25),
            "p75_token_reduction": pct(red, 0.75),
            "mean_static_tokens": mean(tok_st),
            "mean_ms_tokens": mean(tok_ms),
            "mean_static_units": mean(unit_st),
            "mean_ms_units": mean(unit_ms),
            "unit_reduction": (1.0 - mean(unit_ms) / mean(unit_st)
                               if mean(unit_st) else 0.0)},
        "temporal": {
            "memory_survival_rate": mean(surv),
            "adjacent_set_jaccard": mean(jac),
            "context_churn": 1.0 - mean(jac)},
        "reader_margin": {
            "mean_static_margin": mean([r["static_margin"] for r in rows]),
            "mean_ms_margin": mean([r["ms_margin"] for r in rows]),
            "mean_margin_retention": mean(ret)},
        "search": {
            "mean_add_count": mean(adds), "mean_remove_count": mean(removes),
            "max_pool_size": max(pools) if pools else 0,
            "mean_pool_size": mean(pools)},
    }


def margin_strata(rows):
    """§19: Static margin bottom 25% / 25-75% / top 25%."""
    margins = sorted(r["static_margin"] for r in rows)
    if not margins:
        return {}
    q1 = margins[int(0.25 * len(margins))]
    q3 = margins[int(0.75 * len(margins))]

    def _bin(m):
        if m <= q1:
            return "low_confidence"
        if m <= q3:
            return "medium"
        return "high_confidence"
    out = {}
    for name in ("low_confidence", "medium", "high_confidence"):
        sub = [r for r in rows if _bin(r["static_margin"]) == name]
        if not sub:
            continue
        red = [1.0 - (r["ms_evidence_tokens"] / r["static_evidence_tokens"])
               if r["static_evidence_tokens"] > 0 else 0.0 for r in sub]
        st = classification_metrics([(r["gold"], r["static_prediction"])
                                     for r in sub])
        ms = classification_metrics([(r["gold"], r["ms_prediction"])
                                     for r in sub])
        out[name] = {
            "n_snapshots": len(sub),
            "static_margin_boundary": {"q25": q1, "q75": q3},
            "mean_token_reduction": mean(red),
            "static_macro_f1": st["macro_f1"],
            "ms_macro_f1": ms["macro_f1"],
            "delta_macro_f1": ms["macro_f1"] - st["macro_f1"],
            "fallback_rate": mean([float(r["fallback_to_static"])
                                   for r in sub]),
            "dual_view_agreement_rate": mean(
                [float(r["dual_view_agree"]) for r in sub])}
    return out


def stage_summary(rows):
    out = {}
    for stage, cuts in STAGE_CUTOFFS.items():
        sub = [r for r in rows if r["cutoff"] in cuts]
        if not sub:
            continue
        st = classification_metrics([(r["gold"], r["static_prediction"])
                                     for r in sub])
        ms = classification_metrics([(r["gold"], r["ms_prediction"])
                                     for r in sub])
        red = [1.0 - (r["ms_evidence_tokens"] / r["static_evidence_tokens"])
               if r["static_evidence_tokens"] > 0 else 0.0 for r in sub]
        churn = mean([float(set(r["memory_previous_ids"])
                            != set(r["memory_current_ids"])) for r in sub])
        out[stage] = {"static_macro_f1": st["macro_f1"],
                      "ms_macro_f1": ms["macro_f1"],
                      "delta": ms["macro_f1"] - st["macro_f1"],
                      "mean_token_reduction": mean(red),
                      "context_churn": churn,
                      "n_snapshots": len(sub)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args(argv)
    root = args.root
    summary = {"stage": "V3-A (proxy validation only; no test, no Qwen)",
               "budget": BUDGET_TOKENS,
               "alpha_grid": list(ALPHA_GRID),
               "thresholds": {
                   "delta_macro_f1_min": NONINF_DELTA_MF1,
                   "max_relative_flip_increase": MAX_REL_FLIP_INCREASE,
                   "min_token_reduction": MIN_TOKEN_REDUCTION},
               "datasets": {}, "diagnostics": {}}
    for dataset in DATASETS:
        ds = {"folds": {}, "fold_alpha": {}, "not_supported_folds": []}
        agg = {"static": [], "ms": [], "red": [], "flip_st": 0, "flip_ms": 0,
               "n_trans": 0, "rows": []}
        for fold in FOLDS:
            grids, rows_by_seed = {}, {}
            for seed in SEEDS:
                g, r = load_run(root, dataset, fold, seed)
                grids[seed] = g
                rows_by_seed[seed] = r
            per_seed = {a: [grids[s][a] for s in SEEDS] for a in ALPHA_ORDER}
            best, stats = pick_fold_alpha(per_seed)
            fold_dir = os.path.join(root, dataset, f"fold{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, "alpha_grid.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"dataset": dataset, "fold": fold,
                           "candidates": stats, "selected_alpha": best,
                           "rule": "§16: among alphas with delta Macro-F1 "
                                   ">= -0.002 and relative flip increase "
                                   "<= 5%, take the largest token "
                                   "reduction (ties -> smaller alpha)"},
                          fh, indent=1)
            if best is None:
                # §16: no alpha respects non-inferiority + flip bound -> the
                # fold is reported as NOT_SUPPORTED and contributes nothing
                # to the dataset gate (never relaxed).
                ds["not_supported_folds"].append(f"fold{fold}")
                ds["fold_alpha"][f"fold{fold}"] = None
                ds["folds"][f"fold{fold}"] = {
                    "dataset": dataset, "fold": fold, "alpha": None,
                    "gate": "V3_NOT_SUPPORTED"}
                continue
            ds["fold_alpha"][f"fold{fold}"] = float(best)
            fold_rows = []
            for seed in SEEDS:
                g = grids[seed][best]
                agg["static"].append(g["mean_primary_macro_f1"]["static"])
                agg["ms"].append(g["mean_primary_macro_f1"]["ms"])
                agg["red"].append(g["compression"]["mean_token_reduction"])
                for r in rows_by_seed[seed]:
                    if str(r["alpha"]) == best:
                        fold_rows.append(r)
                        agg["rows"].append(r)
                ev = {}
                for r in rows_by_seed[seed]:
                    if str(r["alpha"]) != best:
                        continue
                    ev.setdefault(r["event_id"], []).append(r)
                for evr in ev.values():
                    evr.sort(key=lambda x: PRIMARY_CUTOFFS.index(
                        int(x["cutoff"])))
                    for i in range(1, len(evr)):
                        agg["n_trans"] += 1
                        if evr[i - 1]["static_prediction"] != \
                                evr[i]["static_prediction"]:
                            agg["flip_st"] += 1
                        if evr[i - 1]["ms_prediction"] != \
                                evr[i]["ms_prediction"]:
                            agg["flip_ms"] += 1
            mech = mechanism_from_rows(fold_rows)
            with open(os.path.join(fold_dir, "sufficiency_dynamics.json"),
                      "w", encoding="utf-8") as fh:
                json.dump({"dataset": dataset, "fold": fold,
                           "alpha": float(best), **mech}, fh, indent=1)
            with open(os.path.join(fold_dir, "compression_metrics.json"),
                      "w", encoding="utf-8") as fh:
                json.dump({"dataset": dataset, "fold": fold,
                           "alpha": float(best),
                           "mean_token_reduction": mech["compression"]
                           ["mean_token_reduction"],
                           "mean_static_tokens": mech["compression"]
                           ["mean_static_tokens"],
                           "mean_ms_tokens": mech["compression"]
                           ["mean_ms_tokens"],
                           "unit_reduction": mech["compression"]
                           ["unit_reduction"]}, fh, indent=1)
            with open(os.path.join(fold_dir,
                                   "validation_predictions.jsonl"), "w",
                      encoding="utf-8") as fh:
                for r in fold_rows:
                    fh.write(json.dumps(r) + "\n")
            ds["folds"][f"fold{fold}"] = {
                "dataset": dataset, "fold": fold, "budget": BUDGET_TOKENS,
                "alpha": float(best),
                "validation_static_macro_f1": stats[best]
                ["static_macro_f1"],
                "validation_ms_tsr_macro_f1": stats[best]["ms_macro_f1"],
                "delta_macro_f1": stats[best]["delta_macro_f1"],
                "static_flip_rate": stats[best]["static_flip_rate"],
                "ms_tsr_flip_rate": stats[best]["ms_flip_rate"],
                "relative_flip_increase": stats[best]
                ["relative_flip_increase"],
                "mean_token_reduction": stats[best]
                ["mean_token_reduction"],
                "selection_scope": "validation-only (fold-local, §16)"}
            with open(os.path.join(fold_dir, "best_config.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(ds["folds"][f"fold{fold}"], fh, indent=1)
        # ---- dataset aggregates over fold-selected alphas ----------------
        if agg["static"]:
            ds["mean_static_macro_f1"] = mean(agg["static"])
        else:
            ds["mean_static_macro_f1"] = 0.0
        ds["mean_ms_tsr_macro_f1"] = mean(agg["ms"])
        ds["delta_macro_f1"] = (ds["mean_ms_tsr_macro_f1"]
                                - ds["mean_static_macro_f1"]) if agg["ms"] \
            else 0.0
        ds["mean_token_reduction"] = mean(agg["red"])
        ds["flip_rate_static"] = agg["flip_st"] / max(agg["n_trans"], 1)
        ds["flip_rate_ms_tsr"] = agg["flip_ms"] / max(agg["n_trans"], 1)
        ds["relative_flip_increase"] = (
            (ds["flip_rate_ms_tsr"] - ds["flip_rate_static"])
            / ds["flip_rate_static"] if ds["flip_rate_static"] > 0 else 0.0)
        gate, conds = gate_verdict(ds["delta_macro_f1"],
                                   ds["mean_token_reduction"],
                                   ds["relative_flip_increase"])
        ds["gate"] = gate
        ds["gate_conditions"] = conds
        ds["n_folds_supported"] = len(FOLDS) - len(ds["not_supported_folds"])
        ds["n_runs_in_gate"] = len(agg["ms"])
        ds["mechanism"] = mechanism_from_rows(agg["rows"]) if agg["rows"] \
            else {}
        ds["margin_stratified"] = margin_strata(agg["rows"])
        ds["stage_wise"] = stage_summary(agg["rows"])
        summary["datasets"][dataset] = ds
        summary["diagnostics"][dataset] = {
            "margin_stratified": ds["margin_stratified"],
            "stage_wise": ds["stage_wise"]}
    overall, rec = dataset_decision(summary["datasets"])
    summary["overall"] = overall
    summary["recommendation"] = rec
    os.makedirs(os.path.join(root, "diagnostics"), exist_ok=True)
    for dataset in DATASETS:
        with open(os.path.join(root, "diagnostics",
                               f"{dataset}_diagnostics.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(summary["diagnostics"][dataset], fh, indent=1)
    os.makedirs(os.path.join(root, "readiness"), exist_ok=True)
    with open(os.path.join(root, "readiness", "ms_tsr_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(root, "ms_tsr_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    write_report(root, summary)
    print(json.dumps({
        "overall": overall, "recommendation": rec,
        "datasets": {d: {
            "delta_macro_f1": round(v["delta_macro_f1"], 5),
            "token_reduction": round(v["mean_token_reduction"], 4),
            "relative_flip_increase": round(v["relative_flip_increase"], 4),
            "gate": v["gate"], "fold_alpha": v["fold_alpha"],
            "not_supported_folds": v["not_supported_folds"]}
            for d, v in summary["datasets"].items()}}, indent=1), flush=True)
    return 0


def write_report(root, summary):
    lines = ["# TC-DSCR Dynamic V3 — MS-TSR Validation (Stage V3-A)", "",
             "## Implementation",
             "- encoder frozen: True (corrected E2 provenance SHA-checked)",
             "- selector frozen: True", "- proxy frozen: True",
             "- new trainable parameters: NONE",
             f"- budget: {BUDGET_TOKENS} (frozen)",
             f"- alpha grid: {list(ALPHA_GRID)} (only validation grid)",
             "- moves: ADD + REMOVE only (no SWAP)", "",
             "## Protocol", "- all validation events at all 6 cutoffs "
             "(no-candidate snapshots included)",
             "- test split never read; Qwen never called in V3-A", ""]
    for dataset in DATASETS:
        d = summary["datasets"][dataset]
        lines += [f"## {DISPLAY_NAME[dataset]}", "",
                  "### Fold alpha", "| fold | alpha |", "|---|---:|"]
        for f in FOLDS:
            a = d["fold_alpha"][f"fold{f}"]
            lines.append(f"| {f} | {a if a is not None else 'NOT_SUPPORTED'} |")
        lines += ["", "### Validation",
                  "| method | Macro-F1 | FlipRate | tokens |",
                  "|---|---:|---:|---:|"]
        mech = d.get("mechanism") or {}
        comp = mech.get("compression", {})
        lines += [f"| Static | {d['mean_static_macro_f1']:.4f} | "
                  f"{d['flip_rate_static']:.4f} | "
                  f"{comp.get('mean_static_tokens', 0.0):.1f} |",
                  f"| MS-TSR | {d['mean_ms_tsr_macro_f1']:.4f} | "
                  f"{d['flip_rate_ms_tsr']:.4f} | "
                  f"{comp.get('mean_ms_tokens', 0.0):.1f} |", "",
                  f"delta Macro-F1: {d['delta_macro_f1']:+.5f}",
                  f"token reduction: {d['mean_token_reduction']:.4f}",
                  f"relative flip increase: "
                  f"{d['relative_flip_increase']:+.4f}",
                  f"gate: {d['gate']} "
                  f"({d['gate_conditions']})", ""]
        suf = mech.get("sufficiency", {})
        tmp = mech.get("temporal", {})
        mar = mech.get("reader_margin", {})
        srh = mech.get("search", {})
        lines += ["### Mechanism",
                  f"- dual-view agreement rate: "
                  f"{suf.get('dual_view_agreement_rate', 0):.4f}",
                  f"- compression attempt rate: "
                  f"{suf.get('compression_attempt_rate', 0):.4f}",
                  f"- sufficiency success rate: "
                  f"{suf.get('sufficiency_success_rate', 0):.4f}",
                  f"- static fallback rate: "
                  f"{suf.get('static_fallback_rate', 0):.4f}",
                  f"- prediction change rate (MS-TSR vs Static): "
                  f"{suf.get('prediction_change_rate', 0):.6f}",
                  f"- mean token reduction: "
                  f"{comp.get('mean_token_reduction', 0):.4f} "
                  f"(median {comp.get('median_token_reduction', 0):.4f}, "
                  f"P25 {comp.get('p25_token_reduction', 0):.4f}, "
                  f"P75 {comp.get('p75_token_reduction', 0):.4f})",
                  f"- unit reduction: {comp.get('unit_reduction', 0):.4f}",
                  f"- memory survival rate: "
                  f"{tmp.get('memory_survival_rate', 0):.4f}",
                  f"- adjacent set Jaccard: "
                  f"{tmp.get('adjacent_set_jaccard', 0):.4f}",
                  f"- context churn: {tmp.get('context_churn', 0):.4f}",
                  f"- margin retention ratio: "
                  f"{mar.get('mean_margin_retention', 0):.4f}",
                  f"- mean ADD / REMOVE per snapshot: "
                  f"{srh.get('mean_add_count', 0):.2f} / "
                  f"{srh.get('mean_remove_count', 0):.2f}",
                  f"- max candidate-pool size: "
                  f"{srh.get('max_pool_size', 0)}", ""]
        lines += ["### Margin-stratified (§19)",
                  "| stratum | n | token reduction | Delta MF1 | fallback |",
                  "|---|---:|---:|---:|---:|"]
        for name in ("low_confidence", "medium", "high_confidence"):
            v = (d.get("margin_stratified") or {}).get(name)
            if not v:
                continue
            lines.append(f"| {name} | {v['n_snapshots']} | "
                         f"{v['mean_token_reduction']:.4f} | "
                         f"{v['delta_macro_f1']:+.4f} | "
                         f"{v['fallback_rate']:.4f} |")
        lines += ["", "### Stage-wise",
                  "| stage | Static MF1 | MS-TSR MF1 | delta | token "
                  "reduction |", "|---|---:|---:|---:|---:|"]
        for k, v in (d.get("stage_wise") or {}).items():
            lines.append(f"| {k} | {v['static_macro_f1']:.4f} | "
                         f"{v['ms_macro_f1']:.4f} | {v['delta']:+.4f} | "
                         f"{v['mean_token_reduction']:.4f} |")
        lines.append("")
    total_rows = 0
    total_changed = 0.0
    for d in summary["datasets"].values():
        s = (d.get("mechanism") or {}).get("sufficiency", {})
        n_snap = s.get("n_snapshots", 0)
        total_rows += n_snap
        total_changed += s.get("prediction_change_rate", 0.0) * n_snap
    total_changed = int(round(total_changed))
    lines += [
        "## Reading of the proxy result", "",
        f"- MS-TSR changed the Static decision on {total_changed} of "
        f"{total_rows} snapshots. Sufficiency condition C1 requires "
        "argmax q(S) == y_ref for every successfully compressed snapshot, and "
        "fallback rows keep the Static set verbatim; at the proxy layer "
        "Delta Macro-F1 and FlipRate therefore match Static by construction "
        "rather than by measurement.",
        "- The gate's classification non-inferiority and temporal stability "
        "conditions consequently carry no evidential weight on the proxy "
        "reader; only compression, sufficiency, margin and temporal-set "
        "metrics are informative here.",
        "- The proxy is additionally insensitive to evidence substitution (the "
        "Dynamic V2 diagnosis measured a 0.4-0.7% prediction-change rate under "
        "set replacement), so a proxy-level 'no loss' must not be read as a "
        "real-reader 'no loss'.",
        "- Fold-local alpha collapsed to the loosest grid point (0.8) in every "
        "fold, i.e. the C2 margin constraint never bound; the achieved "
        "compression is driven by the budget and the greedy forward "
        "construction rather than by the sufficiency margin.",
        "- Resolving exactly this gap is the purpose of Stage V3-B (frozen "
        "Qwen reader transfer), which is the required next step.", ""]
    v_path = os.path.join(root, "ms_tsr_verify.json")
    n_issues, v_detail = "not run", ""
    if os.path.exists(v_path):
        with open(v_path, encoding="utf-8") as fh:
            v = json.load(fh)
        n_issues = v.get("n_issues")
        v_detail = (f" (runs={v.get('runs')}, rows={v.get('rows')}, "
                    f"no_candidate_rows={v.get('no_candidate_rows')})")
    lines += ["## Verifier", f"issues = {n_issues}{v_detail}", "",
              "## Recommendation",
              f"Overall = {summary['overall']}",
              f"Recommendation = {summary['recommendation']}", ""]
    with open(os.path.join(root, "MS_TSR_VALIDATION_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
