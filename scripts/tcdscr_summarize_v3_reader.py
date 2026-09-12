#!/usr/bin/env python
"""V3-B STEP 9-13: reader-transfer statistics, stratification and report.

Consumes the frozen manifest, the raw generations and the parsed rows, then
writes per-dataset statistics, diagnostics, the overall summary and
``V3_B_READER_TRANSFER_REPORT.md``.

Statistics are kept as pure functions so the tests can exercise them without a
GPU: paired bootstrap resamples indices (multiplicity preserved, §32), McNemar
is the exact binomial form (§33), ECE is a 10-bin diagnostic (§27) and the
citation accounting follows §28-§31.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATASETS = ("pheme", "maweibo")
CUTOFFS = (5, 15, 30, 60, 180, 360)
DISPLAY = {"pheme": "PHEME", "maweibo": "Ma-Weibo"}
CUTOFF_DISPLAY = {5: "5m", 15: "15m", 30: "30m", 60: "1h", 180: "3h",
                  360: "6h"}
PRESSURES = ("NO_CANDIDATE", "LOW", "MEDIUM", "HIGH")
RUMOR = "RUMOR"
NON_RUMOR = "NON_RUMOR"
BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 3090

# §22 / §23 / §41 frozen thresholds
MIN_DELTA_MACRO_F1 = -0.005
MIN_TOKEN_REDUCTION = 0.30
MAX_UNSUPPORTED_CITATION_RATE = 0.01
CLEAR_CI_LOWER = -0.01

DEFAULT_ROOT = "/data/jyz/next/llm/results/tcdscr/dynamic_v3_reader"


# ---------------------------------------------------------------- metrics

def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall \
        else 0.0
    return precision, recall, f1


def classification_metrics(pairs):
    """``pairs`` = list of ``(gold_label, pred_label)`` over one arm."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0,
                "rumor_f1": 0.0, "rumor_precision": 0.0, "rumor_recall": 0.0,
                "confusion": {}}
    tp = sum(1 for g, p in pairs if g == RUMOR and p == RUMOR)
    fp = sum(1 for g, p in pairs if g == NON_RUMOR and p == RUMOR)
    fn = sum(1 for g, p in pairs if g == RUMOR and p == NON_RUMOR)
    tn = sum(1 for g, p in pairs if g == NON_RUMOR and p == NON_RUMOR)
    p_r, r_r, f1_r = _prf(tp, fp, fn)
    p_n, r_n, f1_n = _prf(tn, fn, fp)
    support_r = tp + fn
    support_n = tn + fp
    weighted = (f1_r * support_r + f1_n * support_n) / n
    return {
        "n": n,
        "accuracy": (tp + tn) / n,
        "macro_f1": (f1_r + f1_n) / 2,
        "weighted_f1": weighted,
        "rumor_f1": f1_r,
        "rumor_precision": p_r,
        "rumor_recall": r_r,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + \
        recall else 0.0


def resample_indices(n, iters, seed):
    """Bootstrap index draws *with replacement* (§32).

    Every draw has length ``n`` and duplicates are kept: a sample drawn k times
    contributes k times.  Deduplicating here (the historical ``list -> dict``
    bug) would silently shrink the resample.
    """
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(iters)]


def paired_bootstrap_delta(pairs, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """95% percentile CI of Macro-F1(MS) - Macro-F1(Static) (§32)."""
    import numpy as np
    n = len(pairs)
    point = (classification_metrics([(p["gold_label"], p["ms_label"])
                                     for p in pairs])["macro_f1"]
             - classification_metrics([(p["gold_label"], p["static_label"])
                                       for p in pairs])["macro_f1"])
    if n == 0:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "iters": iters, "n": 0, "seed": seed}
    gold = np.array([1 if p["gold_label"] == RUMOR else 0 for p in pairs])
    static = np.array([1 if p["static_label"] == RUMOR else 0 for p in pairs])
    ms = np.array([1 if p["ms_label"] == RUMOR else 0 for p in pairs])

    def macro(g, p):
        tp = int(np.sum((g == 1) & (p == 1)))
        fp = int(np.sum((g == 0) & (p == 1)))
        fn = int(np.sum((g == 1) & (p == 0)))
        tn = int(np.sum((g == 0) & (p == 0)))
        return (_f1_from_counts(tp, fp, fn) + _f1_from_counts(tn, fn, fp)) / 2

    deltas = np.empty(iters, dtype=float)
    for i, draw in enumerate(resample_indices(n, iters, seed)):
        sel = np.asarray(draw, dtype=int)
        deltas[i] = macro(gold[sel], ms[sel]) - macro(gold[sel], static[sel])
    deltas.sort()
    lo = float(deltas[int(0.025 * iters)])
    hi = float(deltas[min(iters - 1, int(0.975 * iters))])
    return {"point": point, "ci_low": lo, "ci_high": hi, "iters": iters,
            "n": n, "seed": seed}


def binom_two_sided_p(b, c):
    """Exact two-sided binomial p-value with p=0.5 (§33)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def expected_calibration_error(items, bins=10):
    """``items`` = list of ``(confidence, correct)``; equal-width bins (§27)."""
    if not items:
        return 0.0
    total = len(items)
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        bucket = [(c, ok) for c, ok in items
                  if (lo <= c < hi) or (b == bins - 1 and c == hi)]
        if not bucket:
            continue
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        conf = sum(c for c, _ in bucket) / len(bucket)
        ece += len(bucket) / total * abs(acc - conf)
    return ece


def citation_metrics(rows):
    """§28-§31 over one arm's parsed rows."""
    n = len(rows)
    cited = sum(r["n_cited"] for r in rows)
    valid = sum(len(r["valid_evidence_ids"]) for r in rows)
    invalid = sum(len(r["invalid_evidence_ids"]) for r in rows)
    no_citation = sum(1 for r in rows if r["n_cited"] == 0)
    supplied = sum(r.get("n_evidence", 0) for r in rows)
    reason_bad = sum(1 for r in rows if r["reason_invalid_refs"])
    return {
        "n_runs": n,
        "n_cited_total": cited,
        "valid_citation_rate": valid / cited if cited else None,
        "unsupported_citation_count": invalid,
        "unsupported_citation_rate": invalid / cited if cited else None,
        "no_citation_rate": no_citation / n if n else 0.0,
        "mean_cited_evidence_count": cited / n if n else 0.0,
        "cited_over_supplied": cited / supplied if supplied else None,
        "reason_invalid_evidence_reference_rate": reason_bad / n if n else 0.0,
    }


def compression_metrics(pairs):
    """§20: social-context reduction only where Static social tokens > 0."""
    eligible = [p for p in pairs if p["static_social_tokens"] > 0]
    social = [1.0 - p["ms_social_tokens"] / p["static_social_tokens"]
              for p in eligible]
    total = [1.0 - p["ms_total_input_tokens"] / p["static_total_input_tokens"]
             for p in eligible]
    return {
        "n_eligible": len(eligible),
        "n_excluded_zero_static": len(pairs) - len(eligible),
        "mean_static_social_tokens": (sum(p["static_social_tokens"]
                                          for p in eligible) / len(eligible)
                                      if eligible else 0.0),
        "mean_ms_social_tokens": (sum(p["ms_social_tokens"]
                                      for p in eligible) / len(eligible)
                                  if eligible else 0.0),
        "mean_social_token_reduction": (sum(social) / len(social)
                                        if social else None),
        "mean_static_total_tokens": (sum(p["static_total_input_tokens"]
                                         for p in eligible) / len(eligible)
                                     if eligible else 0.0),
        "mean_ms_total_tokens": (sum(p["ms_total_input_tokens"]
                                     for p in eligible) / len(eligible)
                                 if eligible else 0.0),
        "mean_total_prompt_token_reduction": (sum(total) / len(total)
                                              if total else None),
    }


def paired_outcomes(pairs):
    """§24-§25 paired correctness and disagreement decomposition."""
    out = {"both_correct": 0, "both_wrong": 0, "wrong_to_correct": 0,
           "correct_to_wrong": 0, "disagreement": 0,
           "label_changed_both_wrong": 0,
           "label_changed_static_to_correct": 0,
           "label_changed_static_to_wrong": 0}
    for p in pairs:
        sc = p["static_label"] == p["gold_label"]
        mc = p["ms_label"] == p["gold_label"]
        if sc and mc:
            out["both_correct"] += 1
        elif sc and not mc:
            out["correct_to_wrong"] += 1
        elif mc and not sc:
            out["wrong_to_correct"] += 1
        else:
            out["both_wrong"] += 1
        if p["static_label"] != p["ms_label"]:
            out["disagreement"] += 1
            if sc and not mc:
                out["label_changed_static_to_wrong"] += 1
            elif mc and not sc:
                out["label_changed_static_to_correct"] += 1
            else:
                out["label_changed_both_wrong"] += 1
    n = len(pairs)
    out["n"] = n
    out["net_correction_gain"] = (out["wrong_to_correct"]
                                  - out["correct_to_wrong"])
    out["disagreement_rate"] = out["disagreement"] / n if n else 0.0
    return out


def confidence_metrics(pairs):
    """§26/§27 mean confidence overall / correct / wrong, plus ECE per arm."""
    def _summ(conf_key, correct_key):
        if not pairs:
            return {"mean": None, "mean_correct": None, "mean_wrong": None}
        conf = [p[conf_key] for p in pairs]
        corr = [p[conf_key] for p in pairs if p[correct_key] == 1]
        wrong = [p[conf_key] for p in pairs if p[correct_key] == 0]
        return {
            "mean": sum(conf) / len(conf),
            "mean_correct": sum(corr) / len(corr) if corr else None,
            "mean_wrong": sum(wrong) / len(wrong) if wrong else None,
        }
    s = _summ("static_confidence", "static_correct")
    m = _summ("ms_confidence", "ms_correct")
    delta = (None if s["mean"] is None or m["mean"] is None
             else m["mean"] - s["mean"])
    return {
        "static": s, "ms": m, "mean_confidence_delta": delta,
        "ece_static": expected_calibration_error(
            [(p["static_confidence"], p["static_correct"] == 1)
             for p in pairs]),
        "ece_ms": expected_calibration_error(
            [(p["ms_confidence"], p["ms_correct"] == 1) for p in pairs]),
        "ece_bins": 10,
    }


def stratum_metrics(pairs, key_fn):
    groups = {}
    for p in pairs:
        groups.setdefault(key_fn(p), []).append(p)
    out = {}
    for key, sub in groups.items():
        st = classification_metrics([(p["gold_label"], p["static_label"])
                                     for p in sub])
        ms = classification_metrics([(p["gold_label"], p["ms_label"])
                                     for p in sub])
        out[str(key)] = {
            "n": len(sub),
            "static_macro_f1": st["macro_f1"],
            "ms_macro_f1": ms["macro_f1"],
            "delta_macro_f1": ms["macro_f1"] - st["macro_f1"],
            "static_accuracy": st["accuracy"],
            "ms_accuracy": ms["accuracy"],
            "mean_social_token_reduction": (
                sum(1.0 - p["ms_social_tokens"] / p["static_social_tokens"]
                    for p in sub if p["static_social_tokens"] > 0)
                / max(1, sum(1 for p in sub if p["static_social_tokens"] > 0))),
            "disagreement_rate": (sum(1 for p in sub
                                      if p["static_label"] != p["ms_label"])
                                  / len(sub)),
            "fallback_rate": (sum(1 for p in sub if p["fallback_to_static"])
                              / len(sub)),
        }
    return out


def margin_strata(pairs):
    """§36: quantile split of the V3-A Static margin inside the pilot."""
    margins = sorted(p["static_margin"] for p in pairs)
    if not margins:
        return {}
    q1 = margins[int(0.25 * len(margins))]
    q3 = margins[int(0.75 * len(margins))]

    def _bin(p):
        if p["static_margin"] <= q1:
            return "low_confidence"
        if p["static_margin"] <= q3:
            return "medium"
        return "high_confidence"
    ordered = ("low_confidence", "medium", "high_confidence")
    raw = stratum_metrics(pairs, _bin)
    return {k: raw[k] for k in ordered if k in raw}


# ---------------------------------------------------------------- gate

def dataset_gate(delta_macro_f1, token_reduction, unsupported_rate,
                 ci_low):
    """§41/§42 dataset gate and transfer label."""
    conds = {
        "reader_non_inferiority": delta_macro_f1 >= MIN_DELTA_MACRO_F1,
        "context_reduction": (token_reduction is not None
                              and token_reduction >= MIN_TOKEN_REDUCTION),
        "grounding_integrity": (unsupported_rate is None
                                or unsupported_rate
                                <= MAX_UNSUPPORTED_CITATION_RATE),
    }
    passed = all(conds.values())
    if not passed:
        label = "TRANSFER_FAIL"
    elif ci_low >= CLEAR_CI_LOWER:
        label = "CLEAR_TRANSFER"
    else:
        label = "WEAK_TRANSFER"
    return ("PASS" if passed else "FAIL"), conds, label


def overall_decision(per_dataset):
    """§43 overall verdict from the per-dataset gates."""
    if not per_dataset:
        return "FAIL", "PROXY_READER_TRANSFER_FAIL"
    passed = [d for d in per_dataset.values() if d["gate"] == "PASS"]
    if len(passed) == len(per_dataset):
        return "PASS", "START_V3_C_HELD_OUT_TEST"
    if passed:
        return "PARTIAL", "STOP_FOR_RESEARCH_REVIEW"
    return "FAIL", "PROXY_READER_TRANSFER_FAIL"


# ---------------------------------------------------------------- io

def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dataset(root, dataset, limit=None):
    manifest_path = os.path.join(root, "sampling_manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    samples = [s for s in manifest["samples"] if s["dataset"] == dataset]
    if limit is not None:
        samples = samples[:limit]
    raw = load_jsonl(os.path.join(root, "raw_generations",
                                  f"{dataset}.jsonl"))
    parsed = load_jsonl(os.path.join(root, "parsed", f"{dataset}.jsonl"))
    return manifest, samples, raw, parsed


def build_pairs(samples, parsed, raw):
    """Join the two arms; only pairs with both arms parsed enter the metrics."""
    n_evidence = {}
    for r in raw:
        n_evidence[(r["sample_id"], r["arm"])] = r["n_evidence"]
    by_key = {(r["sample_id"], r["arm"]): r for r in parsed}
    pairs, unpaired, parse_failures = [], [], {"static": 0, "ms": 0}
    for s in samples:
        st = by_key.get((s["sample_id"], "static"))
        ms = by_key.get((s["sample_id"], "ms"))
        for arm, row in (("static", st), ("ms", ms)):
            if row is None or row["parse_failure"]:
                parse_failures[arm] += 1
        if st is None or ms is None or st["parse_failure"] or \
                ms["parse_failure"]:
            unpaired.append(s["sample_id"])
            continue
        pairs.append({
            "sample_id": s["sample_id"],
            "dataset": s["dataset"],
            "event_id": s["event_id"],
            "cutoff": int(s["cutoff"]),
            "fold": s["fold"],
            "gold": s["gold"],
            "gold_label": st["gold_label"],
            "pressure_bin": s["pressure_bin"],
            "fallback_to_static": bool(s["fallback_to_static"]),
            "static_margin": float(s["static_margin"]),
            "ms_margin": float(s["ms_margin"]),
            "dual_view_agree": bool(s["dual_view_agree"]),
            "static_label": st["parsed_label"],
            "ms_label": ms["parsed_label"],
            "static_confidence": st["confidence"],
            "ms_confidence": ms["confidence"],
            "static_correct": st["correct"],
            "ms_correct": ms["correct"],
            "static_social_tokens": int(s["static_social_tokens"]),
            "ms_social_tokens": int(s["ms_social_tokens"]),
            "static_total_input_tokens": int(s["static_total_input_tokens"]),
            "ms_total_input_tokens": int(s["ms_total_input_tokens"]),
            "static_n_evidence": n_evidence.get((s["sample_id"], "static"), 0),
            "ms_n_evidence": n_evidence.get((s["sample_id"], "ms"), 0),
            "static_prompt_evidence_ids": sorted(
                (st.get("valid_evidence_ids") or [])),
            "ms_prompt_evidence_ids": sorted(
                (ms.get("valid_evidence_ids") or [])),
        })
    return pairs, unpaired, parse_failures


def determinism_audit(pairs, raw):
    """§37: fallback pairs share one context, so their outputs must match."""
    raw_by = {(r["sample_id"], r["arm"]): r for r in raw}
    fallback = [p for p in pairs if p["fallback_to_static"]]
    identical_context = 0
    identical_output = 0
    warnings = []
    for p in fallback:
        rs = raw_by.get((p["sample_id"], "static"))
        rm = raw_by.get((p["sample_id"], "ms"))
        if rs is None or rm is None:
            continue
        if rs["prompt_hash"] == rm["prompt_hash"]:
            identical_context += 1
            if rs["raw_output"] == rm["raw_output"]:
                identical_output += 1
            else:
                warnings.append(p["sample_id"])
    return {
        "n_fallback_pairs": len(fallback),
        "n_identical_prompt_hash": identical_context,
        "n_identical_raw_output": identical_output,
        "determinism_warnings": len(warnings),
        "determinism_warning_samples": warnings[:50],
    }


def dataset_statistics(root, dataset, limit=None):
    manifest, samples, raw, parsed = load_dataset(root, dataset, limit)
    pairs, unpaired, parse_failures = build_pairs(samples, parsed, raw)
    static_rows = [r for r in parsed if r["arm"] == "static"
                   and not r["parse_failure"]]
    ms_rows = [r for r in parsed if r["arm"] == "ms"
               and not r["parse_failure"]]

    detection = {
        "static": classification_metrics([(p["gold_label"], p["static_label"])
                                          for p in pairs]),
        "ms": classification_metrics([(p["gold_label"], p["ms_label"])
                                      for p in pairs]),
    }
    for arm in ("static", "ms"):
        rows = static_rows if arm == "static" else ms_rows
        detection[f"{arm}_all_parsed"] = classification_metrics(
            [(r["gold_label"], r["parsed_label"]) for r in rows])
    bos = paired_bootstrap_delta(pairs)
    outcomes = paired_outcomes(pairs)
    b = outcomes["correct_to_wrong"]
    c = outcomes["wrong_to_correct"]
    stats = {
        "dataset": dataset,
        "n_samples": len(samples),
        "n_pairs_effective": len(pairs),
        "n_unpaired": len(unpaired),
        "unpaired_sample_ids": unpaired[:50],
        "parse_failures": parse_failures,
        "detection": detection,
        "delta_macro_f1": bos["point"],
        "bootstrap": bos,
        "mcnemar": {"static_correct_ms_wrong": b,
                    "static_wrong_ms_correct": c,
                    "discordant": b + c,
                    "p_value": binom_two_sided_p(b, c)},
        "paired_outcomes": outcomes,
        "compression": compression_metrics(pairs),
        "confidence": confidence_metrics(pairs),
        "citation": {
            "static": citation_metrics(_with_supplied(static_rows, raw,
                                                     "static")),
            "ms": citation_metrics(_with_supplied(ms_rows, raw, "ms")),
        },
        "strata": {
            "cutoff": _cutoff_strata(pairs),
            "pressure": {k: v for k, v in
                         stratum_metrics(pairs,
                                         lambda p: p["pressure_bin"]).items()
                         if k in PRESSURES},
            "margin": margin_strata(pairs),
            "fallback": {
                "fallback": stratum_metrics(
                    [p for p in pairs if p["fallback_to_static"]],
                    lambda p: "fallback"),
                "compressed": stratum_metrics(
                    [p for p in pairs if not p["fallback_to_static"]],
                    lambda p: "compressed"),
            },
        },
        "determinism": determinism_audit(pairs, raw),
        "token_limits": {
            "max_total_input_tokens": max(
                [s["static_total_input_tokens"] for s in samples]
                + [s["ms_total_input_tokens"] for s in samples], default=0),
            "max_social_tokens": max(
                [s["static_social_tokens"] for s in samples]
                + [s["ms_social_tokens"] for s in samples], default=0),
            "n_context_overflow": sum(
                1 for r in raw
                if r.get("parse_status") == "CONTEXT_OVERFLOW"),
        },
    }
    comp = stats["compression"]
    ms_cit = stats["citation"]["ms"]
    gate, conds, label = dataset_gate(
        stats["delta_macro_f1"], comp["mean_social_token_reduction"],
        ms_cit["unsupported_citation_rate"], bos["ci_low"])
    stats["gate"] = gate
    stats["gate_conditions"] = conds
    stats["transfer_label"] = label
    return stats


def _cutoff_strata(pairs):
    raw = stratum_metrics(pairs, lambda p: p["cutoff"])
    out = {}
    for c in CUTOFFS:
        if str(c) in raw:
            out[str(c)] = {**raw[str(c)], "label": CUTOFF_DISPLAY[c]}
    return out


def _with_supplied(rows, raw, arm):
    """Attach per-run supplied evidence counts for cited/supplied (§30)."""
    counts = {(r["sample_id"], r["arm"]): r["n_evidence"] for r in raw}
    return [{**r, "n_evidence": counts.get((r["sample_id"], arm), 0)}
            for r in rows]


# ---------------------------------------------------------------- report

def _fmt(v, digits=4):
    return "n/a" if v is None else f"{v:.{digits}f}"


def write_report(root, summary):
    lines = ["# TC-DSCR V3-B — Frozen Qwen Reader Transfer", "",
             "## Protocol",
             f"- model: {summary['model_path']}",
             f"- checkpoint: config_sha256={summary['model']['config_sha256']}"
             f", weight_shards_sha256={summary['model']['weight_shards_sha256']}",
             f"- dtype/device: {summary['model']['dtype']} / "
             f"{summary['model']['device']}",
             f"- decoding: do_sample={summary['model']['decoding']['do_sample']}"
             f", num_beams={summary['model']['decoding']['num_beams']}, "
             f"max_new_tokens={summary['model']['decoding']['max_new_tokens']}"
             f", temperature={summary['model']['decoding']['temperature']}",
             "- validation only: True (no test events read)",
             "- scope: validation-only transfer-feasibility pilot; this is "
             "not held-out test performance (§44)",
             f"- samples: PHEME n={summary['datasets']['pheme']['n_samples']}, "
             f"Ma-Weibo n={summary['datasets']['maweibo']['n_samples']}",
             f"- sampling seed: {summary['sampling_seed']} "
             f"(context source seed {summary['context_source_seed']})",
             f"- alpha: {summary['alpha']} (frozen V3-A)",
             f"- budget: {summary['budget']}",
             f"- prompt version: {summary['prompt_version']}",
             f"- manifest samples_sha256: {summary['samples_sha256']}", ""]
    for ds in DATASETS:
        d = summary["datasets"][ds]
        det = d["detection"]
        comp = d["compression"]
        out = d["paired_outcomes"]
        conf = d["confidence"]
        lines += [f"## {DISPLAY[ds]}", "",
                  "### Detection",
                  "| Arm | Accuracy | Macro-F1 | Weighted-F1 | Rumor-F1 |",
                  "|---|---:|---:|---:|---:|"]
        for arm, name in (("static", "Static"), ("ms", "MS-TSR")):
            m = det[arm]
            lines.append(f"| {name} | {_fmt(m['accuracy'])} | "
                         f"{_fmt(m['macro_f1'])} | {_fmt(m['weighted_f1'])} | "
                         f"{_fmt(m['rumor_f1'])} |")
        lines += ["",
                  f"delta Macro-F1: {d['delta_macro_f1']:+.5f}",
                  f"bootstrap 95% CI: [{_fmt(d['bootstrap']['ci_low'])}, "
                  f"{_fmt(d['bootstrap']['ci_high'])}] "
                  f"({d['bootstrap']['iters']} iters, seed "
                  f"{BOOTSTRAP_SEED}, multiplicity preserved)",
                  f"McNemar p: {d['mcnemar']['p_value']:.4f} "
                  f"(b={d['mcnemar']['static_correct_ms_wrong']}, "
                  f"c={d['mcnemar']['static_wrong_ms_correct']})",
                  f"effective pairs: {d['n_pairs_effective']} / "
                  f"{d['n_samples']} (parse failures static="
                  f"{d['parse_failures']['static']}, ms="
                  f"{d['parse_failures']['ms']})", "",
                  "### Compression",
                  f"Static social tokens: "
                  f"{_fmt(comp['mean_static_social_tokens'], 1)}",
                  f"MS social tokens: {_fmt(comp['mean_ms_social_tokens'], 1)}",
                  f"Reduction: {_fmt(comp['mean_social_token_reduction'])}",
                  f"Static total prompt tokens: "
                  f"{_fmt(comp['mean_static_total_tokens'], 1)}",
                  f"MS total prompt tokens: "
                  f"{_fmt(comp['mean_ms_total_tokens'], 1)}",
                  f"Reduction: "
                  f"{_fmt(comp['mean_total_prompt_token_reduction'])}",
                  f"excluded zero-Static-context samples: "
                  f"{comp['n_excluded_zero_static']}",
                  f"max total input tokens (all samples): "
                  f"{d['token_limits']['max_total_input_tokens']}",
                  f"context overflow (CONTEXT_OVERFLOW): "
                  f"{d['token_limits']['n_context_overflow']} "
                  "(flagged, never truncated)",
                  "",
                  "### Paired Outcomes",
                  f"wrong -> correct: {out['wrong_to_correct']}",
                  f"correct -> wrong: {out['correct_to_wrong']}",
                  f"both correct: {out['both_correct']}",
                  f"both wrong: {out['both_wrong']}",
                  f"net correction gain: {out['net_correction_gain']}", "",
                  "### Disagreement",
                  f"rate: {_fmt(out['disagreement_rate'])}",
                  f"label changed, both wrong: "
                  f"{out['label_changed_both_wrong']}",
                  f"label changed Static->correct: "
                  f"{out['label_changed_static_to_correct']}",
                  f"label changed Static->wrong: "
                  f"{out['label_changed_static_to_wrong']}", "",
                  "### Confidence / Calibration",
                  f"mean confidence Static: "
                  f"{_fmt(conf['static']['mean'])}",
                  f"mean confidence MS: {_fmt(conf['ms']['mean'])}",
                  f"confidence delta: {_fmt(conf['mean_confidence_delta'])}",
                  f"Static confidence correct/wrong: "
                  f"{_fmt(conf['static']['mean_correct'])} / "
                  f"{_fmt(conf['static']['mean_wrong'])}",
                  f"MS confidence correct/wrong: "
                  f"{_fmt(conf['ms']['mean_correct'])} / "
                  f"{_fmt(conf['ms']['mean_wrong'])}",
                  f"ECE Static / MS (10 bins): {_fmt(conf['ece_static'])} / "
                  f"{_fmt(conf['ece_ms'])}", "",
                  "### Grounding",
                  "| metric | Static | MS-TSR |", "|---|---:|---:|"]
        for key in ("valid_citation_rate", "unsupported_citation_rate",
                    "no_citation_rate", "mean_cited_evidence_count",
                    "cited_over_supplied",
                    "reason_invalid_evidence_reference_rate"):
            lines.append(f"| {key} | "
                         f"{_fmt(d['citation']['static'][key])} | "
                         f"{_fmt(d['citation']['ms'][key])} |")
        lines += ["",
                  f"unsupported citations (MS): "
                  f"{d['citation']['ms']['unsupported_citation_count']}", "",
                  "### Stratified — cutoff",
                  "| cutoff | n | Static MF1 | MS MF1 | delta | token reduction "
                  "| disagreement |", "|---|---:|---:|---:|---:|---:|---:|"]
        for c in CUTOFFS:
            v = d["strata"]["cutoff"].get(str(c))
            if not v:
                continue
            lines.append(f"| {CUTOFF_DISPLAY[c]} | {v['n']} | "
                         f"{_fmt(v['static_macro_f1'])} | "
                         f"{_fmt(v['ms_macro_f1'])} | "
                         f"{v['delta_macro_f1']:+.5f} | "
                         f"{_fmt(v['mean_social_token_reduction'])} | "
                         f"{_fmt(v['disagreement_rate'])} |")
        lines += ["", "### Stratified — selection pressure",
                  "| pressure | n | Static MF1 | MS MF1 | delta | token "
                  "reduction |", "|---|---:|---:|---:|---:|---:|"]
        for p in PRESSURES:
            v = d["strata"]["pressure"].get(p)
            if not v:
                continue
            lines.append(f"| {p} | {v['n']} | {_fmt(v['static_macro_f1'])} | "
                         f"{_fmt(v['ms_macro_f1'])} | "
                         f"{v['delta_macro_f1']:+.5f} | "
                         f"{_fmt(v['mean_social_token_reduction'])} |")
        lines += ["", "### Stratified — proxy margin",
                  "| stratum | n | Static MF1 | MS MF1 | delta | token "
                  "reduction | disagreement |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for m in ("low_confidence", "medium", "high_confidence"):
            v = d["strata"]["margin"].get(m)
            if not v:
                continue
            lines.append(f"| {m} | {v['n']} | {_fmt(v['static_macro_f1'])} | "
                         f"{_fmt(v['ms_macro_f1'])} | "
                         f"{v['delta_macro_f1']:+.5f} | "
                         f"{_fmt(v['mean_social_token_reduction'])} | "
                         f"{_fmt(v['disagreement_rate'])} |")
        lines += ["", "### Fallback vs compressed",
                  "| subset | n | Static MF1 | MS MF1 | delta | token "
                  "reduction |", "|---|---:|---:|---:|---:|---:|"]
        for k in ("fallback", "compressed"):
            v = (d["strata"]["fallback"].get(k) or {}).get(k)
            if not v:
                continue
            lines.append(f"| {k} | {v['n']} | {_fmt(v['static_macro_f1'])} | "
                         f"{_fmt(v['ms_macro_f1'])} | "
                         f"{v['delta_macro_f1']:+.5f} | "
                         f"{_fmt(v['mean_social_token_reduction'])} |")
        lines += ["",
                  f"### Gate — {d['gate']} / {d['transfer_label']}",
                  f"conditions: {d['gate_conditions']}", ""]
    det = summary["determinism"]
    lines += ["## Determinism Audit",
              f"fallback pairs: {det['n_fallback_pairs']}",
              f"identical prompt hash: {det['n_identical_prompt_hash']}",
              f"identical raw output: {det['n_identical_raw_output']}",
              f"DETERMINISM_WARNING: {det['determinism_warnings']}", "",
              "## Verifier", "issues = not run", "",
              "## Overall", f"{summary['overall']}", "",
              "## Recommendation", f"{summary['recommendation']}", ""]
    with open(os.path.join(root, "V3_B_READER_TRANSFER_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    root = args.root

    with open(os.path.join(root, "sampling_manifest.json"),
              encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(os.path.join(root, "run_manifest.json"), encoding="utf-8") as fh:
        run_manifest = json.load(fh)

    os.makedirs(os.path.join(root, "statistics"), exist_ok=True)
    os.makedirs(os.path.join(root, "diagnostics"), exist_ok=True)

    summary = {
        "model_path": run_manifest["model_path"],
        "model": run_manifest,
        "prompt_version": manifest["prompt_version"],
        "sampling_seed": manifest["seed"],
        "context_source_seed": manifest["context_source_seed"],
        "alpha": manifest["alpha"],
        "budget": manifest["budget"],
        "samples_sha256": manifest["samples_sha256"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": {}, "determinism": {},
    }
    det_all = {"n_fallback_pairs": 0, "n_identical_prompt_hash": 0,
               "n_identical_raw_output": 0, "determinism_warnings": 0,
               "determinism_warning_samples": []}
    for ds in DATASETS:
        stats = dataset_statistics(root, ds, args.limit)
        summary["datasets"][ds] = stats
        det_all["n_fallback_pairs"] += stats["determinism"]["n_fallback_pairs"]
        det_all["n_identical_prompt_hash"] += \
            stats["determinism"]["n_identical_prompt_hash"]
        det_all["n_identical_raw_output"] += \
            stats["determinism"]["n_identical_raw_output"]
        det_all["determinism_warnings"] += \
            stats["determinism"]["determinism_warnings"]
        det_all["determinism_warning_samples"].extend(
            stats["determinism"]["determinism_warning_samples"])
        with open(os.path.join(root, "statistics", f"{ds}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(stats, fh, indent=1, ensure_ascii=False)
        with open(os.path.join(root, "diagnostics", f"{ds}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"determinism": stats["determinism"],
                       "parse_failures": stats["parse_failures"],
                       "unpaired_sample_ids": stats["unpaired_sample_ids"],
                       "citation": stats["citation"]}, fh, indent=1,
                      ensure_ascii=False)
        print(json.dumps({
            "dataset": ds, "n_pairs": stats["n_pairs_effective"],
            "delta_macro_f1": round(stats["delta_macro_f1"], 5),
            "ci": [round(stats["bootstrap"]["ci_low"], 5),
                   round(stats["bootstrap"]["ci_high"], 5)],
            "token_reduction": None if
            stats["compression"]["mean_social_token_reduction"] is None else
            round(stats["compression"]["mean_social_token_reduction"], 4),
            "unsupported_citation_rate": stats["citation"]["ms"][
                "unsupported_citation_rate"],
            "gate": stats["gate"], "label": stats["transfer_label"]},
            indent=1))
    summary["determinism"] = det_all
    summary["overall"], summary["recommendation"] = overall_decision(
        summary["datasets"])
    with open(os.path.join(root, "reader_transfer_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    write_report(root, summary)
    print(json.dumps({"overall": summary["overall"],
                      "recommendation": summary["recommendation"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
