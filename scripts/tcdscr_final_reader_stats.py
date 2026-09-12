#!/usr/bin/env python
"""Final Evidence Closure — Gap B + Gap C reader statistics.

Gold labels are merged here only, after all generations were frozen.  Reads
the frozen reader manifest, prompts, raw generations and parsed outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "project")
for _path in (HERE, PROJECT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DATASETS = ("pheme", "maweibo")
ARMS = ("STATIC_FULL", "UTILITY_TOKEN_MATCHED", "RANDOM_TOKEN_MATCHED",
        "MS_TSR")
N_ITER = 10000
BOOT_SEED = 4096
OUT_ROOT = "/data/jyz/next/llm/results/tcdscr/final_evidence/reader"


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)


def label_to_int(label):
    return 1 if str(label).upper() == "RUMOR" else 0


def counts4(gold, pred):
    return np.array([1 if gold == 1 and pred == 1 else 0,
                     1 if gold == 0 and pred == 1 else 0,
                     1 if gold == 1 and pred == 0 else 0,
                     1 if gold == 0 and pred == 0 else 0], dtype=np.int64)


def macro_f1_from_counts(c):
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]

    def f1(a, b, d):
        pr = np.divide(a, a + b, out=np.zeros_like(a, dtype=float),
                       where=(a + b) > 0)
        re = np.divide(a, a + d, out=np.zeros_like(a, dtype=float),
                       where=(a + d) > 0)
        return np.divide(2 * pr * re, pr + re, out=np.zeros_like(pr),
                         where=(pr + re) > 0)
    return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2.0


def full_metrics(counts):
    tp, fp, fn, tn = (int(counts[0]), int(counts[1]), int(counts[2]),
                      int(counts[3]))
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0

    def f1(a, b, d):
        pr = a / (a + b) if a + b else 0.0
        re = a / (a + d) if a + d else 0.0
        return 2 * pr * re / (pr + re) if pr + re else 0.0
    mf1 = (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2.0
    s1, s0 = tp + fn, tn + fp
    wf1 = (f1(tp, fp, fn) * s1 + f1(tn, fn, fp) * s0) / max(s1 + s0, 1)
    return {"n": n, "accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1,
            "rumor_f1": f1(tp, fp, fn), "confusion": {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    obs = pmf(k)
    return float(min(1.0, sum(pmf(i) for i in range(n + 1)
                              if pmf(i) <= obs + 1e-12)))


def ece(confs, corrects, bins=10):
    confs = np.asarray(confs, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    total = len(confs)
    if total == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    val = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs >= lo) & (confs < hi if i < bins - 1 else confs <= hi)
        if mask.sum() == 0:
            continue
        val += (mask.sum() / total) * abs(corrects[mask].mean()
                                          - confs[mask].mean())
    return float(val)


def load_gold(dataset):
    from tcdscr.config.schema import config_from_env
    from tcdscr.data.temporal_split import build_primary_fold_split
    from tcdscr_common import event_label_registry, load_split_events
    cfg = config_from_env(dataset)
    registry = event_label_registry(dataset, cfg)
    labels = {}
    folds = {}
    for fold in range(5):
        split = build_primary_fold_split(registry, fold, seed=3090)
        events = load_split_events(dataset, cfg, split)
        for ev in events["test"]:
            labels[ev["event_id"]] = int(ev["label"])
            folds.setdefault(ev["event_id"], fold)
    return labels, folds


def dataset_stats(out_root, dataset, manifest_by_id, prompts, parsed, raw,
                  gold, folds):
    samples = [m for m in manifest_by_id.values() if m["dataset"] == dataset]
    kept = [m for m in samples if (m["sample_id"], "MS_TSR") in parsed]
    n_missing = len(samples) - len(kept)

    counts = {a: [] for a in ARMS}
    meta = []
    for m in kept:
        sid = m["sample_id"]
        g = gold.get(m["event_id"])
        for a in ARMS:
            counts[a].append(counts4(g, label_to_int(
                parsed[(sid, a)]["label"])))
        meta.append({"sample_id": sid, "fold": m["fold"],
                     "cutoff": m["cutoff"], "gold": g,
                     "pred": {a: label_to_int(parsed[(sid, a)]["label"])
                              for a in ARMS}})
    counts = {a: np.stack(v) for a, v in counts.items()}

    detection = {a: full_metrics(counts[a].sum(axis=0)) for a in ARMS}
    # stratified paired bootstrap over (fold, cutoff) cells
    cells = {}
    for i, m in enumerate(meta):
        cells.setdefault((m["fold"], m["cutoff"]), []).append(i)
    rng = np.random.default_rng(BOOT_SEED)
    idx_by_iter = []
    for _ in range(N_ITER):
        parts = []
        for key in sorted(cells):
            idx = cells[key]
            parts.append(rng.choice(idx, size=len(idx), replace=True))
        idx_by_iter.append(np.concatenate(parts))

    def boot_delta(a, b):
        out = np.empty(N_ITER)
        for i, idx in enumerate(idx_by_iter):
            ca = counts[a][idx].sum(axis=0)
            cb = counts[b][idx].sum(axis=0)
            out[i] = float(macro_f1_from_counts(ca)
                           - macro_f1_from_counts(cb))
        return out
    comparisons = {}
    for name, a, b in (("ms_vs_static", "MS_TSR", "STATIC_FULL"),
                       ("ms_vs_utility_tm", "MS_TSR",
                        "UTILITY_TOKEN_MATCHED"),
                       ("ms_vs_random_tm", "MS_TSR", "RANDOM_TOKEN_MATCHED"),
                       ("utility_tm_vs_random_tm", "UTILITY_TOKEN_MATCHED",
                        "RANDOM_TOKEN_MATCHED"),
                       ("utility_tm_vs_static", "UTILITY_TOKEN_MATCHED",
                        "STATIC_FULL"),
                       ("random_tm_vs_static", "RANDOM_TOKEN_MATCHED",
                        "STATIC_FULL")):
        delta = boot_delta(a, b)
        lo, hi = np.percentile(delta, [2.5, 97.5])
        comparisons[name] = {
            "point": float(detection[a]["macro_f1"]
                           - detection[b]["macro_f1"]),
            "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}
    # paired outcomes + exact McNemar for the two primary comparisons
    paired = {}
    for name, a, b in (("ms_vs_static", "MS_TSR", "STATIC_FULL"),
                       ("ms_vs_utility_tm", "MS_TSR",
                        "UTILITY_TOKEN_MATCHED")):
        both = a_ok = b_ok = both_wrong = 0
        for m in meta:
            pa, pb = m["pred"][a] == m["gold"], m["pred"][b] == m["gold"]
            if pa and pb:
                both += 1
            elif pa and not pb:
                a_ok += 1
            elif pb and not pa:
                b_ok += 1
            else:
                both_wrong += 1
        discordant = a_ok + b_ok
        paired[name] = {
            "both_correct": both, "both_wrong": both_wrong,
            "a_correct_b_wrong": a_ok, "a_wrong_b_correct": b_ok,
            "disagreement": discordant,
            "net_correction_gain": a_ok - b_ok,
            "disagreement_rate": (discordant / len(meta) if meta else None),
            "mcnemar_exact_p": binom_two_sided(min(a_ok, b_ok), discordant)}
    # citation grounding
    citation = {}
    for a in ARMS:
        cited = unsupported = no_cite = 0
        cited_counts = []
        for m in kept:
            sid = m["sample_id"]
            row = parsed[(sid, a)]
            supplied = set(prompts[(sid, a)]["evidence_ids"].keys())
            ids = row.get("evidence_ids") or []
            if not ids:
                no_cite += 1
            cited += len(ids)
            unsupported += sum(1 for e in ids if e not in supplied)
            cited_counts.append(len(ids))
        citation[a] = {
            "n_runs": len(kept), "n_cited_total": cited,
            "valid_citation_rate": (1.0 - unsupported / cited) if cited
            else 1.0,
            "unsupported_citation_count": unsupported,
            "unsupported_citation_rate": (unsupported / cited) if cited
            else 0.0,
            "no_citation_rate": no_cite / len(kept) if kept else None,
            "mean_cited_evidence_count": (float(np.mean(cited_counts))
                                          if cited_counts else None)}
    # confidence / ECE
    confidence = {}
    for a in ARMS:
        confs = [float(parsed[(m["sample_id"], a)]["confidence"])
                 for m in kept]
        correct = [1.0 if m["pred"][a] == m["gold"] else 0.0 for m in meta]
        confs = confs[:len(correct)]
        conf_arr = np.asarray(confs, dtype=float)
        corr_arr = np.asarray(correct, dtype=float)
        confidence[a] = {
            "mean": float(conf_arr.mean()) if len(conf_arr) else None,
            "mean_correct": (float(conf_arr[corr_arr == 1].mean())
                             if (corr_arr == 1).any() else None),
            "mean_wrong": (float(conf_arr[corr_arr == 0].mean())
                           if (corr_arr == 0).any() else None),
            "ece": ece(conf_arr, corr_arr, bins=10), "ece_bins": 10}
    # tokens / latency
    tokens = {}
    static_tokens = np.array([m["static_social_tokens"] for m in kept],
                             dtype=float)
    for a in ARMS:
        arr = np.array([m[{"STATIC_FULL": "static_social_tokens",
                           "MS_TSR": "ms_social_tokens",
                           "UTILITY_TOKEN_MATCHED":
                               "utility_tm_social_tokens",
                           "RANDOM_TOKEN_MATCHED":
                               "random_tm_social_tokens"}[a]]
                        for m in kept], dtype=float)
        eligible = static_tokens > 0
        tokens[a] = {
            "mean_social_tokens": float(arr.mean()) if len(arr) else None,
            "mean_reduction_vs_static": (
                float((1 - arr[eligible] / static_tokens[eligible]).mean())
                if eligible.any() else None),
            "n_eligible": int(eligible.sum())}
    lat = [raw[(m["sample_id"], a)]["latency_sec"] for m in kept
           for a in ARMS if (m["sample_id"], a) in raw]
    gen = [raw[(m["sample_id"], a)]["generated_tokens"] for m in kept
           for a in ARMS if (m["sample_id"], a) in raw]
    inp = [prompts[(m["sample_id"], a)]["total_input_tokens"] for m in kept
           for a in ARMS if (m["sample_id"], a) in prompts]
    cost = {
        "mean_latency_sec": float(np.mean(lat)) if lat else None,
        "median_latency_sec": float(np.median(lat)) if lat else None,
        "total_input_tokens": int(sum(inp)),
        "total_generated_tokens": int(sum(gen)),
        "mean_input_tokens_per_generation": float(np.mean(inp))
        if inp else None,
        "monetary_cost": None,
        "monetary_cost_note": "no price is fabricated; only token/latency "
                              "cost is reported"}
    gaps = np.array([m["utility_token_gap"] for m in kept], dtype=float)
    gaps_r = np.array([m["random_token_gap"] for m in kept], dtype=float)
    matching = {
        "utility_token_gap": {
            "mean": float(gaps.mean()), "median": float(np.median(gaps)),
            "p90": float(np.percentile(gaps, 90)),
            "min": float(gaps.min()), "max": float(gaps.max())},
        "random_token_gap": {
            "mean": float(gaps_r.mean()),
            "median": float(np.median(gaps_r)),
            "p90": float(np.percentile(gaps_r, 90)),
            "min": float(gaps_r.min()), "max": float(gaps_r.max())},
        "never_overshoot": bool(gaps.min() >= 0 and gaps_r.min() >= 0)}

    # statuses
    ms_static = comparisons["ms_vs_static"]
    reduction = tokens["MS_TSR"]["mean_reduction_vs_static"]
    unsup = citation["MS_TSR"]["unsupported_citation_rate"]
    gate_ok = (ms_static["point"] >= -0.005 and (reduction or 0) >= 0.30
               and unsup <= 0.01)
    gate = "HELD_OUT_TRANSFER_PASS" if gate_ok else "HELD_OUT_TRANSFER_FAIL"
    if gate_ok:
        transfer = ("STRONG_TRANSFER" if ms_static["ci_low"] >= -0.01
                    else "WEAK_TRANSFER")
    else:
        transfer = None
    d_util = comparisons["ms_vs_utility_tm"]["point"]
    if d_util >= 0.005:
        interp = "MS_SELECTION_ADVANTAGE"
    elif d_util <= -0.005:
        interp = "SIMPLE_COMPRESSION_BETTER"
    else:
        interp = "PRACTICALLY_SIMILAR"
    def group_metrics(key_fn):
        out = {}
        for key in sorted({key_fn(m) for m in meta}):
            idx = [i for i, m in enumerate(meta) if key_fn(m) == key]
            out[str(key)] = {a: full_metrics(counts[a][idx].sum(axis=0))
                             for a in ARMS}
        return out
    per_fold = group_metrics(lambda m: m["fold"])
    per_cutoff = group_metrics(lambda m: m["cutoff"])
    return {
        "dataset": dataset,
        "n_samples": len(samples), "n_evaluated": len(kept),
        "n_missing_generations": n_missing,
        "detection": detection, "comparisons": comparisons,
        "per_fold": per_fold, "per_cutoff": per_cutoff,
        "paired_outcomes": paired, "citation": citation,
        "confidence": confidence, "tokens": tokens, "cost": cost,
        "token_matching": matching,
        "gate": gate, "transfer_label": transfer,
        "gate_conditions": {
            "reader_non_inferiority": bool(ms_static["point"] >= -0.005),
            "context_reduction": bool((reduction or 0) >= 0.30),
            "grounding_integrity": bool(unsup <= 0.01)},
        "ms_vs_utility_tm_interpretation": interp,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args(argv)
    manifest = read_json(os.path.join(args.out_root,
                                      "reader_sampling_manifest.json"))
    manifest_by_id = {m["sample_id"]: m for m in manifest["samples"]}
    summary = {"stage": "final_evidence_reader", "arms": list(ARMS),
               "prompt_version": manifest["prompt_version"],
               "sampling_seed": manifest["sampling_seed"],
               "context_source_seed": manifest["context_source_seed"],
               "alpha": manifest["alpha"], "budget": manifest["budget"],
               "bootstrap": {"iterations": N_ITER, "seed": BOOT_SEED,
                             "stratified": "fold x cutoff cells",
                             "multiplicity": "preserved"},
               "datasets": {}}
    for dataset in args.datasets.split(","):
        prompts = {(r["sample_id"], r["arm"]): r for r in read_jsonl(
            os.path.join(args.out_root, "prompts", f"{dataset}.jsonl"))}
        parsed = {(r["sample_id"], r["arm"]): r for r in read_jsonl(
            os.path.join(args.out_root, "parsed", f"{dataset}.jsonl"))}
        raw = {(r["sample_id"], r["arm"]): r for r in read_jsonl(
            os.path.join(args.out_root, "raw_generations",
                         f"{dataset}.jsonl"))}
        gold, folds = load_gold(dataset)
        summary["datasets"][dataset] = dataset_stats(
            args.out_root, dataset, manifest_by_id, prompts, parsed, raw,
            gold, folds)
    write_json(os.path.join(args.out_root, "reader_summary.json"), summary)
    print(json.dumps({ds: {"gate": v["gate"],
                           "transfer": v["transfer_label"],
                           "ms_vs_static": v["comparisons"]["ms_vs_static"],
                           "interp": v["ms_vs_utility_tm_interpretation"]}
                      for ds, v in summary["datasets"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
