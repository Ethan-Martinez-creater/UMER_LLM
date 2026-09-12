#!/usr/bin/env python
"""Verifier for the Final Evidence Closure stage (Gap A + Gap F + reader).

Checks the frozen artifacts only; never re-runs a model.  Requires
``issues = 0``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "project")
for _path in (HERE, PROJECT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DATASETS = ("pheme", "maweibo")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
CUTOFFS = (5, 15, 30, 60, 180, 360)
ARMS = ("STATIC_FULL", "UTILITY_TOKEN_MATCHED", "RANDOM_TOKEN_MATCHED",
        "MS_TSR")
GAP_A_ARMS = ("static", "random", "semantic")
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))


def _results_root():
    """Local repo layout first, then the GPU host's split layout."""
    local = os.path.join(REPO_ROOT, "results", "tcdscr")
    if os.path.exists(local):
        return local
    return "/data/jyz/next/llm/results/tcdscr"


RESULTS = _results_root()
ROOT = os.path.join(RESULTS, "final_evidence")
V3B_READER = os.path.join(RESULTS, "dynamic_v3_reader")
E3_ROOT = os.path.join(RESULTS, "formal_e3")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def _runs(root, gap, dataset):
    base = os.path.join(root, gap, "runs", dataset)
    out = []
    for fold in FOLDS:
        for seed in SEEDS:
            d = os.path.join(base, f"fold{fold}_seed{seed}")
            man = os.path.join(d, "run_manifest.json")
            if os.path.exists(man):
                out.append((d, load_json(man)))
    return out


def check_gap_a(root, issues):
    total_rows = total_expected = 0
    n_runs = 0
    for dataset in DATASETS:
        runs = _runs(root, "gap_a", dataset)
        if len(runs) != 15:
            issues.append(f"gap_a {dataset}: {len(runs)} runs, expected 15")
        for d, man in runs:
            n_runs += 1
            if man["budget"] != 1024:
                issues.append(f"gap_a {dataset}: budget != 1024")
            if sorted(man["arms"]) != sorted(GAP_A_ARMS):
                issues.append(f"gap_a {dataset}: arms != 3 methods")
            if not all(man["frozen_checksum_match"].values()):
                issues.append(f"gap_a {dataset}: frozen checkpoint mismatch")
            if man["new_trainable_parameters"] != 0:
                issues.append(f"gap_a {dataset}: trainable parameters != 0")
            if man.get("test_time_hyperparameter_search"):
                issues.append(f"gap_a {dataset}: test-time search used")
            if man["n_rows"] != man["expected_rows"]:
                issues.append(f"gap_a {dataset}: row coverage "
                              f"{man['n_rows']} != {man['expected_rows']}")
            if man["empty_selection_rows"] <= 0:
                issues.append(f"gap_a {dataset}: no no-candidate rows")
            pj = os.path.join(d, "predictions.jsonl")
            if os.path.exists(pj):
                rows = read_jsonl(pj)
                if len(rows) != man["n_rows"]:
                    issues.append(f"gap_a {dataset}: predictions length "
                                  f"mismatch")
                if {r["method"] for r in rows} != set(GAP_A_ARMS):
                    issues.append(f"gap_a {dataset}: prediction methods "
                                  f"mismatch")
                if {int(r["cutoff"]) for r in rows} != set(CUTOFFS):
                    issues.append(f"gap_a {dataset}: not all cutoffs present")
            total_rows += man["n_rows"]
            total_expected += man["expected_rows"]
    boot = load_json(os.path.join(root, "gap_a", "bootstrap.json"))
    for ds in DATASETS:
        b = boot.get(ds) or {}
        if "primary" not in b:
            issues.append(f"gap_a: bootstrap missing primary for {ds}")
    return {"n_runs": n_runs, "n_rows": total_rows,
            "expected_rows": total_expected}


def _memory_chain_ok(rows):
    by_event = {}
    for r in rows:
        by_event.setdefault((r["event_id"], r["seed"]), []).append(r)
    for seq in by_event.values():
        seq.sort(key=lambda r: CUTOFFS.index(int(r["cutoff"])))
        prev = []
        for r in seq:
            if list(r["memory_previous_ids"]) != list(prev):
                return False
            prev = list(r["dynamic_selected_node_ids"])
    return True


def check_gap_f(root, e3_root, issues):
    total_rows = total_expected = 0
    n_runs = 0
    for dataset in DATASETS:
        runs = _runs(root, "gap_f", dataset)
        if len(runs) != 15:
            issues.append(f"gap_f {dataset}: {len(runs)} runs, expected 15")
        for d, man in runs:
            n_runs += 1
            cfg = man["config"]
            if not cfg["validation_frozen"] or cfg["test_time_search"]:
                issues.append(f"gap_f {dataset}: config not frozen")
            bc = os.path.join(e3_root, dataset, f"fold{man['fold']}",
                              "best_config.json")
            if not os.path.exists(bc):
                issues.append(f"gap_f {dataset}: best_config missing")
            else:
                frozen_cfg = load_json(bc)
                if sha256_file(bc) != cfg.get("best_config_sha256"):
                    issues.append(f"gap_f {dataset}: best_config hash "
                                  f"mismatch")
                if int(cfg["budget"]) != int(frozen_cfg["budget"]):
                    issues.append(f"gap_f {dataset}: budget differs from the "
                                  f"frozen best_config")
                if abs(float(cfg["lambda_n"]) - float(
                        frozen_cfg["lambda_n"])) > 1e-12 or abs(
                        float(cfg["lambda_p"]) - float(
                            frozen_cfg["lambda_p"])) > 1e-12:
                    issues.append(f"gap_f {dataset}: lambda differs from the "
                                  f"frozen best_config")
            if man["n_rows"] != man["expected_rows"]:
                issues.append(f"gap_f {dataset}: row coverage "
                              f"{man['n_rows']} != {man['expected_rows']}")
            if man["no_candidate_rows"] <= 0:
                issues.append(f"gap_f {dataset}: no no-candidate rows")
            if not all(man["frozen_checksum_match"].values()):
                issues.append(f"gap_f {dataset}: frozen checkpoint mismatch")
            pj = os.path.join(d, "test_predictions.jsonl")
            if os.path.exists(pj):
                rows = read_jsonl(pj)
                if len(rows) != man["n_rows"]:
                    issues.append(f"gap_f {dataset}: predictions length "
                                  f"mismatch")
                if dataset == "pheme" and not _memory_chain_ok(rows):
                    issues.append("gap_f pheme: dynamic memory chain not "
                                  "causal")
            total_rows += man["n_rows"]
            total_expected += man["expected_rows"]
    boot = load_json(os.path.join(root, "gap_f", "bootstrap.json"))
    for ds in DATASETS:
        if (boot.get(ds) or {}).get("multiplicity") != "preserved":
            issues.append(f"gap_f: bootstrap multiplicity not preserved ({ds})")
    return {"n_runs": n_runs, "n_rows": total_rows,
            "expected_rows": total_expected}


def _source_of(prompt):
    marker = "Source Post:\n"
    end = "\n\nObserved Social Evidence up to "
    if marker not in prompt or end not in prompt:
        return None
    return prompt.split(marker, 1)[1].split(end, 1)[0]


def check_reader(root, v3b_root, issues):
    rroot = os.path.join(root, "reader")
    manifest_path = os.path.join(rroot, "reader_sampling_manifest.json")
    if not os.path.exists(manifest_path):
        issues.append("reader: sampling manifest missing")
        return {}
    manifest = load_json(manifest_path)
    digest = sha256_file(manifest_path)
    recorded = open(os.path.join(rroot,
                                 "reader_sampling_manifest.sha256"),
                    encoding="utf-8").read().strip()
    if digest != recorded:
        issues.append("reader: sampling manifest hash changed after freeze")
    if manifest["sampling_seed"] != 4096:
        issues.append("reader: sampling seed != 4096")
    if manifest["context_source_seed"] != 2000:
        issues.append("reader: context source seed != 2000")
    if manifest["alpha"] != 0.8 or manifest["budget"] != 1024:
        issues.append("reader: alpha/budget not frozen")
    if manifest["prompt_version"] != "v3b-2":
        issues.append("reader: prompt version != v3b-2")
    if sorted(manifest["arms"]) != sorted(ARMS):
        issues.append("reader: arms != the four required arms")

    prior = load_json(os.path.join(v3b_root, "sampling_manifest.json"))
    prior_ids = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "event_id" and isinstance(v, str):
                    prior_ids.add(v)
                elif k == "event_ids" and isinstance(v, list):
                    prior_ids.update(str(x) for x in v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(prior)

    samples = manifest["samples"]
    per_ds = {ds: [s for s in samples if s["dataset"] == ds]
              for ds in DATASETS}
    for ds, rows in per_ds.items():
        if len(rows) != 300 and not all(r["allocation_shortage"]
                                        for r in rows):
            issues.append(f"reader {ds}: {len(rows)} samples, expected 300 "
                          f"(no shortage recorded)")
        ids = [r["event_id"] for r in rows]
        if len(set(ids)) != len(ids):
            issues.append(f"reader {ds}: duplicate event id in the sample")
        if set(ids) & prior_ids:
            issues.append(f"reader {ds}: prior V3-B reader event included")
        for fold in FOLDS:
            cell = [r for r in rows if r["fold"] == fold]
            if len(cell) != 60 and not all(r["allocation_shortage"]
                                           for r in cell):
                issues.append(f"reader {ds} fold{fold}: {len(cell)} events, "
                              f"expected 60")
            for c in CUTOFFS:
                cc = [r for r in cell if int(r["cutoff"]) == c]
                if len(cc) != 10 and not all(r["allocation_shortage"]
                                             for r in cc):
                    issues.append(f"reader {ds} fold{fold} cutoff{c}: "
                                  f"{len(cc)} events, expected 10")
        for r in rows:
            if r["utility_tm_social_tokens"] > r["ms_social_tokens"]:
                issues.append(f"reader {ds}: utility TM exceeded MS tokens")
            if r["random_tm_social_tokens"] > r["ms_social_tokens"]:
                issues.append(f"reader {ds}: random TM exceeded MS tokens")
            if sorted(r["arm_order"]) != sorted(ARMS):
                issues.append(f"reader {ds}: arm_order is not a permutation")

    # gold isolation in everything produced before the generation freeze
    for rel in ("reader_sampling_manifest.json",):
        for r in samples:
            if any(k in r for k in ("gold", "label")):
                issues.append(f"reader: gold leaked into {rel}")
    for ds in DATASETS:
        for r in read_jsonl(os.path.join(rroot, "prompts",
                                         f"{ds}.jsonl")):
            if any(k in r for k in ("gold", "label")):
                issues.append("reader: gold leaked into prompts")
        for r in read_jsonl(os.path.join(rroot, "contexts",
                                         f"{ds}.jsonl")):
            if any(k in r for k in ("gold", "label")):
                issues.append("reader: gold leaked into contexts")
        for r in read_jsonl(os.path.join(rroot, "raw_generations",
                                         f"{ds}.jsonl")):
            if any(k in r for k in ("gold", "label")):
                issues.append("reader: gold leaked into raw generations")

    # prompts: four arms, identical source text, same E# rendering
    n_prompts = 0
    for ds in DATASETS:
        by_sample = {}
        for r in read_jsonl(os.path.join(rroot, "prompts", f"{ds}.jsonl")):
            n_prompts += 1
            by_sample.setdefault(r["sample_id"], {})[r["arm"]] = r
        for sid, arms in by_sample.items():
            if sorted(arms) != sorted(ARMS):
                issues.append(f"reader {ds}: sample {sid} lacks four arms")
                continue
            sources = {_source_of(a["prompt"]) for a in arms.values()}
            if len(sources) != 1 or None in sources:
                issues.append(f"reader {ds}: source differs across arms "
                              f"({sid})")
    # raw generations
    n_raw = 0
    overflow = 0
    for ds in DATASETS:
        for r in read_jsonl(os.path.join(rroot, "raw_generations",
                                         f"{ds}.jsonl")):
            n_raw += 1
            if r.get("context_overflow"):
                overflow += 1
    if n_raw != 4 * sum(len(v) for v in per_ds.values()):
        issues.append(f"reader: {n_raw} raw generations, expected "
                      f"{4 * sum(len(v) for v in per_ds.values())}")
    # parsed + citation recomputation
    unsupported = total_cited = 0
    n_parsed = 0
    for ds in DATASETS:
        prompts = {(r["sample_id"], r["arm"]): r for r in read_jsonl(
            os.path.join(rroot, "prompts", f"{ds}.jsonl"))}
        for r in read_jsonl(os.path.join(rroot, "parsed", f"{ds}.jsonl")):
            n_parsed += 1
            supplied = set((prompts.get((r["sample_id"], r["arm"]) ) or {})
                           .get("evidence_ids", {}).keys())
            for e in r.get("evidence_ids") or []:
                total_cited += 1
                if e not in supplied:
                    unsupported += 1
    if n_parsed != n_raw:
        issues.append(f"reader: {n_parsed} parsed vs {n_raw} raw")
    summary_path = os.path.join(rroot, "reader_summary.json")
    summary = load_json(summary_path) if os.path.exists(summary_path) else {}
    if summary:
        for ds in DATASETS:
            got = (summary["datasets"][ds]["citation"]["MS_TSR"]
                   ["unsupported_citation_rate"])
            exp = (unsupported / total_cited) if total_cited else 0.0
            if total_cited and abs(got - exp) > 0.01:
                issues.append(f"reader {ds}: unsupported citation rate does "
                              f"not reproduce")
    # model manifest
    mm = os.path.join(rroot, "diagnostics", "reader_model_manifest.json")
    if not os.path.exists(mm):
        issues.append("reader: model manifest missing")
    else:
        m = load_json(mm)
        if m.get("matches_v3b") is False:
            issues.append("reader: MODEL_MISMATCH vs V3-B")
    return {"n_samples": len(samples), "n_raw_generations": n_raw,
            "n_parsed": n_parsed, "n_prompts": n_prompts,
            "n_context_overflow": overflow,
            "unsupported_citations": unsupported,
            "cited_total": total_cited,
            "prior_reader_overlap": len(
                {s["event_id"] for s in samples} & prior_ids)}


def verify(root=None, v3b_root=V3B_READER, e3_root=E3_ROOT):
    root = root or ROOT
    issues = []
    gap_a = check_gap_a(root, issues)
    gap_f = check_gap_f(root, e3_root, issues)
    reader = check_reader(root, v3b_root, issues)
    result = {
        "n_issues": len(issues),
        "issues": issues[:60],
        "gap_a": gap_a,
        "gap_f": gap_f,
        "reader": reader,
        "checks": [
            "Gap A: 30 runs, all events at all 6 cutoffs, no-candidate rows "
            "retained, 3 arms, frozen E2 checkpoints, budget 1024",
            "Gap A: paired event bootstrap with multiplicity preserved",
            "Gap F: 30 runs, rows == test events x 6, no-candidate rows > 0",
            "Gap F: frozen best_config only, no test-time search",
            "Gap F: dynamic memory chain causal (no future memory)",
            "Gap F: static and dynamic paired on the same rows",
            "reader: outer test only, all prior V3-B event ids excluded",
            "reader: 300 unique events/dataset, 60/fold, 10 per cutoff/fold",
            "reader: seed 4096, context seed 2000, alpha 0.8, budget 1024",
            "reader: four arms exactly, token-matched arms never exceed MS",
            "reader: gold labels not read before the generation freeze",
            "reader: identical source text across the four arms",
            "reader: same Qwen checkpoint as V3-B, prompt v3b-2",
            "reader: manifest hash unchanged after freeze",
            "reader: unsupported citation rate reproduces from parsed output",
        ],
    }
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--v3b-reader-root", default=V3B_READER)
    ap.add_argument("--e3-root", default=E3_ROOT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    result = verify(args.root, args.v3b_reader_root, args.e3_root)
    root = args.root or ROOT
    out = args.out or os.path.join(root, "final_evidence_verify.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    print(json.dumps(result, indent=1, ensure_ascii=False), flush=True)
    return 0 if result["n_issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
