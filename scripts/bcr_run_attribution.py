#!/usr/bin/env python
"""M1-E Task B/C — run the frozen feature-group ablations (SERVER).

Trains the four variants this audit owns (D0, D2, D3, D5) through the exact
M1 LORO protocol and reuses the frozen D1 (= B4) and D4 (= B5) results and
predictions verbatim. Everything is a POST-HOC DIAGNOSTIC: no gate is
defined, no frozen model is modified, no reader is called.

Output: ``results/bcr_utility_v1/m1e/attribution/<dataset>.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "project"))
sys.path.insert(0, str(REPO / "scripts"))

from bcr_utility.attribution import feature_ablation as fa  # noqa: E402
from bcr_utility.config import protocol as P  # noqa: E402
from bcr_utility.evaluation.unseen_reader import build_feature_table  # noqa: E402


class AttributionRunRefused(RuntimeError):
    """Raised when the attribution run cannot proceed as specified."""


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, payload) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return {"path": path, "bytes": os.path.getsize(path),
            "sha256": _sha_file(path)}


def load_inputs(repo_root, dataset):
    index = _load_json(os.path.join(P.bootstrap_dir(repo_root),
                                    P.ATOMIC_INDEX_FILENAME))
    entries = (index.get("datasets") or {}).get(dataset, {}).get("entries")
    if not entries:
        raise AttributionRunRefused(f"{dataset}: atomic index missing")
    fdir = P.m1_path(repo_root, "features")
    e0 = _load_jsonl(os.path.join(fdir, f"e0_{dataset}.jsonl"))
    e1 = _load_jsonl(os.path.join(fdir, f"e1_{dataset}.jsonl"))
    e2 = _load_jsonl(os.path.join(fdir, f"e2_{dataset}.jsonl"))
    e3 = _load_jsonl(os.path.join(fdir, f"e3_{dataset}.jsonl"))
    split = _load_json(os.path.join(
        P.historical_root(repo_root), "manifests", dataset,
        "event_split.json"))
    fingerprints = _load_json(P.m1_path(repo_root, "fingerprints",
                                        "fingerprints.json"))
    feature_table = build_feature_table(dataset, entries, e0, e1, e2,
                                        e3_rows=e3)
    return entries, feature_table, split, fingerprints


def run_dataset(repo_root, dataset) -> dict:
    t0 = time.time()
    entries, table, split, fingerprints = load_inputs(repo_root, dataset)
    print(f"[m1e] {dataset}: {len(entries)} atomic keys, "
          f"{len(table)} feature rows")

    results = {}
    for variant in P.ATTRIBUTION_TRAINED:
        results[variant] = fa.run_variant(dataset, variant, entries, table,
                                          split, fingerprints)
        f1 = {h: r["eval_metrics"]["macro_f1"]
              for h, r in results[variant]["rotations"].items()}
        print(f"[m1e] {dataset}/{variant}: "
              + " ".join(f"{h}={v:.4f}" for h, v in f1.items())
              + f" ({time.time() - t0:.1f}s)")

    # frozen D1 (=B4) and D4 (=B5): reused verbatim, never redefined
    frozen_predictions = {}
    reused = {}
    for variant, model_kind in P.ATTRIBUTION_REUSED.items():
        frozen_predictions[variant] = fa.load_frozen_predictions(
            repo_root, dataset, model_kind)
        view = fa.frozen_m1_view(repo_root, dataset, model_kind)
        metrics = {held: rot["eval_metrics"]
                   for held, rot in view["rotations"].items()}
        reused[variant] = fa.frozen_variant_summary(view, variant, model_kind,
                                                    metrics)
        print(f"[m1e] {dataset}/{variant}: reused frozen {model_kind} "
              + " ".join(f"{h}={v:.4f}"
                         for h, v in view["per_reader_macro_f1"].items()))

    baseline = results["D0_Z"]
    comparisons = {}
    for variant in P.ATTRIBUTION_VARIANTS:
        if variant == "D0_Z":
            continue
        if variant in results:
            comparisons[variant] = fa.compare_to_baseline(
                results[variant], baseline)
        else:
            comparisons[variant] = fa.compare_frozen_to_baseline(
                frozen_predictions[variant], baseline)
    # Task C: the fingerprint's incremental value with E3 already present
    fingerprint_increment = fa.compare_frozen_to_baseline(
        frozen_predictions["D4_Z_F_S_C"], results["D5_Z_S_C"])
    fingerprint_increment["baseline"] = "D5_Z_S_C"
    fingerprint_increment["variant"] = "D4_Z_F_S_C"

    # secondary metrics per variant, per reader
    secondary = {}
    for variant, result in results.items():
        secondary[variant] = {
            held: {k: rot["eval_metrics"][k]
                   for k in ("macro_f1", "utility_spearman", "mae",
                             "harmful_auprc", "harmful_f1", "helpful_auprc")}
            for held, rot in result["rotations"].items()}
    for variant, model_kind in P.ATTRIBUTION_REUSED.items():
        view = fa.frozen_m1_view(repo_root, dataset, model_kind)
        secondary[variant] = {
            held: {k: rot["eval_metrics"][k]
                   for k in ("macro_f1", "utility_spearman", "mae",
                             "harmful_auprc", "harmful_f1", "helpful_auprc")}
            for held, rot in view["rotations"].items()}

    payload = {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1e_attribution",
        "dataset": dataset,
        "scope": "post_hoc_diagnostic_only",
        "defines_gate": False,
        "variants": {v: fa.summarise_variant(results[v])
                     for v in P.ATTRIBUTION_TRAINED},
        "reused_variants": reused,
        "comparisons_vs_D0": comparisons,
        "fingerprint_increment_D4_vs_D5": fingerprint_increment,
        "secondary_metrics": secondary,
        "seconds": time.time() - t0,
    }
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO))
    parser.add_argument("--datasets", nargs="+", choices=P.DATASETS,
                        default=list(P.DATASETS))
    args = parser.parse_args(argv)
    written = {}
    for dataset in args.datasets:
        payload = run_dataset(args.repo_root, dataset)
        info = _write_json(P.m1e_path(args.repo_root,
                                      P.M1E_ATTRIBUTION_DIRNAME,
                                      f"{dataset}.json"), payload)
        written[dataset] = info
        print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
