#!/usr/bin/env python
"""M1-E Task A/D/E/F — frozen-evidence pinning and read-only audits (LOCAL).

``pins``          fail-closed git+SHA256 pinning of the M1 evidence (Task A)
``shift``         Ma-Weibo vs PHEME E3 distribution + association (Task D)
``concentration`` reader/cutoff/class decomposition of B5-B0 (Task E)
``b3``            B3 same-evidence cross-reader contract audit (Task F)
``all``           every audit above

No reader is called, no model is trained, no frozen artifact is written.
All outputs land under ``results/bcr_utility_v1/m1e/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "project"))
sys.path.insert(0, str(REPO / "scripts"))

from bcr_utility.attribution import b3_contract, concentration  # noqa: E402
from bcr_utility.attribution import dataset_shift as ds  # noqa: E402
from bcr_utility.attribution import evidence_pins as pins  # noqa: E402
from bcr_utility.attribution import feature_ablation as fa  # noqa: E402
from bcr_utility.config import protocol as P  # noqa: E402
from bcr_utility.evaluation import utility_metrics as um  # noqa: E402


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


# --------------------------------------------------------------------------
def task_pins(repo_root) -> dict:
    return pins.pin_m1_evidence(repo_root)


def task_shift(repo_root) -> dict:
    e3 = {}
    for dataset in P.DATASETS:
        path = P.m1_path(repo_root, "features", f"e3_{dataset}.jsonl")
        e3[dataset] = _load_jsonl(path)
    by_cell = {dataset: ds.e3_by_reader_cutoff(rows)
               for dataset, rows in e3.items()}
    report = {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1e_dataset_shift",
        "scope": "post_hoc_diagnostic_only",
        "metric": "wasserstein_1d",
        "distributions": ds.distribution_report(by_cell),
        "shift": ds.shift_report(by_cell),
        "association": {},
        "caveat": ("Association only, never causation. The utility_eval "
                   "association is a post-hoc diagnostic and was not used to "
                   "retrain, re-select or re-tune anything."),
    }
    for dataset in P.DATASETS:
        index = _load_json(os.path.join(P.bootstrap_dir(repo_root),
                                        P.ATOMIC_INDEX_FILENAME))
        entries = (index.get("datasets") or {}).get(dataset, {}).get("entries")
        split = _load_json(os.path.join(
            P.historical_root(repo_root), "manifests", dataset,
            "event_split.json"))
        report["association"][dataset] = ds.association_report(
            e3[dataset], entries, split)
    return report


def task_concentration(repo_root) -> dict:
    out = {"protocol": P.PROTOCOL_VERSION,
           "stage": "m1e_concentration",
           "scope": "post_hoc_diagnostic_only",
           "uses_existing_predictions_only": True,
           "datasets": {}}
    for dataset in P.DATASETS:
        out["datasets"][dataset] = {}
        for variant, primary in (("D4_Z_F_S_C", P.MODEL_B5),
                                 ("D1_Z_F", P.MODEL_B4)):
            try:
                rows = concentration.load_paired_predictions(
                    repo_root, dataset, primary, P.MODEL_B0)
            except concentration.ConcentrationRefused as exc:
                out["datasets"][dataset][variant] = {"error": str(exc)}
                continue
            out["datasets"][dataset][variant] = concentration.concentration_report(
                rows, primary, P.MODEL_B0)
    return out


def task_b3(repo_root) -> dict:
    out = {"protocol": P.PROTOCOL_VERSION,
           "stage": "m1e_b3_contract",
           "scope": "post_hoc_diagnostic_only",
           "datasets": {}}
    for dataset in P.DATASETS:
        out["datasets"][dataset] = b3_contract.audit(repo_root, dataset)
    return out


_METRIC_KEYS = ("macro_f1", "utility_spearman", "mae", "harmful_auprc",
                "harmful_f1", "helpful_auprc")


def task_vs_b0(repo_root) -> dict:
    """Every variant against the frozen M1 ``B0`` baseline (post-hoc).

    The M1 gate compares B4/B5 with B0, so this is the comparison that shows
    what the attribution variants would have scored on the *frozen* gate
    baseline. It is diagnostic only and defines no gate.
    """
    out = {"protocol": P.PROTOCOL_VERSION,
           "stage": "m1e_vs_frozen_b0",
           "scope": "post_hoc_diagnostic_only",
           "defines_gate": False,
           "baseline": P.MODEL_B0,
           "bootstrap": {"unit": P.BOOTSTRAP_UNIT,
                         "iterations": P.BOOTSTRAP_ITERATIONS,
                         "seed": P.BOOTSTRAP_SEED},
           "datasets": {}}
    for dataset in P.DATASETS:
        baseline = fa.load_frozen_b0_baseline(repo_root, dataset)
        per_variant = {}
        for variant in P.ATTRIBUTION_VARIANTS:
            try:
                frozen = fa.load_variant_predictions(repo_root, dataset,
                                                     variant)
            except fa.AttributionRefused as exc:
                per_variant[variant] = {"error": str(exc)}
                continue
            comparison = fa.compare_frozen_to_baseline(frozen, baseline)
            comparison["variant"] = variant
            comparison["per_reader_metrics"] = {
                held: {k: um.evaluate_predictions(entry["eval_rows"],
                                                  entry["prediction"])[k]
                       for k in _METRIC_KEYS}
                for held, entry in frozen.items()}
            per_variant[variant] = comparison
        out["datasets"][dataset] = per_variant
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True,
                        choices=("pins", "shift", "concentration", "b3",
                                 "vs_b0", "all"))
    parser.add_argument("--repo-root", default=str(REPO))
    args = parser.parse_args(argv)
    tasks = {"pins": (task_pins, P.M1E_EVIDENCE_PINS_FILENAME),
             "shift": (task_shift, P.M1E_SHIFT_FILENAME),
             "concentration": (task_concentration,
                               P.M1E_CONCENTRATION_FILENAME),
             "b3": (task_b3, P.M1E_B3_CONTRACT_FILENAME),
             "vs_b0": (task_vs_b0, P.M1E_VS_B0_FILENAME)}
    selected = tasks if args.task == "all" else {args.task: tasks[args.task]}
    written = {}
    for name, (fn, filename) in selected.items():
        payload = fn(args.repo_root)
        info = _write_json(P.m1e_path(args.repo_root, filename), payload)
        written[name] = info
        print(f"[m1e] {name}: {info['path']} ({info['bytes']} B)")
    print(json.dumps(written, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
