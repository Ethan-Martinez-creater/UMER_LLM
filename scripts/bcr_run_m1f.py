#!/usr/bin/env python
"""M1-F — run the strong-baseline & validity audit (SERVER).

Bounded falsification round over frozen artifacts only:

1. fail-closed pin of every frozen input (``evidence_pins.json``);
2. task-validity audit (``TASK_VALIDITY.md`` / ``task_validity.json``);
3. the S0–S4 strong-baseline ladder per dataset (``baselines/``);
4. Task C/D active signed audit (``active_signed_audit.json``);
5. Task E within-snapshot evidence discrimination
   (``within_snapshot_audit.json``);
6. Task F fixed-threshold diagnostics (``robustness_diagnostic.json``);
7. Task G PHEME diagnostic (``pheme_diagnostic.json``);
8. the M1-F decision (``M1F_VERDICT.json``) and the report
   (``M1F_REPORT.md``).

No reader is loaded, no utility label is generated, no E0/E1/E2/E3 cache is
regenerated and nothing under ``m1/``, ``m1e/`` or the historical namespaces is
written.
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

from bcr_utility.config import protocol as P          # noqa: E402
from bcr_utility.m1f import (analysis, baselines, pins,  # noqa: E402
                             report as m1f_report, task_validity)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: str, payload) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return {"path": path, "bytes": os.path.getsize(path),
            "sha256": _sha256_file(path)}


def _write_text(path: str, text: str) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return {"path": path, "bytes": os.path.getsize(path),
            "sha256": _sha256_file(path)}


def run(repo_root: str, datasets, iterations: int, seed: int,
        environment: str = "") -> dict:
    t0 = time.time()
    out_dir = P.m1f_dir(repo_root)
    os.makedirs(out_dir, exist_ok=True)
    written = {}

    # 1 — fail-closed pins
    pin = pins.pin_m1f_inputs(repo_root)
    written[P.M1F_EVIDENCE_PINS_FILENAME] = _write_json(
        P.m1f_path(repo_root, P.M1F_EVIDENCE_PINS_FILENAME), pin)
    print(f"[m1f] pinned {pin['n_artifacts']} artifacts across "
          f"{len(pin['groups'])} groups at {pin['commit'][:12]}")

    # 2 — Task A
    validity = task_validity.task_validity(repo_root)
    written[P.M1F_TASK_VALIDITY_FILENAME] = _write_json(
        P.m1f_path(repo_root, P.M1F_TASK_VALIDITY_FILENAME), validity)
    written[P.M1F_TASK_VALIDITY_REPORT_FILENAME] = _write_text(
        P.m1f_path(repo_root, P.M1F_TASK_VALIDITY_REPORT_FILENAME),
        task_validity.task_validity_markdown(validity))
    print(f"[m1f] task validity: utility is rumour-label agreement utility "
          f"(verification utility = "
          f"{validity['utility_meaning']['is_verification_utility']})")

    # 3 — Task B ladder
    base_dir = os.path.join(out_dir, P.M1F_BASELINES_DIRNAME)
    base_payloads, results_by_dataset, comparators = {}, {}, {}
    for dataset in datasets:
        payload, results, comparator = baselines.run_dataset(
            repo_root, dataset, base_dir)
        base_payloads[dataset] = payload
        results_by_dataset[dataset] = results
        comparators[dataset] = comparator
        written[f"{P.M1F_BASELINES_DIRNAME}/{dataset}.json"] = _write_json(
            os.path.join(base_dir, f"{dataset}.json"), payload)
        f1 = {v: payload["variants"][v]["per_reader_macro_f1"]
              for v in P.M1F_VARIANTS}
        print(f"[m1f] {dataset}: comparator={comparator['selected']} "
              f"({time.time() - t0:.1f}s)")
        for variant in P.M1F_VARIANTS:
            print(f"[m1f]   {dataset}/{variant}: "
                  + " ".join(f"{h}={v:.4f}" for h, v in f1[variant].items()))

    # 4 — alignment and analysis
    aligned = {ds: analysis.align_rotations(results_by_dataset[ds])
               for ds in datasets}
    primary_dataset = P.PRIMARY_DATASET
    if primary_dataset not in datasets:
        raise SystemExit(f"primary dataset {primary_dataset} not run")
    comparisons = {ds: analysis.primary_comparison(
        aligned[ds], P.M1F_S4, comparators[ds]["selected"], iterations, seed)
        for ds in datasets}
    bundles = {ds: {v: analysis.variant_bundle(aligned[ds], v)
                    for v in P.M1F_VARIANTS} for ds in datasets}
    print(f"[m1f] primary comparison computed ({time.time() - t0:.1f}s)")

    within = {ds: analysis.within_snapshot_audit(aligned[ds], ds, iterations,
                                                 seed) for ds in datasets}
    print(f"[m1f] within-snapshot audit computed ({time.time() - t0:.1f}s)")

    active_payload = {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1f_active_signed_audit",
        "scope": "task_c_overall_and_task_d_active_signed",
        "primary_dataset": primary_dataset,
        "official_threshold": P.M1F_OFFICIAL_THRESHOLD,
        "primary": comparisons[primary_dataset],
        "variants": bundles[primary_dataset],
        "datasets": {
            ds: {"diagnostic_only": ds != primary_dataset,
                 "primary_comparison": comparisons[ds],
                 "variants": bundles[ds]}
            for ds in datasets},
    }
    written[P.M1F_ACTIVE_AUDIT_FILENAME] = _write_json(
        P.m1f_path(repo_root, P.M1F_ACTIVE_AUDIT_FILENAME), active_payload)

    within_primary = dict(within[primary_dataset])
    within_primary["datasets"] = {
        ds: {k: v for k, v in within[ds].items()
             if k not in ("variants", "per_snapshot")}
        for ds in datasets}
    written[P.M1F_WITHIN_SNAPSHOT_FILENAME] = _write_json(
        P.m1f_path(repo_root, P.M1F_WITHIN_SNAPSHOT_FILENAME), within_primary)

    # 5 — Task F diagnostics (primary dataset)
    diagnostics = {f"{t:.2f}": {} for t in P.M1F_THRESHOLDS}
    for variant in P.M1F_VARIANTS:
        entry = bundles[primary_dataset][variant]
        for key, record in entry["threshold_diagnostics"].items():
            diagnostics[key][variant] = record
    robustness = {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1f_robustness_diagnostic",
        "dataset": primary_dataset,
        "official_threshold": P.M1F_OFFICIAL_THRESHOLD,
        "thresholds": list(P.M1F_THRESHOLDS),
        "threshold_selected": False,
        "diagnostics": diagnostics,
        "continuous_utility": {
            v: {"spearman": bundles[primary_dataset][v]["overall"]
                ["utility_spearman"],
                "mae": bundles[primary_dataset][v]["overall"]["mae"]}
            for v in P.M1F_VARIANTS},
        "activity": {
            "gold_activity_rate": {v: bundles[primary_dataset][v]
                                   ["activity_rate_gold"]
                                   for v in P.M1F_VARIANTS},
            "predicted_activity_rate": {v: bundles[primary_dataset][v]
                                        ["activity_rate_predicted"]
                                        for v in P.M1F_VARIANTS},
            "near_boundary_rate": {v: bundles[primary_dataset][v]
                                   ["near_boundary_predicted"]
                                   for v in P.M1F_VARIANTS},
            "gold_near_boundary_rate": {v: bundles[primary_dataset][v]
                                        ["near_boundary_gold"]
                                        for v in P.M1F_VARIANTS},
            "near_boundary_threshold": P.M1F_OFFICIAL_THRESHOLD,
            "band": P.M1F_NEAR_BOUNDARY_BAND,
        },
        "conditional_sign": {v: bundles[primary_dataset][v]
                             ["conditional_sign"]
                             for v in P.M1F_VARIANTS},
        "ab_label_swap": {
            "status": P.M1F_AB_SWAP_STATUS,
            "reason": "the frozen E3 rows carry only the original A=RUMOR / "
                      "B=NON_RUMOR orientation; a swapped prompt would need "
                      "new reader forward passes, which M1-F forbids",
            "reader_inference_run": False,
        },
    }
    written[P.M1F_ROBUSTNESS_FILENAME] = _write_json(
        P.m1f_path(repo_root, P.M1F_ROBUSTNESS_FILENAME), robustness)

    # 6 — Task G PHEME diagnostic
    pheme_dataset = P.SECONDARY_DATASET
    pheme_payload = None
    if pheme_dataset in datasets:
        pheme_within = dict(within[pheme_dataset])
        pheme_within["variants"] = {
            v: {k: val for k, val in entry.items() if k != "per_snapshot"}
            for v, entry in within[pheme_dataset]["variants"].items()}
        pheme_payload = {
            "protocol": P.PROTOCOL_VERSION,
            "stage": "m1f_pheme_diagnostic",
            "dataset": pheme_dataset,
            "diagnostic_only": True,
            "decides_gate": False,
            "comparator": comparators[pheme_dataset],
            "primary_comparison": comparisons[pheme_dataset],
            "variants": bundles[pheme_dataset],
            "within_snapshot": pheme_within,
            "note": "PHEME is a diagnostic only: it cannot decide the primary "
                    "gate and cannot rescue a failed Ma-Weibo result.",
        }
        written[P.M1F_PHEME_FILENAME] = _write_json(
            P.m1f_path(repo_root, P.M1F_PHEME_FILENAME), pheme_payload)

    # 7 — decision (Ma-Weibo only)
    secondary = comparisons[primary_dataset]["secondary_metric_deltas"]
    active_delta = secondary["active_macro_f1"]["observed"]
    harmful_delta = secondary["harmful_auprc"]["observed"]
    decision = analysis.decide(
        comparisons[primary_dataset], within[primary_dataset],
        validity_blocked=bool(validity["conclusion"]["validity_blocked"]),
        active_improves=active_delta > 0.0,
        harmful_auprc_not_decreased=harmful_delta >= 0.0)
    verdict = {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1f",
        "status": "COMPLETED",
        "verdict": decision["verdict"],
        "primary_dataset": primary_dataset,
        "defines_new_gate": False,
        "m1_verdict_unchanged": True,
        "m1e_verdict_unchanged": True,
        "m2_entered": False,
        "authorizes_m2": False,
        "checks": decision["checks"],
        "conditions": decision["conditions"],
        "observed": decision["observed"],
        "gain_only_neutral_detail": decision["gain_only_neutral_detail"],
        "comparator": comparisons[primary_dataset]["comparator"],
        "pheme_diagnostic": {"diagnostic_only": True, "decides_gate": False},
        "reader_inference_run": False,
        "utility_labels_generated": False,
        "e3_regenerated": False,
        "new_model_deployed": False,
        "baseline_commit": P.M1F_BASELINE_COMMIT,
        "note": decision["note"],
        "seconds": time.time() - t0,
    }
    written[P.M1F_VERDICT_FILENAME] = _write_json(
        P.m1f_path(repo_root, P.M1F_VERDICT_FILENAME), verdict)

    # 8 — report
    report_data = {
        "commit": pin["commit"],
        "environment": environment,
        "task_validity": validity,
        "baselines": base_payloads,
        "active": active_payload,
        "within": within[primary_dataset],
        "robustness": robustness,
        "pheme": pheme_payload or {"diagnostic_only": True,
                                   "decides_gate": False,
                                   "comparator": {"selected": "n/a"},
                                   "primary_comparison": comparisons[
                                       primary_dataset],
                                   "note": "PHEME was not part of this run."},
        "verdict": verdict,
        "pins": {"n_groups": len(pin["groups"]),
                 "n_artifacts": pin["n_artifacts"]},
    }
    written[P.M1F_REPORT_FILENAME] = _write_text(
        P.m1f_path(repo_root, P.M1F_REPORT_FILENAME),
        m1f_report.build_report(report_data))
    print(f"[m1f] VERDICT = {verdict['verdict']} "
          f"({time.time() - t0:.1f}s)")
    return {"verdict": verdict, "written": written}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO))
    parser.add_argument("--datasets", nargs="+", choices=P.DATASETS,
                        default=list(P.DATASETS))
    parser.add_argument("--iterations", type=int,
                        default=P.BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=P.BOOTSTRAP_SEED)
    parser.add_argument("--environment", default="")
    args = parser.parse_args(argv)
    result = run(args.repo_root, args.datasets, args.iterations, args.seed,
                 environment=args.environment)
    print(json.dumps({k: v for k, v in result["verdict"].items()
                      if k in ("verdict", "checks", "observed")}, indent=1,
                     ensure_ascii=False))
    for name, info in result["written"].items():
        print(f"[m1f] wrote {name}: {info['bytes']} B {info['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
