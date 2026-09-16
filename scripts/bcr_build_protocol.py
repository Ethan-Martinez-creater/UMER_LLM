#!/usr/bin/env python
"""Freeze the ``bcr_v1`` protocol surface (plan §11.2).

Writes ``results/bcr_utility_v1/bootstrap/protocol.json``: the frozen
constants, the inherited CR-TSER values they must agree with, the read-only
historical namespaces, the reused-dependency digests and the M0 stage contract.

This script reads no data and no model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
PROJECT = REPO / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from bcr_utility.config import protocol as P                      # noqa: E402
from bcr_utility.data import historical_import                    # noqa: E402
from bcr_utility.probes import probe_contexts                     # noqa: E402


def build(repo_root: str) -> dict:
    return {
        "protocol": P.PROTOCOL_VERSION,
        "baseline_commit": P.BASELINE_COMMIT,
        "stage": "M0",
        "namespace": {
            "results_root": P.RESULTS_ROOT,
            "bootstrap_dir": f"{P.RESULTS_ROOT}/{P.BOOTSTRAP_DIRNAME}",
            "read_only_historical_namespaces": list(P.HISTORICAL_NAMESPACES),
        },
        "frozen_constants": P.frozen_constants(),
        "inherited_from_cr_tser": dict(P.INHERITED_FROM_CR_TSER),
        "not_inherited": {
            "structured_interaction": (
                "BCR does not adopt the CR-TSER structural-interaction "
                "hypothesis; I2-I5 are read-only historical evidence and are "
                "never BCR supervision."),
            "non_supervision_intervention_types":
                list(P.NON_SUPERVISION_INTERVENTION_TYPES),
        },
        "utility_target": {
            "definition": "u_r(e) = p_r(gold|SRC) - p_r(gold|SRC \\ e)",
            "threshold": P.UTILITY_THRESHOLD,
            "classes": list(P.SIGN_CLASSES),
            "atomic_intervention_type": P.ATOMIC_INTERVENTION_TYPE,
        },
        "probe_protocol": {
            "source_split": P.PROBE_SOURCE_SPLIT,
            "events_per_dataset": P.PROBE_EVENTS_PER_DATASET,
            "events_per_cutoff": P.PROBE_EVENTS_PER_CUTOFF,
            "items_total": P.PROBE_ITEMS_TOTAL,
            "seed": P.PROBE_SEED,
            "contexts": probe_contexts.context_plan(),
        },
        "reused_cr_tser_modules": {
            rel: historical_import.canonical_sha256_file(
                str(Path(repo_root) / Path(rel)))
            for rel in P.REUSED_CR_TSER_MODULES
        },
        "m0_contract": {
            "runs_reader_inference": False,
            "downloads_models": False,
            "generates_utility_labels": False,
            "trains_predictors": False,
            "phases_not_started": ["M1", "M2", "M3", "M4", "M5"],
        },
    }


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = build(args.repo_root)
    target = args.out or str(Path(args.repo_root) / P.RESULTS_ROOT
                             / P.BOOTSTRAP_DIRNAME / "protocol.json")
    historical_import.write_json(target, payload)
    print(json.dumps({"wrote": target,
                      "protocol": payload["protocol"],
                      "baseline_commit": payload["baseline_commit"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
