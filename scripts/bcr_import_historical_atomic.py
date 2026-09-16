#!/usr/bin/env python
"""SERVER-only: bootstrap the BCR M0 artifacts from the frozen caches (§11).

Requires the historical ``utility_labels/<dataset>/labels.jsonl`` caches, which
live on SERVER/DGPA. It writes, into ``results/bcr_utility_v1/bootstrap/``:

* ``historical_identity.json`` — verified digests of every historical input;
* ``atomic_index.json`` — the ``I1_atomic`` evidence index per dataset;
* ``geometry_diagnostic.json`` — activity / ordinal / signed / interaction
  diagnostics per dataset.

No reader is loaded, no model is downloaded and no utility label is generated.
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

from bcr_utility.config import protocol as P                       # noqa: E402
from bcr_utility.data import atomic_manifest, historical_import    # noqa: E402
from bcr_utility.evaluation import reader_geometry                 # noqa: E402


def bootstrap(repo_root: str, historical_root: str, environment: str,
              datasets=P.DATASETS, iterations=P.BOOTSTRAP_ITERATIONS):
    identity = historical_import.build_historical_identity(
        repo_root, historical_root, environment=environment)
    written = {}

    out_dir = Path(repo_root) / P.RESULTS_ROOT / P.BOOTSTRAP_DIRNAME
    written["historical_identity"] = historical_import.write_json(
        str(out_dir / P.HISTORICAL_IDENTITY_FILENAME), identity)

    atomic = {"protocol": P.PROTOCOL_VERSION, "stage": "M0", "datasets": {}}
    geometry = {"protocol": P.PROTOCOL_VERSION, "stage": "M0",
                "bootstrap": {"unit": P.BOOTSTRAP_UNIT,
                              "iterations": iterations,
                              "seed": P.BOOTSTRAP_SEED},
                "datasets": {}}
    for dataset in datasets:
        labels_sha = identity["datasets"][dataset]["labels"]["sha256"]
        index = atomic_manifest.build_atomic_index(
            historical_root, dataset, labels_sha)
        atomic_manifest.validate_atomic_index(index, dataset)
        atomic["datasets"][dataset] = index
        geometry["datasets"][dataset] = reader_geometry.geometry_report(
            index, iterations=iterations)
    written["atomic_index"] = historical_import.write_json(
        str(out_dir / P.ATOMIC_INDEX_FILENAME), atomic)
    written["geometry_diagnostic"] = historical_import.write_json(
        str(out_dir / P.GEOMETRY_DIAGNOSTIC_FILENAME), geometry)

    summary = {
        "historical_root": identity["historical_root"],
        "datasets": {
            dataset: {
                "labels_sha256": identity["datasets"][dataset]["labels"]["sha256"],
                "rows": identity["datasets"][dataset]["labels"]["rows"],
                "atomic_keys": atomic["datasets"][dataset]["n_evidence_keys"],
                "atomic_events": atomic["datasets"][dataset]["n_events"],
            } for dataset in datasets},
        "written": written,
    }
    return summary


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--historical-root", default=None)
    ap.add_argument("--environment", default="SERVER/DGPA")
    ap.add_argument("--dataset", action="append", choices=list(P.DATASETS))
    ap.add_argument("--iterations", type=int, default=P.BOOTSTRAP_ITERATIONS,
                    help="bootstrap iterations (formal runs use the frozen "
                         f"{P.BOOTSTRAP_ITERATIONS})")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    historical_root = args.historical_root or P.historical_root(args.repo_root)
    summary = bootstrap(args.repo_root, historical_root, args.environment,
                        datasets=tuple(args.dataset or P.DATASETS),
                        iterations=args.iterations)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
