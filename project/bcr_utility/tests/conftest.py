"""Synthetic fixtures for the BCR-Utility M0 suite.

Everything is tiny and synthetic: no test reads the real datasets, the real
label caches or a model. The reused CR-TSER modules are imported for real so
the shared contract under test is the frozen implementation, not a copy.

``synth_repo`` builds a miniature historical namespace (two datasets, frozen
manifests, a frozen-looking label cache) inside ``tmp_path`` and patches the
frozen protocol constants to the synthetic digests. That lets the verifier run
end to end without the multi-gigabyte originals, while still failing closed on
every mismatch the real verifier checks.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
PROJECT_DIR = HERE.parents[2]          # <repo>/project
REPO_DIR = HERE.parents[3]             # <repo>
for _path in (PROJECT_DIR, REPO_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bcr_utility.config import protocol as P                       # noqa: E402
from bcr_utility.data import historical_import                     # noqa: E402

WRITERS = ("qwen", "mistral", "internlm")

#: Synthetic atomic layout: two events, three cutoffs, one reply each.
SYNTH_ATOMIC = {
    "maweibo": {
        "u0": {15: ["n1"], 60: ["n1"], 360: ["n1"]},
        "u1": {15: ["n2"], 60: ["n2"], 360: ["n2"]},
    },
    "pheme": {
        "u0": {15: ["n1"], 60: ["n1"], 360: ["n1"]},
        "u1": {15: ["n2"], 60: ["n2"], 360: ["n2"]},
    },
}

#: Deterministic per-reader utilities. Every value is active under the frozen
#: ±0.05 band. ``qwen`` and ``internlm`` agree in sign; ``mistral`` has the
#: same magnitude ordering as ``qwen`` but the opposite sign — which is exactly
#: the gap the geometry diagnostic has to keep visible.
SYNTH_UTILITY = {
    "maweibo": {
        "qwen": {"u0": 0.40, "u1": 0.30},
        "internlm": {"u0": 0.35, "u1": 0.25},
        "mistral": {"u0": -0.45, "u1": -0.30},
    },
    "pheme": {
        "qwen": {"u0": 0.20, "u1": 0.30},
        "internlm": {"u0": 0.22, "u1": 0.28},
        "mistral": {"u0": -0.25, "u1": -0.26},
    },
}


def pytest_configure(config):
    """Keep the pytest base temp inside the repository (portable temp root)."""
    if not config.option.basetemp:
        config.option.basetemp = str(REPO_DIR / ".pytest_tmp_bcr")


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_entries(dataset: str) -> list:
    """``[(event_id, cutoff, node_id), ...]`` of the synthetic atomic layout."""
    layout = SYNTH_ATOMIC[dataset]
    return [(eid, cutoff, node)
            for eid in sorted(layout)
            for cutoff in sorted(layout[eid])
            for node in layout[eid][cutoff]]


def expected_atomic_keys(dataset: str) -> int:
    return len(atomic_entries(dataset))


class SynthRepo:
    """A miniature repo root with a BCR bootstrap namespace."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.historical = self.root / P.HISTORICAL_RESULTS_ROOT
        self.bootstrap = self.root / P.RESULTS_ROOT / P.BOOTSTRAP_DIRNAME
        self.labels = {}
        self.manifests = {}

    # ---- writers -------------------------------------------------------
    def build(self) -> "SynthRepo":
        for dataset in P.DATASETS:
            self._build_dataset(dataset)
        return self

    def _build_dataset(self, dataset: str) -> None:
        manifests = self.historical / "manifests" / dataset
        n_foundation = 80
        foundation = [f"f{i:03d}" for i in range(n_foundation)]
        split = {
            "foundation_train": foundation,
            "utility_train": ["u0", "u1"],
            "utility_dev": ["d0"],
            "utility_eval": ["e0"],
            "unused": [],
            "label_counts": {name: {"0": 1, "1": 1} for name in
                             P.HISTORICAL_SPLIT_NAMES},
            "sizes": {"foundation_train": n_foundation, "utility_train": 2,
                      "utility_dev": 1, "utility_eval": 1},
            "dataset": dataset,
            "source": {"kind": "synthetic", "path": "/dev/null",
                       "sha256": "0" * 64, "n_files": 1, "exists": True},
            "viable_event_count": 100,
            "total_events": 100,
            "viability_filtered": 0,
            "split_seed": P.SEED,
        }
        _write(manifests / "event_split.json", split)
        _write(manifests / "source.json", split["source"])
        _write(manifests / "hashes.json", {
            "dataset": dataset,
            "source": split["source"],
            "event_split_sha256": "1" * 64,
            "cutoffs": list(P.CUTOFFS_MIN),
            "loro_rotations": [list(r) for r in
                               (("qwen", "mistral", "internlm"),
                                ("qwen", "internlm", "mistral"),
                                ("mistral", "internlm", "qwen"))],
            "readers": {key: {"model_id": P.READER_MODEL_IDS[key],
                              "path": f"/synthetic/{key}",
                              "weight_hash": "w" * 8, "tokenizer_hash": "t" * 8}
                        for key in P.READER_KEYS},
            "viable_events": 100, "n_snapshots": 3, "n_interventions": 3,
            "n_zero_reply_snapshots": 0, "max_snapshot_nodes": 10,
            "snapshot_cap": None,
        })
        _write_jsonl(manifests / "snapshot_manifest.jsonl",
                     [{"dataset": dataset, "event_id": "u0", "cutoff": 15,
                       "num_nodes": 3, "num_replies": 2, "zero_reply": False}])
        _write_jsonl(manifests / "intervention_manifest.jsonl",
                     [{"dataset": dataset, "event_id": "u0", "cutoff": 15,
                       "intervention_id": "I1:n1", "type": "I1_atomic",
                       "status": "OK", "remove_node_ids": ["n1"],
                       "valid_in_gt": True}])

        rows = []
        for reader in WRITERS:
            for eid, cutoff, node in atomic_entries(dataset):
                utility = SYNTH_UTILITY[dataset][reader][eid]
                rows.append(self._row(dataset, eid, cutoff, node, reader,
                                      f"I1:{node}", "I1_atomic", utility, []))
            rows.append(self._row(dataset, "u0", 15, None, reader, "I0",
                                  "I0_base", 0.0, []))
            for itype, name in (("I2_parent_child", "I2"),
                                ("I3_matched_nonadjacent", "I3"),
                                ("I4_subtree", "I4"),
                                ("I5_matched_disconnected", "I5")):
                rows.append(self._row(dataset, "u0", 15, None, reader, name,
                                      itype, 0.0, ["n1"]))
        path = self.historical / "utility_labels" / dataset / "labels.jsonl"
        _write_jsonl(path, rows)

        self.labels[dataset] = {
            "rel_path": f"utility_labels/{dataset}/labels.jsonl",
            "sha256": historical_import.sha256_file(str(path)),
            "bytes": path.stat().st_size,
            "rows": len(rows),
            "rows_per_reader": {r: len(rows) // len(WRITERS) for r in WRITERS},
        }
        self.manifests[dataset] = {
            rel: historical_import.canonical_sha256_file(
                str(self.historical / rel))
            for rel in P.FROZEN_HISTORICAL_MANIFEST_SHA256
            if rel.startswith(f"manifests/{dataset}/")}

    @staticmethod
    def _row(dataset, eid, cutoff, node, reader, intervention_id,
             intervention_type, utility, affected):
        sign = ("HELPFUL" if utility >= P.UTILITY_THRESHOLD else
                "HARMFUL" if utility <= -P.UTILITY_THRESHOLD else "NEUTRAL")
        return {
            "dataset": dataset, "event_id": eid, "cutoff": int(cutoff),
            "reader": reader, "intervention_id": intervention_id,
            "intervention_type": intervention_type,
            "affected_reply_ids": [node] if node else list(affected),
            "utility": float(utility), "sign": sign,
            "correct_before": True, "correct_after": True,
        }

    # ---- patching ------------------------------------------------------
    def patch(self, monkeypatch) -> "SynthRepo":
        """Point the frozen constants at the synthetic namespace."""
        labels = {
            dataset: dict(self.labels[dataset], rel_path=(
                f"utility_labels/{dataset}/labels.jsonl"))
            for dataset in P.DATASETS}
        monkeypatch.setattr(P, "FROZEN_HISTORICAL_LABELS", labels)
        monkeypatch.setattr(P, "FROZEN_HISTORICAL_ROWS_PER_READER", {
            dataset: dict(self.labels[dataset]["rows_per_reader"])
            for dataset in P.DATASETS})
        monkeypatch.setattr(P, "FROZEN_ATOMIC_KEYS", {
            dataset: expected_atomic_keys(dataset)
            for dataset in P.DATASETS})
        manifests = {}
        for dataset in P.DATASETS:
            manifests.update(self.manifests[dataset])
        monkeypatch.setattr(P, "FROZEN_HISTORICAL_MANIFEST_SHA256", manifests)
        monkeypatch.setattr(P, "REUSED_CR_TSER_MODULES", ())
        monkeypatch.setattr(P, "HISTORICAL_NAMESPACES", ())
        return self

    # ---- helpers -------------------------------------------------------
    def split(self, dataset: str) -> dict:
        return json.loads((self.historical / "manifests" / dataset
                           / "event_split.json").read_text(encoding="utf-8"))

    def availability(self, dataset: str) -> dict:
        """Every foundation event is eligible at every cutoff by default."""
        split = self.split(dataset)
        per_event = {}
        for eid in split["foundation_train"]:
            per_event[eid] = {str(c): {"n_units": 5, "src_selected": 3,
                                       "n_visible_nodes": 12}
                              for c in P.CUTOFFS_MIN}
        return {
            "protocol": P.PROTOCOL_VERSION, "stage": "M0",
            "source_split": P.PROBE_SOURCE_SPLIT,
            "datasets": {dataset: {
                "n_foundation_events": len(split["foundation_train"]),
                "event_split_sha256": historical_import.canonical_sha256_file(
                    str(self.historical / "manifests" / dataset
                        / "event_split.json")),
                "gold_labels_for_balance_only": {
                    eid: int(i % 2 == 0) for i, eid in
                    enumerate(split["foundation_train"])},
                "availability": per_event,
            }},
        }

    def write_availability(self, datasets=P.DATASETS):
        payload = {"protocol": P.PROTOCOL_VERSION, "stage": "M0",
                   "source_split": P.PROBE_SOURCE_SPLIT, "datasets": {}}
        for dataset in datasets:
            payload["datasets"][dataset] = self.availability(
                dataset)["datasets"][dataset]
        dest = self.bootstrap / P.PROBE_AVAILABILITY_FILENAME
        _write(dest, payload)
        return payload

    def digest(self, rel: str) -> str:
        return historical_import.canonical_sha256_file(
            str(self.historical / rel))

    # ---- full M0 pipeline on the synthetic namespace -------------------
    def seed_bootstrap(self, iterations: int = None) -> dict:
        """Produce every M0 artifact exactly as the stage scripts do."""
        from bcr_utility.data import atomic_manifest
        from bcr_utility.evaluation import reader_geometry
        from bcr_utility.probes import probe_manifest

        iterations = P.BOOTSTRAP_ITERATIONS if iterations is None else iterations
        identity = historical_import.build_historical_identity(
            str(self.root), str(self.historical), environment="TEST")
        _write(self.bootstrap / P.HISTORICAL_IDENTITY_FILENAME, identity)

        atomic = {"protocol": P.PROTOCOL_VERSION, "stage": "M0",
                  "datasets": {}}
        geometry = {"protocol": P.PROTOCOL_VERSION, "stage": "M0",
                    "bootstrap": {"unit": P.BOOTSTRAP_UNIT,
                                  "iterations": P.BOOTSTRAP_ITERATIONS,
                                  "seed": P.BOOTSTRAP_SEED},
                    "datasets": {}}
        for dataset in P.DATASETS:
            index = atomic_manifest.build_atomic_index(
                str(self.historical), dataset,
                identity["datasets"][dataset]["labels"]["sha256"])
            atomic["datasets"][dataset] = index
            geometry["datasets"][dataset] = reader_geometry.geometry_report(
                index, iterations=iterations)
        _write(self.bootstrap / P.ATOMIC_INDEX_FILENAME, atomic)
        _write(self.bootstrap / P.GEOMETRY_DIAGNOSTIC_FILENAME, geometry)

        availability = self.write_availability()
        manifest = {"protocol": P.PROTOCOL_VERSION, "stage": "M0",
                    "source_split": P.PROBE_SOURCE_SPLIT,
                    "seed": P.PROBE_SEED,
                    "design": {"events_per_dataset": P.PROBE_EVENTS_PER_DATASET,
                               "events_per_cutoff": P.PROBE_EVENTS_PER_CUTOFF},
                    "datasets": {}, "overlap_audit": {}}
        for dataset in P.DATASETS:
            entry = availability["datasets"][dataset]
            split = self.split(dataset)
            built = probe_manifest.build_probe_manifest(
                split, dataset, entry["availability"],
                entry["gold_labels_for_balance_only"], seed=P.PROBE_SEED)
            built["overlap_audit"] = probe_manifest.audit_overlap(
                built, split)
            manifest["datasets"][dataset] = built
            manifest["overlap_audit"][dataset] = built["overlap_audit"]
        _write(self.bootstrap / P.PROBE_MANIFEST_FILENAME, manifest)
        self.atomic = atomic
        self.geometry = geometry
        self.probe_manifest = manifest
        return {"identity": identity, "atomic": atomic, "geometry": geometry,
                "probe_manifest": manifest}


@pytest.fixture
def synth_repo(tmp_path, monkeypatch) -> SynthRepo:
    return SynthRepo(tmp_path).build().patch(monkeypatch)
