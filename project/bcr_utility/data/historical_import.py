"""Historical read-only importer for the frozen CR-TSER v2r1 caches (plan §2, §11).

M0 may not regenerate anything. This module therefore *proves* that the
historical inputs it is about to bootstrap from are the ones the CR-TSER line
closed on, and refuses to continue otherwise:

* ``utility_labels/<dataset>/labels.jsonl`` is matched byte-for-byte against
  the frozen SHA256, byte size, total row count and per-reader row count;
* every frozen v2r1 manifest is matched against its canonical (LF-normalised)
  digest, so the same check holds on a CRLF Windows checkout and on Linux;
* the label rows are checked for the expected intervention families, the exact
  reader panel and the frozen cutoffs.

Nothing here writes to ``results/cr_tser*/``; the module only ever opens
historical paths for reading.
"""
from __future__ import annotations

import hashlib
import json
import os

from ..config import protocol as P


class HistoricalImportRefused(RuntimeError):
    """Raised when a historical artifact does not match its frozen identity."""


# --------------------------------------------------------------------------
# digest helpers
# --------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    """Raw content digest (no line-ending normalisation)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256_file(path: str) -> str:
    """LF-normalised digest (repository files, cross-platform identity)."""
    with open(path, "rb") as fh:
        return sha256_bytes(P.canonical_bytes(fh.read()))


def _file_bytes(path: str) -> int:
    return os.path.getsize(path)


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------
# path resolution
# --------------------------------------------------------------------------
def labels_rel_path(dataset: str) -> str:
    return P.FROZEN_HISTORICAL_LABELS[dataset]["rel_path"]


def labels_path(historical_root: str, dataset: str) -> str:
    return os.path.join(str(historical_root), labels_rel_path(dataset))


def manifest_path(historical_root: str, rel_path: str) -> str:
    return os.path.join(str(historical_root), rel_path)


# --------------------------------------------------------------------------
# label cache inspection
# --------------------------------------------------------------------------
def inspect_labels(historical_root: str, dataset: str) -> dict:
    """Read + verify one frozen label cache; raise on any mismatch."""
    if dataset not in P.FROZEN_HISTORICAL_LABELS:
        raise HistoricalImportRefused(f"unknown dataset {dataset!r}")
    spec = P.FROZEN_HISTORICAL_LABELS[dataset]
    path = labels_path(historical_root, dataset)
    if not os.path.exists(path):
        raise HistoricalImportRefused(
            f"{dataset}: frozen label cache missing at {path}")

    got_bytes = _file_bytes(path)
    if got_bytes != spec["bytes"]:
        raise HistoricalImportRefused(
            f"{dataset}: label cache is {got_bytes} bytes, frozen "
            f"{spec['bytes']}")
    got_sha = sha256_file(path)
    if got_sha != spec["sha256"]:
        raise HistoricalImportRefused(
            f"{dataset}: label cache digest {got_sha} does not match the "
            f"frozen {spec['sha256']}")

    rows_per_reader = {key: 0 for key in P.READER_KEYS}
    type_counts = {}
    cutoffs = set()
    events = set()
    unknown_readers = set()
    n_rows = 0
    for row in iter_jsonl(path):
        n_rows += 1
        reader = row.get("reader")
        if reader in rows_per_reader:
            rows_per_reader[reader] += 1
        else:
            unknown_readers.add(reader)
        itype = row.get("intervention_type")
        type_counts[itype] = type_counts.get(itype, 0) + 1
        cutoffs.add(int(row["cutoff"]))
        events.add(str(row["event_id"]))

    if n_rows != spec["rows"]:
        raise HistoricalImportRefused(
            f"{dataset}: label cache has {n_rows} rows, frozen {spec['rows']}")
    expected_per_reader = P.FROZEN_HISTORICAL_ROWS_PER_READER[dataset]
    if rows_per_reader != expected_per_reader:
        raise HistoricalImportRefused(
            f"{dataset}: per-reader row counts {rows_per_reader} != frozen "
            f"{expected_per_reader}")
    if unknown_readers:
        raise HistoricalImportRefused(
            f"{dataset}: label cache carries readers outside the bcr_v1 panel: "
            f"{sorted(str(r) for r in unknown_readers)}")
    if set(type_counts) - {P.ATOMIC_INTERVENTION_TYPE,
                           P.BASE_INTERVENTION_TYPE,
                           *P.NON_SUPERVISION_INTERVENTION_TYPES}:
        raise HistoricalImportRefused(
            f"{dataset}: unexpected intervention families {sorted(type_counts)}")
    if cutoffs != set(P.CUTOFFS_MIN):
        raise HistoricalImportRefused(
            f"{dataset}: label cache cutoffs {sorted(cutoffs)} != frozen "
            f"{list(P.CUTOFFS_MIN)}")

    return {
        "rel_path": spec["rel_path"],
        "sha256": got_sha,
        "bytes": got_bytes,
        "rows": n_rows,
        "rows_per_reader": rows_per_reader,
        "intervention_type_counts": dict(sorted(type_counts.items())),
        "cutoffs": sorted(cutoffs),
        "n_events": len(events),
    }


# --------------------------------------------------------------------------
# manifest inspection
# --------------------------------------------------------------------------
def inspect_manifests(historical_root: str) -> dict:
    """Verify every frozen v2r1 manifest by canonical digest."""
    out = {}
    for rel, expected in P.FROZEN_HISTORICAL_MANIFEST_SHA256.items():
        path = manifest_path(historical_root, rel)
        if not os.path.exists(path):
            raise HistoricalImportRefused(f"frozen manifest missing: {rel}")
        digest = canonical_sha256_file(path)
        if digest != expected:
            raise HistoricalImportRefused(
                f"frozen manifest drifted: {rel} -> {digest} (expected "
                f"{expected})")
        lines = None
        if rel.endswith(".jsonl"):
            lines = sum(1 for line in open(path, encoding="utf-8")
                        if line.strip())
        out[rel] = {"sha256": digest, "bytes": _file_bytes(path),
                    "lines": lines}
    return out


def inspect_event_split(historical_root: str, dataset: str) -> dict:
    rel = f"manifests/{dataset}/event_split.json"
    path = manifest_path(historical_root, rel)
    if not os.path.exists(path):
        raise HistoricalImportRefused(f"event split missing for {dataset}")
    split = read_json(path)
    sizes = {name: len(split.get(name, []))
             for name in P.HISTORICAL_SPLIT_NAMES}
    if split.get("split_seed") != P.SEED:
        raise HistoricalImportRefused(
            f"{dataset}: split seed {split.get('split_seed')} != {P.SEED}")
    if sizes.get("foundation_train") != 80:
        raise HistoricalImportRefused(
            f"{dataset}: foundation_train has {sizes.get('foundation_train')} "
            "events, expected 80")
    seen = set()
    for name in P.HISTORICAL_SPLIT_NAMES:
        for eid in split.get(name, []):
            if eid in seen:
                raise HistoricalImportRefused(
                    f"{dataset}: event {eid!r} appears in two splits")
            seen.add(eid)
    return {
        "rel_path": rel,
        "sha256": canonical_sha256_file(path),
        "sizes": sizes,
        "label_counts": {k: {str(kk): vv for kk, vv in v.items()}
                         for k, v in (split.get("label_counts") or {}).items()},
        "split_seed": split.get("split_seed"),
        "viable_event_count": split.get("viable_event_count"),
        "viable_event_ids": sorted(seen),
    }


def inspect_readers(historical_root: str, dataset: str) -> dict:
    """The frozen R1 reader contract carried by ``hashes.json``."""
    path = manifest_path(historical_root, f"manifests/{dataset}/hashes.json")
    hashes = read_json(path)
    readers = hashes.get("readers") or {}
    if sorted(readers) != sorted(P.READER_KEYS):
        raise HistoricalImportRefused(
            f"{dataset}: manifest reader set {sorted(readers)} != frozen "
            f"{sorted(P.READER_KEYS)}")
    for key, entry in readers.items():
        if entry.get("model_id") != P.READER_MODEL_IDS[key]:
            raise HistoricalImportRefused(
                f"{dataset}: reader {key} model id {entry.get('model_id')!r} "
                f"!= {P.READER_MODEL_IDS[key]!r}")
    if tuple(hashes.get("cutoffs") or ()) != P.CUTOFFS_MIN:
        raise HistoricalImportRefused(
            f"{dataset}: manifest cutoffs {hashes.get('cutoffs')} != "
            f"{list(P.CUTOFFS_MIN)}")
    return {
        "readers": {k: readers[k]["model_id"] for k in sorted(readers)},
        "cutoffs": list(hashes.get("cutoffs") or ()),
        "source_fingerprint": (hashes.get("source") or {}).get("sha256"),
        "event_split_sha256": hashes.get("event_split_sha256"),
        "n_snapshots": hashes.get("n_snapshots"),
        "n_interventions": hashes.get("n_interventions"),
    }


# --------------------------------------------------------------------------
# reused-dependency provenance
# --------------------------------------------------------------------------
def reused_dependency_digests(repo_root: str) -> dict:
    """Canonical digest of every CR-TSER file BCR imports (plan §3)."""
    out = {}
    for rel in P.REUSED_CR_TSER_MODULES:
        path = os.path.join(str(repo_root), rel.replace("/", os.sep))
        if not os.path.exists(path):
            raise HistoricalImportRefused(
                f"reused CR-TSER module missing: {rel}")
        out[rel] = canonical_sha256_file(path)
    return out


# --------------------------------------------------------------------------
# historical identity artifact
# --------------------------------------------------------------------------
def build_historical_identity(repo_root: str, historical_root: str = None,
                              environment: str = "SERVER") -> dict:
    """Collect and verify the complete M0 historical identity.

    Requires the label caches, so it runs where they exist (SERVER/DGPA).
    """
    historical_root = historical_root or P.historical_root(repo_root)
    payload = {
        "protocol": P.PROTOCOL_VERSION,
        "baseline_commit": P.BASELINE_COMMIT,
        "historical_root": P.HISTORICAL_RESULTS_ROOT,
        "environment": environment,
        "datasets": {},
        "reused_dependencies": reused_dependency_digests(repo_root),
        "frozen_constants": P.frozen_constants(),
        "read_only_namespaces": list(P.HISTORICAL_NAMESPACES),
    }
    for dataset in P.DATASETS:
        payload["datasets"][dataset] = {
            "labels": inspect_labels(historical_root, dataset),
            "manifests": inspect_manifests(historical_root),
            "event_split": inspect_event_split(historical_root, dataset),
            "reader_contract": inspect_readers(historical_root, dataset),
        }
    return payload


def load_historical_identity(path: str) -> dict:
    if not os.path.exists(path):
        raise HistoricalImportRefused(f"historical identity missing: {path}")
    return read_json(path)


def write_json(path: str, payload) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    return path
