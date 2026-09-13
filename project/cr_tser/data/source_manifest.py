"""Frozen data-source identity for the pilot (plan §4.1, §5, §31).

Every stage must consume the *same* frozen dataset. This module owns the
single fingerprint of a dataset source so that:

* the manifest records ``path`` + ``sha256`` for the source it was built from;
* label generation, training, selection and the report verify that the source
  they load is byte-identical to the frozen one, and fail closed otherwise.

For Weibo22 the plan's temporal requirement means the source of record is the
**normalized export** (``CRTSER_WEIBO22_NORMALIZED``), never the raw KPG
release, which has no timestamps.
"""
from __future__ import annotations

import hashlib
import json
import os

SOURCE_KINDS = ("pheme_raw", "weibo22_normalized", "weibo22_raw")


class SourceIdentityError(RuntimeError):
    """Raised when a stage is not reading the frozen data source."""


def dataset_source_path(dataset: str, paths) -> tuple:
    """``(kind, path)`` of the source of record for one dataset."""
    if dataset == "pheme":
        return "pheme_raw", paths.pheme_raw
    if dataset == "weibo22":
        if paths.weibo22_normalized:
            return "weibo22_normalized", paths.weibo22_normalized
        return "weibo22_raw", paths.weibo22_raw
    raise ValueError(f"unknown dataset {dataset!r}")


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_path(path: str) -> dict:
    """Deterministic fingerprint of a file or directory of data files."""
    if not path or not os.path.exists(path):
        return {"path": path, "exists": False, "sha256": "",
                "bytes": 0, "n_files": 0}
    if os.path.isfile(path):
        return {"path": path, "exists": True, "sha256": _hash_file(path),
                "bytes": os.path.getsize(path), "n_files": 1}
    h = hashlib.sha256()
    total = 0
    count = 0
    for root, dirs, files in os.walk(path):
        dirs.sort()  # deterministic traversal: subdirectories in sorted order
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path)
            size = os.path.getsize(full)
            h.update(f"{rel}:{size}:".encode())
            h.update(_hash_file(full).encode())
            h.update(b"\x00")
            total += size
            count += 1
    return {"path": path, "exists": True, "sha256": h.hexdigest(),
            "bytes": total, "n_files": count}


def source_fingerprint(dataset: str, paths) -> dict:
    """Frozen source identity recorded in every dataset manifest."""
    kind, path = dataset_source_path(dataset, paths)
    record = {"dataset": dataset, "kind": kind}
    record.update(fingerprint_path(path))
    return record


def assert_same_source(frozen: dict, current: dict, context: str = "") -> None:
    """Fail closed when a stage is not on the frozen source (plan §31)."""
    if not frozen:
        return
    for field in ("kind", "sha256", "n_files"):
        if frozen.get(field) != current.get(field):
            raise SourceIdentityError(
                f"{context}: data source identity changed "
                f"({field}: frozen={frozen.get(field)!r} "
                f"current={current.get(field)!r}); refusing to continue")


def load_frozen_source(manifest_dir: str) -> dict:
    """Read the frozen source record from a dataset manifest directory."""
    path = os.path.join(manifest_dir, "source.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_frozen_source(manifest_dir: str, record: dict) -> str:
    os.makedirs(manifest_dir, exist_ok=True)
    path = os.path.join(manifest_dir, "source.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    return path
