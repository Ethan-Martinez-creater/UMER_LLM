"""M1-F Task — fail-closed pinning of every frozen input this audit reads.

Two independent identities per tracked artifact, both required (the M1-E
``evidence_pins`` contract, widened to the M1-F input set):

1. **git identity** — the worktree blob must equal the blob committed at HEAD;
2. **content identity** — the raw SHA256 and byte count are recorded, so a
   later reader can compare without git.

The frozen historical utility caches are deliberately untracked on SERVER
(they are large server-side artifacts), so for them the frozen SHA256/bytes
pinned in ``protocol.FROZEN_HISTORICAL_LABELS`` are the identity; a missing
cache is reported as ``server_only`` and becomes a *pending* check, never a
fabricated pass. Nothing is ever written below ``m1/``, ``m1e/`` or the
historical namespaces by this module.
"""
from __future__ import annotations

import hashlib
import os
import subprocess

from ..config import protocol as P


class PinRefused(RuntimeError):
    """Raised when a frozen M1-F input is not exactly the approved one."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo_root, *args) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo_root),
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise PinRefused(
            f"git {' '.join(args)} failed: {proc.stderr.strip()[:200]}")
    return proc.stdout


def _tracked_paths(repo_root, pathspecs) -> list:
    out = _git(repo_root, "ls-files", "--", *pathspecs)
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def _committed_blobs(repo_root, commit: str, pathspecs) -> dict:
    out = _git(repo_root, "ls-tree", "-r", commit, "--", *pathspecs)
    blobs = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        meta, path = line.split("\t", 1)
        fields = meta.split()
        if len(fields) < 3:
            raise PinRefused(f"unparsable ls-tree line: {line!r}")
        blobs[path.strip()] = fields[2]
    return blobs


def _worktree_blob(repo_root, rel_path: str) -> str:
    return _git(repo_root, "hash-object", "--", rel_path).strip()


def _rel_status(repo_root, pathspecs) -> list:
    out = _git(repo_root, "status", "--porcelain", "--", *pathspecs)
    return [line for line in out.splitlines()
            if line.strip() and not line.startswith("??")]


def pinned_groups() -> dict:
    """The frozen input groups the M1-F verifier fails closed on."""
    m1 = f"{P.RESULTS_ROOT}/{P.M1_DIRNAME}"
    m1e = f"{P.RESULTS_ROOT}/{P.M1E_DIRNAME}"
    return {
        "m1_verdict": [f"{m1}/{P.M1_VERDICT_FILENAME}"],
        "m1e_verdict": [f"{m1e}/{P.M1E_VERDICT_FILENAME}"],
        "m1_predictions": [
            f"{m1}/evaluation/{P.M1_PREDICTIONS_FILENAME}",
            f"{m1}/evaluation/predictions_light_touch.jsonl",
        ],
        "m1_frozen_evaluations": [
            f"{m1}/evaluation/{P.M1_EVALUATION_FILENAME}",
            f"{m1}/evaluation/evaluation_light_touch.json",
            f"{m1}/evaluation/gate.json",
            f"{m1}/evaluation/gate_light_touch.json",
        ],
        "e0_e1_e2_e3_caches": [f"{m1}/features"],
        "probe_artifacts": [
            f"{m1}/probe_responses",
            f"{m1}/fingerprints",
            f"{P.RESULTS_ROOT}/{P.BOOTSTRAP_DIRNAME}/"
            f"{P.PROBE_MANIFEST_FILENAME}",
        ],
        "d5_frozen_predictions": [
            f"{m1e}/{P.M1E_ATTRIBUTION_DIRNAME}/"
            f"predictions_{d}_D5_Z_S_C.jsonl" for d in P.DATASETS
        ],
        "atomic_index": [
            f"{P.RESULTS_ROOT}/{P.BOOTSTRAP_DIRNAME}/"
            f"{P.ATOMIC_INDEX_FILENAME}"
        ],
        "historical_manifests": [f"{P.HISTORICAL_RESULTS_ROOT}/manifests"],
    }


def pin_tracked_group(repo_root, name: str, pathspecs, commit: str) -> dict:
    """Pin one tracked group: blob parity + raw content identity."""
    blobs = _committed_blobs(repo_root, commit, pathspecs)
    if not blobs:
        raise PinRefused(f"{name}: nothing committed at {commit} for "
                         f"{list(pathspecs)}")
    paths = _tracked_paths(repo_root, pathspecs)
    problems = []
    records = {}
    for rel in paths:
        full = os.path.join(str(repo_root), rel.replace("/", os.sep))
        if rel not in blobs:
            problems.append(f"{rel}: not committed at {commit}")
            continue
        if not os.path.exists(full):
            problems.append(f"{rel}: missing from the worktree")
            continue
        got = _worktree_blob(repo_root, rel)
        if got != blobs[rel]:
            problems.append(f"{rel}: worktree blob {got[:12]} != committed "
                            f"{blobs[rel][:12]}")
            continue
        records[rel] = {"git_blob": got, "sha256": sha256_file(full),
                        "bytes": os.path.getsize(full)}
    extra = sorted(set(blobs) - set(paths))
    if extra:
        problems.append(f"{len(extra)} committed artifacts missing from the "
                        f"worktree: {extra[:3]}")
    changes = _rel_status(repo_root, pathspecs)
    if changes:
        problems.append(f"tracked changes under {name}: " +
                        "; ".join(changes[:3]))
    if problems:
        raise PinRefused("; ".join(problems[:8]))
    return {"group": name, "n_artifacts": len(records), "artifacts": records}


def pin_historical_caches(repo_root) -> dict:
    """Hash the frozen label caches against their pinned digests."""
    records = {}
    missing = []
    problems = []
    for dataset, spec in P.FROZEN_HISTORICAL_LABELS.items():
        rel = f"{P.HISTORICAL_RESULTS_ROOT}/{spec['rel_path']}"
        full = os.path.join(str(repo_root), rel.replace("/", os.sep))
        if not os.path.exists(full):
            missing.append(dataset)
            continue
        digest = sha256_file(full)
        size = os.path.getsize(full)
        if digest != spec["sha256"] or size != spec["bytes"]:
            problems.append(f"{dataset}: cache digest/bytes drift "
                            f"({digest[:12]}, {size})")
            continue
        records[dataset] = {"path": rel, "sha256": digest, "bytes": size}
    if problems:
        raise PinRefused("; ".join(problems[:6]))
    return {
        "group": "historical_utility_caches",
        "n_artifacts": len(records),
        "artifacts": records,
        "server_only_missing": sorted(missing),
        "all_present": not missing,
    }


def pin_m1f_inputs(repo_root, commit: str = None) -> dict:
    """Every frozen input group, fail-closed (M1-F plan §3)."""
    commit = commit or _git(repo_root, "rev-parse", "HEAD").strip()
    groups = {}
    for name, pathspecs in pinned_groups().items():
        groups[name] = pin_tracked_group(repo_root, name, pathspecs, commit)
    caches = pin_historical_caches(repo_root)
    groups["historical_utility_caches"] = caches
    return {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1f",
        "baseline_commit": P.M1F_BASELINE_COMMIT,
        "commit": commit,
        "groups": groups,
        "n_artifacts": sum(g["n_artifacts"] for g in groups.values()),
        "historical_caches_all_present": caches["all_present"],
        "historical_caches_server_only_missing":
            caches["server_only_missing"],
        "ok": True,
    }
