"""M1-E Task A — fail-closed pinning of the frozen M1 evidence.

Two independent checks, both required:

1. **git identity** — every M1 artifact path must hash to the same git blob
   as at the approved commit (``fef08cd…``), and the M1 subtree must show no
   tracked modification;
2. **content identity** — each artifact's raw SHA256 is recorded, so a later
   reader can compare without git.

Nothing under ``results/bcr_utility_v1/m1/`` is written by this module.
"""
from __future__ import annotations

import hashlib
import os
import subprocess

from ..config import protocol as P


class EvidencePinRefused(RuntimeError):
    """Raised when the frozen M1 evidence is not exactly the approved one."""


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
        raise EvidencePinRefused(
            f"git {' '.join(args)} failed: {proc.stderr.strip()[:200]}")
    return proc.stdout


def m1_artifact_paths(repo_root) -> list:
    """Every tracked file under the M1 artifact root."""
    root = os.path.join(str(repo_root), P.RESULTS_ROOT, P.M1_DIRNAME)
    if not os.path.isdir(root):
        raise EvidencePinRefused(f"M1 artifact root missing: {root}")
    paths = []
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, str(repo_root)).replace(os.sep, "/")
            paths.append(rel)
    return sorted(paths)


def _committed_blobs(repo_root, commit: str) -> dict:
    """``{path: blob}`` from ``git ls-tree -r`` (portable output format)."""
    out = _git(repo_root, "ls-tree", "-r", commit, "--",
               f"{P.RESULTS_ROOT}/{P.M1_DIRNAME}")
    blobs = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        meta, path = line.split("\t", 1)
        fields = meta.split()
        if len(fields) < 3:
            raise EvidencePinRefused(f"unparsable ls-tree line: {line!r}")
        blobs[path.strip()] = fields[2]
    if not blobs:
        raise EvidencePinRefused(f"no M1 blobs at {commit}")
    return blobs


def _worktree_blob(repo_root, rel_path: str) -> str:
    return _git(repo_root, "hash-object", "--", rel_path).strip()


def pin_m1_evidence(repo_root, commit: str = None) -> dict:
    """Verify and record the frozen M1 evidence (Task A)."""
    commit = commit or _git(repo_root, "rev-parse", "HEAD").strip()
    blobs = _committed_blobs(repo_root, commit)
    paths = m1_artifact_paths(repo_root)
    problems = []
    records = {}
    for rel in paths:
        full = os.path.join(str(repo_root), rel.replace("/", os.sep))
        if rel not in blobs:
            problems.append(f"{rel}: not committed at {commit}")
            continue
        got = _worktree_blob(repo_root, rel)
        if got != blobs[rel]:
            problems.append(f"{rel}: worktree blob {got[:12]} != "
                            f"committed {blobs[rel][:12]}")
            continue
        records[rel] = {"git_blob": got, "sha256": sha256_file(full),
                        "bytes": os.path.getsize(full)}
    extra = sorted(set(blobs) - set(paths))
    if extra:
        problems.append(f"{len(extra)} committed M1 artifacts missing from "
                        f"the worktree: {extra[:3]}")

    status = _git(repo_root, "status", "--porcelain", "--",
                  f"{P.RESULTS_ROOT}/{P.M1_DIRNAME}")
    tracked_changes = [line for line in status.splitlines()
                       if line.strip() and not line.startswith("??")]
    if tracked_changes:
        problems.append("tracked changes under the M1 root: "
                        + "; ".join(tracked_changes[:3]))

    verdict_path = os.path.join(str(repo_root), P.RESULTS_ROOT, P.M1_DIRNAME,
                                P.M1_VERDICT_FILENAME)
    if not os.path.exists(verdict_path):
        problems.append("M1_VERDICT.json missing")
        verdict = {}
    else:
        import json
        with open(verdict_path, encoding="utf-8") as fh:
            verdict = json.load(fh)
        if verdict.get("final_outcome") != P.M1E_EXPECTED_M1_VERDICT:
            problems.append(
                f"M1 verdict is {verdict.get('final_outcome')!r}, expected "
                f"{P.M1E_EXPECTED_M1_VERDICT!r}")

    if problems:
        raise EvidencePinRefused("; ".join(problems[:8]))
    return {
        "commit": commit,
        "n_artifacts": len(records),
        "artifacts": records,
        "m1_verdict": verdict.get("final_outcome"),
        "zero_touch_passed": verdict.get("zero_touch_passed"),
        "light_touch_passed": verdict.get("light_touch_passed"),
        "tracked_changes": 0,
        "ok": True,
    }
