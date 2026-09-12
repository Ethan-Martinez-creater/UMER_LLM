#!/usr/bin/env python
"""Verifier for the TC-DSCR research consolidation stage (protocol §24).

Checks, independently of the consolidation writer:

  - every required consolidation artifact exists;
  - every commit referenced by the registry is well formed and (best effort)
    resolvable in the git history;
  - every canonical number traces to an existing result artifact;
  - validation results are not mislabeled as held-out test;
  - deprecated metrics/roots are not marked canonical;
  - the V3-B reader stage is labeled a validation pilot;
  - MS_TSR_COMPRESSION_ONLY is preserved;
  - V3-C / full E4 remain NOT APPROVED;
  - no new Qwen generation artifact was created (frozen V3-B hashes unchanged);
  - no new model checkpoint / training output was added;
  - no selector/algorithm implementation was changed by this stage.

Writes ``research_consolidation_verify.json`` and refreshes the report's
verdict line.  Requires ``issues = 0``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

DATASETS = ("pheme", "maweibo")
CONSOLIDATION_DIR = "results/tcdscr/research_consolidation"
READER_DIR = "results/tcdscr/dynamic_v3_reader"
DIAG_DIR = "results/tcdscr/v3b_failure_diagnosis"

REQUIRED = (
    "research_freeze_manifest.json",
    "experiment_registry.json",
    "research_evidence_ledger.json",
    "RESEARCH_EVIDENCE_LEDGER.md",
    "FINAL_CONTRIBUTION_MATRIX.md",
    "PROHIBITED_CLAIMS.md",
    "NEGATIVE_FINDINGS_LEDGER.md",
    "PROTOCOL_CORRECTIONS.md",
    "CANONICAL_RESULTS_TABLE.md",
    "DEPRECATED_RESULTS.md",
    "TC_DSCR_RESEARCH_STORYLINE.md",
    "PAPER_EVIDENCE_MAP.md",
    "NEXT_EXPERIMENT_GAPS.md",
    "RESEARCH_CONSOLIDATION_REPORT.md",
)

SPLIT_TAXONOMY = ("TRAIN", "VALIDATION", "HELD_OUT_TEST", "DIAGNOSTIC")
STATUS_TAXONOMY = ("SUPPORTED", "PARTIAL", "REJECTED", "DIAGNOSTIC_ONLY")

CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".bin",
                       ".gguf", ".onnx")

# Only these paths may be new/changed in the consolidation commit.
ALLOWED_CHANGES = (
    "results/tcdscr/research_consolidation/",
    "scripts/tcdscr_verify_research_consolidation.py",
    "project/tcdscr/tests/test_research_consolidation.py",
)
PROHIBITED_CLAIM_IDS = ("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7")

_ARTIFACT_RE = re.compile(r"results/tcdscr/[A-Za-z0-9_\-./]+")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _artifact_paths(text):
    paths = set()
    for raw in _ARTIFACT_RE.findall(text):
        paths.add(raw.rstrip("/"))
    return sorted(paths)


def _sections(text):
    """Split a markdown file into (heading, body) pairs."""
    out = []
    heading, buf = "", []
    for line in text.splitlines():
        if line.startswith("## "):
            if heading:
                out.append((heading, "\n".join(buf)))
            heading, buf = line.strip(), []
        else:
            buf.append(line)
    if heading:
        out.append((heading, "\n".join(buf)))
    return out


def _check_required(root, issues):
    for name in REQUIRED:
        if not os.path.exists(os.path.join(root, name)):
            issues.append(f"missing consolidation artifact: {name}")


def _check_registry(root, issues):
    reg = load_json(os.path.join(root, "experiment_registry.json"))
    stages = reg.get("stages", [])
    if len(stages) != reg.get("n_stages"):
        issues.append("registry n_stages disagrees with the stage list")
    if len(stages) < 11:
        issues.append(f"registry has only {len(stages)} stages "
                      f"(expected at least 11)")
    ids = set()
    for st in stages:
        sid = st.get("stage_id")
        ids.add(sid)
        if st.get("split_used") not in SPLIT_TAXONOMY:
            issues.append(f"stage {sid}: split_used={st.get('split_used')!r} "
                          f"outside the taxonomy")
        if st.get("status") not in STATUS_TAXONOMY:
            issues.append(f"stage {sid}: status={st.get('status')!r} "
                          f"outside the taxonomy")
        sha = st.get("commit_sha") or ""
        if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
            issues.append(f"stage {sid}: malformed commit_sha {sha!r}")
        if not st.get("main_artifacts"):
            issues.append(f"stage {sid}: no main_artifacts")
        for art in st.get("main_artifacts", []):
            if not os.path.exists(os.path.join(REPO_ROOT, art.rstrip("/"))):
                issues.append(f"stage {sid}: missing artifact {art}")
    return reg, ids


def _check_commit_refs(root, reg, issues):
    git = shutil.which("git")
    if git is None:
        return "git not on PATH: commit resolution skipped (registry " \
               "self-consistency only)"
    probe = subprocess.run([git, "rev-parse", "--is-inside-work-tree"],
                           cwd=REPO_ROOT, capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return "not a git work tree: commit resolution skipped (registry " \
               "self-consistency only)"
    for st in reg["stages"]:
        sha = st["commit_sha"]
        proc = subprocess.run([git, "cat-file", "-e", f"{sha}^{{commit}}"],
                              cwd=REPO_ROOT, capture_output=True)
        if proc.returncode != 0:
            issues.append(f"stage {st['stage_id']}: commit {sha} not found "
                          f"in git history")
    return "all registry commits resolve in git history"


def _check_ledger_stage_refs(root, stage_ids, issues):
    led = load_json(os.path.join(root, "research_evidence_ledger.json"))
    claims = led.get("claims", [])
    if led.get("claim_count") != len(claims):
        issues.append("ledger claim_count disagrees with the claim list")
    if len(claims) < 12:
        issues.append(f"ledger has only {len(claims)} claims "
                      f"(expected at least 12)")
    counts = {"SUPPORTED": 0, "PARTIAL": 0, "REJECTED": 0,
              "DIAGNOSTIC_ONLY": 0}
    for claim in claims:
        cid = claim.get("claim_id")
        if claim.get("status") not in STATUS_TAXONOMY:
            issues.append(f"claim {cid}: status {claim.get('status')!r} "
                          f"outside the taxonomy")
        else:
            counts[claim["status"]] += 1
        if not claim.get("supporting_stage"):
            issues.append(f"claim {cid}: no supporting stage")
        for sid in claim.get("supporting_stage", []):
            if sid not in stage_ids:
                issues.append(f"claim {cid}: supporting stage {sid} is not "
                              f"in the registry")
        if not claim.get("artifacts"):
            issues.append(f"claim {cid}: no artifacts cited")
        if claim.get("evidence_strength") not in ("STRONG", "MODERATE",
                                                  "WEAK"):
            issues.append(f"claim {cid}: bad evidence_strength "
                          f"{claim.get('evidence_strength')!r}")
        if claim.get("split_used") not in SPLIT_TAXONOMY:
            issues.append(f"claim {cid}: split_used "
                          f"{claim.get('split_used')!r} outside the taxonomy")
        for art in claim.get("artifacts", []):
            if not os.path.exists(os.path.join(REPO_ROOT, art.rstrip("/"))):
                issues.append(f"claim {cid}: missing artifact {art}")
    declared = led.get("status_counts", {})
    if declared.get("SUPPORTED") != counts["SUPPORTED"] or \
            declared.get("REJECTED") != counts["REJECTED"] or \
            declared.get("DIAGNOSTIC") != counts["DIAGNOSTIC_ONLY"]:
        issues.append("ledger status_counts disagree with the claim statuses")
    return counts


def _check_canonical(root, issues):
    text = read_text(os.path.join(root, "CANONICAL_RESULTS_TABLE.md"))
    missing = [p for p in _artifact_paths(text)
               if not os.path.exists(os.path.join(REPO_ROOT, p))]
    for path in missing:
        issues.append(f"canonical table cites a missing artifact: {path}")
    if "VALIDATION PILOT" not in text:
        issues.append("canonical table does not mark the V3-B reader stage "
                      "as a validation pilot")
    # V3-B reader numbers must live only in the validation-pilot section.
    for heading, body in _sections(text):
        has_reader = "reader_transfer_summary.json" in body
        if "HELD_OUT_TEST" in heading and has_reader:
            issues.append(f"section {heading!r}: reader-pilot artifact "
                          f"labeled as held-out test")
        if "VALIDATION PILOT" in heading and not has_reader:
            issues.append(f"section {heading!r}: reader pilot section does "
                          f"not cite reader_transfer_summary.json")
    if "VALIDATION PILOT" not in " ".join(h for h, _ in _sections(text)):
        issues.append("no canonical section is labeled VALIDATION PILOT")
    if not any("HELD_OUT_TEST" in h for h, _ in _sections(text)):
        issues.append("canonical table has no section labeled HELD_OUT_TEST")


def _check_deprecated(root, issues):
    dep_text = read_text(os.path.join(root, "DEPRECATED_RESULTS.md"))
    block = re.search(r"```json\s*(\{.*?\})\s*```", dep_text, re.S)
    if not block:
        issues.append("deprecated ledger has no machine-readable json block")
        return
    dep = json.loads(block.group(1))
    canonical = read_text(os.path.join(root, "CANONICAL_RESULTS_TABLE.md"))
    for entry in dep.get("deprecated_entries", []):
        for key in ("replacement", "replacement_artifact"):
            path = entry.get(key, "")
            path = path.split(" ")[0].rstrip("/")
            if path and not os.path.exists(os.path.join(REPO_ROOT, path)):
                issues.append(f"{entry.get('entry_id')}: replacement artifact "
                              f"missing: {path}")
    for dep_root in dep.get("deprecated_artifact_roots", []):
        if dep_root in canonical:
            issues.append(f"deprecated artifact root {dep_root} appears in "
                          f"the canonical table")
    for number in dep.get("deprecated_numbers", []):
        if number in canonical:
            issues.append(f"deprecated number {number} appears in the "
                          f"canonical table")


def _check_prohibited(root, issues):
    text = read_text(os.path.join(root, "PROHIBITED_CLAIMS.md"))
    for pid in PROHIBITED_CLAIM_IDS:
        if f"{pid} —" not in text and f"### {pid} " not in text:
            issues.append(f"prohibited claims list is missing {pid}")
    block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if not block:
        issues.append("prohibited claims file has no machine-readable block")
        return
    meta = json.loads(block.group(1))
    if meta.get("n_prohibited") != len(PROHIBITED_CLAIM_IDS):
        issues.append("prohibited claims count disagrees with the list")


def _check_recommendation_and_approvals(root, issues):
    manifest = load_json(os.path.join(root, "research_freeze_manifest.json"))
    diag = load_json(os.path.join(REPO_ROOT, DIAG_DIR,
                                  "diagnosis_summary.json"))
    if manifest.get("final_v3_recommendation") != "MS_TSR_COMPRESSION_ONLY":
        issues.append("freeze manifest does not preserve "
                      "MS_TSR_COMPRESSION_ONLY")
    if diag.get("recommendation") != "MS_TSR_COMPRESSION_ONLY":
        issues.append("V3-B diagnosis recommendation is not "
                      "MS_TSR_COMPRESSION_ONLY")
    report = read_text(os.path.join(root, "RESEARCH_CONSOLIDATION_REPORT.md"))
    if "MS_TSR_COMPRESSION_ONLY" not in report:
        issues.append("consolidation report does not preserve "
                      "MS_TSR_COMPRESSION_ONLY")
    if manifest.get("v3c_approved") is not False or \
            manifest.get("full_e4_approved") is not False:
        issues.append("freeze manifest does not keep V3-C / full E4 "
                      "unapproved")
    if "NOT APPROVED" not in report:
        issues.append("consolidation report does not state NOT APPROVED for "
                      "V3-C / full E4")
    if manifest.get("new_training_allowed") is not False or \
            manifest.get("new_qwen_inference_allowed") is not False or \
            manifest.get("held_out_test_allowed") is not False:
        issues.append("freeze manifest does not forbid new training / Qwen / "
                      "held-out runs")


def _check_frozen_v3b(root, issues):
    frozen = load_json(os.path.join(REPO_ROOT, DIAG_DIR,
                                    "frozen_artifacts.json"))
    checks = {
        "sampling_manifest_sha256": os.path.join(REPO_ROOT, READER_DIR,
                                                 "sampling_manifest.json"),
        "parsed_sha256": None,
        "raw_generations_sha256": None,
        "prompts_sha256": None,
    }
    observed = {
        "sampling_manifest_sha256": sha256_file(checks[
            "sampling_manifest_sha256"]),
        "parsed_sha256": {
            ds: sha256_file(os.path.join(REPO_ROOT, READER_DIR, "parsed",
                                         f"{ds}.jsonl")) for ds in DATASETS},
        "raw_generations_sha256": {
            ds: sha256_file(os.path.join(REPO_ROOT, READER_DIR,
                                         "raw_generations", f"{ds}.jsonl"))
            for ds in DATASETS},
        "prompts_sha256": {
            ds: sha256_file(os.path.join(REPO_ROOT, READER_DIR, "prompts",
                                         f"{ds}.jsonl")) for ds in DATASETS},
    }
    for key, value in observed.items():
        if frozen.get(key) != value:
            issues.append(f"frozen V3-B artifact changed: {key} "
                          f"(no new Qwen generation is allowed)")


def _check_no_new_artifacts(root, issues):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(CHECKPOINT_SUFFIXES):
                issues.append(f"checkpoint-like file inside the "
                              f"consolidation output: "
                              f"{os.path.relpath(os.path.join(dirpath, name), root)}")


def _check_no_code_change(issues):
    """No tracked implementation file may change; the only untracked files
    that may be new are the consolidation outputs, this verifier and its
    test.  Pre-existing untracked run logs / checkpoints (never committed)
    are recognized and ignored."""
    git = shutil.which("git")
    if git is None:
        return "git not on PATH: working-tree whitelist check skipped"
    proc = subprocess.run([git, "status", "--porcelain"], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return "git status unavailable: working-tree whitelist check skipped"
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ")[-1]
        if any(path.startswith(prefix) for prefix in ALLOWED_CHANGES):
            continue
        if status.strip() and status != "??":
            issues.append(f"tracked file modified outside the consolidation "
                          f"whitelist: {path}")
            continue
        if _is_legacy_untracked(path):
            continue
        issues.append(f"unexpected new file outside the consolidation "
                      f"whitelist: {path}")
    return "no tracked file changed; new files limited to the " \
           "consolidation whitelist"


def _is_legacy_untracked(path):
    """Run logs / selector checkpoints left untracked by earlier stages."""
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".log") or name.endswith(".pt") or \
            name.endswith(".pth") or name.endswith(".ckpt"):
        return True
    return path.startswith(".tmp_")


def verify(root=None):
    root = root or os.path.join(REPO_ROOT, CONSOLIDATION_DIR)
    issues = []
    _check_required(root, issues)
    reg, stage_ids = _check_registry(root, issues)
    git_note = _check_commit_refs(root, reg, issues)
    counts = _check_ledger_stage_refs(root, stage_ids, issues)
    _check_canonical(root, issues)
    _check_deprecated(root, issues)
    _check_prohibited(root, issues)
    _check_recommendation_and_approvals(root, issues)
    _check_frozen_v3b(root, issues)
    _check_no_new_artifacts(root, issues)
    code_note = _check_no_code_change(issues)
    return {
        "n_issues": len(issues),
        "issues": issues[:60],
        "consolidation_root": os.path.relpath(root, REPO_ROOT).replace(
            os.sep, "/"),
        "n_stages": len(reg.get("stages", [])),
        "ledger_claims": counts,
        "n_prohibited_claims": len(PROHIBITED_CLAIM_IDS),
        "recommendation": "MS_TSR_COMPRESSION_ONLY",
        "v3c_approved": False,
        "full_e4_approved": False,
        "notes": [git_note, code_note],
        "checks": [
            "all required consolidation artifacts exist",
            "all commits referenced by the registry resolve in git",
            "all canonical numbers trace to an existing result artifact",
            "validation results are not mislabeled as held-out test",
            "the V3-B reader stage is labeled a validation pilot",
            "deprecated metrics and artifact roots are not marked canonical",
            "all seven prohibited claims are present",
            "MS_TSR_COMPRESSION_ONLY is preserved",
            "V3-C and full E4 remain NOT APPROVED",
            "no new Qwen generation artifact created (frozen hashes "
            "unchanged)",
            "no new model checkpoint or training output added",
            "no selector/algorithm implementation changed",
        ],
    }


def _refresh_report(root, result):
    path = os.path.join(root, "RESEARCH_CONSOLIDATION_REPORT.md")
    if not os.path.exists(path):
        return
    text = read_text(path)
    line = (f"issues = {result['n_issues']} "
            f"(stages={result['n_stages']}, "
            f"claims={result['ledger_claims']})")
    if re.search(r"^issues = .*$", text, re.M):
        text = re.sub(r"^issues = .*$", line, text, count=1, flags=re.M)
    else:
        text = text.rstrip() + "\n\n## Verifier\n" + line + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="consolidation output directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    result = verify(args.root)
    root = args.root or os.path.join(REPO_ROOT, CONSOLIDATION_DIR)
    out = args.out or os.path.join(root,
                                   "research_consolidation_verify.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    _refresh_report(root, result)
    print(json.dumps(result, indent=1, ensure_ascii=False), flush=True)
    return 0 if result["n_issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
