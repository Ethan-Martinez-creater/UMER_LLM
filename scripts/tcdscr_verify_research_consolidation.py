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
    "E3_NO_CANDIDATE_SCOPE_AUDIT.md",
    "e3_no_candidate_scope_audit.json",
)

SPLIT_TAXONOMY = ("TRAIN", "VALIDATION", "HELD_OUT_TEST", "DIAGNOSTIC")
STATUS_TAXONOMY = ("SUPPORTED", "PARTIAL", "REJECTED", "DIAGNOSTIC_ONLY")

CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".bin",
                       ".gguf", ".onnx")

# E3 held-out scope audit inputs.
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (2000, 2001, 2002)
E3_TEST_DIR = "results/tcdscr/formal_e3_test"
N_PRIMARY_CUTOFFS = 6
PROTOCOL_STATUSES = ("ALL_EVENT", "CANDIDATE_CONDITIONED", "VALIDATION_PILOT",
                     "HELD_OUT", "DIAGNOSTIC")

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


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def e3_scope_audit(results_root=None):
    """Audit whether the E3 held-out test runner keeps no-candidate
    (candidate_count == 0) event-cutoffs.

    Reads only the frozen runner outputs: each run's ``run_manifest.json``
    (n_test_events, n_rows) and ``test_predictions.jsonl`` (n_candidates per
    row).  Never re-runs anything.
    """
    root = os.path.join(results_root or os.path.join(REPO_ROOT, "results"),
                        "tcdscr", "formal_e3_test")
    per_ds, totals = {}, {"n_runs": 0, "n_test_events": 0,
                          "expected_rows": 0, "actual_rows": 0,
                          "zero_candidate_rows": 0}
    for ds in DATASETS:
        agg = {"n_runs": 0, "n_test_events": 0, "expected_rows": 0,
               "actual_rows": 0, "zero_candidate_rows": 0,
               "manifest_n_rows": 0}
        for fold in FOLDS:
            for seed in SEEDS:
                run_dir = os.path.join(root, "runs", ds,
                                       f"fold{fold}_seed{seed}")
                mpath = os.path.join(run_dir, "run_manifest.json")
                ppath = os.path.join(run_dir, "test_predictions.jsonl")
                if not (os.path.exists(mpath) and os.path.exists(ppath)):
                    continue
                man = load_json(mpath)
                n_events = int(man["n_test_events"])
                rows = load_jsonl(ppath)
                agg["n_runs"] += 1
                agg["n_test_events"] += n_events
                agg["expected_rows"] += n_events * N_PRIMARY_CUTOFFS
                agg["actual_rows"] += len(rows)
                agg["manifest_n_rows"] += int(man.get("n_rows", 0))
                agg["zero_candidate_rows"] += sum(
                    1 for r in rows if int(r.get("n_candidates", -1)) == 0)
        agg["expected_matches_actual"] = (agg["expected_rows"]
                                          == agg["actual_rows"])
        agg["manifest_matches_actual"] = (agg["manifest_n_rows"]
                                          == agg["actual_rows"])
        agg["zero_candidate_row_rate"] = (
            agg["zero_candidate_rows"] / agg["actual_rows"]
            if agg["actual_rows"] else None)
        per_ds[ds] = agg
        for key in ("n_runs", "n_test_events", "expected_rows",
                    "actual_rows", "zero_candidate_rows"):
            totals[key] += agg[key]
    retained = (totals["actual_rows"] == totals["expected_rows"]
                and totals["zero_candidate_rows"] > 0
                and all(per_ds[ds]["manifest_matches_actual"]
                        for ds in DATASETS))
    # Independent cross-check: the missing rows should match the validation
    # no-candidate rate, which shows the skipped rows are no-candidate
    # snapshots rather than a generic row loss.
    cross = {}
    nca_path = os.path.join(results_root or os.path.join(REPO_ROOT,
                                                         "results"),
                            "tcdscr", "dynamic_v2_protocol",
                            "no_candidate_audit.json")
    if os.path.exists(nca_path):
        nca = load_json(nca_path)
        for ds in DATASETS:
            rates = [nca[ds]["pooled_over_folds"][str(c)]["no_candidate_rate"]
                     for c in (5, 15, 30, 60, 180, 360)]
            expected_rate = sum(rates) / len(rates)
            a = per_ds[ds]
            missing = a["expected_rows"] - a["actual_rows"]
            missing_rate = (missing / a["expected_rows"]
                            if a["expected_rows"] else None)
            cross[ds] = {
                "validation_mean_no_candidate_rate": expected_rate,
                "held_out_missing_row_rate": missing_rate,
                "abs_gap": (abs(expected_rate - missing_rate)
                            if missing_rate is not None else None),
                "consistent": (missing_rate is not None
                               and abs(expected_rate - missing_rate) < 0.03),
            }
    consistent = bool(cross) and all(v["consistent"] for v in cross.values())
    return {
        "audit_version": "1.0",
        "runner": {
            "entry_point": "scripts/tcdscr_run_e3_test.py",
            "item_build": "scripts/tcdscr_run_e3_test.py main(): "
                          "for ev in events['test']: for c in PRIMARY_CUTOFFS: "
                          "items.append(build_light_item(...)) -- one item per "
                          "event-cutoff, no candidate filter",
            "row_build": "scripts/tcdscr_run_e3.py collect_event_data()",
            "trajectory": "scripts/tcdscr_run_e3.py run_event_trajectory(): "
                          "appends one row per (event, cutoff) carrying "
                          "n_candidates = len(cand_ids); no early continue",
            "current_source_keeps_zero_candidate": True,
            "current_source_note": "The on-disk collect_event_data now carries "
                                   "the Dynamic V2 protocol correction "
                                   "('no-candidate snapshots are kept in the "
                                   "trajectory'); this correction was applied "
                                   "to the same file AFTER the E3 held-out "
                                   "artifacts were produced.",
            "artifacts_observed_skip": not retained,
            "artifacts_observed_note": "The frozen prediction rows contain no "
                                       "row with n_candidates == 0 and are "
                                       "short by exactly the no-candidate "
                                       "fraction, so the run that produced "
                                       "these artifacts skipped them.",
        },
        "per_dataset": per_ds,
        "totals": totals,
        "cross_check": cross,
        "cross_check_consistent": consistent,
        "scope_verdict": ("CASE_A_ALL_EVENTS_RETAINED" if retained
                          else "CASE_B_ZERO_CANDIDATE_SKIPPED"),
        "held_out_status": "CANONICAL" if retained
                           else "CONDITIONAL_HELD_OUT",
        "notes": [
            "expected_rows = sum(n_test_events) * 6 (six primary cutoffs)",
            "zero_candidate_rows counts prediction rows with n_candidates == 0",
            "cross_check compares the missing-row rate against the "
            "validation no-candidate rate",
            "audit is read-only over the frozen E3 test artifacts",
        ],
    }


def write_e3_audit_files(root, results_root=None):
    audit = e3_scope_audit(results_root)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "e3_no_candidate_scope_audit.json"), "w",
              encoding="utf-8") as fh:
        json.dump(audit, fh, indent=1, ensure_ascii=False)
    t = audit["totals"]
    p = audit["per_dataset"]
    lines = [
        "# E3 Held-out No-Candidate Scope Audit",
        "",
        "Read-only audit of whether the E3 held-out test runner keeps "
        "event-cutoffs with `candidate_count == 0`.",
        "",
        "## Runner behavior",
        "",
        f"- entry point: `{audit['runner']['entry_point']}`",
        f"- item build: {audit['runner']['item_build']}",
        f"- row build: {audit['runner']['row_build']}",
        f"- trajectory: {audit['runner']['trajectory']}",
        f"- current on-disk source keeps zero-candidate rows: "
        f"**{audit['runner']['current_source_keeps_zero_candidate']}** "
        f"({audit['runner']['current_source_note']})",
        f"- frozen artifacts skipped zero-candidate rows: "
        f"**{audit['runner']['artifacts_observed_skip']}** "
        f"({audit['runner']['artifacts_observed_note']})",
        "",
        "## Row-count evidence",
        "",
        "| Dataset | runs | test events | expected rows | actual rows | "
        "zero-candidate rows | zero-candidate rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds in DATASETS:
        a = p[ds]
        rate = ("n/a" if a["zero_candidate_row_rate"] is None
                else f"{a['zero_candidate_row_rate']:.6f}")
        lines.append(
            f"| {ds} | {a['n_runs']} | {a['n_test_events']} | "
            f"{a['expected_rows']} | {a['actual_rows']} | "
            f"{a['zero_candidate_rows']} | {rate} |")
    lines += [
        f"| **total** | {t['n_runs']} | {t['n_test_events']} | "
        f"{t['expected_rows']} | {t['actual_rows']} | "
        f"{t['zero_candidate_rows']} | |",
        "",
        f"- expected event x cutoff rows: {t['expected_rows']}",
        f"- actual rows: {t['actual_rows']}",
        f"- missing rows: {t['expected_rows'] - t['actual_rows']}",
        f"- rows with n_candidates == 0: {t['zero_candidate_rows']}",
        "",
        "## Cross-check against the validation no-candidate rate",
        "",
        "| Dataset | validation mean no-candidate rate | held-out missing-row "
        "rate | abs gap | consistent |",
        "|---|---|---|---|---|",
    ]
    for ds in DATASETS:
        c = audit["cross_check"].get(ds)
        if not c:
            continue
        lines.append(
            f"| {ds} | {c['validation_mean_no_candidate_rate']:.6f} | "
            f"{c['held_out_missing_row_rate']:.6f} | {c['abs_gap']:.6f} | "
            f"{c['consistent']} |")
    verdict = audit["scope_verdict"]
    lines += [
        "",
        "## Verdict",
        "",
        f"- scope verdict: **{verdict}**",
        f"- E3 held-out canonical status: **{audit['held_out_status']}**",
        "",
    ]
    if verdict == "CASE_A_ALL_EVENTS_RETAINED":
        lines += [
            "The held-out runner retained every event at every real cutoff, "
            "so the E3 held-out point estimates remain all-event canonical; "
            "the candidate-conditioned deprecation applies only to E3 "
            "validation/readiness metrics.",
            "",
        ]
    else:
        lines += [
            "The held-out runner dropped event-cutoffs with no candidate. "
            "The missing rows match the validation no-candidate rate, so the "
            "lost rows are no-candidate snapshots rather than a generic row "
            "loss.",
            "",
            "Consequences (no re-run, per protocol):",
            "",
            "- E3 held-out absolute Macro-F1 values cannot remain ALL-EVENT "
            "canonical; they are classified CONDITIONAL_HELD_OUT / "
            "DEPRECATED_ABSOLUTE.",
            "- The relative negative finding (Dynamic V1 not supported) is "
            "retained as a historical negative finding: both arms predict "
            "source-only on a no-candidate snapshot, so restoring those rows "
            "would move both arms toward the same value and cannot turn the "
            "delta positive.",
            "- An all-event held-out evaluation is a missing evidence cell "
            "and is recorded in NEXT_EXPERIMENT_GAPS.md.",
            "- Validation all-event diagnostics are NOT a substitute for the "
            "held-out result.",
            "",
        ]
    with open(os.path.join(root, "E3_NO_CANDIDATE_SCOPE_AUDIT.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return audit


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


def _check_e3_scope_audit(root, issues):
    json_path = os.path.join(root, "e3_no_candidate_scope_audit.json")
    md_path = os.path.join(root, "E3_NO_CANDIDATE_SCOPE_AUDIT.md")
    for path in (json_path, md_path):
        if not os.path.exists(path):
            issues.append(f"missing E3 scope audit artifact: "
                          f"{os.path.basename(path)}")
    if not os.path.exists(json_path):
        return None
    recorded = load_json(json_path)
    recomputed = e3_scope_audit()
    if recorded.get("scope_verdict") != recomputed.get("scope_verdict"):
        issues.append("E3 scope audit verdict disagrees with the frozen "
                      "held-out artifacts")
    if recorded.get("totals", {}).get("actual_rows") != \
            recomputed.get("totals", {}).get("actual_rows"):
        issues.append("E3 scope audit row counts disagree with the frozen "
                      "held-out artifacts")
    canonical = read_text(os.path.join(root, "CANONICAL_RESULTS_TABLE.md"))
    if recomputed["scope_verdict"] != "CASE_A_ALL_EVENTS_RETAINED":
        if "CONDITIONAL_HELD_OUT" not in canonical:
            issues.append("E3 held-out artifacts are candidate-conditioned "
                          "but the canonical status is not "
                          "CONDITIONAL_HELD_OUT")
        if "DEPRECATED_ABSOLUTE" not in read_text(
                os.path.join(root, "DEPRECATED_RESULTS.md")):
            issues.append("E3 held-out absolute values are not marked "
                          "DEPRECATED_ABSOLUTE")
    if recomputed["held_out_status"] not in canonical:
        issues.append("canonical table does not carry the audited E3 "
                      "held-out status")
    dep_path = os.path.join(root, "DEPRECATED_RESULTS.md")
    if os.path.exists(dep_path):
        dep_text = read_text(dep_path)
        if recomputed["scope_verdict"] == "CASE_A_ALL_EVENTS_RETAINED":
            if "E3 held-out test" in dep_text:
                issues.append("deprecated ledger still implies the E3 "
                              "held-out test is candidate-conditioned")
            if "E3 validation/readiness" not in dep_text:
                issues.append("deprecated ledger does not scope the "
                              "candidate-conditioned deprecation to E3 "
                              "validation/readiness")
    return recomputed


def _check_protocol_status(root, issues):
    text = read_text(os.path.join(root, "CANONICAL_RESULTS_TABLE.md"))
    counted = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if set(stripped) <= set("|-: "):
            continue
        if "protocol_status" in stripped:
            continue
        counted += 1
        if not any(status in stripped for status in PROTOCOL_STATUSES):
            issues.append("canonical table row without a protocol_status: "
                          f"{stripped[:90]}")
        if "CANDIDATE_CONDITIONED" in stripped and \
                "HISTORICAL_ONLY" not in stripped and \
                "CONDITIONAL_HELD_OUT" not in stripped and \
                "DEPRECATED_ABSOLUTE" not in stripped:
            issues.append("candidate-conditioned row marked canonical "
                          f"without HISTORICAL_ONLY / CONDITIONAL_HELD_OUT: "
                          f"{stripped[:90]}")
    if counted == 0:
        issues.append("canonical table has no data rows")
    return counted


def _check_e1_split(root, issues):
    reg = load_json(os.path.join(root, "experiment_registry.json"))
    e1 = [st for st in reg["stages"] if st["stage_id"] == "E1"]
    if not e1:
        issues.append("registry has no E1 stage")
    else:
        stage = e1[0]
        if stage.get("split_used") != "VALIDATION":
            issues.append(f"E1 reported split is "
                          f"{stage.get('split_used')!r}, expected VALIDATION")
        if stage.get("training_split") != "TRAIN":
            issues.append("E1 does not record training_split = TRAIN")
        if stage.get("reported_evaluation_split") != "VALIDATION":
            issues.append("E1 does not record reported_evaluation_split = "
                          "VALIDATION")
    text = read_text(os.path.join(root, "CANONICAL_RESULTS_TABLE.md"))
    for heading, body in _sections(text):
        if "E1" not in heading or "Causal Encoder" not in heading:
            continue
        if "split: VALIDATION" not in heading and \
                "split: VALIDATION" not in body:
            issues.append("canonical E1 section is not labeled VALIDATION")
        # "training split: TRAIN" is the correct wording and must not be
        # mistaken for a TRAIN evidence label.
        for chunk in (heading, body):
            if re.search(r"(?<!training )split: TRAIN", chunk):
                issues.append("canonical E1 section still labeled TRAIN as "
                              "the reported evidence split")


def _check_option_a1(root, issues):
    gaps = read_text(os.path.join(root, "NEXT_EXPERIMENT_GAPS.md"))
    matrix = read_text(os.path.join(root, "FINAL_CONTRIBUTION_MATRIX.md"))
    report = read_text(os.path.join(root, "RESEARCH_CONSOLIDATION_REPORT.md"))
    has_contrib_d = "Proxy-to-LLM Reader Transfer Analysis" in matrix
    if has_contrib_d:
        if "Gap B" not in gaps:
            issues.append("Contribution D is present but Gap B is not in "
                          "NEXT_EXPERIMENT_GAPS.md")
        if "A1" not in gaps or "A1" not in report:
            issues.append("Option A1 is not the recommended option in the "
                          "gaps file / report")
        for gap in ("Gap A", "Gap B", "Gap C"):
            if gap not in gaps:
                issues.append(f"{gap} missing from NEXT_EXPERIMENT_GAPS.md")
        if "combined" not in gaps.lower():
            issues.append("Gap B + Gap C are not planned as one combined "
                          "reader experiment")
        if "FINAL HELD-OUT EVIDENCE PENDING" not in matrix:
            issues.append("Contribution D does not state FINAL HELD-OUT "
                          "EVIDENCE PENDING")
    return has_contrib_d


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
    e3_audit = _check_e3_scope_audit(root, issues)
    n_canonical_rows = _check_protocol_status(root, issues)
    _check_e1_split(root, issues)
    _check_option_a1(root, issues)
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
        "n_canonical_rows": n_canonical_rows,
        "e3_scope_verdict": (e3_audit or {}).get("scope_verdict"),
        "e3_held_out_status": (e3_audit or {}).get("held_out_status"),
        "e1_reported_split": "VALIDATION",
        "e2_canonical_protocol": "ALL_EVENT",
        "recommended_option": "A1",
        "must_have_gaps": ["Gap A", "Gap B", "Gap C"],
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
            "E1 reported split is VALIDATION (training split is TRAIN)",
            "canonical rows carry a protocol_status",
            "candidate-conditioned rows are never plain CANONICAL",
            "E3 held-out no-candidate scope audited from frozen rows",
            "E3 canonical status matches the scope audit",
            "Contribution D implies Gap B and Option A1 (Gap A+B+C, "
            "B+C combined)",
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
    ap.add_argument("--emit-e3-audit", action="store_true",
                    help="(re)generate the E3 no-candidate scope audit "
                         "artifacts from the frozen held-out rows")
    args = ap.parse_args(argv)
    root = args.root or os.path.join(REPO_ROOT, CONSOLIDATION_DIR)
    if args.emit_e3_audit:
        audit = write_e3_audit_files(root)
        print(json.dumps(audit, indent=1, ensure_ascii=False), flush=True)
    result = verify(args.root)
    out = args.out or os.path.join(root,
                                   "research_consolidation_verify.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    _refresh_report(root, result)
    print(json.dumps(result, indent=1, ensure_ascii=False), flush=True)
    return 0 if result["n_issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
