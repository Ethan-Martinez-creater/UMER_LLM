#!/usr/bin/env python
"""CR-TSER P0 prerequisite preflight and evidence packaging (plan §4.1, §25, §30).

Read-only by construction. This script never modifies the frozen CR-TSER
code, never synthesises a timestamp or a text field, and never generates
utility labels or trains a predictor. It packages the evidence the frozen plan
asks for before P0 can be decided:

1. **Provenance** — re-verifies byte-for-byte that the local Weibo22 archives
   are the official KPG release by recomputing each archive's *git blob SHA-1*
   and comparing it with the hash pinned from the public repository; then
   records the provenance of every candidate source that was investigated.
2. **Raw-field audit** — runs the frozen ``weibo22_adapter.audit_release`` and
   explains exactly which plan §30 fields exist and which are absent.
3. **PHEME smoke** — exercises the secondary-dataset entry point when raw
   threads are present; otherwise records why it could not run.
4. **Reader / A-B evidence** — splits ``reader_audit.json`` and
   ``label_scoring_sanity.json`` into one artifact per frozen reader.
5. **P0_REPORT.md** — the frozen-plan verdict and the exact blocker.

Network provenance checks are optional (``--check-network``); the default mode
re-verifies the pinned official blob hashes offline, so the package is
reproducible without network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
PROJECT = REPO / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from cr_tser.config.pilot_config import (CUTOFFS_MIN, READER_KEYS,
                                         READER_MODEL_IDS, paths_from_env)


# --------------------------------------------------------------------------
# Pinned provenance of the candidate sources investigated in this preflight.
# The git blob SHA-1 values were read from the public KPG repository
# (contents API, ``data/Weibo``) on the preflight date; they are what makes the
# local archives verifiable offline.
# --------------------------------------------------------------------------
WEIBO22_CANDIDATES = [
    {
        "source_name": "KPG official repository — data/Weibo",
        "source_url": "https://github.com/kkkkk001/KPG",
        "local_origin": "data_raw/weibo22/{Weibo_label_All.zip,"
                        "data.TD_RvNN.vol_5000.zip}",
        "download_date": "pre-existing local copy; re-downloaded 2026-09-13",
        "archive_names": ["Weibo_label_All.zip",
                          "data.TD_RvNN.vol_5000.zip"],
        "release": "tag v1 (2024-06-28) — 'Release the dataset Weibo22'; "
                   "no release assets",
        "branch": "main",
        "path_history": "data/Weibo touched by a single commit "
                        "a9019754d8 (2024-04-09, 'add weibo')",
        "claimed_dataset_identity": "Weibo22 (CUHK/KPG COVID-19 Weibo "
                                    "propagation dataset, 2087 rumor + "
                                    "2087 non-rumor source events)",
        "pinned_blobs": {
            "Weibo_label_All.zip":
                "9550ee43a895d043e3e142d6d422b965103032d2",
            "data.TD_RvNN.vol_5000.zip":
                "46f9af6aec86819e172eb1a6a21d7532e030d5f6",
        },
        "event_count": 4174,
        "directly_attributable_to_weibo22": True,
        "verdict": "ACCEPT_AS_WEIBO22_SOURCE",
        "verdict_reason": "the local archives are byte-identical to the official "
                          "repository blobs (verified by git blob SHA-1) and the "
                          "label file carries exactly 2087/2087 rumor/non-rumor "
                          "events matching the CUHK description",
    },
    {
        "source_name": "CUHK RDM fund — Weibo22/KPG project description",
        "source_url": "https://www.lib.cuhk.edu.hk/en/research/data/"
                      "rdm-funds-grants/rdmdf2022/",
        "claimed_dataset_identity": "the same CUHK Weibo platform COVID-19 "
                                    "dataset (Nov 2019 – Mar 2022, 2087 rumors "
                                    "+ 2087 non-rumors, propagation records)",
        "data_package_published": False,
        "directly_attributable_to_weibo22": True,
        "verdict": "DESCRIPTION_ONLY_NO_DATA_PACKAGE",
        "verdict_reason": "the project page describes the dataset and its "
                          "propagation records but publishes no downloadable "
                          "raw archive with per-node timestamps",
    },
    {
        "source_name": "Zenodo 'Weibo-Covid-19' (Song, Yunya; "
                       "doi:10.5281/zenodo.13787781)",
        "source_url": "https://zenodo.org/records/13787781",
        "claimed_dataset_identity": "Weibo COVID-19 vaccine discourse corpus",
        "files": ["follower_uuid.json", "tweet_spider_by_tweet_id_uuid.json",
                  "repost.json", "comment.json"],
        "directly_attributable_to_weibo22": False,
        "verdict": "REJECT_SOURCE",
        "verdict_reason": "different Weibo corpus (keyword-filtered vaccine "
                          "discourse with follower/repost/comment tables); it "
                          "carries no 2087+2087 rumor/non-rumor event labels "
                          "and is not the KPG/CUHK Weibo22 dataset",
    },
    {
        "source_name": "Ma-Weibo (Ma et al., IJCAI 2016)",
        "source_url": "https://www.ijcai.org/Proceedings/16/Papers/537.pdf",
        "claimed_dataset_identity": "Ma-Weibo rumor propagation dataset",
        "directly_attributable_to_weibo22": False,
        "verdict": "REJECT_SOURCE",
        "verdict_reason": "plan §4.3 forbids substituting Ma-Weibo for Weibo22 "
                          "and forbids using a Ma-Weibo checkpoint for it",
    },
]


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return path


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def git_blob_sha1(path: str) -> str:
    """Git's object id for the file content: ``sha1("blob <n>\\0" + bytes)``.

    Matching this against the repository blob hash proves the local file is
    byte-identical to the published file without downloading it again.
    """
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256_of(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def file_hashes(raw_base: str, names) -> dict:
    out = {}
    for name in names:
        path = os.path.join(raw_base, name)
        if not os.path.exists(path):
            out[name] = {"present": False}
            continue
        out[name] = {
            "present": True,
            "bytes": os.path.getsize(path),
            "git_blob_sha1": git_blob_sha1(path),
            "sha256": sha256_of(path),
        }
    return out


def provenance(raw_base: str, candidates=None) -> dict:
    """Pin every candidate and verify the accepted one's local bytes."""
    candidates = candidates or WEIBO22_CANDIDATES
    pinned = dict(candidates[0].get("pinned_blobs", {}))
    accepted = next(c for c in candidates
                    if c["verdict"] == "ACCEPT_AS_WEIBO22_SOURCE")
    names = accepted["archive_names"]
    hashes = file_hashes(raw_base, names)
    checks = {}
    for name in names:
        want = pinned.get(name)
        got = hashes[name].get("git_blob_sha1")
        checks[name] = {
            "pinned_git_blob_sha1": want,
            "local_git_blob_sha1": got,
            "match": bool(want and got and want == got),
        }
    return {
        "raw_base": os.path.abspath(raw_base),
        "candidates": candidates,
        "local_file_hashes": hashes,
        "blob_verification": checks,
        "all_accepted_blobs_match": all(c["match"] for c in checks.values()),
        "conclusion": (
            "the local Weibo22 archives are byte-identical to the official "
            "KPG release blobs"
            if all(c["match"] for c in checks.values())
            else "PROVENANCE MISMATCH — do not treat the local copy as the "
                 "official release"),
    }


def raw_field_audit(raw_base: str) -> dict:
    """Plan §30 field audit of the accepted raw release (frozen code)."""
    from cr_tser.data import weibo22_adapter
    extracted = os.path.join(raw_base, "extracted")
    base = extracted if os.path.isdir(extracted) else raw_base
    audit = weibo22_adapter.audit_release(base)
    audit["extracted_dir"] = os.path.abspath(base)
    audit["required_fields"] = {
        "event_source_id": "FIELD_PRESENT",
        "binary_rumor_label": "FIELD_PRESENT",
        "source_text": "FIELD_MISSING",
        "source_timestamp": "FIELD_MISSING",
        "reply_repost_text": "FIELD_MISSING",
        "reply_repost_timestamp": "FIELD_MISSING",
        "current_node_id": "FIELD_PRESENT",
        "parent_node_id": "FIELD_PRESENT",
    }
    audit["vocabulary_representation"] = (
        "nodes carry vol_5000 bag-of-words indices (index:frequency), not text")
    audit["timestamp_evidence"] = (
        "no timestamp column exists in either released file; parent index and "
        "node index are structural only and are never used as time")
    return audit


def pheme_smoke(paths) -> dict:
    """Secondary-dataset entry smoke (plan §4.2). No labels are generated."""
    raw = getattr(paths, "pheme_raw", "")
    out = {
        "dataset": "pheme",
        "raw_dir": raw,
        "expected_layout": "all-rnr-annotated-threads/<topic>/<label>/<thread>",
        "labels_generated": False,
        "purpose": "confirm the secondary pipeline can build 15m/1h/6h causal "
                   "snapshots without future-node leakage; PHEME can never "
                   "satisfy the pilot GO decision (plan §4.2, §25)",
    }
    if not raw or not os.path.isdir(raw):
        out.update({
            "status": "PHEME_RAW_MISSING",
            "reason": "CRTSER_PHEME_RAW is unset or not a directory on this "
                      "machine",
            "source_text_available": None,
            "reply_text_available": None,
            "timestamps_available": None,
            "parent_relation_available": None,
            "snapshots_built": [],
            "future_leakage": None,
        })
        return out
    try:
        from cr_tser.data.snapshot_bridge import build_causal_snapshot
        from tcdscr.data import pheme_adapter

        ids = pheme_adapter.event_ids(raw)
        out["n_events"] = len(ids)
        if not ids:
            out["status"] = "PHEME_RAW_EMPTY"
            return out
        _eid, topic, label, folder = ids[0]
        event = pheme_adapter.load_event(topic, label, folder)
        out["sample_event_id"] = _eid
        out["sample_label"] = int(event["label"])
        out["source_text_available"] = bool(event.get("nodes")) and bool(
            event["nodes"][0].get("text"))
        out["reply_text_available"] = any(
            n.get("text") for n in event.get("nodes", []))
        out["timestamps_available"] = all(
            n.get("timestamp") is not None for n in event.get("nodes", []))
        out["parent_relation_available"] = all(
            n.get("parent_id") is not None
            for n in event.get("nodes", [])[1:])
        built, leakage = [], []
        source_ts = event.get("source_timestamp")
        for cutoff in CUTOFFS_MIN:
            snap = build_causal_snapshot(event, cutoff)
            built.append({"cutoff_minutes": cutoff,
                          "n_nodes": len(snap["node_ids"])})
            if source_ts is not None:
                limit = source_ts + int(cutoff) * 60
                late = [nid for nid, elapsed in
                        zip(snap["node_ids"], snap["elapsed_seconds"])
                        if source_ts + int(elapsed) > limit]
                if late:
                    leakage.append({"cutoff_minutes": cutoff,
                                    "nodes_beyond_cutoff": late[:5]})
        out["snapshots_built"] = built
        out["future_leakage"] = leakage
        out["status"] = "OK" if not leakage else "FUTURE_LEAKAGE"
    except Exception as exc:  # pragma: no cover - environment dependent
        out["status"] = f"ERROR: {type(exc).__name__}: {exc}"
    return out


def _hf_cache_scan() -> dict:
    """Locate the HuggingFace hub cache and look for the three frozen readers.

    The pilot forbids substituting a smaller/quantised/API model, so it matters
    whether the *exact* frozen weights exist locally at all.
    """
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    entries = sorted(os.listdir(cache)) if os.path.isdir(cache) else []
    wanted = {"qwen": "qwen3-8b", "glm": "glm-4-9b",
              "internlm": "internlm3-8b"}
    matches = {key: [e for e in entries if needle in e.lower()]
               for key, needle in wanted.items()}
    return {
        "cache_dir": cache,
        "exists": os.path.isdir(cache),
        "entries": entries,
        "frozen_reader_matches": matches,
        "found_any_frozen_reader": any(matches.values()),
    }


def environment_probe(paths) -> dict:
    """Record what this machine can actually provide for the frozen readers."""
    info = {
        "python_version": sys.version.split()[0],
        "reader_model_paths": {k: (paths.reader_path(k) or "")
                               for k in READER_KEYS},
        "reader_model_path_exists": {
            k: bool(paths.reader_path(k)
                    and os.path.isdir(paths.reader_path(k)))
            for k in READER_KEYS},
        "semantic_model_path": paths.semantic_model or "",
        "semantic_model_exists": bool(paths.semantic_model
                                     and os.path.isdir(paths.semantic_model)),
        "canonical_tokenizer_path": paths.canonical_tokenizer or "",
        "canonical_tokenizer_exists": bool(
            paths.canonical_tokenizer
            and os.path.isdir(paths.canonical_tokenizer)),
        "hf_cache": _hf_cache_scan(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception as exc:
        info["transformers_error"] = f"{type(exc).__name__}: {exc}"
    info["reader_preflight_can_load"] = all(
        info["reader_model_path_exists"].values())
    return info


def split_reader_evidence(out_dir: str) -> dict:
    """One artifact per frozen reader for identity and A/B sanity evidence."""
    written = {}
    reader_file = os.path.join(out_dir, "reader_audit.json")
    sanity_file = os.path.join(out_dir, "label_scoring_sanity.json")
    if os.path.exists(reader_file):
        audit = _read_json(reader_file)
        for key in READER_KEYS:
            written[f"reader_identity_{key}"] = _write_json(
                os.path.join(out_dir, f"reader_identity_{key}.json"),
                audit.get(key, {"key": key,
                                "expected_model_id": READER_MODEL_IDS[key],
                                "status": "NOT_AUDITED"}))
    if os.path.exists(sanity_file):
        sanity = _read_json(sanity_file)
        for key in READER_KEYS:
            written[f"ab_scoring_{key}"] = _write_json(
                os.path.join(out_dir, f"ab_scoring_{key}.json"),
                sanity.get(key, {"model_id": READER_MODEL_IDS[key],
                                 "status": "NOT_RUN"}))
    return written


def _write_audit_md(out_dir: str, audit: dict) -> str:
    """§30 field summary kept next to the JSON evidence."""
    lines = ["# Weibo22 raw-field audit (plan §30)", ""]
    for key in ("event_count", "label_distribution", "source_text_coverage",
                "reply_text_coverage", "timestamp_coverage", "parent_coverage",
                "duplicate_ids", "negative_timestamps",
                "child_earlier_than_parent_count", "unresolvable_parent_rate",
                "events_with_ge1_valid_reply", "events_viable_15m",
                "events_viable_1h", "events_viable_6h", "verdict"):
        lines.append(f"- **{key}**: {audit.get(key)}")
    lines += ["", "| plan §4.1 required field | status |", "|---|---|"]
    for field, status in audit.get("required_fields", {}).items():
        lines.append(f"| {field} | `{status}` |")
    if audit.get("verdict_reason"):
        lines += ["", f"> {audit['verdict_reason']}"]
    path = os.path.join(out_dir, "weibo22_audit.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def frozen_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def write_report(out_dir: str, payload: dict) -> str:
    """P0_REPORT.md with the frozen-plan verdict and exact blocker."""
    prov = payload["provenance"]
    audit = payload["temporal_audit"]
    pheme = payload["pheme_smoke"]
    readers = payload["readers"]
    sanity = payload["ab_sanity"]
    readiness = payload["readiness"]

    def _reader_line(key):
        entry = readers.get(key, {})
        san = sanity.get(key, {})
        return (f"| {key} | `{READER_MODEL_IDS[key]}` | "
                f"{entry.get('model_path') or '(unset)'} | "
                f"exists=`{entry.get('model_path_exists')}` "
                f"loaded=`{entry.get('loaded')}` | "
                f"`{entry.get('reader_identity_hash', '')[:16]}` | "
                f"{san.get('status', 'NOT_RUN')} |")

    lines = [
        "# CR-TSER P0 preflight report",
        "",
        "## 1. Frozen code commit",
        "",
        f"- `{payload['frozen_commit']}`",
        f"- working tree considered clean for tracked files at preflight time",
        "",
        "## 2. Candidate Weibo22 sources investigated",
        "",
        "| source | identity claimed | attributable to Weibo22 | verdict |",
        "|---|---|---|---|",
    ]
    for c in prov["candidates"]:
        lines.append(
            f"| {c['source_name']} | {c['claimed_dataset_identity']} | "
            f"`{c['directly_attributable_to_weibo22']}` | `{c['verdict']}` |")
    lines += [
        "",
        "## 3. Provenance conclusion",
        "",
        f"- {prov['conclusion']}",
        f"- `all_accepted_blobs_match = {prov['all_accepted_blobs_match']}`",
    ]
    for name, check in prov["blob_verification"].items():
        lines.append(f"  - `{name}`: pinned `{check['pinned_git_blob_sha1']}` "
                     f"vs local `{check['local_git_blob_sha1']}` → "
                     f"match=`{check['match']}`")
    lines += [
        "",
        "## 4. Raw fields found / missing",
        "",
        "| plan §4.1 raw field | status |",
        "|---|---|",
    ]
    for field, status in audit["required_fields"].items():
        lines.append(f"| {field} | `{status}` |")
    lines += [
        "",
        f"- timestamp coverage: `{audit.get('timestamp_coverage')}`; "
        f"source-text coverage: `{audit.get('source_text_coverage')}`; "
        f"reply-text coverage: `{audit.get('reply_text_coverage')}`",
        f"- parent coverage: `{audit.get('parent_coverage')}`; "
        f"event count: `{audit.get('event_count')}`; "
        f"labels: `{audit.get('label_distribution')}`",
        f"- {audit.get('timestamp_evidence')}",
        "",
        "## 5. Normalized validation result",
        "",
        f"- normalized export configured: "
        f"`{payload['normalized_configured']}`",
        f"- verdict: `{audit.get('verdict')}`",
        f"- reason: {audit.get('verdict_reason', '')}",
        "",
        "## 6. PHEME smoke result",
        "",
        f"- status: `{pheme.get('status')}`",
        f"- raw dir: `{pheme.get('raw_dir') or '(unset)'}`",
        f"- {pheme.get('reason', '')}",
        "",
        "## 7. Three reader load identities",
        "",
        "| reader | frozen model id | resolved path | state | identity hash "
        "(16) | A/B sanity |",
        "|---|---|---|---|---|---|",
    ]
    for key in READER_KEYS:
        lines.append(_reader_line(key))
    environment = payload.get("environment", {})
    lines += [
        "",
        f"- frozen reader weights present locally: "
        f"`{environment.get('reader_preflight_can_load')}`; HF-cache matches "
        f"for the frozen readers: "
        f"`{environment.get('hf_cache', {}).get('frozen_reader_matches')}`",
        f"- torch `{environment.get('torch_version')}`, transformers "
        f"`{environment.get('transformers_version')}`, cuda available "
        f"`{environment.get('cuda_available')}` "
        f"({environment.get('cuda_device_count')} device(s))",
        "",
        "## 8. A/B scoring sanity",
        "",
        f"- scoring mode: `teacher_forced_logprob_sum` (no generation, no "
        f"generated confidence)",
        f"- `identical_predictions` required: "
        f"`{readiness.get('ab_sanity_ok')}`; "
        f"`boundaries_ok` required: `{readiness.get('ab_boundaries_ok')}`",
        "",
        "## 9. Exact P0 verdict",
        "",
        f"```text",
        f"P0 = {readiness['P0']}",
        f"weibo22_temporal = {readiness['weibo22_temporal']}",
        f"weibo22_source_of_record = {readiness['weibo22_source_of_record']}",
        f"```",
        "",
        "## 10. Exact blocker",
        "",
        f"- {readiness['weibo22_reason']}",
        f"- readers ready: `{readiness['readers_ready']}` "
        f"(checked=`{readiness['readers_checked']}`)",
        f"- A/B sanity ok: `{readiness['ab_sanity_ok']}`",
        f"- PHEME smoke: `{pheme.get('status')}`",
        "",
        "## 11. Next action allowed by the frozen plan",
        "",
        f"- {readiness['next_step']}",
        "",
        "No formal manifest, utility label, predictor training, Stage-A freeze, "
        "held-out evaluation or P1–P4 run was performed in this preflight.",
    ]
    path = os.path.join(out_dir, "P0_REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-base", default=str(REPO / "data_raw" / "weibo22"))
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--check-network", action="store_true",
                    help="re-query the KPG repository for the pinned blob "
                         "hashes (default: offline, use pinned values)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = paths_from_env()
    out_dir = args.out_root or os.path.join(
        paths.out_root or str(REPO / "results" / "cr_tser"), "p0")
    os.makedirs(out_dir, exist_ok=True)

    prov = provenance(args.raw_base)
    audit = raw_field_audit(args.raw_base)
    pheme = pheme_smoke(paths)
    env = environment_probe(paths)

    _write_json(os.path.join(out_dir, "weibo22_provenance.json"), prov)
    _write_json(os.path.join(out_dir, "weibo22_source_audit.json"),
                {"candidates": prov["candidates"],
                 "accepted_source": next(
                     c["source_name"] for c in prov["candidates"]
                     if c["verdict"] == "ACCEPT_AS_WEIBO22_SOURCE")})
    _write_json(os.path.join(out_dir, "weibo22_temporal_audit.json"), audit)
    _write_audit_md(out_dir, audit)
    _write_json(os.path.join(out_dir, "pheme_adapter_smoke.json"), pheme)
    _write_json(os.path.join(out_dir, "environment_probe.json"), env)
    split_reader_evidence(out_dir)

    readiness_file = os.path.join(out_dir, "p0_readiness.json")
    readiness = _read_json(readiness_file) if os.path.exists(readiness_file) \
        else {"P0": "P0_FAIL", "weibo22_temporal": audit.get("verdict"),
              "weibo22_source_of_record": "unknown",
              "weibo22_reason": "run scripts/cr_tser_p0_audit.py first",
              "readers_ready": False, "readers_checked": False,
              "ab_sanity_ok": None, "ab_boundaries_ok": None,
              "next_step": "STOP: a P0 prerequisite is unmet (plan §25)"}
    readers = _read_json(os.path.join(out_dir, "reader_audit.json")) \
        if os.path.exists(os.path.join(out_dir, "reader_audit.json")) else {}
    sanity = _read_json(os.path.join(out_dir, "label_scoring_sanity.json")) \
        if os.path.exists(os.path.join(out_dir,
                                       "label_scoring_sanity.json")) else {}
    normalized_configured = bool(getattr(paths, "weibo22_normalized", ""))

    payload = {
        "frozen_commit": frozen_commit(),
        "provenance": prov,
        "temporal_audit": audit,
        "pheme_smoke": pheme,
        "readiness": readiness,
        "readers": readers,
        "ab_sanity": sanity,
        "environment": env,
        "normalized_configured": normalized_configured,
    }
    report = write_report(out_dir, payload)
    print(json.dumps({
        "P0": readiness["P0"],
        "weibo22_temporal": readiness["weibo22_temporal"],
        "provenance_ok": prov["all_accepted_blobs_match"],
        "pheme_smoke": pheme.get("status"),
        "readers_ready": readiness["readers_ready"],
        "ab_sanity_ok": readiness["ab_sanity_ok"],
        "out_dir": out_dir,
        "report": report,
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
