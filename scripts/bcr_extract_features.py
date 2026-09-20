#!/usr/bin/env python
"""M1-B — fingerprints and ZERO-TOUCH feature extraction (M1 plan §8–§9).

Stages:

* ``fingerprints`` — merge the six M1-A probe shards, run the fail-closed
  coverage/identity audit and freeze the raw vectors + compact fingerprints;
* ``features`` — per dataset build E0 (MiniLM/SRC/text), E1 (frozen NLI) and
  E2 (tokenizer-only) rows for exactly the frozen ``I1_atomic`` keys, in one
  snapshot pass, then audit and freeze the caches;
* ``all`` — both.

All caches land under ``results/bcr_utility_v1/m1/`` with source hashes,
schema version, extractor identity, row counts and SHA256 (M1 plan §9).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "project"))
sys.path.insert(0, str(REPO / "scripts"))

import cr_tser_common as common  # noqa: E402

from bcr_utility.config import protocol as P  # noqa: E402
from bcr_utility.features import (evidence_features as ef,  # noqa: E402
                                  nli_features as nf,
                                  tokenizer_features as tf)
from bcr_utility.probes import fingerprint as fp  # noqa: E402

FEATURE_SCHEMA_VERSION = "bcr_m1_features_v1"
FINGERPRINT_SCHEMA_VERSION = "bcr_m1_fingerprint_v1"


class ExtractRefused(RuntimeError):
    """Raised when an extraction stage cannot proceed exactly."""


def _sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path, rows) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"path": path, "rows": len(rows), "bytes": os.path.getsize(path),
            "sha256": _sha_file(path)}


def _write_json(path, payload) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return {"path": path, "bytes": os.path.getsize(path),
            "sha256": _sha_file(path)}


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _atomic_entries(repo_root, dataset):
    index = _load_json(os.path.join(P.bootstrap_dir(repo_root),
                                    P.ATOMIC_INDEX_FILENAME))
    entries = (index.get("datasets") or {}).get(dataset, {}).get("entries")
    if not entries:
        raise ExtractRefused(f"{dataset}: atomic index missing/empty")
    return entries


def _manifest(repo_root):
    path = os.path.join(P.bootstrap_dir(repo_root),
                        P.PROBE_MANIFEST_FILENAME)
    return _load_json(path), _sha_file(path)


# --------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------
def run_fingerprints(repo_root) -> dict:
    manifest, manifest_sha = _manifest(repo_root)
    rows = []
    shards = {}
    for dataset in P.DATASETS:
        for reader in P.READER_KEYS:
            shard = os.path.join(
                P.m1_path(repo_root, "probe_responses"),
                f"probe_responses_{dataset}_{reader}.jsonl")
            if not os.path.exists(shard):
                raise ExtractRefused(f"probe shard missing: {shard}")
            with open(shard, encoding="utf-8") as fh:
                shard_rows = [json.loads(line) for line in fh if line.strip()]
            shards[f"{dataset}/{reader}"] = len(shard_rows)
            rows += shard_rows
    built = fp.build_fingerprints(rows, manifest)
    out_dir = P.m1_path(repo_root, "fingerprints")
    raw_payload = {
        "schema": FINGERPRINT_SCHEMA_VERSION,
        "protocol": P.PROTOCOL_VERSION,
        "probe_manifest_sha256": manifest_sha,
        "fields": list(fp.RAW_VECTOR_FIELDS),
        "readers": built["raw"],
    }
    raw_info = _write_json(os.path.join(out_dir, "raw_vectors.json"),
                           raw_payload)
    compact = {
        "schema": FINGERPRINT_SCHEMA_VERSION,
        "protocol": P.PROTOCOL_VERSION,
        "probe_manifest_sha256": manifest_sha,
        "dim": P.FINGERPRINT_DIM,
        "layout": {
            "datasets": list(P.DATASETS),
            "cutoffs": list(P.CUTOFFS_MIN),
            "contexts": list(P.PROBE_CONTEXTS),
            "base_metrics": list(P.FINGERPRINT_BASE_METRICS),
            "delta_metrics": list(P.FINGERPRINT_DELTA_METRICS),
        },
        "readers": {r: {"model_id": built["readers"][r]["model_id"],
                        "compact": built["readers"][r]["compact"],
                        "raw_sha256": built["readers"][r]["raw_sha256"]}
                    for r in P.READER_KEYS},
    }
    compact["sha256"] = fp.fingerprints_digest({"readers": compact["readers"]})
    compact_info = _write_json(os.path.join(out_dir, "fingerprints.json"),
                               compact)
    audit_payload = {
        "schema": FINGERPRINT_SCHEMA_VERSION,
        "probe_manifest_sha256": manifest_sha,
        "shard_rows": shards,
        "response_audit": built["audit"],
        "fingerprints_sha256": compact["sha256"],
        "forbidden_inputs": list(P.PROBE_FORBIDDEN_FIELDS),
        "ok": True,
    }
    audit_info = _write_json(os.path.join(out_dir, "fingerprint_audit.json"),
                             audit_payload)
    print(f"[m1b] fingerprints frozen: {compact['sha256'][:16]}... "
          f"({compact_info['bytes']} B)")
    return {"fingerprints": compact_info, "raw_vectors": raw_info,
            "audit": audit_info, "sha256": compact["sha256"]}


# --------------------------------------------------------------------------
# E0/E1/E2
# --------------------------------------------------------------------------
def _dataset_cleaner(dataset):
    from tcdscr.data.text_cleaning import clean_text_weibo, clean_tweet_pheme
    return clean_tweet_pheme if dataset == "pheme" else clean_text_weibo


def _nli_identity(nli_path, nli):
    from cr_tser.readers.base_reader import model_weight_hash, tokenizer_hash
    config_path = os.path.join(nli_path, "config.json")
    return {
        "model_id": P.NLI_MODEL_ID,
        "model_path": nli_path,
        "weight_hash": model_weight_hash(nli_path),
        "tokenizer_hash": tokenizer_hash(nli_path),
        "config_sha256": _sha_file(config_path)
        if os.path.exists(config_path) else "",
        "id2label": nli["id2label"],
        "device": nli["device"],
    }


def _reader_tokenizers(paths):
    """Tokenizer-only loads of the three frozen readers (E2; no weights).

    ``trust_remote_code`` follows each reader's own frozen reader class —
    InternLM's tokenizer requires it, Qwen/Mistral forbid it — so the E2
    counts describe exactly the tokenizers the readers use.
    """
    from transformers import AutoTokenizer
    trust = {"qwen": False, "mistral": False, "internlm": True}
    return {r: AutoTokenizer.from_pretrained(
        paths.reader_path(r), local_files_only=True,
        trust_remote_code=trust[r]) for r in P.READER_KEYS}


def run_features(dataset: str, paths, repo_root, nli_path: str,
                 device: str = "cpu") -> dict:
    t0 = time.time()
    entries = _atomic_entries(repo_root, dataset)
    expected_keys = [e["key"] for e in entries]
    event_ids = sorted({str(e["event_id"]) for e in entries})
    events = {str(e["event_id"]): e
              for e in common.load_dataset_events(dataset, paths)
              if str(e["event_id"]) in set(event_ids)}
    missing = sorted(set(event_ids) - set(events))
    if missing:
        raise ExtractRefused(f"{dataset}: {len(missing)} atomic events not "
                             f"loadable: {missing[:3]}")
    encoder = common.CrSemanticEncoder(paths.semantic_model, dataset)
    canonical = common.canonical_tokenizer(paths.canonical_tokenizer)
    reader_tokenizers = _reader_tokenizers(paths)
    nli = nf.load_nli(nli_path, device=device)
    nli_identity = _nli_identity(nli_path, nli)
    print(f"[m1b] {dataset}: NLI {nli_identity['model_id']} "
          f"weights={nli_identity['weight_hash'][:16]}... device={device}")

    entries_by_snapshot = {}
    for entry in entries:
        entries_by_snapshot.setdefault(
            (str(entry["event_id"]), int(entry["cutoff"])), []).append(entry)

    e0_rows, e1_text_rows, units_by_key = [], [], {}
    n_snap = len(entries_by_snapshot)
    for pos, ((event_id, cutoff), snapshot_entries) in enumerate(
            sorted(entries_by_snapshot.items())):
        event = events[event_id]
        art = common.snapshot_artifacts(event, cutoff, encoder, canonical)
        if art["zero_reply"]:
            raise ExtractRefused(
                f"{event_id}/{cutoff}: zero-reply snapshot carries atomic "
                "keys; refusing to continue")
        snapshot = art["snapshot"]
        e0_rows += ef.rows_for_snapshot(snapshot_entries, snapshot, art)
        source_text = next(n["text"] for n in event["nodes"]
                           if n["node_id"] == event["source_id"])
        unit_by_id = {u["node_id"]: u for u in art["units"]}
        for entry in snapshot_entries:
            unit = unit_by_id[entry["node_id"]]
            units_by_key[entry["key"]] = unit
            e1_text_rows.append({
                "key": entry["key"],
                "source_text": source_text,
                "reply_text": unit["reply_text"],
                "parent_text": unit.get("parent_text"),
            })
        if (pos + 1) % 60 == 0:
            print(f"[m1b] {dataset}: {pos + 1}/{n_snap} snapshots "
                  f"({time.time() - t0:.1f}s)")
    e1_rows = nf.e1_rows(e1_text_rows, nli, _dataset_cleaner(dataset))
    e2_rows = tf.e2_rows(units_by_key, reader_tokenizers, canonical)

    audits = {
        "e0": ef.validate_e0_rows(e0_rows, expected_keys),
        "e1": nf.validate_e1_rows(e1_rows, expected_keys),
        "e2": tf.validate_e2_rows(e2_rows, expected_keys),
    }
    out_dir = P.m1_path(repo_root, "features")
    e0_info = _write_jsonl(os.path.join(out_dir, f"e0_{dataset}.jsonl"),
                           e0_rows)
    e1_info = _write_jsonl(os.path.join(out_dir, f"e1_{dataset}.jsonl"),
                           e1_rows)
    e2_info = _write_jsonl(os.path.join(out_dir, f"e2_{dataset}.jsonl"),
                           e2_rows)
    audit = {
        "schema": FEATURE_SCHEMA_VERSION,
        "protocol": P.PROTOCOL_VERSION,
        "dataset": dataset,
        "n_atomic_keys": len(expected_keys),
        "e0": {**audits["e0"], **e0_info,
               "features": list(P.E0_FEATURE_NAMES),
               "struct_features_b1_only": list(P.B1_STRUCT_NAMES)},
        "e1": {**audits["e1"], **e1_info,
               "features": list(P.E1_FEATURE_NAMES),
               "extractor": nli_identity},
        "e2": {**audits["e2"], **e2_info,
               "features": list(P.E2_FEATURE_NAMES),
               "readers": list(P.READER_KEYS)},
        "seconds": time.time() - t0,
        "ok": True,
    }
    return audit


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("fingerprints", "features", "all"),
                    required=True)
    ap.add_argument("--dataset", choices=P.DATASETS, default=None)
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--nli-path", default=os.environ.get(
        "CRTSER_NLI_MODEL", P.NLI_SERVER_PATH))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--audit-out", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    summary = {"stage": args.stage, "protocol": P.PROTOCOL_VERSION}
    if args.stage in ("fingerprints", "all"):
        summary["fingerprints"] = run_fingerprints(args.repo_root)
    if args.stage in ("features", "all"):
        datasets = [args.dataset] if args.dataset else list(P.DATASETS)
        summary["features"] = {}
        for dataset in datasets:
            audit = run_features(dataset, paths, args.repo_root,
                                 args.nli_path, device=args.device)
            summary["features"][dataset] = audit
    audit_out = args.audit_out or P.m1_path(args.repo_root, "features",
                                            "feature_audit.json")
    if args.stage == "fingerprints":
        audit_out = args.audit_out or P.m1_path(
            args.repo_root, "fingerprints", "fingerprint_audit_full.json")
    info = _write_json(audit_out, summary)
    print(f"[m1b] audit written: {audit_out} ({info['bytes']} B)")
    print(json.dumps({k: v for k, v in summary.items() if k != "features"},
                     indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
