#!/usr/bin/env python
"""Fail-closed V2 -> v2r1 utility-label cache migration (amendment R1 §5, §12).

The completed Qwen and InternLM labels stay scientifically valid across the
reader amendment: a row's score depends on the frozen source, the frozen
event/cutoff/intervention, the frozen prompt and *its own* reader identity —
never on which third reader participates in LORO. Migration is therefore a
verified copy, not a re-scoring.

Every guard fails closed:

* only ``MIGRATABLE_READERS`` may migrate — a retired ``glm`` row can never
  reach the target cache (amendment R1 §4);
* the target namespace must already carry the same frozen source and the same
  manifest/reader digests as the source namespace;
* every retained row keeps its §32 identity fields byte-identical, so the
  prompt/context/reader fingerprints cannot change under migration;
* duplicate cache keys are fatal, in the source and in the product;
* the target cache must be empty before the first migration: nothing is ever
  overwritten or appended to an existing cache.

The round that implements this file must not run it against the formal V2
cache (amendment R1 §12).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_tser_common as common  # noqa: E402

from cr_tser.data.source_manifest import (SourceIdentityError,  # noqa: E402
                                          assert_same_source)
from cr_tser.intervention.evidence_units import (  # noqa: E402
    structured_group_key)

#: Only the two reader keys that survive amendment R1 may migrate.
MIGRATABLE_READERS = ("qwen", "internlm")
#: The retired R0 reader: present in historical V2 evidence, never migrated.
RETIRED_READERS = ("glm",)
#: Every field that must survive migration unchanged (plan §32 identity).
IDENTITY_FIELDS = ("base_context_hash", "intervened_context_hash",
                   "reader_hash", "reader_identity_hash", "tokenizer_hash",
                   "chat_template_hash", "prompt_hash", "prompt_ids_hash")


class MigrationRefused(RuntimeError):
    """Raised whenever a migration guard fails; no file is written."""


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_rows(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cache_key(row) -> str:
    return structured_group_key(row["dataset"], row["event_id"], row["cutoff"],
                                f"{row['reader']}@{row['intervention_id']}")


def _manifest_dir(root: str, dataset: str) -> str:
    return os.path.join(root, "manifests", dataset)


def _assert_target_namespace_matches(source_root: str, target_root: str,
                                     dataset: str) -> dict:
    """The two namespaces must describe the same frozen data and readers."""
    src_dir = _manifest_dir(source_root, dataset)
    dst_dir = _manifest_dir(target_root, dataset)
    if not os.path.isdir(dst_dir):
        raise MigrationRefused(
            f"target namespace has no manifests for {dataset!r}: {dst_dir}; "
            "freeze the v2r1 manifests before migrating any label")
    src_source = _load_json(os.path.join(src_dir, "source.json"))
    dst_source = _load_json(os.path.join(dst_dir, "source.json"))
    try:
        assert_same_source(src_source, dst_source,
                           context=f"migration/{dataset}")
    except SourceIdentityError as exc:
        raise MigrationRefused(
            f"source identity drift between the two namespaces: {exc}") from exc
    src_hashes = _load_json(os.path.join(src_dir, "hashes.json"))
    dst_hashes = _load_json(os.path.join(dst_dir, "hashes.json"))
    mismatched = {}
    for field in ("event_split_sha256", "cutoffs", "viable_events",
                  "n_snapshots", "n_interventions"):
        if src_hashes.get(field) != dst_hashes.get(field):
            mismatched[field] = (src_hashes.get(field), dst_hashes.get(field))
    for reader in MIGRATABLE_READERS:
        src_reader = src_hashes.get("readers", {}).get(reader, {})
        dst_reader = dst_hashes.get("readers", {}).get(reader, {})
        if not dst_reader:
            raise MigrationRefused(
                f"target manifest has no {reader!r} reader record; the v2r1 "
                "manifest must pin every migratable reader")
        for field in ("model_id", "weight_hash", "tokenizer_hash"):
            if src_reader.get(field) != dst_reader.get(field):
                mismatched[f"readers.{reader}.{field}"] = (
                    src_reader.get(field), dst_reader.get(field))
    if mismatched:
        raise MigrationRefused(
            f"manifest drift between namespaces: {mismatched}")
    return {"source": src_source, "hashes": src_hashes}


def _select_rows(rows) -> tuple:
    """Split the source cache into migratable rows and skipped rows."""
    kept, skipped_glm, skipped_other = [], 0, 0
    for row in rows:
        reader = row.get("reader")
        if reader in MIGRATABLE_READERS:
            kept.append(row)
        elif reader in RETIRED_READERS:
            skipped_glm += 1
        else:
            skipped_other += 1
    return kept, skipped_glm, skipped_other


def _assert_no_forbidden_reader(rows) -> None:
    offenders = sorted({r.get("reader") for r in rows
                        if r.get("reader") not in MIGRATABLE_READERS})
    if offenders:
        raise MigrationRefused(
            f"retired/unknown readers would enter the target cache: "
            f"{offenders}; only {list(MIGRATABLE_READERS)} may migrate")


def _assert_unique(rows, where: str) -> None:
    seen = set()
    duplicates = 0
    for row in rows:
        key = _cache_key(row)
        if key in seen:
            duplicates += 1
        seen.add(key)
    if duplicates:
        raise MigrationRefused(
            f"{duplicates} duplicate cache keys in {where}; refusing to write "
            "a cache with duplicated rows")


def _assert_identity_preserved(source_rows, migrated_rows) -> None:
    if len(source_rows) != len(migrated_rows):
        raise MigrationRefused("row count changed during migration")
    for before, after in zip(source_rows, migrated_rows):
        if before != after:
            drift = [f for f in IDENTITY_FIELDS
                     if before.get(f) != after.get(f)]
            raise MigrationRefused(
                f"row identity changed during migration for "
                f"{_cache_key(before)}: {drift or 'non-identity field'}")


def _assert_row_reader_identity(rows, hashes: dict) -> None:
    """Every migratable row must carry the identity the manifest pins.

    A cache produced by a different reader build (a re-tokenized or re-templated
    checkpoint, a different dtype) would otherwise migrate silently.
    """
    pinned = hashes.get("readers", {})
    for row in rows:
        reader = row.get("reader")
        expected = pinned.get(reader, {})
        if not expected:
            raise MigrationRefused(
                f"manifest pins no reader {reader!r}; cannot verify its rows")
        for field, key in (("reader_hash", "weight_hash"),
                           ("tokenizer_hash", "tokenizer_hash")):
            if row.get(field) != expected.get(key):
                raise MigrationRefused(
                    f"reader identity drift for {reader!r} at "
                    f"{_cache_key(row)}: {field}={row.get(field)!r} "
                    f"manifest={expected.get(key)!r}")


def _target_cache_path(target_root: str, dataset: str) -> str:
    return os.path.join(target_root, "utility_labels", dataset, "labels.jsonl")


def migrate(dataset: str, source_root: str, target_root: str,
            apply: bool = False, out_path: str | None = None) -> dict:
    """Verified copy of the migratable rows; nothing is written unless asked."""
    if dataset not in ("maweibo", "pheme"):
        raise MigrationRefused(f"unknown dataset {dataset!r}")
    if os.path.abspath(source_root) == os.path.abspath(target_root):
        raise MigrationRefused("source and target namespaces are the same")

    src_cache = os.path.join(source_root, "utility_labels", dataset,
                             "labels.jsonl")
    if not os.path.isfile(src_cache):
        raise MigrationRefused(f"source cache does not exist: {src_cache}")
    target_cache = out_path or _target_cache_path(target_root, dataset)
    if os.path.exists(target_cache) and os.path.getsize(target_cache) > 0:
        raise MigrationRefused(
            f"target cache is not empty: {target_cache}; the first migration "
            "requires an empty cache (amendment R1 §12)")

    manifest = _assert_target_namespace_matches(source_root, target_root,
                                                dataset)

    rows = _load_rows(src_cache)
    if not rows:
        raise MigrationRefused(f"source cache is empty: {src_cache}")
    _assert_unique(rows, "the source cache")
    migrated, skipped_glm, skipped_other = _select_rows(rows)
    if not migrated:
        raise MigrationRefused("no migratable rows found")
    _assert_row_reader_identity(migrated, manifest["hashes"])
    _assert_no_forbidden_reader(migrated)
    _assert_unique(migrated, "the migrated selection")
    _assert_identity_preserved(
        [r for r in rows if r.get("reader") in MIGRATABLE_READERS], migrated)

    report = {
        "dataset": dataset,
        "source_root": os.path.abspath(source_root),
        "target_root": os.path.abspath(target_root),
        "source_cache": os.path.abspath(src_cache),
        "target_cache": os.path.abspath(target_cache),
        "source_rows": len(rows),
        "migrated_rows": len(migrated),
        "skipped_retired_rows": skipped_glm,
        "skipped_unknown_reader_rows": skipped_other,
        "readers_migrated": sorted({r["reader"] for r in migrated}),
        "duplicate_keys": 0,
        "identity_fields_preserved": list(IDENTITY_FIELDS),
        "dry_run": not apply,
    }
    if apply:
        os.makedirs(os.path.dirname(os.path.abspath(target_cache)),
                    exist_ok=True)
        with open(target_cache, "x", encoding="utf-8") as fh:
            for row in migrated:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["written_bytes"] = os.path.getsize(target_cache)
    return report


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("maweibo", "pheme"), required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--target-root", required=True)
    ap.add_argument("--out", default=None,
                    help="target cache path (default: <target-root>/"
                         "utility_labels/<dataset>/labels.jsonl)")
    ap.add_argument("--apply", action="store_true",
                    help="write the target cache; without it this is a dry run")
    ap.add_argument("--out-report", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = migrate(args.dataset, args.source_root, args.target_root,
                         apply=args.apply, out_path=args.out)
    except MigrationRefused as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}, indent=1))
        return 3
    print(json.dumps(report, indent=1))
    if args.out_report:
        common.write_json(args.out_report, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
