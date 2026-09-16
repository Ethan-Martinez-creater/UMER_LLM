"""BCR bootstrap atomic evidence index (plan §11.4, §11.5).

The new research line only needs the **atomic removal** utility
``u_r(e) = p_r(gold|SRC) - p_r(gold|SRC \\ e)``. M0 rebuilds that view from the
frozen historical caches rather than regenerating anything:

* only ``I1_atomic`` rows enter the index; ``I0`` base rows and the structured
  ``I2``–``I5`` groups are counted and explicitly excluded (plan §5, §12);
* the canonical evidence identity is imported from the frozen CR-TSER
  implementation, so a BCR key can never disagree with a historical key;
* every row is re-labelled with the frozen tri-class rule and compared against
  the ``sign`` the historical cache already carries — a mismatch fails closed.

``atomic_keys`` counts distinct ``(event, cutoff, reply_node)`` identities.
"""
from __future__ import annotations

import math
import os

from ..config import protocol as P
from .historical_import import (iter_jsonl, labels_path, read_json,
                                sha256_bytes, sha256_file)

try:  # reused read-only from the frozen CR-TSER implementation
    from cr_tser.intervention.evidence_units import evidence_key
    from cr_tser.models.utility_heads import tri_class_label
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "BCR reuses cr_tser.evidence_units / cr_tser.models.utility_heads "
        "read-only; the repository's project directory must be importable"
    ) from exc


class AtomicIndexRefused(RuntimeError):
    """Raised when the atomic index cannot be built fail-closed."""


def _finite(value, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AtomicIndexRefused(
            f"{what}: non-numeric utility {value!r}") from exc
    if math.isnan(number) or math.isinf(number):
        raise AtomicIndexRefused(f"{what}: non-finite utility {value!r}")
    return number


def extract_atomic_index(labels_file: str, dataset: str,
                         readers=P.READER_KEYS) -> dict:
    """``{key: {reader -> utility/sign/correctness}}`` from ``I1_atomic`` rows.

    The returned mapping is keyed by the frozen canonical evidence key, so the
    same reply at a different cutoff (or in another dataset) can never collide.
    """
    if dataset not in P.DATASETS:
        raise AtomicIndexRefused(f"unknown dataset {dataset!r}")
    readers = tuple(readers)
    if sorted(readers) != sorted(P.READER_KEYS):
        raise AtomicIndexRefused(
            f"reader panel {list(readers)} != frozen {list(P.READER_KEYS)}")

    index = {}
    excluded = {}
    n_i1_rows = 0
    for row in iter_jsonl(labels_file):
        itype = row.get("intervention_type")
        if itype != P.ATOMIC_INTERVENTION_TYPE:
            excluded[itype] = excluded.get(itype, 0) + 1
            continue
        n_i1_rows += 1
        reader = row.get("reader")
        if reader not in readers:
            raise AtomicIndexRefused(
                f"{dataset}: I1 row for unexpected reader {reader!r}")
        affected = list(row.get("affected_reply_ids") or [])
        if len(affected) != 1:
            raise AtomicIndexRefused(
                f"{dataset}: I1_atomic row must affect exactly one reply, got "
                f"{affected!r}")
        node_id = affected[0]
        key = evidence_key(dataset, row["event_id"], row["cutoff"], node_id)
        utility = _finite(row["utility"], f"{key}/{reader}")
        correctness_before = row.get("correctness_before")
        correctness_after = row.get("correctness_after")
        if not isinstance(correctness_before, bool) or \
                not isinstance(correctness_after, bool):
            raise AtomicIndexRefused(
                f"{key}/{reader}: correctness flags must be booleans")
        sign = tri_class_label(utility, correctness_before, correctness_after)
        if row.get("sign") != sign:
            raise AtomicIndexRefused(
                f"{key}/{reader}: cached sign {row.get('sign')!r} != frozen "
                f"tri-class label {sign!r}")
        entry = index.setdefault(key, {
            "key": key,
            "event_id": str(row["event_id"]),
            "cutoff": int(row["cutoff"]),
            "node_id": str(node_id),
            "utility": {}, "sign": {},
            "correctness_before": {}, "correctness_after": {},
        })
        if reader in entry["utility"]:
            raise AtomicIndexRefused(
                f"{key}/{reader}: duplicate I1 row for the same reader")
        entry["utility"][reader] = utility
        entry["sign"][reader] = sign
        entry["correctness_before"][reader] = correctness_before
        entry["correctness_after"][reader] = correctness_after

    if not index:
        raise AtomicIndexRefused(f"{dataset}: no I1_atomic rows found")
    incomplete = [k for k, e in index.items()
                  if sorted(e["utility"]) != sorted(readers)]
    if incomplete:
        raise AtomicIndexRefused(
            f"{dataset}: {len(incomplete)} evidence keys are missing a reader, "
            f"e.g. {incomplete[:3]}")

    entries = sorted(index.values(),
                     key=lambda e: (e["event_id"], e["cutoff"], e["node_id"]))
    return {
        "entries": entries,
        "excluded_intervention_type_counts": {
            str(k): v for k, v in sorted(excluded.items(), key=str)},
        "n_i1_rows": n_i1_rows,
    }


def keys_digest(entries) -> str:
    """Order-independent digest of the evidence keys carried by an index."""
    payload = "\n".join(sorted(e["key"] for e in entries))
    return sha256_bytes(payload.encode("utf-8"))


def build_atomic_index(historical_root: str, dataset: str,
                       labels_sha256: str) -> dict:
    """The M0 ``atomic_index.json`` artifact for one dataset."""
    path = labels_path(historical_root, dataset)
    if not os.path.exists(path):
        raise AtomicIndexRefused(f"{dataset}: frozen label cache missing")
    got = sha256_file(path)
    if got != labels_sha256:
        raise AtomicIndexRefused(
            f"{dataset}: label cache digest {got} != verified {labels_sha256}")
    extracted = extract_atomic_index(path, dataset)
    entries = extracted["entries"]
    expected = P.FROZEN_ATOMIC_KEYS[dataset]
    if len(entries) != expected:
        raise AtomicIndexRefused(
            f"{dataset}: {len(entries)} atomic evidence keys, frozen gate "
            f"expects {expected}")
    return {
        "protocol": P.PROTOCOL_VERSION,
        "dataset": dataset,
        "source_labels": {
            "rel_path": P.FROZEN_HISTORICAL_LABELS[dataset]["rel_path"],
            "sha256": labels_sha256,
            "rows": P.FROZEN_HISTORICAL_LABELS[dataset]["rows"],
        },
        "reader_keys": list(P.READER_KEYS),
        "utility_threshold": P.UTILITY_THRESHOLD,
        "sign_classes": list(P.SIGN_CLASSES),
        "atomic_intervention_type": P.ATOMIC_INTERVENTION_TYPE,
        #: Counted for auditability: they are *excluded* from BCR supervision.
        "excluded_intervention_type_counts":
            extracted["excluded_intervention_type_counts"],
        "n_i1_rows": extracted["n_i1_rows"],
        "n_evidence_keys": len(entries),
        "n_events": len({e["event_id"] for e in entries}),
        "cutoffs": sorted({e["cutoff"] for e in entries}),
        "keys_sha256": keys_digest(entries),
        "entries": entries,
    }


def load_atomic_index(path: str) -> dict:
    if not os.path.exists(path):
        raise AtomicIndexRefused(f"atomic index missing: {path}")
    return read_json(path)


def validate_atomic_index(index: dict, dataset: str = None) -> None:
    """Structural validation for a loaded/frozen index (used by the verifier)."""
    dataset = dataset or index.get("dataset")
    if dataset not in P.DATASETS:
        raise AtomicIndexRefused(f"unknown dataset {dataset!r}")
    if index.get("protocol") != P.PROTOCOL_VERSION:
        raise AtomicIndexRefused(
            f"atomic index protocol {index.get('protocol')!r} != "
            f"{P.PROTOCOL_VERSION!r}")
    if sorted(index.get("reader_keys") or []) != sorted(P.READER_KEYS):
        raise AtomicIndexRefused("atomic index reader panel drifted")
    if index.get("utility_threshold") != P.UTILITY_THRESHOLD:
        raise AtomicIndexRefused("atomic index utility threshold drifted")
    entries = index.get("entries") or []
    if len(entries) != index.get("n_evidence_keys"):
        raise AtomicIndexRefused("atomic index entry count disagrees")
    expected = P.FROZEN_ATOMIC_KEYS[dataset]
    if len(entries) != expected:
        raise AtomicIndexRefused(
            f"{dataset}: {len(entries)} keys != frozen {expected}")
    if keys_digest(entries) != index.get("keys_sha256"):
        raise AtomicIndexRefused("atomic index key digest disagrees")
    for entry in entries:
        if sorted(entry.get("utility") or {}) != sorted(P.READER_KEYS):
            raise AtomicIndexRefused(f"{entry.get('key')}: reader panel drifted")
        for reader in P.READER_KEYS:
            utility = _finite(entry["utility"][reader], entry["key"])
            sign = tri_class_label(utility, entry["correctness_before"][reader],
                                   entry["correctness_after"][reader])
            if sign != entry["sign"][reader]:
                raise AtomicIndexRefused(
                    f"{entry['key']}/{reader}: sign does not follow the frozen "
                    "tri-class rule")


def utility_matrix(index: dict, readers=P.READER_KEYS):
    """``(keys, readers, {reader: [utility, ...]})`` in a stable key order."""
    entries = index["entries"]
    keys = [e["key"] for e in entries]
    columns = {r: [float(e["utility"][r]) for e in entries] for r in readers}
    return keys, list(readers), columns


def event_of_key(key: str) -> str:
    """Event id of a canonical evidence key (``dataset|event|cutoff|node``)."""
    parts = str(key).split("|")
    if len(parts) != 4:
        raise AtomicIndexRefused(f"malformed evidence key {key!r}")
    return parts[1]


__all__ = ["AtomicIndexRefused", "build_atomic_index", "event_of_key",
           "extract_atomic_index", "keys_digest", "load_atomic_index",
           "utility_matrix", "validate_atomic_index"]
