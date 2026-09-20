"""Reader behavioral fingerprints from frozen probe responses (M1 plan §8).

One deterministic fingerprint per reader, built **only** from the frozen
M1-A probe responses: dataset x cutoff x context-grouped margin / entropy /
flip / token statistics. The fingerprint never sees utility labels, held-out
utility outcomes, gold labels or a reader-ID embedding — the audit below
refuses any row that even carries such a field.

Layout of the compact vector (fixed order, ``FINGERPRINT_DIM = 216``)::

    for dataset in (maweibo, pheme):
      for cutoff in (15, 60, 360):
        for context in (P0, P1, P2, P3):
          [mean_signed_margin, mean_abs_margin, std_abs_margin,
           mean_entropy, mean_log1p_reader_prompt_tokens,
           mean_log1p_canonical_qwen_tokens]
          if context != P0:
            [mean_delta_margin, mean_abs_delta_margin, flip_rate,
             mean_delta_entropy]        # relative to the same item's P0

Nothing in this module loads a model or reads a dataset.
"""
from __future__ import annotations

import hashlib
import json
import math

from ..config import protocol as P


class FingerprintRefused(RuntimeError):
    """Raised when probe responses violate the M1 fingerprint contract."""


RAW_VECTOR_FIELDS = (
    "signed_margin", "abs_margin", "entropy", "p_rumor", "p_nonrumor",
    "reader_prompt_tokens", "canonical_qwen_tokens",
)

#: Numeric fields every probe response row must carry.
REQUIRED_ROW_FIELDS = ("dataset", "event_id", "cutoff", "context", "reader",
                       "model_id", "score_A", "score_B", "p_rumor",
                       "p_nonrumor", "prediction", "signed_margin",
                       "abs_margin", "entropy", "reader_prompt_tokens",
                       "canonical_qwen_tokens", "n_units")

_EPS_VAR = 1e-12


def binary_entropy(p: float) -> float:
    """Binary entropy (base 2) of the two-way A/B posterior."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def response_metrics(score_a: float, score_b: float, p_rumor: float) -> dict:
    """Margin/entropy record of one teacher-forced A/B response."""
    signed = float(score_a) - float(score_b)
    return {
        "signed_margin": signed,
        "abs_margin": abs(signed),
        "entropy": binary_entropy(p_rumor),
    }


def row_key(row) -> tuple:
    return (str(row["dataset"]), str(row["event_id"]), int(row["cutoff"]),
            str(row["context"]))


def field_is_forbidden(field) -> bool:
    """A field is forbidden when one of its ``_``-separated tokens names a
    utility/gold/label concept (``signed_margin`` is a margin, not a sign),
    or when it exactly names a forbidden composite (``reader_id_embedding``).
    """
    low = str(field).lower().replace("-", "_")
    tokens = {t for t in low.split("_") if t}
    for word in P.PROBE_FORBIDDEN_FIELDS:
        if "_" in word:
            if low == word:
                return True
        elif word in tokens:
            return True
    return False


def _check_forbidden_fields(row) -> None:
    for field in row:
        if field_is_forbidden(field):
            raise FingerprintRefused(
                f"probe response carries the forbidden field {field!r}; "
                "utility labels, gold labels and reader-ID embeddings never "
                "enter a fingerprint (M1 plan §7, §8)")


def _check_finite(row) -> None:
    for field in RAW_VECTOR_FIELDS + ("score_A", "score_B"):
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise FingerprintRefused(
                f"{row_key(row)}: field {field!r} is not finite: {value!r}")


def audit_responses(rows, manifest, readers=P.READER_KEYS) -> dict:
    """Fail-closed coverage/identity audit of the raw probe responses.

    Checks (M1 plan §8, §18): full 72x4 coverage per reader against the
    frozen probe manifest, no duplicates, no NaN/Inf, exact reader identity
    (model ids) and no utility/gold fields anywhere.
    """
    rows = list(rows)
    problems = []
    seen = {}
    per_reader = {r: 0 for r in readers}
    model_ids = {}
    for row in rows:
        for field in REQUIRED_ROW_FIELDS:
            if field not in row:
                problems.append(f"missing field {field!r} in {row.get('reader')}"
                                f"/{row.get('event_id')}")
                break
        else:
            _check_forbidden_fields(row)
            _check_finite(row)
            key = (row["reader"],) + row_key(row)
            if key in seen:
                problems.append(f"duplicate response row {key}")
            seen[key] = True
            reader = row["reader"]
            if reader not in per_reader:
                problems.append(f"unexpected reader {reader!r}")
                continue
            per_reader[reader] += 1
            model_ids.setdefault(reader, set()).add(row["model_id"])
            if row["context"] not in P.PROBE_CONTEXTS:
                problems.append(f"unknown context {row['context']!r}")
    for reader, ids in model_ids.items():
        if ids != {P.READER_MODEL_IDS[reader]}:
            problems.append(f"{reader}: model id drift {sorted(ids)}")

    expected = set()
    for dataset in P.DATASETS:
        items = ((manifest.get("datasets") or {}).get(dataset) or {}
                 ).get("items") or []
        for item in items:
            for context in P.PROBE_CONTEXTS:
                for reader in readers:
                    expected.add((reader, dataset, str(item["event_id"]),
                                  int(item["cutoff"]), context))
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    if missing:
        problems.append(f"{len(missing)} missing responses, e.g. {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} responses outside the frozen manifest, "
                        f"e.g. {extra[:3]}")
    counts_ok = all(per_reader.get(r) == P.PROBE_CONTEXT_EVALS_PER_READER
                    for r in readers)
    if not counts_ok:
        problems.append(f"per-reader counts {per_reader}")
    if len(rows) != P.PROBE_TOTAL_EVALS:
        problems.append(f"total rows {len(rows)} != {P.PROBE_TOTAL_EVALS}")
    audit = {
        "n_rows": len(rows),
        "expected_rows": P.PROBE_TOTAL_EVALS,
        "per_reader_counts": per_reader,
        "per_reader_expected": P.PROBE_CONTEXT_EVALS_PER_READER,
        "duplicates": len(rows) - len(seen),
        "missing": [list(m) for m in missing[:10]],
        "extra": [list(e) for e in extra[:10]],
        "model_ids": {r: sorted(i) for r, i in model_ids.items()},
        "problems": problems,
        "ok": not problems,
    }
    if problems:
        raise FingerprintRefused("; ".join(problems[:8]))
    return audit


def build_raw_vectors(rows) -> dict:
    """Per-reader raw response matrices (fixed row order, fixed columns)."""
    per_reader = {r: [] for r in P.READER_KEYS}
    for row in rows:
        per_reader[row["reader"]].append(row)
    out = {}
    for reader, rrows in per_reader.items():
        ordered = sorted(rrows, key=row_key)
        matrix = [[float(r[f]) for f in RAW_VECTOR_FIELDS] for r in ordered]
        keys = [list(row_key(r)) for r in ordered]
        digest = hashlib.sha256(json.dumps(
            {"keys": keys, "matrix": matrix}, sort_keys=True).encode()
        ).hexdigest()
        out[reader] = {
            "reader": reader,
            "fields": list(RAW_VECTOR_FIELDS),
            "keys": keys,
            "matrix": matrix,
            "n_rows": len(ordered),
            "sha256": digest,
        }
    return out


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _std(values) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _base_block(rows) -> list:
    return [
        _mean(r["signed_margin"] for r in rows),
        _mean(r["abs_margin"] for r in rows),
        _std(r["abs_margin"] for r in rows),
        _mean(r["entropy"] for r in rows),
        _mean(math.log1p(r["reader_prompt_tokens"]) for r in rows),
        _mean(math.log1p(r["canonical_qwen_tokens"]) for r in rows),
    ]


def _delta_block(rows, p0_by_item) -> list:
    """P0-relative statistics of one P1/P2/P3 group (M1 plan §8)."""
    d_margin, abs_d_margin, flips, d_entropy = [], [], [], []
    for row in rows:
        p0 = p0_by_item.get((row["dataset"], row["event_id"], row["cutoff"]))
        if p0 is None:
            raise FingerprintRefused(
                f"{row_key(row)}: no P0 response for the same probe item")
        delta = row["signed_margin"] - p0["signed_margin"]
        d_margin.append(delta)
        abs_d_margin.append(abs(delta))
        flips.append(1.0 if row["prediction"] != p0["prediction"] else 0.0)
        d_entropy.append(row["entropy"] - p0["entropy"])
    return [_mean(d_margin), _mean(abs_d_margin), _mean(flips),
            _mean(d_entropy)]


def compact_fingerprint(reader_rows) -> dict:
    """The deterministic 216D compact fingerprint of one reader."""
    groups = {}
    for row in reader_rows:
        groups.setdefault((row["dataset"], int(row["cutoff"]),
                           row["context"]), []).append(row)
    p0_by_item = {}
    for row in reader_rows:
        if row["context"] == "P0":
            p0_by_item[(row["dataset"], row["event_id"],
                        int(row["cutoff"]))] = row
    vector = []
    blocks = {}
    for dataset in P.DATASETS:
        for cutoff in P.CUTOFFS_MIN:
            for context in P.PROBE_CONTEXTS:
                rows = groups.get((dataset, cutoff, context)) or []
                if len(rows) != P.PROBE_EVENTS_PER_CUTOFF:
                    raise FingerprintRefused(
                        f"{dataset}/{cutoff}/{context}: {len(rows)} rows != "
                        f"{P.PROBE_EVENTS_PER_CUTOFF}")
                block = _base_block(rows)
                names = list(P.FINGERPRINT_BASE_METRICS)
                if context != "P0":
                    block += _delta_block(rows, p0_by_item)
                    names += list(P.FINGERPRINT_DELTA_METRICS)
                for name, value in zip(names, block):
                    if not math.isfinite(value):
                        raise FingerprintRefused(
                            f"{dataset}/{cutoff}/{context}/{name} is not "
                            "finite")
                blocks[f"{dataset}|{cutoff}|{context}"] = dict(zip(names,
                                                                   block))
                vector += block
    if len(vector) != P.FINGERPRINT_DIM:
        raise FingerprintRefused(
            f"fingerprint dim {len(vector)} != {P.FINGERPRINT_DIM}")
    return {"vector": vector, "groups": blocks, "dim": P.FINGERPRINT_DIM}


def build_fingerprints(rows, manifest) -> dict:
    """Audit the responses, then build raw + compact per-reader artifacts."""
    audit = audit_responses(rows, manifest)
    raw = build_raw_vectors(rows)
    readers = {}
    for reader in P.READER_KEYS:
        rrows = [r for r in rows if r["reader"] == reader]
        compact = compact_fingerprint(rrows)
        readers[reader] = {
            "reader": reader,
            "model_id": P.READER_MODEL_IDS[reader],
            "compact": compact,
            "raw_sha256": raw[reader]["sha256"],
        }
    return {"audit": audit, "raw": raw, "readers": readers}


def fingerprint_vector(fingerprints: dict, reader: str) -> list:
    return list(fingerprints["readers"][reader]["compact"]["vector"])


def fit_fingerprint_scaler(fingerprints: dict, fit_readers) -> dict:
    """Mean/std over the *training* readers only (M1 plan §12).

    The held-out reader is transformed with these statistics; a zero-variance
    dimension gets std 1.0 (deterministic epsilon handling).
    """
    vectors = [fingerprint_vector(fingerprints, r) for r in fit_readers]
    dim = len(vectors[0])
    means, stds = [], []
    for j in range(dim):
        col = [v[j] for v in vectors]
        m = _mean(col)
        s = _std(col)
        means.append(m)
        stds.append(s if s > _EPS_VAR else 1.0)
    return {"mean": means, "std": stds, "fit_readers": list(fit_readers),
            "epsilon": _EPS_VAR}


def transform_fingerprint(vector, scaler: dict) -> list:
    return [(float(v) - m) / s
            for v, m, s in zip(vector, scaler["mean"], scaler["std"])]


def fingerprint_distances(fingerprints: dict, scaler: dict, readers) -> dict:
    """Pairwise Euclidean distances between z-scored compact fingerprints."""
    z = {r: transform_fingerprint(fingerprint_vector(fingerprints, r), scaler)
         for r in readers}
    out = {}
    for i, a in enumerate(readers):
        for b in readers[i + 1:]:
            d = math.sqrt(sum((x - y) ** 2 for x, y in zip(z[a], z[b])))
            out[f"{a}|{b}"] = d
    return out


def fingerprints_digest(fingerprints: dict) -> str:
    """Content digest over the three compact vectors."""
    payload = {r: fingerprints["readers"][r]["compact"]["vector"]
               for r in P.READER_KEYS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()
                          ).hexdigest()
