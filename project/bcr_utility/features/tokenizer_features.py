"""E2 tokenizer-only reader compatibility features (M1 plan §9).

Per evidence key **and reader**: how that reader's tokenizer fragments the
reply/parent/unit text. No reader forward pass is involved — only the frozen
tokenizer files of each approved reader checkpoint are loaded.
"""
from __future__ import annotations

import math

from ..config import protocol as P
from .evidence_features import FeatureExtractionRefused

try:  # reused read-only from the frozen CR-TSER / TC-DSCR implementations
    from cr_tser.intervention.evidence_units import unit_token_cost
    from tcdscr.llm.reader_prompt import render_evidence
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "BCR reuses cr_tser evidence units and the TC-DSCR renderer "
        "read-only") from exc


def _count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def e2_feature_dict(unit: dict, reader_tokenizer, canonical_tokenizer) -> dict:
    """The six frozen E2 scalars of one (evidence key, reader) pair."""
    parent_text = unit.get("parent_text") or ""
    reply_tokens = _count(reader_tokenizer, unit["reply_text"])
    parent_tokens = _count(reader_tokenizer, parent_text) if parent_text else 0
    unit_tokens = _count(reader_tokenizer, render_evidence(unit, 0))
    canonical = unit_token_cost(canonical_tokenizer, unit)
    if unit_tokens < 1 or canonical < 1:
        raise FeatureExtractionRefused(
            f"{unit['node_id']}: empty rendered unit tokens "
            f"({unit_tokens}/{canonical})")
    features = {
        "reply_tokens": float(reply_tokens),
        "parent_tokens": float(parent_tokens),
        "unit_tokens": float(unit_tokens),
        "canonical_unit_tokens": float(canonical),
        "reply_frac": float(reply_tokens) / float(unit_tokens),
        "unit_vs_canonical": float(unit_tokens) / float(canonical),
    }
    if tuple(features) != P.E2_FEATURE_NAMES:
        raise FeatureExtractionRefused("E2 feature order drift")
    return features


def e2_rows(units_by_key, tokenizers: dict, canonical_tokenizer) -> list:
    """E2 rows for ``{key: unit}`` x the frozen reader panel."""
    rows = []
    for key in sorted(units_by_key):
        unit = units_by_key[key]
        for reader in P.READER_KEYS:
            if reader not in tokenizers:
                raise FeatureExtractionRefused(
                    f"reader tokenizer missing: {reader}")
            rows.append({
                "key": key,
                "reader": reader,
                "e2": e2_feature_dict(unit, tokenizers[reader],
                                      canonical_tokenizer),
            })
    return rows


def validate_e2_rows(rows, expected_keys, readers=P.READER_KEYS) -> dict:
    seen = {(r["key"], r["reader"]) for r in rows}
    problems = []
    expected = {(k, r) for k in expected_keys for r in readers}
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        problems.append(f"{len(missing)} missing: {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} unexpected: {extra[:3]}")
    for row in rows:
        if tuple(row["e2"]) != P.E2_FEATURE_NAMES:
            problems.append(f"{row['key']}: feature order drift")
            break
        for name, value in row["e2"].items():
            if not math.isfinite(value):
                problems.append(f"{row['key']}: {name} not finite")
                break
    if problems:
        raise FeatureExtractionRefused("; ".join(problems[:8]))
    return {"rows": len(rows), "per_reader": len(rows) // len(readers),
            "ok": True}
