"""E0 reader-agnostic evidence features (M1 plan §9).

One row per frozen ``I1_atomic`` evidence key. Every value is recomputed from
the primary snapshot through the read-only CR-TSER orchestration (causal
snapshot -> Reply-Parent units -> SRC), so the cache can never drift from the
frozen evidence definition.

The ten CR-TSER structural/time scalars are stored alongside (``struct``);
they are **B1-control-only** inputs and must never enter B0/B3/B4
(M1 plan §9).
"""
from __future__ import annotations

import math

from ..config import protocol as P

try:  # reused read-only from the frozen CR-TSER implementation
    from cr_tser.data.structural_stats import structural_scalars
    from cr_tser.intervention.semantic_reference import cosine
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "BCR reuses cr_tser structural stats and SRC cosine read-only") from exc


class FeatureExtractionRefused(RuntimeError):
    """Raised when an E0/E1/E2 row cannot be built exactly as specified."""


def e0_features(unit: dict, source_emb, parent_emb, reply_emb,
                src: dict, cutoff) -> dict:
    """The eleven frozen E0 scalars of one evidence key (M1 plan §9)."""
    node_id = unit["node_id"]
    parent_text = unit.get("parent_text")
    parent_available = 1 if parent_text else 0
    cos_parent_source = cosine(parent_emb, source_emb) if parent_available \
        else 0.0
    cos_reply_parent = cosine(reply_emb, parent_emb) if parent_available \
        else 0.0
    reply_chars = len(unit["reply_text"] or "")
    parent_chars = len(parent_text or "") if parent_available else 0
    costs = src.get("unit_token_costs") or {}
    if node_id not in costs:
        raise FeatureExtractionRefused(
            f"{node_id}: missing from the SRC unit token costs")
    relevance = (src.get("relevance") or {}).get(node_id)
    percentile = (src.get("rank_percentile") or {}).get(node_id)
    if relevance is None or percentile is None:
        raise FeatureExtractionRefused(
            f"{node_id}: missing from the SRC relevance / rank percentile")
    return {
        "cos_reply_source": float(cosine(reply_emb, source_emb)),
        "cos_parent_source": float(cos_parent_source),
        "cos_reply_parent": float(cos_reply_parent),
        "src_relevance": float(relevance),
        "src_rank_percentile": float(percentile),
        "canonical_token_cost": float(costs[node_id]),
        "reply_chars": float(reply_chars),
        "parent_chars": float(parent_chars),
        "combined_chars": float(reply_chars + parent_chars),
        "elapsed_seconds": float(unit["elapsed_seconds"]),
        "cutoff_minutes": float(cutoff),
    }


def struct_features(snapshot: dict, cutoff) -> dict:
    """``{node_id: {scalar: value}}`` — B1-control-only (M1 plan §9)."""
    rows = structural_scalars(snapshot, cutoff)
    return {nid: dict(zip(P.B1_STRUCT_NAMES, map(float, row)))
            for nid, row in zip(snapshot["node_ids"], rows)}


def rows_for_snapshot(entries, snapshot: dict, artifacts: dict) -> list:
    """E0 (+B1 struct) rows of every atomic entry of one (event, cutoff)."""
    units = artifacts.get("units") or []
    src = artifacts.get("src")
    if not units or src is None:
        raise FeatureExtractionRefused(
            f"{snapshot['event_id']}/{snapshot['cutoff_minutes']}: no "
            "evidence units / SRC for an event that carries atomic keys")
    unit_by_id = {u["node_id"]: u for u in units}
    reply_emb = artifacts["reply_emb"]
    source_emb = reply_emb[snapshot["source_id"]]
    structs = struct_features(snapshot, snapshot["cutoff_minutes"])
    rows = []
    for entry in entries:
        node_id = entry["node_id"]
        unit = unit_by_id.get(node_id)
        if unit is None:
            raise FeatureExtractionRefused(
                f"{entry['key']}: the atomic key has no matching evidence "
                "unit in the recomputed snapshot")
        parent_id = unit.get("parent_id")
        parent_emb = reply_emb.get(parent_id) if parent_id else None
        struct = structs.get(node_id)
        if struct is None:
            raise FeatureExtractionRefused(
                f"{entry['key']}: node missing from the snapshot scalars")
        rows.append({
            "key": entry["key"],
            "event_id": entry["event_id"],
            "cutoff": int(entry["cutoff"]),
            "node_id": node_id,
            "parent_available": 1 if unit.get("parent_text") else 0,
            "e0": e0_features(unit, source_emb, parent_emb,
                              reply_emb[node_id], src,
                              snapshot["cutoff_minutes"]),
            "struct": struct,
        })
    return rows


def validate_e0_rows(rows, expected_keys) -> dict:
    """Coverage/identity audit of one dataset's E0 cache."""
    keys = [r["key"] for r in rows]
    problems = []
    if len(set(keys)) != len(keys):
        problems.append("duplicate keys")
    missing = sorted(set(expected_keys) - set(keys))
    extra = sorted(set(keys) - set(expected_keys))
    if missing:
        problems.append(f"{len(missing)} missing keys: {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} unexpected keys: {extra[:3]}")
    for row in rows:
        for name in P.E0_FEATURE_NAMES:
            value = row["e0"].get(name)
            if not isinstance(value, (int, float)) or \
                    not math.isfinite(value):
                problems.append(f"{row['key']}: e0.{name}={value!r}")
                break
        for name in P.B1_STRUCT_NAMES:
            value = row["struct"].get(name)
            if not isinstance(value, (int, float)) or \
                    not math.isfinite(value):
                problems.append(f"{row['key']}: struct.{name}={value!r}")
                break
    if problems:
        raise FeatureExtractionRefused("; ".join(problems[:8]))
    return {"rows": len(rows), "unique_keys": len(set(keys)),
            "parent_available_rows": sum(r["parent_available"] for r in rows),
            "ok": True}
