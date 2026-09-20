"""Synthetic M1 fixtures shared by the M1 test modules.

Everything is tiny, deterministic and model-free: no dataset, no reader
checkpoint, no NLI weights. Values are derived from a stable character hash
so test rows exercise the real code paths without needing the real pipeline.
"""
from __future__ import annotations

import math

from ..config import protocol as P
from ..probes import fingerprint as fp

PROBE_EVENTS_PER_DATASET = 36
ATOMIC_EVENTS_PER_DATASET = 90
KEYS_PER_SNAPSHOT = 2


def det_value(*parts) -> float:
    """Deterministic pseudo-value in [-4, 4) from arbitrary parts."""
    x = 0
    for part in parts:
        for ch in str(part):
            x = (x * 131 + ord(ch)) % 100003
    return (x % 2000 - 1000) / 250.0


def synth_manifest() -> dict:
    datasets = {}
    for ds in P.DATASETS:
        items = []
        for i in range(PROBE_EVENTS_PER_DATASET):
            items.append({
                "event_id": f"{ds}_p{i:03d}",
                "cutoff": (15, 60, 360)[i % 3],
                "n_visible_units": 5,
                "src_selected": 2,
                "n_visible_nodes": 6,
            })
        datasets[ds] = {"items": items}
    return {"datasets": datasets}


def synth_probe_rows() -> list:
    rows = []
    for ds in P.DATASETS:
        for i in range(PROBE_EVENTS_PER_DATASET):
            event_id = f"{ds}_p{i:03d}"
            cutoff = (15, 60, 360)[i % 3]
            for context in P.PROBE_CONTEXTS:
                for reader in P.READER_KEYS:
                    score_a = det_value(ds, i, context, reader, "A")
                    score_b = det_value(ds, i, context, reader, "B")
                    m = max(score_a, score_b)
                    ea, eb = math.exp(score_a - m), math.exp(score_b - m)
                    p = ea / (ea + eb)
                    signed = score_a - score_b
                    rows.append({
                        "dataset": ds,
                        "event_id": event_id,
                        "cutoff": cutoff,
                        "context": context,
                        "n_units": 0 if context == "P0" else
                        (1 if context in ("P1", "P2") else 2),
                        "reader": reader,
                        "model_id": P.READER_MODEL_IDS[reader],
                        "reader_identity_hash": f"identity_{reader}",
                        "weight_hash": f"w_{reader}",
                        "tokenizer_hash": f"t_{reader}",
                        "chat_template_hash": f"c_{reader}",
                        "dtype": "bfloat16",
                        "score_A": score_a,
                        "score_B": score_b,
                        "p_rumor": p,
                        "p_nonrumor": 1.0 - p,
                        "prediction": "A" if score_a >= score_b else "B",
                        "signed_margin": signed,
                        "abs_margin": abs(signed),
                        "entropy": fp.binary_entropy(p),
                        "reader_prompt_tokens": 100 + (i % 7),
                        "canonical_qwen_tokens": 110 + (i % 5),
                        "prompt_hash": f"ph_{ds}_{i}_{context}_{reader}",
                        "prompt_ids_hash": f"pi_{ds}_{i}_{context}_{reader}",
                    })
    return rows


def synth_fingerprints() -> dict:
    built = fp.build_fingerprints(synth_probe_rows(), synth_manifest())
    return {"schema": "test", "probe_manifest_sha256": "m" * 64,
            "dim": P.FINGERPRINT_DIM,
            "readers": {r: {"model_id": P.READER_MODEL_IDS[r],
                            "compact": built["readers"][r]["compact"],
                            "raw_sha256": built["readers"][r]["raw_sha256"]}
                        for r in P.READER_KEYS},
            "sha256": fp.fingerprints_digest(
                {"readers": built["readers"]})}


def synth_split() -> dict:
    events = [f"u{i:03d}" for i in range(ATOMIC_EVENTS_PER_DATASET)]
    return {
        "foundation_train": [f"f{i:03d}" for i in range(80)],
        "utility_train": events[:50],
        "utility_dev": events[50:65],
        "utility_eval": events[65:90],
    }


def synth_atomic_entries(dataset: str) -> list:
    from cr_tser.models.utility_heads import tri_class_label
    entries = []
    for i in range(ATOMIC_EVENTS_PER_DATASET):
        event_id = f"u{i:03d}"
        for cutoff in P.CUTOFFS_MIN:
            for k in range(KEYS_PER_SNAPSHOT):
                node = f"n{k}"
                utility = {r: det_value(dataset, event_id, cutoff, node, r)
                           / 4.0 for r in P.READER_KEYS}
                entries.append({
                    "key": f"{dataset}|{event_id}|{cutoff}|{node}",
                    "event_id": event_id,
                    "cutoff": cutoff,
                    "node_id": node,
                    "utility": utility,
                    "sign": {r: tri_class_label(utility[r], True, True)
                             for r in P.READER_KEYS},
                    "correctness_before": {r: True for r in P.READER_KEYS},
                    "correctness_after": {r: True for r in P.READER_KEYS},
                })
    return entries


def synth_e0_rows(entries) -> list:
    rows = []
    for e in entries:
        e0 = {name: abs(det_value(e["key"], name)) / 4.0
              for name in P.E0_FEATURE_NAMES}
        struct = {name: abs(det_value(e["key"], "s", name)) / 4.0
                  for name in P.B1_STRUCT_NAMES}
        rows.append({"key": e["key"], "event_id": e["event_id"],
                     "cutoff": e["cutoff"], "node_id": e["node_id"],
                     "parent_available": 1, "e0": e0, "struct": struct})
    return rows


def synth_e1_rows(entries) -> list:
    rows = []
    for e in entries:
        e1 = {name: abs(det_value(e["key"], name)) % 1.0
              for name in P.E1_FEATURE_NAMES}
        rows.append({"key": e["key"], "e1": e1})
    return rows


def synth_e2_rows(entries) -> list:
    rows = []
    for e in entries:
        for reader in P.READER_KEYS:
            e2 = {name: abs(det_value(e["key"], reader, name)) / 2.0 + 0.5
                  for name in P.E2_FEATURE_NAMES}
            rows.append({"key": e["key"], "reader": reader, "e2": e2})
    return rows


def synth_e3_rows(entries) -> list:
    rows = []
    for e in entries:
        for reader in P.READER_KEYS:
            e3 = {name: abs(det_value(e["key"], reader, "e3", name)) % 3.0
                  for name in P.E3_FEATURE_NAMES}
            e3["nll_gap"] = (e3["evidence_nll_per_token"]
                             - e3["conditional_evidence_nll_per_token"])
            rows.append({"key": e["key"], "reader": reader, "e3": e3})
    return rows
