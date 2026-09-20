"""M1-E Task E — reader / cutoff / class concentration of the B5-B0 gain.

Decomposes the frozen Ma-Weibo improvement using the *existing* M1
predictions only (no reader inference is re-run, no model is retrained):

* per held-out reader,
* per cutoff (15m / 60m / 360m),
* per gold class (HELPFUL / NEUTRAL / HARMFUL),

and answers whether the aggregate gain is dominated by a single reader,
cutoff or class.
"""
from __future__ import annotations

import json
import os

from ..config import protocol as P
from ..evaluation import utility_metrics as um

try:  # reused read-only from the frozen CR-TSER implementation
    from cr_tser.intervention.evidence_units import evidence_key_parts
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "BCR reuses cr_tser evidence units read-only") from exc


class ConcentrationRefused(RuntimeError):
    """Raised when the concentration audit cannot run as specified."""


def _prediction_path(repo_root, primary_model: str) -> str:
    """The predictions file of the stage that produced ``primary_model``.

    ``B0`` appears in both files; the *primary* model decides which stage's
    file the pair is read from.
    """
    if primary_model == P.MODEL_B5:
        name = "predictions_light_touch.jsonl"
    elif primary_model in (P.MODEL_B4, P.MODEL_B3):
        name = "predictions.jsonl"
    else:
        raise ConcentrationRefused(
            f"no frozen predictions stage for {primary_model!r}")
    path = P.m1_path(repo_root, "evaluation", name)
    if not os.path.exists(path):
        raise ConcentrationRefused(f"predictions missing: {path}")
    return path


def load_paired_predictions(repo_root, dataset: str, primary_model: str,
                            baseline_model: str = P.MODEL_B0) -> list:
    """Rows carrying both models' predictions, joined on
    (held_out, key, reader) from the same stage file."""
    path = _prediction_path(repo_root, primary_model)

    def read(model_kind):
        out = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("dataset") != dataset or \
                        row.get("model") != model_kind:
                    continue
                out[(row["held_out"], row["key"])] = row
        return out

    primary, baseline = read(primary_model), read(baseline_model)
    shared = sorted(set(primary) & set(baseline))
    if not shared:
        raise ConcentrationRefused(
            f"{dataset}: no shared rows between {primary_model} and "
            f"{baseline_model}")
    rows = []
    for held, key in shared:
        p, b = primary[(held, key)], baseline[(held, key)]
        if p["gold_sign"] != b["gold_sign"] or \
                p["reader"] != b["reader"]:
            raise ConcentrationRefused(f"{held}/{key}: gold/reader mismatch")
        rows.append({
            "held_out": held, "key": key, "reader": p["reader"],
            "event_id": str(p["event_id"]),
            "cutoff": evidence_key_parts(key)["cutoff"],
            "gold_sign": p["gold_sign"],
            "gold_utility": float(p["gold_utility"]),
            "baseline_pred": b["pred_sign"], "primary_pred": p["pred_sign"],
        })
    return rows


def _macro_f1(rows, field) -> float:
    return um.macro_f1([r["gold_sign"] for r in rows],
                       [r[field] for r in rows])


def _cell(rows) -> dict:
    if not rows:
        return {"n": 0, "baseline_f1": None, "primary_f1": None,
                "delta": None}
    b, p = _macro_f1(rows, "baseline_pred"), _macro_f1(rows, "primary_pred")
    return {"n": len(rows), "baseline_f1": b, "primary_f1": p, "delta": p - b}


def concentration_report(rows, primary_model: str,
                         baseline_model: str = P.MODEL_B0) -> dict:
    """Per reader / cutoff / class decomposition of the aggregate delta."""
    readers = {}
    for held in P.READER_KEYS:
        subset = [r for r in rows if r["held_out"] == held]
        if not subset:
            continue
        cell = _cell(subset)
        cell["per_cutoff"] = {
            str(c): _cell([r for r in subset if r["cutoff"] == c])
            for c in P.CUTOFFS_MIN}
        cell["per_class_f1"] = {
            cls: {
                "baseline_f1": um.class_f1(
                    [r["gold_sign"] for r in subset],
                    [r["baseline_pred"] for r in subset], cls),
                "primary_f1": um.class_f1(
                    [r["gold_sign"] for r in subset],
                    [r["primary_pred"] for r in subset], cls),
                "n": sum(1 for r in subset if r["gold_sign"] == cls),
            }
            for cls in P.SIGN_CLASSES}
        for cls in cell["per_class_f1"]:
            entry = cell["per_class_f1"][cls]
            entry["delta"] = entry["primary_f1"] - entry["baseline_f1"]
        readers[held] = cell

    reader_deltas = {h: c["delta"] for h, c in readers.items()}
    mean_delta = sum(reader_deltas.values()) / len(reader_deltas) \
        if reader_deltas else float("nan")

    cutoff_deltas = {}
    for cutoff in P.CUTOFFS_MIN:
        per_reader = [readers[h]["per_cutoff"][str(cutoff)]["delta"]
                      for h in readers]
        cutoff_deltas[str(cutoff)] = sum(per_reader) / len(per_reader)

    class_deltas = {}
    for cls in P.SIGN_CLASSES:
        per_reader = [readers[h]["per_class_f1"][cls]["delta"]
                      for h in readers]
        class_deltas[cls] = sum(per_reader) / len(per_reader)

    def dominance(deltas: dict) -> dict:
        positive = {k: v for k, v in deltas.items() if v > 0}
        total = sum(positive.values())
        if total <= 0:
            return {"largest": None, "share": None,
                    "positive_keys": sorted(positive)}
        largest = max(positive, key=lambda k: positive[k])
        return {"largest": largest, "share": positive[largest] / total,
                "positive_keys": sorted(positive),
                "largest_delta": positive[largest]}

    return {
        "primary_model": primary_model,
        "baseline_model": baseline_model,
        "n_rows": len(rows),
        "mean_reader_delta_macro_f1": mean_delta,
        "per_reader": readers,
        "reader_deltas": reader_deltas,
        "cutoff_deltas": cutoff_deltas,
        "class_deltas": class_deltas,
        "dominance": {
            "reader": dominance(reader_deltas),
            "cutoff": dominance(cutoff_deltas),
            "class": dominance(class_deltas),
        },
        "scope": "post_hoc_diagnostic_only",
        "uses_existing_predictions_only": True,
    }
