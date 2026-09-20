"""M1-E Task B/C — pre-registered feature-group ablations (post-hoc only).
The frozen B5 input is split into the four groups the audit fixes in advance:

``Z`` ZERO-TOUCH evidence compatibility = E0 + E1 + E2
``F`` behavioral fingerprint (the 216D compact vector)
``S`` source-state E3 = source-only A/B margin, source-only entropy, source NLL
``C`` evidence-familiarity E3 = evidence NLL/token, conditional evidence NLL,
    NLL gap

Variants (names are diagnostic labels; the frozen B4/B5 definitions are never
redefined)::

    D0 = Z            (no fingerprint; the ZERO-TOUCH-with-E2 baseline)
    D1 = Z + F        (= the frozen B4, reused verbatim)
    D2 = Z + F + S
    D3 = Z + F + C
    D4 = Z + F + S + C (= the frozen B5, reused verbatim)
    D5 = Z + S + C    (E3 present, fingerprint removed)

Every variant runs the exact M1 protocol: the frozen three-reader LORO
rotations, utility_train/dev/eval, the 8-config grid, three seeds, and the
event-level bootstrap (10000 iterations, seed 7319). These are POST-HOC
DIAGNOSTICS and define no new gate.
"""
from __future__ import annotations

import json
import os

from ..config import protocol as P
from ..evaluation import bootstrap as boot
from ..evaluation import utility_metrics as um
from ..evaluation.unseen_reader import (LoroRefused, apply_scaler,
                                        fit_feature_scaler, seed_averaged_eval,
                                        select_and_train)
from ..probes import fingerprint as fp


class AttributionRefused(RuntimeError):
    """Raised when an attribution variant violates the frozen contract."""


_S_INDEX = [P.E3_FEATURE_NAMES.index(n) for n in P.E3_SOURCE_STATE_NAMES]
_C_INDEX = [P.E3_FEATURE_NAMES.index(n)
            for n in P.E3_EVIDENCE_FAMILIARITY_NAMES]


def variant_uses_fingerprint(variant: str) -> bool:
    if variant not in P.ATTRIBUTION_VARIANTS:
        raise AttributionRefused(f"unknown variant {variant!r}")
    return variant in P.ATTRIBUTION_FINGERPRINT_VARIANTS


def variant_uses_e3_group(variant: str, group: str) -> bool:
    """``group`` is ``"S"`` or ``"C"``; D1/D4 include them by definition."""
    if variant not in P.ATTRIBUTION_VARIANTS:
        raise AttributionRefused(f"unknown variant {variant!r}")
    if variant == "D4_Z_F_S_C":
        return True
    if variant == "D5_Z_S_C":
        return True
    if group == "S":
        return variant == "D2_Z_F_S"
    if group == "C":
        return variant == "D3_Z_F_C"
    raise AttributionRefused(f"unknown E3 group {group!r}")


def variant_evidence_dim(variant: str) -> int:
    """Width of the evidence vector before the fingerprint interaction."""
    base = len(P.E0_FEATURE_NAMES) + len(P.E1_FEATURE_NAMES) \
        + len(P.E2_FEATURE_NAMES)
    extra = 0
    if variant_uses_e3_group(variant, "S"):
        extra += len(P.E3_SOURCE_STATE_NAMES)
    if variant_uses_e3_group(variant, "C"):
        extra += len(P.E3_EVIDENCE_FAMILIARITY_NAMES)
    return base + extra


def variant_evidence_vector(table_row: dict, reader: str, variant: str) -> list:
    """The frozen evidence vector of one (key, reader) under one variant."""
    vector = list(table_row["e0e1"]) + list(table_row["e2"][reader])
    if variant_uses_e3_group(variant, "S") or \
            variant_uses_e3_group(variant, "C"):
        e3 = table_row.get("e3")
        if not e3 or reader not in e3:
            raise AttributionRefused(
                f"variant {variant} needs E3 rows; none present")
        values = e3[reader]
        if variant_uses_e3_group(variant, "S"):
            vector += [values[i] for i in _S_INDEX]
        if variant_uses_e3_group(variant, "C"):
            vector += [values[i] for i in _C_INDEX]
    if len(vector) != variant_evidence_dim(variant):
        raise AttributionRefused(
            f"variant {variant}: evidence dim {len(vector)} != "
            f"{variant_evidence_dim(variant)}")
    return vector


def build_rows(variant: str, atomic_entries, table, readers, event_ids) -> list:
    """Supervised rows of one variant; only ``readers`` x ``event_ids``."""
    wanted = {str(e) for e in event_ids}
    rows = []
    for entry in atomic_entries:
        if str(entry["event_id"]) not in wanted:
            continue
        features = table[entry["key"]]
        for reader in readers:
            rows.append({
                "key": entry["key"],
                "event_id": str(entry["event_id"]),
                "reader": reader,
                "x": variant_evidence_vector(features, reader, variant),
                "u": float(entry["utility"][reader]),
                "s": um.SIGN_TO_INDEX[entry["sign"][reader]],
                "sign": entry["sign"][reader],
                "utility": float(entry["utility"][reader]),
            })
    return rows


def run_variant(dataset: str, variant: str, atomic_entries, table, split,
                fingerprints: dict, train_kwargs=None) -> dict:
    """One variant through the frozen LORO protocol (train + evaluate)."""
    uses_fp = variant_uses_fingerprint(variant)
    model_kind = P.MODEL_B4 if uses_fp else P.MODEL_B0
    folds = {name: set(str(e) for e in split[name])
             for name in ("utility_train", "utility_dev", "utility_eval")}
    rotations = {}
    for rotation in P.LORO_ROTATIONS:
        train_readers = [rotation[0], rotation[1]]
        held_out = rotation[2]
        fp_scaler = fp.fit_fingerprint_scaler(fingerprints, train_readers)
        fingerprint_vectors = {r: fp.transform_fingerprint(
            fp.fingerprint_vector(fingerprints, r), fp_scaler)
            for r in P.READER_KEYS} if uses_fp else None
        train_rows = build_rows(variant, atomic_entries, table, train_readers,
                                folds["utility_train"])
        dev_rows = build_rows(variant, atomic_entries, table, train_readers,
                              folds["utility_dev"])
        eval_rows = build_rows(variant, atomic_entries, table, [held_out],
                               folds["utility_eval"])
        if not train_rows or not dev_rows or not eval_rows:
            raise AttributionRefused(
                f"{dataset}/{variant}/{held_out}: empty fold")
        if {r["reader"] for r in train_rows + dev_rows} != set(train_readers):
            raise AttributionRefused(
                f"{dataset}/{variant}/{held_out}: train/dev reader leak")
        if {r["reader"] for r in eval_rows} != {held_out}:
            raise AttributionRefused(
                f"{dataset}/{variant}/{held_out}: eval reader leak")
        scaler = fit_feature_scaler(train_rows)
        train_s = apply_scaler(train_rows, scaler)
        dev_s = apply_scaler(dev_rows, scaler)
        eval_s = apply_scaler(eval_rows, scaler)
        selected = select_and_train(model_kind, train_s, dev_s,
                                    fingerprint_vectors=fingerprint_vectors,
                                    train_kwargs=train_kwargs)
        prediction = seed_averaged_eval(selected, eval_s, model_kind,
                                        fingerprint_vectors=fingerprint_vectors)
        rotations[held_out] = {
            "held_out": held_out,
            "train_readers": list(train_readers),
            "selected_config": selected["selected_config"],
            "dev_macro_f1": selected["dev_macro_f1"],
            "n_parameters": selected["n_parameters"],
            "fold_sizes": {"train": len(train_rows), "dev": len(dev_rows),
                           "eval": len(eval_rows)},
            "scaler_rows": scaler["n_rows"],
            "evidence_dim": variant_evidence_dim(variant),
            "eval_metrics": um.evaluate_predictions(eval_rows, prediction),
            "eval_rows": eval_rows,
            "prediction": prediction,
        }
    return {"dataset": dataset, "variant": variant,
            "uses_fingerprint": uses_fp, "model_kind": model_kind,
            "rotations": rotations}


def compare_to_baseline(variant_result: dict, baseline_result: dict,
                        iterations=P.BOOTSTRAP_ITERATIONS,
                        seed=P.BOOTSTRAP_SEED) -> dict:
    """Paired event-level ``variant - baseline`` over the three rotations."""
    per_reader = {}
    payloads = []
    for rotation in P.LORO_ROTATIONS:
        held = rotation[2]
        v = variant_result["rotations"][held]
        b = baseline_result["rotations"][held]
        per_reader[held] = boot.paired_event_delta(
            v["eval_rows"], v["prediction"]["sign_probs"],
            b["prediction"]["sign_probs"], iterations=iterations, seed=seed)
        payloads.append((v["eval_rows"], v["prediction"]["sign_probs"],
                         b["prediction"]["sign_probs"]))
    aggregate = boot.aggregate_delta(payloads, iterations=iterations,
                                     seed=seed)
    deltas = {held: per_reader[held]["observed"] for held in per_reader}
    return {
        "baseline": baseline_result["variant"],
        "per_reader": {held: {k: v for k, v in per_reader[held].items()
                              if k != "reps"} for held in per_reader},
        "per_reader_delta": deltas,
        "aggregate": {
            "mean_delta_macro_f1": aggregate["observed"],
            "ci_low": aggregate["ci_low"],
            "ci_high": aggregate["ci_high"],
            "iterations": aggregate["iterations"],
            "seed": aggregate["seed"],
            "positive_readers": sum(1 for d in deltas.values() if d > 0),
            "worst_reader_delta": min(deltas.values()),
        },
    }


def load_predictions_jsonl(path: str, dataset: str,
                           model_label: str) -> dict:
    """``{held: {"eval_rows", "prediction"}}`` from any M1/M1-E
    predictions file whose ``model`` field equals ``model_label``."""
    if not os.path.exists(path):
        raise AttributionRefused(f"predictions missing: {path}")
    by_reader = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != dataset or row.get("model") != model_label:
                continue
            entry = by_reader.setdefault(row["held_out"],
                                         {"eval_rows": [], "probs": [],
                                          "utilities": []})
            entry["eval_rows"].append({
                "key": row["key"], "event_id": str(row["event_id"]),
                "reader": row["reader"], "sign": row["gold_sign"],
                "utility": float(row["gold_utility"]),
            })
            entry["probs"].append([float(row["prob_helpful"]),
                                   float(row["prob_neutral"]),
                                   float(row["prob_harmful"])])
            entry["utilities"].append(float(row["pred_utility"]))
    if not by_reader:
        raise AttributionRefused(
            f"{dataset}/{model_label}: no rows in {path}")
    return {held: {"held_out": held, "eval_rows": entry["eval_rows"],
                   "prediction": {"sign_probs": entry["probs"],
                                  "utility": entry["utilities"]}}
            for held, entry in by_reader.items()}


def load_frozen_predictions(repo_root, dataset: str,
                            model_kind: str) -> dict:
    """Rebuild ``{held: {"eval_rows", "prediction"}}`` from the frozen
    M1 ``predictions*.jsonl`` so D1/D4 join the same paired bootstrap as the
    newly trained variants without re-running anything."""
    name = "predictions.jsonl" if model_kind == P.MODEL_B4 \
        else "predictions_light_touch.jsonl"
    return load_predictions_jsonl(P.m1_path(repo_root, "evaluation", name),
                                  dataset, model_kind)


def load_variant_predictions(repo_root, dataset: str, variant: str) -> dict:
    """M1-E variant predictions written by ``bcr_run_attribution.py``."""
    path = P.m1e_path(repo_root, P.M1E_ATTRIBUTION_DIRNAME,
                      f"predictions_{dataset}_{variant}.jsonl")
    return load_predictions_jsonl(path, dataset, variant)


def load_frozen_b0_baseline(repo_root, dataset: str) -> dict:
    """The frozen M1 ``B0`` baseline in the comparison result shape.

    Read from **both** stage files and asserted to agree, so a B0 baseline
    from either stage is provably the same frozen model output.
    """
    zero_touch = load_predictions_jsonl(
        P.m1_path(repo_root, "evaluation", "predictions.jsonl"), dataset,
        P.MODEL_B0)
    light_touch = load_predictions_jsonl(
        P.m1_path(repo_root, "evaluation", "predictions_light_touch.jsonl"),
        dataset, P.MODEL_B0)
    for held in sorted(set(zero_touch) & set(light_touch)):
        a, b = zero_touch[held], light_touch[held]
        if a["prediction"]["sign_probs"] != b["prediction"]["sign_probs"]:
            raise AttributionRefused(
                f"{dataset}/{held}: the frozen B0 predictions differ between "
                "the two stage files")
    return {"variant": P.MODEL_B0,
            "rotations": {held: {"eval_rows": entry["eval_rows"],
                                 "prediction": entry["prediction"]}
                          for held, entry in light_touch.items()}}


def compare_frozen_to_baseline(frozen: dict, baseline_result: dict,
                               iterations=P.BOOTSTRAP_ITERATIONS,
                               seed=P.BOOTSTRAP_SEED) -> dict:
    """Same paired bootstrap for a reused (frozen) variant."""
    per_reader = {}
    payloads = []
    for rotation in P.LORO_ROTATIONS:
        held = rotation[2]
        v = frozen[held]
        b = baseline_result["rotations"][held]
        per_reader[held] = boot.paired_event_delta(
            v["eval_rows"], v["prediction"]["sign_probs"],
            b["prediction"]["sign_probs"], iterations=iterations, seed=seed)
        payloads.append((v["eval_rows"], v["prediction"]["sign_probs"],
                         b["prediction"]["sign_probs"]))
    aggregate = boot.aggregate_delta(payloads, iterations=iterations,
                                     seed=seed)
    deltas = {held: per_reader[held]["observed"] for held in per_reader}
    return {
        "baseline": baseline_result["variant"],
        "per_reader": {held: {k: v for k, v in per_reader[held].items()
                              if k != "reps"} for held in per_reader},
        "per_reader_delta": deltas,
        "aggregate": {
            "mean_delta_macro_f1": aggregate["observed"],
            "ci_low": aggregate["ci_low"],
            "ci_high": aggregate["ci_high"],
            "iterations": aggregate["iterations"],
            "seed": aggregate["seed"],
            "positive_readers": sum(1 for d in deltas.values() if d > 0),
            "worst_reader_delta": min(deltas.values()),
        },
    }


def frozen_variant_summary(frozen: dict, variant: str, model_kind: str,
                           metrics: dict) -> dict:
    """Report view of a reused variant, with frozen per-reader metrics."""
    return {
        "dataset": frozen["dataset"],
        "variant": variant,
        "uses_fingerprint": variant_uses_fingerprint(variant),
        "frozen_model": model_kind,
        "reused": True,
        "rotations": frozen["rotations"],
        "per_reader_macro_f1": frozen["per_reader_macro_f1"],
        "mean_macro_f1": frozen["mean_macro_f1"],
        "worst_reader_macro_f1": frozen["worst_reader_macro_f1"],
        "per_reader_metrics": metrics,
    }


def summarise_variant(result: dict) -> dict:
    """Report-ready view of one variant (no eval rows / raw predictions)."""
    rotations = {}
    for held, rot in result["rotations"].items():
        rotations[held] = {k: v for k, v in rot.items()
                           if k not in ("eval_rows", "prediction")}
    per_reader_f1 = {held: rot["eval_metrics"]["macro_f1"]
                     for held, rot in result["rotations"].items()}
    return {
        "dataset": result["dataset"],
        "variant": result["variant"],
        "uses_fingerprint": result["uses_fingerprint"],
        "evidence_dim": result["rotations"][P.LORO_ROTATIONS[0][2]][
            "evidence_dim"],
        "rotations": rotations,
        "per_reader_macro_f1": per_reader_f1,
        "mean_macro_f1": sum(per_reader_f1.values()) / len(per_reader_f1),
        "worst_reader_macro_f1": min(per_reader_f1.values()),
    }


def frozen_m1_view(repo_root, dataset: str, model_kind: str) -> dict:
    """Read the frozen M1 evaluation of B4/B5 (D1/D4 are reused verbatim)."""
    name = "evaluation.json" if model_kind == P.MODEL_B4 \
        else "evaluation_light_touch.json"
    path = P.m1_path(repo_root, "evaluation", name)
    if not os.path.exists(path):
        raise AttributionRefused(f"frozen M1 evaluation missing: {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    result = (data.get("datasets") or {}).get(dataset)
    if not result:
        raise AttributionRefused(f"{dataset}: not in {name}")
    rotations = {}
    for held, rot in (result.get("rotations") or {}).items():
        record = (rot.get("models") or {}).get(model_kind)
        if record is None:
            raise AttributionRefused(f"{dataset}/{held}: {model_kind} missing")
        rotations[held] = {
            "held_out": held,
            "train_readers": rot.get("train_readers"),
            "selected_config": record.get("selected_config"),
            "dev_macro_f1": record.get("dev_macro_f1"),
            "n_parameters": record.get("n_parameters"),
            "fold_sizes": record.get("fold_sizes"),
            "eval_metrics": record.get("eval_metrics"),
            "eval_metrics_disagreement":
                record.get("eval_metrics_disagreement"),
        }
    per_reader_f1 = {held: rot["eval_metrics"]["macro_f1"]
                     for held, rot in rotations.items()}
    return {
        "dataset": dataset,
        "variant": None,
        "frozen_model": model_kind,
        "reused": True,
        "rotations": rotations,
        "per_reader_macro_f1": per_reader_f1,
        "mean_macro_f1": sum(per_reader_f1.values()) / len(per_reader_f1),
        "worst_reader_macro_f1": min(per_reader_f1.values()),
    }
