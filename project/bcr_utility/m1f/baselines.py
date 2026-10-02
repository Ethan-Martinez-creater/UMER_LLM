"""M1-F Task B — the pre-registered strong-baseline ladder S0–S4.

Runs the frozen M1 LORO protocol (three rotations, ``utility_train`` /
``utility_dev`` / ``utility_eval``, the 8-config grid, three seeds) for the two
*learned* simple baselines and materialises the two constant ones; ``S4`` is the
frozen M1-E ``D5_Z_S_C`` prediction reused verbatim.

Two boundaries the ladder must respect:

* the strong simple comparator is chosen from ``S0``–``S3`` using the
  **training readers' ``utility_dev`` Macro-F1 only** — the held-out reader's
  ``utility_eval`` rows never enter the choice (plan §6);
* a held-out reader's labels never enter training, dev selection, scaler
  fitting, the class prior or any threshold.
"""
from __future__ import annotations

import json
import os
import time

from ..attribution import feature_ablation as fa
from ..config import protocol as P
from ..evaluation import utility_metrics as um
from ..evaluation.unseen_reader import (apply_scaler, build_feature_table,
                                       fit_feature_scaler, seed_averaged_eval,
                                       select_and_train)
from . import variants


class BaselineRefused(RuntimeError):
    """Raised when a baseline run violates the frozen M1-F ladder."""


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(repo_root, dataset: str) -> tuple:
    """Frozen M0/M1 inputs of one dataset (atomic entries, feature table, split)."""
    index = _load_json(P.result_path(repo_root, P.ATOMIC_INDEX_FILENAME))
    entries = (index.get("datasets") or {}).get(dataset, {}).get("entries")
    if not entries:
        raise BaselineRefused(f"{dataset}: atomic index entries missing")
    fdir = P.m1_path(repo_root, "features")
    e0 = _load_jsonl(os.path.join(fdir, f"e0_{dataset}.jsonl"))
    e1 = _load_jsonl(os.path.join(fdir, f"e1_{dataset}.jsonl"))
    e2 = _load_jsonl(os.path.join(fdir, f"e2_{dataset}.jsonl"))
    e3 = _load_jsonl(os.path.join(fdir, f"e3_{dataset}.jsonl"))
    split_path = os.path.join(P.historical_root(repo_root), "manifests",
                              dataset, "event_split.json")
    split = _load_json(split_path)
    table = build_feature_table(dataset, entries, e0, e1, e2, e3_rows=e3)
    return entries, table, split


def base_rows(atomic_entries, table, readers, event_ids) -> list:
    """Supervised rows without a learned input vector (S0/S1 only)."""
    wanted = {str(e) for e in event_ids}
    rows = []
    for entry in atomic_entries:
        if str(entry["event_id"]) not in wanted:
            continue
        features = table[entry["key"]]
        for reader in readers:
            if reader not in entry["utility"] or reader not in entry["sign"]:
                raise BaselineRefused(f"{entry['key']}: reader {reader} "
                                      "missing from the atomic index")
            utility = float(entry["utility"][reader])
            rows.append({
                "key": entry["key"], "event_id": str(entry["event_id"]),
                "cutoff": int(features["cutoff"]), "reader": reader,
                "u": utility, "s": um.SIGN_TO_INDEX[entry["sign"][reader]],
                "sign": entry["sign"][reader], "utility": utility,
            })
    return rows


def _folds(split):
    return {name: set(str(e) for e in split[name])
            for name in ("utility_train", "utility_dev", "utility_eval")}


def _check_readers(stage, rows, expected):
    got = {r["reader"] for r in rows}
    if got != set(expected):
        raise BaselineRefused(f"{stage}: readers {sorted(got)} != "
                             f"{sorted(expected)}")


def run_trained_variant(dataset, variant, entries, table, split) -> dict:
    """One learned simple baseline through the frozen LORO protocol."""
    folds = _folds(split)
    rotations = {}
    for rotation in P.LORO_ROTATIONS:
        train_readers = [rotation[0], rotation[1]]
        held = rotation[2]
        train_rows = variants.build_rows(variant, entries, table, train_readers,
                                         folds["utility_train"])
        dev_rows = variants.build_rows(variant, entries, table, train_readers,
                                       folds["utility_dev"])
        eval_rows = variants.build_rows(variant, entries, table, [held],
                                        folds["utility_eval"])
        if not train_rows or not dev_rows or not eval_rows:
            raise BaselineRefused(f"{dataset}/{variant}/{held}: empty fold")
        _check_readers("train", train_rows, train_readers)
        _check_readers("dev", dev_rows, train_readers)
        _check_readers("eval", eval_rows, [held])
        scaler = fit_feature_scaler(train_rows)
        train_s = apply_scaler(train_rows, scaler)
        dev_s = apply_scaler(dev_rows, scaler)
        eval_s = apply_scaler(eval_rows, scaler)
        selected = select_and_train(P.MODEL_B0, train_s, dev_s)
        dev_pred = seed_averaged_eval(selected, dev_s, P.MODEL_B0)
        eval_pred = seed_averaged_eval(selected, eval_s, P.MODEL_B0)
        rotations[held] = {
            "held_out": held, "train_readers": list(train_readers),
            "selected_config": selected["selected_config"],
            "config_selection_dev_macro_f1": selected["dev_macro_f1"],
            "n_parameters": selected["n_parameters"],
            "scaler_rows": scaler["n_rows"],
            "fold_sizes": {"train": len(train_rows), "dev": len(dev_rows),
                           "eval": len(eval_rows)},
            "evidence_dim": variants.variant_dim(variant),
            "dev_rows": dev_rows, "dev_prediction": dev_pred,
            "eval_rows": eval_rows, "prediction": eval_pred,
            "eval_metrics": um.evaluate_predictions(eval_rows, eval_pred),
        }
    return {"dataset": dataset, "variant": variant, "kind": "trained",
            "uses_fingerprint": False, "model_kind": P.MODEL_B0,
            "rotations": rotations}


def run_constant_variant(dataset, variant, entries, table, split) -> dict:
    """S0 (all-NEUTRAL) / S1 (training class prior): no fitting at all."""
    folds = _folds(split)
    rotations = {}
    for rotation in P.LORO_ROTATIONS:
        train_readers = [rotation[0], rotation[1]]
        held = rotation[2]
        train_rows = base_rows(entries, table, train_readers,
                               folds["utility_train"])
        dev_rows = base_rows(entries, table, train_readers,
                             folds["utility_dev"])
        eval_rows = base_rows(entries, table, [held], folds["utility_eval"])
        if not train_rows or not dev_rows or not eval_rows:
            raise BaselineRefused(f"{dataset}/{variant}/{held}: empty fold")
        if variant == P.M1F_S0:
            dev_pred = variants.s0_prediction(dev_rows)
            eval_pred = variants.s0_prediction(eval_rows)
            prior = "NEUTRAL"
        elif variant == P.M1F_S1:
            prior = variants.prior_sign(train_rows)
            dev_pred = variants.s1_prediction(dev_rows, train_rows)
            eval_pred = variants.s1_prediction(eval_rows, train_rows)
        else:
            raise BaselineRefused(f"{variant} is not a constant variant")
        rotations[held] = {
            "held_out": held, "train_readers": list(train_readers),
            "selected_config": None, "n_parameters": 0,
            "scaler_rows": 0,
            "fold_sizes": {"train": len(train_rows), "dev": len(dev_rows),
                           "eval": len(eval_rows)},
            "evidence_dim": 0,
            "constant_sign": prior,
            "dev_rows": dev_rows, "dev_prediction": dev_pred,
            "eval_rows": eval_rows, "prediction": eval_pred,
            "eval_metrics": um.evaluate_predictions(eval_rows, eval_pred),
        }
    return {"dataset": dataset, "variant": variant, "kind": "constant",
            "uses_fingerprint": False, "model_kind": variant,
            "rotations": rotations}


def run_reused_variant(dataset, frozen: dict) -> dict:
    """S4: the frozen D5 prediction, reported in the same shape."""
    rotations = {}
    for held, entry in frozen.items():
        rotations[held] = {
            "held_out": held, "train_readers": None,
            "selected_config": None, "n_parameters": 0, "scaler_rows": None,
            "fold_sizes": {"eval": len(entry["eval_rows"])},
            "evidence_dim": 32,
            "dev_rows": None, "dev_prediction": None,
            "eval_rows": entry["eval_rows"], "prediction": entry["prediction"],
            "eval_metrics": um.evaluate_predictions(entry["eval_rows"],
                                                   entry["prediction"]),
        }
    return {"dataset": dataset, "variant": P.M1F_S4, "kind": "reused",
            "uses_fingerprint": False, "model_kind": P.MODEL_B0,
            "frozen_variant": P.M1F_REUSED_VARIANTS[P.M1F_S4],
            "rotations": rotations}


def dev_macro_f1(rotations) -> dict:
    """Macro-F1 of each rotation's **training-reader** dev prediction."""
    out = {}
    for held, rot in rotations.items():
        dev_rows, dev_pred = rot.get("dev_rows"), rot.get("dev_prediction")
        if not dev_rows or dev_pred is None:
            out[held] = None
            continue
        out[held] = um.macro_f1([r["sign"] for r in dev_rows],
                                um.predicted_signs(dev_pred))
    return out


def select_comparator(dev_scores: dict) -> dict:
    """The dev-selected strong simple comparator (training readers only)."""
    pool = list(P.M1F_COMPARATOR_POOL)
    means = {}
    for variant in pool:
        values = [v for v in dev_scores.get(variant, {}).values()
                  if v is not None]
        if not values:
            raise BaselineRefused(f"{variant}: no dev Macro-F1 to select on")
        means[variant] = sum(values) / len(values)
    selected = max(pool, key=lambda v: (means[v], -pool.index(v)))
    return {
        "selected": selected,
        "pool": pool,
        "dev_macro_f1_mean": means,
        "selected_by": "mean utility_dev Macro-F1 over the three LORO "
                       "rotations, training readers only; utility_eval was "
                       "never consulted",
    }


def export_predictions(path, dataset, variant, rotations) -> dict:
    """Persist one variant's held-out predictions (same schema as M1-E)."""
    rows_written = 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for held in sorted(rotations):
            rot = rotations[held]
            for row, u_hat, probs in zip(rot["eval_rows"],
                                         rot["prediction"]["utility"],
                                         rot["prediction"]["sign_probs"]):
                fh.write(json.dumps({
                    "dataset": dataset, "held_out": held, "model": variant,
                    "key": row["key"], "event_id": row["event_id"],
                    "cutoff": row.get("cutoff"), "reader": row["reader"],
                    "gold_sign": row["sign"], "gold_utility": row["utility"],
                    "pred_utility": u_hat,
                    "pred_sign": max(range(len(P.SIGN_CLASSES)),
                                     key=lambda c: probs[c]),
                    "prob_helpful": probs[0], "prob_neutral": probs[1],
                    "prob_harmful": probs[2],
                }, ensure_ascii=False) + "\n")
                rows_written += 1
    return {"path": path, "rows": rows_written, "sha256": sha256_file(path)}


def summarise(result: dict) -> dict:
    """Report view of one variant (no raw rows or predictions)."""
    rotations = {}
    for held, rot in result["rotations"].items():
        rotations[held] = {k: v for k, v in rot.items()
                           if k not in ("eval_rows", "prediction", "dev_rows",
                                        "dev_prediction")}
        rotations[held]["dev_macro_f1"] = dev_macro_f1({held: rot})[held]
    return {
        "dataset": result["dataset"], "variant": result["variant"],
        "kind": result["kind"],
        "frozen_variant": result.get("frozen_variant"),
        "uses_fingerprint": result["uses_fingerprint"],
        "model_kind": result["model_kind"],
        "rotations": rotations,
        "per_reader_macro_f1": {h: r["eval_metrics"]["macro_f1"]
                                for h, r in result["rotations"].items()},
    }


def run_dataset(repo_root, dataset: str, out_dir: str) -> dict:
    """The full S0–S4 ladder of one dataset plus its dev-selected comparator."""
    t0 = time.time()
    entries, table, split = load_inputs(repo_root, dataset)
    results = {}
    for variant in P.M1F_CONSTANT_VARIANTS:
        results[variant] = run_constant_variant(dataset, variant, entries,
                                                table, split)
    for variant in P.M1F_TRAINED_VARIANTS:
        results[variant] = run_trained_variant(dataset, variant, entries,
                                               table, split)
    frozen = variants.frozen_s4_predictions(repo_root, dataset)
    results[P.M1F_S4] = run_reused_variant(dataset, frozen)

    dev_scores = {v: dev_macro_f1(r["rotations"]) for v, r in results.items()}
    comparator = select_comparator(dev_scores)

    predictions = {}
    for variant, result in results.items():
        if result["kind"] == "reused":
            continue
        path = os.path.join(out_dir, f"predictions_{dataset}_{variant}.jsonl")
        predictions[variant] = export_predictions(path, dataset, variant,
                                                  result["rotations"])
    payload = {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1f_baselines",
        "dataset": dataset,
        "scope": "falsification_audit",
        "defines_gate": False,
        "variants": {v: summarise(r) for v, r in results.items()},
        "dev_macro_f1": dev_scores,
        "comparator": comparator,
        "predictions": predictions,
        "seconds": time.time() - t0,
    }
    return payload, results, comparator
