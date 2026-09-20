"""M1-C leave-one-reader-out mechanism pilot (M1 plan §10–§15).

Rotations (frozen)::

    train mistral + internlm -> hold qwen
    train qwen + internlm    -> hold mistral
    train qwen + mistral     -> hold internlm

Per rotation: ``utility_train`` trains, ``utility_dev`` selects, and the
held-out reader's ``utility_eval`` rows are evaluated exactly once with the
seed-averaged selected model.

Leakage boundaries enforced structurally:

* feature scalers are fit on the training readers' ``utility_train`` rows;
* fingerprint normalisation is fit on the training readers only;
* sign class weights come from the training fold only;
* the held-out reader's utility labels never enter training, dev selection,
  scaler fitting, class weights or hyperparameter selection — its eval rows
  are materialised only after the config is frozen;
* B3 picks the nearest *training* reader by fingerprint distance, which uses
  no utility labels at all.

B2 is an in-domain diagnostic trained/evaluated on all three readers and is
never an unseen-reader comparator.
"""
from __future__ import annotations

import itertools

from ..config import protocol as P
from ..models.baselines import (TrainRefused, average_predictions,
                                build_evidence_net, predict,
                                sign_class_weights, train_model)
from ..probes import fingerprint as fp
from . import bootstrap as boot
from . import utility_metrics as um


class LoroRefused(RuntimeError):
    """Raised when a LORO contract (split, leakage, coverage) is violated."""


_EPS_VAR = 1e-12


# --------------------------------------------------------------------------
# feature assembly
# --------------------------------------------------------------------------
def e0e1_vector(e0_row: dict, e1_row: dict) -> list:
    return [e0_row["e0"][name] for name in P.E0_FEATURE_NAMES] + \
           [e1_row["e1"][name] for name in P.E1_FEATURE_NAMES]


def build_feature_table(dataset: str, atomic_entries, e0_rows, e1_rows,
                        e2_rows, e3_rows=None) -> dict:
    """``{key: {"event_id", "cutoff", "e0e1", "struct", "e2", "e3"}}``.

    ``e3_rows`` exists only after a recorded ZERO-TOUCH failure (B5); a B4
    table simply has no ``e3`` member and B4 rows never look at it.
    """
    e0_by_key = {r["key"]: r for r in e0_rows}
    e1_by_key = {r["key"]: r for r in e1_rows}
    e2_by_key = {}
    for r in e2_rows:
        e2_by_key.setdefault(r["key"], {})[r["reader"]] = r["e2"]
    e3_by_key = {}
    for r in e3_rows or []:
        e3_by_key.setdefault(r["key"], {})[r["reader"]] = r["e3"]
    table = {}
    for entry in atomic_entries:
        key = entry["key"]
        e0 = e0_by_key.get(key)
        e1 = e1_by_key.get(key)
        e2 = e2_by_key.get(key)
        if e0 is None or e1 is None or e2 is None:
            raise LoroRefused(f"{dataset}/{key}: incomplete feature caches "
                              f"(e0={e0 is not None} e1={e1 is not None} "
                              f"e2={e2 is not None})")
        if sorted(e2) != sorted(P.READER_KEYS):
            raise LoroRefused(f"{key}: E2 readers {sorted(e2)}")
        record = {
            "event_id": str(entry["event_id"]),
            "cutoff": int(entry["cutoff"]),
            "e0e1": e0e1_vector(e0, e1),
            "struct": [e0["struct"][name] for name in P.B1_STRUCT_NAMES],
            "e2": {reader: [e2[reader][name] for name in P.E2_FEATURE_NAMES]
                   for reader in P.READER_KEYS},
        }
        if e3_rows is not None:
            e3 = e3_by_key.get(key)
            if e3 is None or sorted(e3) != sorted(P.READER_KEYS):
                raise LoroRefused(f"{key}: E3 readers "
                                  f"{sorted(e3) if e3 else None}")
            record["e3"] = {
                reader: [e3[reader][name] for name in P.E3_FEATURE_NAMES]
                for reader in P.READER_KEYS}
        table[key] = record
    return table


def make_rows(atomic_entries, table, readers, event_ids, model: str,
              held_out: str = None) -> list:
    """Supervised rows ``{x, x_r(B4), u, s, key, event_id, sign, utility}``.

    Only ``readers`` x ``event_ids`` are materialised; the held-out reader is
    simply not in ``readers`` while training/selecting.
    """
    wanted = {str(e) for e in event_ids}
    rows = []
    for entry in atomic_entries:
        if str(entry["event_id"]) not in wanted:
            continue
        features = table[entry["key"]]
        for reader in readers:
            if reader not in entry["utility"] or reader not in entry["sign"]:
                raise LoroRefused(f"{entry['key']}: reader {reader} missing "
                                  "from the atomic index")
            base = list(features["e0e1"])
            if model in (P.MODEL_B1,):
                x = base + list(features["struct"])
            elif model == P.MODEL_B4:
                x = base + list(features["e2"][reader])
            elif model == P.MODEL_B5:
                x = base + list(features["e2"][reader]) \
                    + list(features["e3"][reader])
            else:
                x = base
            rows.append({
                "key": entry["key"],
                "event_id": str(entry["event_id"]),
                "reader": reader,
                "x": x,
                "u": float(entry["utility"][reader]),
                "s": um.SIGN_TO_INDEX[entry["sign"][reader]],
                "sign": entry["sign"][reader],
                "utility": float(entry["utility"][reader]),
            })
    return rows


# --------------------------------------------------------------------------
# scaling
# --------------------------------------------------------------------------
def fit_feature_scaler(rows) -> dict:
    """Mean/std of every feature dim over the given (training) rows."""
    if not rows:
        raise LoroRefused("no rows to fit the feature scaler")
    dim = len(rows[0]["x"])
    means, stds = [], []
    for j in range(dim):
        col = [r["x"][j] for r in rows]
        mean = sum(col) / len(col)
        var = sum((v - mean) ** 2 for v in col) / max(len(col) - 1, 1)
        std = var ** 0.5
        means.append(mean)
        stds.append(std if std > _EPS_VAR else 1.0)
    return {"mean": means, "std": stds, "n_rows": len(rows),
            "epsilon": _EPS_VAR}


def apply_scaler(rows, scaler: dict) -> list:
    return [{**r, "x": [(v - m) / s
                        for v, m, s in zip(r["x"], scaler["mean"],
                                           scaler["std"])]}
            for r in rows]


# --------------------------------------------------------------------------
# grid search
# --------------------------------------------------------------------------
def grid_configs() -> list:
    keys = ("hidden", "dropout", "lr", "weight_decay")
    values = [P.MODEL_GRID[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _build_model(model_kind: str, in_dim: int, config: dict,
                 fingerprint_dim: int = None):
    if model_kind in (P.MODEL_B4, P.MODEL_B5):
        from ..models.conditioned_utility import build_conditioned_net
        return build_conditioned_net(in_dim, fingerprint_dim,
                                     embed=config["hidden"],
                                     dropout=config["dropout"])
    return build_evidence_net(in_dim, hidden=config["hidden"],
                              dropout=config["dropout"])


def select_and_train(model_kind: str, train_rows, dev_rows,
                     fingerprint_vectors=None, train_kwargs=None) -> dict:
    """Frozen grid x seeds; dev Macro-F1 (seed-mean) selects the config.

    Returns the selected config, the dev record and the per-seed trained
    models (kept in memory for the one-shot held-out evaluation).
    """
    configs = grid_configs()
    train_signs = [r["sign"] for r in train_rows]
    weights = sign_class_weights(train_signs)
    in_dim = len(train_rows[0]["x"])
    forward_extra = None
    fingerprint_dim = None
    if model_kind in (P.MODEL_B4, P.MODEL_B5):
        if fingerprint_vectors is None:
            raise LoroRefused(f"{model_kind} needs fingerprint vectors")
        fingerprint_dim = len(fingerprint_vectors[train_rows[0]["reader"]])
        forward_extra = lambda rows: [fingerprint_vectors[r["reader"]]
                                      for r in rows]  # noqa: E731
    scored = []
    for config in configs:
        runs = []
        for seed in P.MODEL_SEEDS:
            model = _build_model(model_kind, in_dim, config, fingerprint_dim)
            result = train_model(model, train_rows, dev_rows, config, seed,
                                 weights, forward_extra=forward_extra,
                                 **(train_kwargs or {}))
            runs.append({"seed": seed, "model": model, "result": result})
        dev_f1 = sum(r["result"]["best_dev_macro_f1"] for r in runs) \
            / len(runs)
        dev_loss = sum(r["result"]["best_dev_loss"] for r in runs) / len(runs)
        scored.append({"config": config, "dev_macro_f1": dev_f1,
                       "dev_loss": dev_loss, "runs": runs})
    best = max(scored, key=lambda s: (s["dev_macro_f1"], -s["dev_loss"]))
    return {"selected_config": best["config"],
            "dev_macro_f1": best["dev_macro_f1"],
            "dev_loss": best["dev_loss"],
            "runs": best["runs"],
            "grid_summary": [{"config": s["config"],
                              "dev_macro_f1": s["dev_macro_f1"],
                              "dev_loss": s["dev_loss"]} for s in scored],
            "n_parameters": best["runs"][0]["result"]["n_parameters"]}


def seed_averaged_eval(selected: dict, eval_rows, model_kind: str,
                       fingerprint_vectors=None) -> dict:
    forward_extra = None
    if model_kind in (P.MODEL_B4, P.MODEL_B5):
        forward_extra = lambda rows: [fingerprint_vectors[r["reader"]]
                                      for r in rows]  # noqa: E731
    preds = [predict(run["model"], eval_rows, forward_extra=forward_extra)
             for run in selected["runs"]]
    return average_predictions(preds)


# --------------------------------------------------------------------------
# B3 nearest-reader transfer
# --------------------------------------------------------------------------
def b3_predict(fingerprints: dict, train_readers, held_out: str,
               atomic_entries, eval_rows) -> dict:
    """Direct utility/sign transfer from the fingerprint-nearest trainer."""
    scaler = fp.fit_fingerprint_scaler(fingerprints, train_readers)
    z_held = fp.transform_fingerprint(
        fp.fingerprint_vector(fingerprints, held_out), scaler)
    distances = {}
    nearest, nearest_d = None, None
    for reader in train_readers:
        z = fp.transform_fingerprint(
            fp.fingerprint_vector(fingerprints, reader), scaler)
        d = sum((a - b) ** 2 for a, b in zip(z_held, z)) ** 0.5
        distances[reader] = d
        if nearest_d is None or d < nearest_d:
            nearest, nearest_d = reader, d
    lookup = {}
    for entry in atomic_entries:
        lookup[entry["key"]] = (float(entry["utility"][nearest]),
                                entry["sign"][nearest])
    utilities, probs = [], []
    for row in eval_rows:
        u, sign = lookup[row["key"]]
        utilities.append(u)
        one_hot = [0.0] * len(P.SIGN_CLASSES)
        one_hot[um.SIGN_TO_INDEX[sign]] = 1.0
        probs.append(one_hot)
    return {"prediction": {"utility": utilities, "sign_probs": probs},
            "nearest_reader": nearest, "distances": distances}


# --------------------------------------------------------------------------
# one rotation
# --------------------------------------------------------------------------
def run_rotation(dataset: str, rotation, atomic_entries, table, split,
                 fingerprints: dict, models=(P.MODEL_B0, P.MODEL_B4),
                 train_kwargs=None) -> dict:
    train_readers = [rotation[0], rotation[1]]
    held_out = rotation[2]
    if tuple(sorted(train_readers + [held_out])) != tuple(sorted(P.READER_KEYS)):
        raise LoroRefused(f"bad rotation {rotation}")
    out = {"dataset": dataset, "held_out": held_out,
           "train_readers": list(train_readers), "models": {}}
    folds = {name: set(str(e) for e in split[name])
             for name in ("utility_train", "utility_dev", "utility_eval")}

    fp_scaler = fp.fit_fingerprint_scaler(fingerprints, train_readers)
    fingerprint_vectors = {r: fp.transform_fingerprint(
        fp.fingerprint_vector(fingerprints, r), fp_scaler)
        for r in P.READER_KEYS}

    eval_cache = {}
    for model_kind in models:
        train_rows = make_rows(atomic_entries, table, train_readers,
                               folds["utility_train"], model_kind)
        dev_rows = make_rows(atomic_entries, table, train_readers,
                             folds["utility_dev"], model_kind)
        eval_rows = make_rows(atomic_entries, table, [held_out],
                              folds["utility_eval"], model_kind)
        if not train_rows or not dev_rows or not eval_rows:
            raise LoroRefused(
                f"{dataset}/{held_out}/{model_kind}: empty fold "
                f"({len(train_rows)}/{len(dev_rows)}/{len(eval_rows)})")
        # structural leakage check: folds contain exactly their own readers
        if {r["reader"] for r in train_rows + dev_rows} != set(train_readers):
            raise LoroRefused(
                f"{dataset}/{held_out}/{model_kind}: train/dev readers "
                f"{sorted({r['reader'] for r in train_rows + dev_rows})} "
                f"!= {sorted(train_readers)}")
        if {r["reader"] for r in eval_rows} != {held_out}:
            raise LoroRefused(
                f"{dataset}/{held_out}/{model_kind}: eval readers "
                f"{sorted({r['reader'] for r in eval_rows})} != [{held_out}]")
        scaler = fit_feature_scaler(train_rows)
        train_s = apply_scaler(train_rows, scaler)
        dev_s = apply_scaler(dev_rows, scaler)
        eval_s = apply_scaler(eval_rows, scaler)

        if model_kind == P.MODEL_B3:
            raise LoroRefused("B3 is handled outside the trained loop")
        selected = select_and_train(model_kind, train_s, dev_s,
                                    fingerprint_vectors=fingerprint_vectors,
                                    train_kwargs=train_kwargs)
        prediction = seed_averaged_eval(selected, eval_s, model_kind,
                                        fingerprint_vectors=fingerprint_vectors)
        record = {
            "selected_config": selected["selected_config"],
            "dev_macro_f1": selected["dev_macro_f1"],
            "dev_loss": selected["dev_loss"],
            "n_parameters": selected["n_parameters"],
            "grid_summary": selected["grid_summary"],
            "scaler_rows": scaler["n_rows"],
            "fold_sizes": {"train": len(train_rows), "dev": len(dev_rows),
                           "eval": len(eval_rows)},
            "eval_metrics": um.evaluate_predictions(eval_rows, prediction),
        }
        out["models"][model_kind] = record
        eval_cache[model_kind] = (eval_rows, prediction)

    # B3 needs no training: it always runs on the same eval rows as B0.
    eval_rows, _b0 = eval_cache[P.MODEL_B0]
    b3 = b3_predict(fingerprints, train_readers, held_out,
                    atomic_entries, eval_rows)
    out["models"][P.MODEL_B3] = {
        "nearest_reader": b3["nearest_reader"],
        "distances": b3["distances"],
        "eval_metrics": um.evaluate_predictions(eval_rows, b3["prediction"]),
    }
    eval_cache[P.MODEL_B3] = (eval_rows, b3["prediction"])

    # disagreement-focused subset (training-reader signs only)
    train_signs_by_key = {}
    for entry in atomic_entries:
        train_signs_by_key[entry["key"]] = {
            r: entry["sign"][r] for r in train_readers}
    eval_rows, b0_pred = eval_cache[P.MODEL_B0]
    mask = um.disagreement_mask(eval_rows, train_signs_by_key)
    out["disagreement_subset"] = {"n": sum(mask), "of": len(mask)}
    for model_kind, (rows, pred) in eval_cache.items():
        if model_kind == P.MODEL_B3:
            continue
        sub_rows, sub_pred = um.subset(rows, mask), \
            um.subset_prediction(pred, mask)
        out["models"][model_kind]["eval_metrics_disagreement"] = \
            um.evaluate_predictions(sub_rows, sub_pred) if sub_rows else None

    out["eval_rows"] = eval_cache
    return out


# --------------------------------------------------------------------------
# B2 in-domain diagnostic
# --------------------------------------------------------------------------
def run_b2_in_domain(dataset: str, atomic_entries, table, split,
                     train_kwargs=None) -> dict:
    """Reader-ID one-hot diagnostic; in-domain only (M1 plan §10)."""
    folds = {name: set(str(e) for e in split[name])
             for name in ("utility_train", "utility_dev", "utility_eval")}
    readers = list(P.READER_KEYS)

    def rows_for(events):
        rows = make_rows(atomic_entries, table, readers, events, P.MODEL_B0)
        one_hot = {r: [1.0 if r == q else 0.0 for q in readers]
                   for r in readers}
        for row in rows:
            row["reader_one_hot"] = one_hot[row["reader"]]
        return rows

    train_rows = rows_for(folds["utility_train"])
    dev_rows = rows_for(folds["utility_dev"])
    eval_rows = rows_for(folds["utility_eval"])
    scaler = fit_feature_scaler(train_rows)
    train_s, dev_s, eval_s = (apply_scaler(r, scaler)
                              for r in (train_rows, dev_rows, eval_rows))
    weights = sign_class_weights([r["sign"] for r in train_s])
    forward_extra = lambda rows: [r["reader_one_hot"] for r in rows]  # noqa: E731
    configs = grid_configs()
    scored = []
    for config in configs:
        runs = []
        for seed in P.MODEL_SEEDS:
            model = build_evidence_net(len(train_s[0]["x"]),
                                       hidden=config["hidden"],
                                       dropout=config["dropout"],
                                       extra_dim=len(readers))
            result = train_model(model, train_s, dev_s, config, seed,
                                 weights, forward_extra=forward_extra,
                                 **(train_kwargs or {}))
            runs.append({"seed": seed, "model": model, "result": result})
        dev_f1 = sum(r["result"]["best_dev_macro_f1"] for r in runs) \
            / len(runs)
        dev_loss = sum(r["result"]["best_dev_loss"] for r in runs) / len(runs)
        scored.append({"config": config, "dev_macro_f1": dev_f1,
                       "dev_loss": dev_loss, "runs": runs})
    best = max(scored, key=lambda s: (s["dev_macro_f1"], -s["dev_loss"]))
    preds = [predict(run["model"], eval_s, forward_extra=forward_extra)
             for run in best["runs"]]
    prediction = average_predictions(preds)
    return {"dataset": dataset, "kind": P.MODEL_B2, "in_domain": True,
            "selected_config": best["config"],
            "dev_macro_f1": best["dev_macro_f1"],
            "eval_metrics": um.evaluate_predictions(eval_rows, prediction)}


# --------------------------------------------------------------------------
# dataset-level pilot + gate
# --------------------------------------------------------------------------
def run_dataset(dataset: str, atomic_entries, table, split, fingerprints,
                models=(P.MODEL_B0, P.MODEL_B4), train_kwargs=None,
                primary_model=P.MODEL_B4) -> dict:
    if primary_model not in models:
        raise LoroRefused(f"primary model {primary_model} not in {models}")
    rotations = {}
    for rotation in P.LORO_ROTATIONS:
        held = rotation[2]
        rotations[held] = run_rotation(dataset, rotation, atomic_entries,
                                       table, split, fingerprints,
                                       models=models,
                                       train_kwargs=train_kwargs)
    per_reader_bootstrap = {}
    agg_payloads = []
    for rotation in P.LORO_ROTATIONS:
        held = rotation[2]
        cache = rotations[held]["eval_rows"]
        eval_rows, b0_pred = cache[P.MODEL_B0]
        _rows, primary_pred = cache[primary_model]
        per_reader_bootstrap[held] = boot.paired_event_delta(
            eval_rows, primary_pred["sign_probs"], b0_pred["sign_probs"])
        agg_payloads.append((eval_rows, primary_pred["sign_probs"],
                             b0_pred["sign_probs"]))
    aggregate = boot.aggregate_delta(agg_payloads)
    per_reader_delta = {held: per_reader_bootstrap[held]["observed"]
                        for held in per_reader_bootstrap}
    result = {
        "dataset": dataset,
        "primary_model": primary_model,
        "primary_comparison": [primary_model, P.MODEL_B0],
        "rotations": {},
        "per_reader_bootstrap": per_reader_bootstrap,
        "aggregate": {
            "mean_delta_macro_f1": aggregate["observed"],
            "ci_low": aggregate["ci_low"],
            "ci_high": aggregate["ci_high"],
            "iterations": aggregate["iterations"],
            "seed": aggregate["seed"],
            "per_reader_delta": per_reader_delta,
            "positive_readers": sum(1 for d in per_reader_delta.values()
                                    if d > 0),
            "worst_reader_delta": min(per_reader_delta.values()),
        },
    }
    for rotation in P.LORO_ROTATIONS:
        held = rotation[2]
        rot = dict(rotations[held])
        rot.pop("eval_rows", None)
        result["rotations"][held] = rot
    return result, aggregate, rotations


def decide_primary_gate(maweibo_result: dict) -> dict:
    """The frozen Ma-Weibo M1_FULL_GO gate (M1 plan §15)."""
    agg = maweibo_result["aggregate"]
    checks = {
        "mean_delta_gte_0.03":
            agg["mean_delta_macro_f1"] >= P.GATE_MEAN_DELTA_MIN,
        "ci_low_gt_0": agg["ci_low"] > 0.0,
        "positive_readers_gte_2":
            agg["positive_readers"] >= P.GATE_POSITIVE_READERS_MIN,
        "worst_reader_gte_-0.05":
            agg["worst_reader_delta"] >= P.GATE_WORST_READER_MIN,
    }
    passed = all(checks.values())
    return {
        "dataset": "maweibo",
        "decides_m1_gate": True,
        "thresholds": {
            "mean_delta_min": P.GATE_MEAN_DELTA_MIN,
            "ci_alpha": P.GATE_CI_ALPHA,
            "positive_readers_min": P.GATE_POSITIVE_READERS_MIN,
            "worst_reader_min": P.GATE_WORST_READER_MIN,
        },
        "checks": checks,
        "aggregate": agg,
        "passed": passed,
        "verdict_if_passed": "M1_FULL_GO",
        "verdict_if_failed": "proceed_to_M1_D",
    }
