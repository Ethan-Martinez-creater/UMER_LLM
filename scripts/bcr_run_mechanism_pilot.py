#!/usr/bin/env python
"""M1-C — three-reader LORO mechanism pilot and the Ma-Weibo gate.

Runs B0/B1/B3/B4 through the frozen LORO rotations per dataset, B2 as an
in-domain diagnostic, evaluates the held-out reader exactly once per
rotation with seed-averaged predictions, bootstraps ``B4 - B0`` at the event
level (10000 iterations, seed 7319) and applies the frozen Ma-Weibo primary
gate. PHEME runs the identical pipeline with ``diagnostic_only = true`` —
it can never rescue or decide the Ma-Weibo gate (M1 plan §15–§16).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "project"))
sys.path.insert(0, str(REPO / "scripts"))

import cr_tser_common as common  # noqa: E402

from bcr_utility.config import protocol as P  # noqa: E402
from bcr_utility.evaluation import unseen_reader as lor  # noqa: E402
from bcr_utility.evaluation import utility_metrics as um  # noqa: E402

ZERO_TOUCH_MODELS = (P.MODEL_B0, P.MODEL_B1, P.MODEL_B4)


class PilotRefused(RuntimeError):
    """Raised when the M1-C pilot cannot run exactly as specified."""


def _sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, payload) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return {"path": path, "bytes": os.path.getsize(path),
            "sha256": _sha_file(path)}


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _inputs(repo_root, dataset):
    bootstrap = P.bootstrap_dir(repo_root)
    index = _load_json(os.path.join(bootstrap, P.ATOMIC_INDEX_FILENAME))
    entries = (index.get("datasets") or {}).get(dataset, {}).get("entries")
    if not entries:
        raise PilotRefused(f"{dataset}: atomic index missing")
    fdir = P.m1_path(repo_root, "features")
    e0 = _load_jsonl(os.path.join(fdir, f"e0_{dataset}.jsonl"))
    e1 = _load_jsonl(os.path.join(fdir, f"e1_{dataset}.jsonl"))
    e2 = _load_jsonl(os.path.join(fdir, f"e2_{dataset}.jsonl"))
    split = _load_json(os.path.join(
        P.historical_root(repo_root), "manifests", dataset,
        "event_split.json"))
    for name in ("utility_train", "utility_dev", "utility_eval"):
        if not split.get(name):
            raise PilotRefused(f"{dataset}: historical split {name} missing")
    fingerprints = _load_json(P.m1_path(repo_root, "fingerprints",
                                        "fingerprints.json"))
    if sorted(fingerprints.get("readers") or {}) != sorted(P.READER_KEYS):
        raise PilotRefused("fingerprint readers drift")
    return entries, e0, e1, e2, split, fingerprints


def _export_predictions(path, dataset, rotations_cache):
    rows_written = 0
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for held, rot in rotations_cache.items():
            for model_kind, (rows, pred) in rot["eval_rows"].items():
                for row, u_hat, probs in zip(rows, pred["utility"],
                                             pred["sign_probs"]):
                    rec = {
                        "dataset": dataset,
                        "held_out": held,
                        "model": model_kind,
                        "key": row["key"],
                        "event_id": row["event_id"],
                        "reader": row["reader"],
                        "gold_sign": row["sign"],
                        "gold_utility": row["utility"],
                        "pred_utility": u_hat,
                        "pred_sign": um.INDEX_TO_SIGN[max(
                            range(len(P.SIGN_CLASSES)),
                            key=lambda c: probs[c])],
                        "prob_helpful": probs[um.SIGN_TO_INDEX["HELPFUL"]],
                        "prob_neutral": probs[um.SIGN_TO_INDEX["NEUTRAL"]],
                        "prob_harmful": probs[um.SIGN_TO_INDEX["HARMFUL"]],
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rows_written += 1
    return rows_written


def run_dataset(repo_root, dataset, decides_gate: bool) -> dict:
    t0 = time.time()
    entries, e0, e1, e2, split, fingerprints = _inputs(repo_root, dataset)
    table = lor.build_feature_table(dataset, entries, e0, e1, e2)
    print(f"[m1c] {dataset}: {len(entries)} atomic keys, "
          f"{len(table)} feature rows ({time.time() - t0:.1f}s)")
    result, aggregate, rotations_cache = lor.run_dataset(
        dataset, entries, table, split, fingerprints,
        models=ZERO_TOUCH_MODELS)
    for held in result["rotations"]:
        for model_kind, record in result["rotations"][held]["models"].items():
            metrics = record.get("eval_metrics") or {}
            print(f"[m1c] {dataset}/hold={held}/{model_kind}: "
                  f"macro_f1={metrics.get('macro_f1'):.4f} "
                  f"({time.time() - t0:.1f}s)")
    b2 = lor.run_b2_in_domain(dataset, entries, table, split)
    print(f"[m1c] {dataset}/B2 in-domain diagnostic: "
          f"macro_f1={b2['eval_metrics']['macro_f1']:.4f}")
    result["b2_in_domain_diagnostic"] = b2
    result["decides_m1_gate"] = bool(decides_gate)
    result["diagnostic_only"] = not bool(decides_gate)
    result["seconds"] = time.time() - t0
    return {"result": result, "aggregate": aggregate,
            "rotations_cache": rotations_cache, "split": split,
            "fingerprints_sha256": fingerprints.get("sha256")}


def _environment(paths, extra) -> dict:
    import torch
    import transformers
    import numpy
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "device": "cpu (deterministic small-MLP training)",
        "platform": platform.platform(),
        **extra,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO))
    parser.add_argument("--datasets", nargs="+", choices=P.DATASETS,
                        default=list(P.DATASETS))
    args = parser.parse_args(argv)
    paths = common.paths_or_exit()
    t0 = time.time()

    pred_path = P.m1_path(args.repo_root, "evaluation",
                          P.M1_PREDICTIONS_FILENAME)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    if os.path.exists(pred_path):
        os.remove(pred_path)

    outcomes = {}
    for dataset in args.datasets:
        decides = dataset == P.PRIMARY_DATASET
        run = run_dataset(args.repo_root, dataset, decides_gate=decides)
        n_pred = _export_predictions(pred_path, dataset,
                                     run["rotations_cache"])
        run["result"]["predictions_file"] = {
            "path": pred_path, "rows_appended": n_pred}
        outcomes[dataset] = run
        agg = run["result"]["aggregate"]
        print(f"[m1c] {dataset}: mean_delta={agg['mean_delta_macro_f1']:.4f} "
              f"CI=[{agg['ci_low']:.4f}, {agg['ci_high']:.4f}] "
              f"positive={agg['positive_readers']}/3 "
              f"worst={agg['worst_reader_delta']:.4f}")

    if P.PRIMARY_DATASET not in outcomes:
        raise PilotRefused("the primary dataset must run for the M1 gate")
    gate = lor.decide_primary_gate(outcomes[P.PRIMARY_DATASET]["result"])

    evaluation = {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1c_zero_touch",
        "models": list(ZERO_TOUCH_MODELS) + [P.MODEL_B2, P.MODEL_B3],
        "primary_comparison": list(P.PRIMARY_COMPARISON),
        "bootstrap": {"unit": P.BOOTSTRAP_UNIT,
                      "iterations": P.BOOTSTRAP_ITERATIONS,
                      "seed": P.BOOTSTRAP_SEED},
        "datasets": {d: outcomes[d]["result"] for d in outcomes},
        "gate": gate,
    }
    eval_info = _write_json(P.m1_path(args.repo_root, "evaluation",
                                      "evaluation.json"), evaluation)
    gate_info = _write_json(P.m1_path(args.repo_root, "evaluation",
                                      P.M1_GATE_FILENAME), gate)
    pred_info = {"path": pred_path, "bytes": os.path.getsize(pred_path),
                 "sha256": _sha_file(pred_path)}

    verdict = {
        "protocol": P.PROTOCOL_VERSION,
        "stage": "m1c",
        "zero_touch_passed": gate["passed"],
        "maweibo_gate": gate,
        "pheme_diagnostic": {
            "diagnostic_only": True,
            "decides_m1_gate": False,
            "aggregate": outcomes[P.SECONDARY_DATASET]["result"]["aggregate"]
            if P.SECONDARY_DATASET in outcomes else None,
        },
        "next": "M1_FULL_GO" if gate["passed"] else "run_M1_D",
        "seconds": time.time() - t0,
    }
    verdict_info = _write_json(P.m1_path(args.repo_root,
                                         P.M1_VERDICT_FILENAME), verdict)
    env_info = _write_json(P.m1_path(
        args.repo_root, "environment.json"), _environment(paths, {
            "fingerprints_sha256": outcomes[P.PRIMARY_DATASET][
                "fingerprints_sha256"],
        }))

    print(json.dumps({
        "zero_touch_passed": gate["passed"],
        "next": verdict["next"],
        "evaluation": eval_info,
        "gate_file": gate_info,
        "predictions": pred_info,
        "verdict_file": verdict_info,
        "environment": env_info,
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
