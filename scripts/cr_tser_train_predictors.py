#!/usr/bin/env python
"""Train the utility predictors under leave-one-reader-out (plan §16, §17, §20, §33).

Steps 1–6 of the plan's training order. The leave-one-reader-out isolation is
enforced at the **entrance**: every rotation filters the utility-label cache
down to its two training readers before any dataset object exists, and
``UtilityDataset`` itself fails closed if a held-out reader survives.

Evidence identity is the canonical ``dataset|event|cutoff|node`` key shared
with the selector, the selection runner and the verifier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_tser_common as common  # noqa: E402

from cr_tser.config.pilot_config import (LORO_ROTATIONS, READER_KEYS,  # noqa: E402
                                         SIGN_CLASSES, TRAIN_SEEDS)
from cr_tser.intervention.evidence_units import evidence_key  # noqa: E402
from cr_tser.models.scalar_structure_baseline import (  # noqa: E402
    ScalarStructurePredictor, scalar_structure_features)
from cr_tser.models.text_baseline import TextOnlyPredictor, text_features  # noqa: E402
from cr_tser.models.utility_heads import q_vector  # noqa: E402
from cr_tser.training.checkpointing import (  # noqa: E402
    save_baseline_checkpoint, save_checkpoint)
from cr_tser.training.train_utility import (group_z, infer_z,  # noqa: E402
                                            predict_baseline, train_baseline,
                                            train_rotation)
from cr_tser.training.utility_dataset import (UtilityDataset,  # noqa: E402
                                              assert_groups_only_readers,
                                              filter_rows_for_readers,
                                              make_group)

TRAIN_SEGMENTS = ("utility_train", "utility_dev", "utility_eval")


def load_labels(out_root, dataset):
    path = os.path.join(out_root, "utility_labels", dataset, "labels.jsonl")
    if not os.path.exists(path):
        path = os.path.join(out_root, "utility_labels", f"{dataset}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"utility labels missing for {dataset}")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_snapshot_cache(dataset, paths, out_root, segments=TRAIN_SEGMENTS):
    """One artifact pass over every labeled (event, cutoff)."""
    split = json.loads((Path(out_root) / "manifests" / dataset /
                        "event_split.json").read_text(encoding="utf-8"))
    wanted = set()
    for seg in segments:
        wanted |= set(split[seg])
    events = {e["event_id"]: e for e in common.load_dataset_events(dataset,
                                                                   paths)
              if e["event_id"] in wanted}
    encoder = common.CrSemanticEncoder(paths.semantic_model, dataset)
    tokenizer = common.canonical_tokenizer(paths.canonical_tokenizer)
    cache = {}
    for eid in sorted(events):
        for cutoff in common.CUTOFFS_MIN:
            art = common.snapshot_artifacts(events[eid], cutoff, encoder,
                                            tokenizer)
            if art["zero_reply"]:
                continue
            cache[(eid, cutoff)] = art
    return cache


def build_groups_from_cache(cache, labels, dataset):
    """Snapshot groups for BiTTE plus row-level features for B0/B1.

    ``labels`` must already be filtered to the rotation's training readers.
    """
    by_snap = defaultdict(list)
    for row in labels:
        if row["intervention_type"] == "I1_atomic":
            by_snap[(row["event_id"], row["cutoff"])].append(row)
    groups, features = [], {}
    for (eid, cutoff) in sorted(by_snap):
        art = cache.get((eid, cutoff))
        if art is None:
            continue
        snap, units, src = art["snapshot"], art["units"], art["src"]
        pos = {nid: i for i, nid in enumerate(snap["node_ids"])}
        n = len(snap["node_ids"])
        sem = art["semantic"]
        struct = torch.tensor(art["scalars"], dtype=torch.float32)
        q = torch.zeros(n, 6)
        source_emb = sem[pos[snap["source_id"]]]
        for unit in units:
            i = pos[unit["node_id"]]
            nid = unit["node_id"]
            cosine = src["relevance"].get(nid, 0.0)
            cost = src["unit_token_costs"].get(nid, 0)
            q[i] = torch.tensor(q_vector(
                cosine, cost, src["rank_percentile"].get(nid, 0.0), cutoff,
                src["n_selected"], src["utilization"]))
            x0 = text_features(sem[i], source_emb, cosine, cost,
                               src["n_selected"], cutoff)
            features[evidence_key(dataset, eid, cutoff, nid)] = {
                "x0": x0,
                "x1": scalar_structure_features(x0, art["scalars"][i]),
            }
        ctx_indices = [pos[nid] for nid in src["selected_node_ids"]]
        unit_rows = []
        for row in by_snap[(eid, cutoff)]:
            nid = row["affected_reply_ids"][0]
            if nid not in pos:
                continue
            unit_rows.append({
                "unit_index": pos[nid], "reader": row["reader"],
                "target": row["utility"], "sign": row["sign"],
                "correct_before": row["correctness_before"],
                "correct_after": row["correctness_after"],
            })
        infer_keys = [(pos[nid], evidence_key(dataset, eid, cutoff, nid))
                      for nid in src["selected_node_ids"]]
        if unit_rows or infer_keys:
            groups.append(make_group(snap, sem, struct, q, unit_rows, cutoff,
                                     ctx_indices, dataset=dataset,
                                     infer_keys=infer_keys))
    return groups, features


def _baseline_rows(labels, features, reader_keys, split, segments):
    wanted = set()
    for seg in segments:
        wanted |= set(split[seg])
    rows = []
    for row in labels:
        if row["intervention_type"] != "I1_atomic":
            continue
        if row["reader"] not in reader_keys or row["event_id"] not in wanted:
            continue
        nid = row["affected_reply_ids"][0]
        key = evidence_key(row.get("dataset", ""), row["event_id"],
                           row["cutoff"], nid)
        if key not in features:
            continue
        rows.append({"key": key, "reader": row["reader"],
                     "target": row["utility"], "sign": row["sign"],
                     "x0": features[key]["x0"], "x1": features[key]["x1"]})
    return rows


def _average_predictions(accum):
    """Seed-average utility + three-class sign probabilities, then argmax."""
    out = {}
    for key, values in accum.items():
        n = len(values)
        mean_probs = [sum(v[1][i] for v in values) / n
                      for i in range(len(SIGN_CLASSES))]
        best = max(range(len(mean_probs)), key=lambda i: mean_probs[i])
        out[key] = {
            "utility": sum(v[0] for v in values) / n,
            "probs": mean_probs,
            "predicted_sign": SIGN_CLASSES[best],
            "class_scores": {c: mean_probs[i]
                             for i, c in enumerate(SIGN_CLASSES)},
        }
    return out


def _reader_predictions(runs, groups, reader_key, reader_index_map, device):
    """B3 predictions for one reader over **every** inference candidate.

    The predicted sign comes from the model's auxiliary sign head (argmax of
    its three-class output). Ground-truth correctness transitions are never
    consulted here — they only ever build the training target.
    """
    accum = {}
    for run in runs:
        with torch.no_grad():
            for g in groups:
                z = infer_z(run["bitte"], g, device)
                r = torch.full((z.shape[0],), reader_index_map[reader_key],
                               dtype=torch.long, device=device)
                _mu, _delta, u_hat, logits = run["model"](z, r)
                probs = torch.softmax(logits.float(), dim=-1).tolist()
                for row, value, prob in zip(g["infer_rows"], u_hat.tolist(),
                                            probs):
                    accum.setdefault(row["unit_key"], []).append(
                        (float(value), prob))
    return _average_predictions(accum)


def _shared_predictions(runs, groups, device):
    """Shared-head (S4) predictions over every inference candidate."""
    accum = {}
    for run in runs:
        with torch.no_grad():
            for g in groups:
                z = infer_z(run["bitte"], g, device)
                mu, logits = run["model"].shared_only(z, 0)
                probs = torch.softmax(logits.float(), dim=-1).tolist()
                for row, value, prob in zip(g["infer_rows"], mu.tolist(),
                                            probs):
                    accum.setdefault(row["unit_key"], []).append(
                        (float(value), prob))
    return _average_predictions(accum)


def train_single_reader(dataset, out_root, cache, all_labels, reader, device):
    """S3a/S3b: a selector trained on **one** reader's supervision (plan §22).

    Same architecture and training protocol as B3, but its dataset contains
    only ``reader``'s rows, so its artifact records ``training_readers ==
    [reader]`` and it is a genuine single-reader baseline rather than a
    reader-conditioned output of the two-reader model.
    """
    split = json.loads((Path(out_root) / "manifests" / dataset /
                        "event_split.json").read_text(encoding="utf-8"))
    reader_labels, dropped = filter_rows_for_readers(all_labels, [reader])
    groups, _features = build_groups_from_cache(cache, reader_labels, dataset)
    assert_groups_only_readers(groups, [reader], context="single-reader groups")
    assert all(row["reader"] == reader for g in groups
               for row in g["unit_rows"])

    def subset(ids):
        return [g for g in groups if g["event_id"] in ids]

    reader_index_map = {reader: 0}
    train_ds = UtilityDataset(subset(set(split["utility_train"])), [reader],
                              reader_index_map)
    dev_ds = UtilityDataset(subset(set(split["utility_dev"])), [reader],
                            reader_index_map)
    eval_groups = subset(set(split["utility_eval"]))
    trained = train_rotation(train_ds, dev_ds, [reader], TRAIN_SEEDS,
                             device=device)
    out_dir = os.path.join(out_root, "predictor", dataset,
                           f"single_{reader}")
    os.makedirs(out_dir, exist_ok=True)
    checkpoints = []
    for run in trained["runs"]:
        meta = {"seed": run["seed"], "best_epoch": run["best"]["epoch"],
                "dev_metric": run["best"]["metric"],
                "dev_score": run["best"]["score"],
                "training_readers": [reader],
                "held_out_rows_filtered": len(dropped),
                "architecture": "single_reader_shared_residual"}
        checkpoints.append(save_checkpoint(
            os.path.join(out_dir, f"b3_seed{run['seed']}.pt"), run["bitte"],
            run["model"], meta))
    predictions = _reader_predictions(trained["runs"], eval_groups, reader,
                                      reader_index_map, device)
    payload = {"dataset": dataset, "training_readers": [reader],
               "held_out_rows_filtered": len(dropped),
               "checkpoints": checkpoints, "predictions": predictions,
               "n_eval_groups": len(eval_groups)}
    common.write_json(os.path.join(out_dir, "predictions.json"), payload)
    return payload


def run_rotation(dataset, out_root, cache, all_labels, rotation, device):
    """One LORO rotation; held-out labels are filtered out at this entrance."""
    train_readers, held = list(rotation[:2]), rotation[2]
    reader_index_map = {k: i for i, k in enumerate(train_readers)}
    train_labels, dropped = filter_rows_for_readers(all_labels, train_readers)
    split = json.loads((Path(out_root) / "manifests" / dataset /
                        "event_split.json").read_text(encoding="utf-8"))

    groups, features = build_groups_from_cache(cache, train_labels, dataset)
    assert_groups_only_readers(groups, train_readers, context="rotation groups")
    assert all(row["reader"] != held for g in groups for row in g["unit_rows"])

    def subset(ids):
        return [g for g in groups if g["event_id"] in ids]

    train_ds = UtilityDataset(subset(set(split["utility_train"])),
                              train_readers, reader_index_map)
    dev_ds = UtilityDataset(subset(set(split["utility_dev"])),
                            train_readers, reader_index_map)
    eval_groups = subset(set(split["utility_eval"]))
    trained = train_rotation(train_ds, dev_ds, train_readers, TRAIN_SEEDS,
                             device=device)

    out_dir = os.path.join(out_root, "predictor", dataset,
                           f"rotation_{train_readers[0]}_{train_readers[1]}")
    os.makedirs(out_dir, exist_ok=True)
    checkpoints = []
    for run in trained["runs"]:
        meta = {"seed": run["seed"], "best_epoch": run["best"]["epoch"],
                "dev_metric": run["best"]["metric"],
                "dev_score": run["best"]["score"],
                "train_readers": train_readers, "held_out": held,
                "evidence_key_contract": "dataset|event|cutoff|node"}
        checkpoints.append(save_checkpoint(
            os.path.join(out_dir, f"b3_seed{run['seed']}.pt"), run["bitte"],
            run["model"], meta))
        common.write_json(os.path.join(out_dir,
                                       f"b3_seed{run['seed']}_history.json"),
                          run["history"])

def _baseline_infer_rows(groups, features, split, segments):
    """B0/B1 inference rows for **every** ``C_ref`` candidate (not just labeled)."""
    wanted = set()
    for seg in segments:
        wanted |= set(split[seg])
    rows = []
    for g in groups:
        if g["event_id"] not in wanted:
            continue
        for r in g["infer_rows"]:
            key = r["unit_key"]
            if key not in features:
                continue
            rows.append({"key": key, "x0": features[key]["x0"],
                         "x1": features[key]["x1"]})
    return rows


def _prediction_coverage(groups, predictions):
    """Coverage of the required inference candidate set, per prediction source."""
    needed = {r["unit_key"] for g in groups for r in g["infer_rows"]}
    out = {}
    for name, preds in predictions.items():
        covered = len(needed & set(preds))
        out[name] = {
            "required": len(needed), "covered": covered,
            "rate": (covered / len(needed)) if needed else 1.0,
            "complete": covered == len(needed),
        }
    return out


def run_rotation(dataset, out_root, cache, all_labels, rotation, device):
    """One LORO rotation; held-out labels are filtered out at this entrance."""
    train_readers, held = list(rotation[:2]), rotation[2]
    reader_index_map = {k: i for i, k in enumerate(train_readers)}
    train_labels, dropped = filter_rows_for_readers(all_labels, train_readers)
    split = json.loads((Path(out_root) / "manifests" / dataset /
                        "event_split.json").read_text(encoding="utf-8"))

    groups, features = build_groups_from_cache(cache, train_labels, dataset)
    assert_groups_only_readers(groups, train_readers, context="rotation groups")
    assert all(row["reader"] != held for g in groups for row in g["unit_rows"])

    def subset(ids):
        return [g for g in groups if g["event_id"] in ids]

    train_ds = UtilityDataset(subset(set(split["utility_train"])),
                              train_readers, reader_index_map)
    dev_ds = UtilityDataset(subset(set(split["utility_dev"])),
                            train_readers, reader_index_map)
    eval_groups = subset(set(split["utility_eval"]))
    trained = train_rotation(train_ds, dev_ds, train_readers, TRAIN_SEEDS,
                             device=device)

    out_dir = os.path.join(out_root, "predictor", dataset,
                           f"rotation_{train_readers[0]}_{train_readers[1]}")
    os.makedirs(out_dir, exist_ok=True)
    checkpoints = []
    for run in trained["runs"]:
        meta = {"seed": run["seed"], "best_epoch": run["best"]["epoch"],
                "dev_metric": run["best"]["metric"],
                "dev_score": run["best"]["score"],
                "train_readers": train_readers, "held_out": held,
                "evidence_key_contract": "dataset|event|cutoff|node"}
        checkpoints.append(save_checkpoint(
            os.path.join(out_dir, f"b3_seed{run['seed']}.pt"), run["bitte"],
            run["model"], meta))
        common.write_json(os.path.join(out_dir,
                                       f"b3_seed{run['seed']}_history.json"),
                          run["history"])

    predictions = {}
    for key in train_readers:
        predictions[key] = _reader_predictions(trained["runs"], eval_groups,
                                               key, reader_index_map, device)
    predictions["shared"] = _shared_predictions(trained["runs"], eval_groups,
                                                device)

    train_rows = _baseline_rows(train_labels, features, train_readers, split,
                                ("utility_train",))
    dev_rows = _baseline_rows(train_labels, features, train_readers, split,
                              ("utility_dev",))
    eval_infer = _baseline_infer_rows(eval_groups, features, split,
                                      ("utility_eval",))
    baseline_out = {}
    for name, factory, idx in (
            ("B0_text", lambda: TextOnlyPredictor(), "x0"),
            ("B1_scalar_structure", lambda: ScalarStructurePredictor(), "x1")):
        runs = train_baseline([{**r, "x": r[idx]} for r in train_rows],
                              [{**r, "x": r[idx]} for r in dev_rows],
                              factory, TRAIN_SEEDS, device=device,
                              reader_keys=train_readers)
        preds = predict_baseline(runs,
                                 [{**r, "x": r[idx]} for r in eval_infer],
                                 device=device)
        baseline_out[name] = {
            "predictions": preds,
            "seeds": [r["seed"] for r in runs],
            "best_epochs": [r["best"]["epoch"] for r in runs],
        }
        for run in runs:
            save_baseline_checkpoint(
                os.path.join(out_dir, f"{name}_seed{run['seed']}.pt"),
                run["model"], {"baseline": name, "seed": run["seed"]})

    coverage_sources = dict(predictions)
    coverage_sources.update({name: entry["predictions"]
                             for name, entry in baseline_out.items()})
    payload = {"dataset": dataset, "train_readers": train_readers,
               "held_out_reader": held, "checkpoints": checkpoints,
               "predictions": predictions, "baselines": baseline_out,
               "n_eval_groups": len(eval_groups),
               "held_out_rows_filtered": len(dropped),
               "prediction_coverage": _prediction_coverage(
                   eval_groups, coverage_sources),
               "single_reader_artifacts": [f"single_{r}"
                                           for r in train_readers],
               "evidence_key_contract": "dataset|event|cutoff|node"}
    common.write_json(os.path.join(out_dir, "predictions.json"), payload)
    return payload


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pheme", "weibo22"), required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="read/write the smoke namespace only")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    out_root = args.out_root or os.path.join(paths.out_root or
                                             str(common.REPO / "results" /
                                                 "cr_tser"))
    if args.smoke:
        out_root = common.smoke_root(out_root)
    common.assert_frozen_source(args.dataset, paths, out_root,
                                "train_predictors")
    cache = build_snapshot_cache(args.dataset, paths, out_root)
    labels = load_labels(out_root, args.dataset)
    print(f"snapshot cache: {len(cache)} | label rows: {len(labels)}")
    results = [run_rotation(args.dataset, out_root, cache, labels, rotation,
                            args.device)
               for rotation in LORO_ROTATIONS]
    # S3a/S3b: one single-reader selector per reader (plan §22 review fix)
    singles = [train_single_reader(args.dataset, out_root, cache, labels,
                                   reader, args.device)
               for reader in READER_KEYS]
    print(json.dumps([{"train": r["train_readers"],
                       "held_out": r["held_out_reader"],
                       "held_out_rows_filtered": r["held_out_rows_filtered"],
                       "checkpoints": len(r["checkpoints"])}
                      for r in results], indent=1))
    print(json.dumps([{"single_reader": s["training_readers"],
                       "predictions": len(s["predictions"])}
                      for s in singles], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
