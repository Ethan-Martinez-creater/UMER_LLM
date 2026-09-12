#!/usr/bin/env python
"""Train the utility predictors under leave-one-reader-out (plan §16, §17, §20, §33).

Steps 1–6 of the plan's training order: build snapshot groups and row-level
baseline features from the cached utility labels, train B0 / B1 / B3 for each
rotation with seeds 7319/7320/7321, early-stop on dev Spearman of the two
**training** readers, freeze checkpoints and predict the evaluation segment.

The held-out reader's labels are never loaded here; the rotation's held-out key
only appears in the output manifest.
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

from cr_tser.config.pilot_config import (LORO_ROTATIONS, SIGN_CLASSES,  # noqa: E402
                                         TRAIN_SEEDS)
from cr_tser.models.scalar_structure_baseline import (  # noqa: E402
    ScalarStructurePredictor, scalar_structure_features)
from cr_tser.models.text_baseline import TextOnlyPredictor, text_features  # noqa: E402
from cr_tser.models.utility_heads import q_vector  # noqa: E402
from cr_tser.training.checkpointing import (  # noqa: E402
    save_baseline_checkpoint, save_checkpoint)
from cr_tser.training.train_utility import (group_z, predict_baseline,  # noqa: E402
                                            train_baseline, train_rotation)
from cr_tser.training.utility_dataset import UtilityDataset, make_group  # noqa: E402


def load_labels(out_root, dataset):
    path = os.path.join(out_root, "utility_labels", f"{dataset}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"utility labels missing: {path}")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_groups(dataset, paths, out_root, segments):
    """Snapshot groups for BiTTE plus row-level features for B0/B1."""
    labels = load_labels(out_root, dataset)
    by_snap = defaultdict(list)
    for row in labels:
        if row["intervention_type"] == "I1_atomic":
            by_snap[(row["event_id"], row["cutoff"])].append(row)
    split = json.loads((Path(out_root) / "manifests" /
                        "event_split.json").read_text(encoding="utf-8"))
    wanted = set()
    for seg in segments:
        wanted |= set(split[seg])
    needed = {eid for (eid, _c) in by_snap if eid in wanted}
    events = {e["event_id"]: e for e in common.load_dataset_events(dataset,
                                                                   paths)
              if e["event_id"] in needed}
    encoder = common.CrSemanticEncoder(paths.semantic_model, dataset)
    tokenizer = common.canonical_tokenizer(paths.canonical_tokenizer)

    groups, features = [], {}
    for (eid, cutoff) in sorted(by_snap):
        if eid not in events:
            continue
        art = common.snapshot_artifacts(events[eid], cutoff, encoder, tokenizer)
        if art["zero_reply"]:
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
            features[f"{eid}:{cutoff}:{nid}"] = {
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
        if unit_rows:
            groups.append(make_group(snap, sem, struct, q, unit_rows, cutoff,
                                     ctx_indices))
    return groups, features


def _baseline_rows(labels, features, reader_keys, segments, split):
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
        key = f"{row['event_id']}:{row['cutoff']}:{nid}"
        if key not in features:
            continue
        rows.append({"key": key, "reader": row["reader"],
                     "target": row["utility"], "sign": row["sign"],
                     "x0": features[key]["x0"], "x1": features[key]["x1"]})
    return rows


def _b3_predictions(runs, groups, reader_key, reader_index_map, device):
    accum = {}
    hi = SIGN_CLASSES.index("HELPFUL")
    ha = SIGN_CLASSES.index("HARMFUL")
    for run in runs:
        with torch.no_grad():
            for g in groups:
                z = group_z(run["bitte"], g, device)
                r = torch.full((z.shape[0],), reader_index_map[reader_key],
                               dtype=torch.long, device=device)
                _mu, _delta, u_hat, logits = run["model"](z, r)
                probs = torch.softmax(logits, dim=-1)
                for row, value, ph, pa in zip(
                        g["unit_rows"], u_hat.tolist(), probs[:, hi].tolist(),
                        probs[:, ha].tolist()):
                    if row["reader"] == reader_key:
                        accum.setdefault(row["unit_key"], []).append(
                            (float(value), float(ph), float(pa)))
    return {k: {"utility": sum(v[0] for v in vals) / len(vals),
                "helpful_score": sum(v[1] for v in vals) / len(vals),
                "harmful_score": sum(v[2] for v in vals) / len(vals)}
            for k, vals in accum.items()}


def _shared_predictions(runs, groups, device):
    accum = {}
    for run in runs:
        with torch.no_grad():
            for g in groups:
                z = group_z(run["bitte"], g, device)
                mu, logits = run["model"].shared_only(z, 0)
                probs = torch.softmax(logits, dim=-1)
                hi = SIGN_CLASSES.index("HELPFUL")
                ha = SIGN_CLASSES.index("HARMFUL")
                for row, value, ph, pa in zip(
                        g["unit_rows"], mu.tolist(), probs[:, hi].tolist(),
                        probs[:, ha].tolist()):
                    accum.setdefault(row["unit_key"], []).append(
                        (float(value), float(ph), float(pa)))
    return {k: {"utility": sum(v[0] for v in vals) / len(vals),
                "helpful_score": sum(v[1] for v in vals) / len(vals),
                "harmful_score": sum(v[2] for v in vals) / len(vals)}
            for k, vals in accum.items()}


def run_rotation(dataset, paths, out_root, groups, features, labels, rotation,
                 device):
    train_readers, held = list(rotation[:2]), rotation[2]
    reader_index_map = {k: i for i, k in enumerate(train_readers)}
    split = json.loads((Path(out_root) / "manifests" /
                        "event_split.json").read_text(encoding="utf-8"))

    def subset(ids):
        return [g for g in groups if g["event_id"] in ids]

    train_ids = set(split["utility_train"])
    dev_ids = set(split["utility_dev"])
    eval_ids = set(split["utility_eval"])
    train_ds = UtilityDataset(subset(train_ids), train_readers,
                              reader_index_map)
    dev_ds = UtilityDataset(subset(dev_ids), train_readers, reader_index_map)
    eval_groups = subset(eval_ids)
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
                "train_readers": train_readers, "held_out": held}
        checkpoints.append(save_checkpoint(
            os.path.join(out_dir, f"b3_seed{run['seed']}.pt"), run["bitte"],
            run["model"], meta))
        common.write_json(os.path.join(out_dir,
                                       f"b3_seed{run['seed']}_history.json"),
                          run["history"])

    predictions = {}
    for key in train_readers:
        predictions[key] = _b3_predictions(trained["runs"], eval_groups, key,
                                           reader_index_map, device)
    predictions["shared"] = _shared_predictions(trained["runs"], eval_groups,
                                                device)

    # ---- B0 / B1 baselines (plan §17) ----
    train_rows = _baseline_rows(labels, features, train_readers,
                                ("utility_train",), split)
    dev_rows = _baseline_rows(labels, features, train_readers,
                              ("utility_dev",), split)
    eval_rows = _baseline_rows(labels, features, train_readers,
                               ("utility_eval",), split)
    baseline_out = {}
    for name, factory, idx in (
            ("B0_text", lambda: TextOnlyPredictor(), "x0"),
            ("B1_scalar_structure", lambda: ScalarStructurePredictor(), "x1")):
        rows_train = [{**r, "x": r[idx]} for r in train_rows]
        rows_dev = [{**r, "x": r[idx]} for r in dev_rows]
        rows_eval = [{**r, "x": r[idx]} for r in eval_rows]
        runs = train_baseline(rows_train, rows_dev, factory, TRAIN_SEEDS,
                              device=device, reader_keys=train_readers)
        preds = predict_baseline(runs, rows_eval, device=device)
        baseline_out[name] = {
            "predictions": {k: {"utility": v} for k, v in preds.items()},
            "seeds": [r["seed"] for r in runs],
            "best_epochs": [r["best"]["epoch"] for r in runs],
        }
        for run in runs:
            save_baseline_checkpoint(
                os.path.join(out_dir, f"{name}_seed{run['seed']}.pt"),
                run["model"], {"baseline": name, "seed": run["seed"]})

    payload = {"dataset": dataset, "train_readers": train_readers,
               "held_out_reader": held, "checkpoints": checkpoints,
               "predictions": predictions, "baselines": baseline_out,
               "n_eval_groups": len(eval_groups)}
    common.write_json(os.path.join(out_dir, "predictions.json"), payload)
    return payload


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pheme", "weibo22"), required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--device", default="cpu")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    out_root = args.out_root or os.path.join(paths.out_root or
                                             str(common.REPO / "results" /
                                                 "cr_tser"))
    groups, features = build_groups(args.dataset, paths, out_root,
                                    ("utility_train", "utility_dev",
                                     "utility_eval"))
    labels = load_labels(out_root, args.dataset)
    print(f"groups: {len(groups)} | unit features: {len(features)}")
    results = [run_rotation(args.dataset, paths, out_root, groups, features,
                            labels, rotation, args.device)
               for rotation in LORO_ROTATIONS]
    print(json.dumps([{"train": r["train_readers"],
                       "held_out": r["held_out_reader"],
                       "checkpoints": len(r["checkpoints"])}
                      for r in results], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
