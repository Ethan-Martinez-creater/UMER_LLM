#!/usr/bin/env python
"""Unseen-reader evaluation of the frozen selection arms (plan §21–§23, §33).

Order of operations is the plan's: predictors are already trained and frozen,
arms are built from **training-reader** predictions only, the evidence subsets
are frozen (ids + token counts + content hash), and only then is the held-out
reader scores every arm's prompt.

Evidence identity: predictor artifacts are keyed by the canonical
``dataset|event|cutoff|node`` string. This runner parses that key back into its
components and hands the selector a per-snapshot ``{node_id: score}`` map, so
predictor output, single-reader arms (S3a/S3b), the shared arm (S4) and the
robust arm (S5) all speak the same contract.

S6 (legacy TC-DSCR Utility-TM) is PHEME-only and must be requested explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_tser_common as common  # noqa: E402

from cr_tser.config.pilot_config import (CUTOFFS_MIN, LORO_ROTATIONS,  # noqa: E402
                                         PARTITION_SEED, SELECTION_ARMS)
from cr_tser.intervention.evidence_units import (  # noqa: E402
    evidence_key_parts, render_units_for_budget)
from cr_tser.models.legacy_utility import legacy_arm_enabled  # noqa: E402
from cr_tser.models.robust_selector import all_arms  # noqa: E402
from cr_tser.readers.base_reader import (build_messages,  # noqa: E402
                                         build_reader_prompt, ReaderSpec,
                                         build_reader)
from cr_tser.readers.sequence_scorer import ab_scores  # noqa: E402
from cr_tser.evaluation.unseen_reader import (rotation_delta,  # noqa: E402
                                              selection_metrics)


def _source_text(event):
    return next(n["text"] for n in event["nodes"]
                if n["node_id"] == event["source_id"])


def _load_predictor(out_root, dataset, train_readers):
    path = os.path.join(out_root, "predictor", dataset,
                        f"rotation_{train_readers[0]}_{train_readers[1]}",
                        "predictions.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"predictor missing: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def predictions_for_snapshot(payload_predictions, dataset, event_id, cutoff,
                             reader_key):
    """``{node_id: utility}`` for one snapshot, from canonical-key artifacts."""
    out = {}
    for key, entry in payload_predictions.get(reader_key, {}).items():
        parts = evidence_key_parts(key)
        if parts["dataset"] == dataset and parts["event_id"] == event_id \
                and parts["cutoff"] == int(cutoff):
            out[parts["node_id"]] = entry["utility"]
    return out


def run(dataset, paths, out_root, device, mock=False, max_snapshots=None,
        legacy=False):
    split = json.loads((Path(out_root) / "manifests" / dataset /
                        "event_split.json").read_text(encoding="utf-8"))
    eval_ids = set(split["utility_eval"])
    events = {e["event_id"]: e for e in
              common.load_dataset_events(dataset, paths)
              if e["event_id"] in eval_ids}
    encoder = common.CrSemanticEncoder(paths.semantic_model, dataset)
    tokenizer = common.canonical_tokenizer(paths.canonical_tokenizer)
    out_dir = os.path.join(out_root, "unseen_reader", dataset)
    os.makedirs(out_dir, exist_ok=True)
    if legacy and not legacy_arm_enabled(dataset):
        raise ValueError(
            f"S6 legacy Utility-TM is PHEME-only; refusing dataset {dataset!r}")

    arms_used = [a for a in SELECTION_ARMS if a != "S6_legacy_utility_tm"] + \
        (["S6_legacy_utility_tm"] if legacy else [])
    rotations = []
    for rotation in LORO_ROTATIONS:
        train_readers, held = list(rotation[:2]), rotation[2]
        payload = _load_predictor(out_root, dataset, train_readers)
        predictions_artifact = payload["predictions"]
        reader = build_reader(held, ReaderSpec(held, paths.reader_path(held)),
                              mock=mock)
        collected = {arm: {"golds": [], "preds": [], "tokens": [],
                           "reference_ids": set(), "subsets": []}
                     for arm in arms_used}
        n_snap = 0
        for event_id in sorted(events):
            for cutoff in CUTOFFS_MIN:
                art = common.snapshot_artifacts(events[event_id], cutoff,
                                                encoder, tokenizer)
                if art["zero_reply"]:
                    continue
                units, src = art["units"], art["src"]
                gold = int(events[event_id]["label"])
                snapshot_predictions = {
                    key: predictions_for_snapshot(predictions_artifact, dataset,
                                                  event_id, cutoff, key)
                    for key in list(train_readers) + ["shared"]}
                if legacy:
                    snapshot_predictions["legacy"] = {}
                arms = all_arms(units, src, snapshot_predictions, tokenizer,
                                train_readers, PARTITION_SEED,
                                include_legacy=legacy)
                for arm_name, arm in arms.items():
                    ctx = [u for u in units
                           if u["node_id"] in set(arm["selected_node_ids"])]
                    text = render_units_for_budget(
                        units, arm["selected_node_ids"])
                    prompt = build_reader_prompt(_source_text(events[event_id]),
                                                 cutoff, ctx, text)
                    out = ab_scores(reader.candidate_logprobs(prompt))
                    bucket = collected[arm_name]
                    bucket["golds"].append(gold)
                    bucket["preds"].append(1 if out["prediction"] == "A" else 0)
                    bucket["tokens"].append(arm["total_tokens"])
                    bucket["reference_ids"] |= set(src["selected_node_ids"])
                    bucket["subsets"].append({
                        "event_id": event_id, "cutoff": cutoff,
                        "selected_node_ids": arm["selected_node_ids"],
                        "total_tokens": arm["total_tokens"],
                        "target_tokens": arm["target_tokens"],
                        "content_hash": arm["content_hash"]})
                n_snap += 1
                if max_snapshots and n_snap >= max_snapshots:
                    break
            if max_snapshots and n_snap >= max_snapshots:
                break
        reader.unload()

        metrics, subsets = {}, {}
        for arm_name, bucket in collected.items():
            if not bucket["golds"]:
                continue
            metrics[arm_name] = selection_metrics(
                bucket["golds"], bucket["preds"], bucket["tokens"])
            subsets[arm_name] = {
                "reference_ids": sorted(bucket["reference_ids"]),
                "selected_node_ids": bucket["subsets"][0]["selected_node_ids"]
                if bucket["subsets"] else [],
                "total_tokens": max((s["total_tokens"]
                                     for s in bucket["subsets"]), default=0),
                "target_tokens": max((s["target_tokens"]
                                      for s in bucket["subsets"]), default=0),
                "frozen_subsets": bucket["subsets"],
            }
        delta = rotation_delta(metrics) if metrics else {"delta": float("nan")}
        token_ok = all(s["total_tokens"] <= s["target_tokens"]
                       for bucket in collected.values()
                       for s in bucket["subsets"])
        delta["token_target_ok"] = token_ok
        record = {
            "dataset": dataset, "train_readers": train_readers,
            "held_out_reader": held, "n_snapshots": n_snap,
            "legacy_s6_enabled": bool(legacy),
            "evidence_key_contract": "dataset|event|cutoff|node",
            "arms": subsets, "metrics": metrics, "delta": delta,
            "reader_identity": reader.identity() if not mock else {"mock": True},
        }
        rotations.append(record)
        common.write_json(os.path.join(out_dir, f"rotation_{held}.json"),
                          record)
    return rotations


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pheme", "weibo22"), required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-snapshots", type=int, default=None)
    ap.add_argument("--legacy", action="store_true",
                    help="enable the PHEME-only S6 legacy arm")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    out_root = args.out_root or os.path.join(paths.out_root or
                                             str(common.REPO / "results" /
                                                 "cr_tser"))
    rotations = run(args.dataset, paths, out_root, args.device,
                    mock=args.smoke, max_snapshots=args.max_snapshots,
                    legacy=args.legacy)
    print(json.dumps([{"held_out": r["held_out_reader"],
                       "delta": r["delta"].get("delta"),
                       "arms": len(r["metrics"])} for r in rotations], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
