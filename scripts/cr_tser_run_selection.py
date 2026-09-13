#!/usr/bin/env python
"""Unseen-reader evaluation of the frozen selection arms (plan §21–§23, §33).

The §33 order is enforced as **two auditable stages**:

``--mode freeze-subsets`` (Stage A)
    Reads training-reader predictors only, builds S0–S5 (plus S6 for PHEME),
    and writes dataset/rotation-scoped frozen subset artifacts with a content
    hash. It **never** loads, constructs or calls a held-out reader.

``--mode score-heldout`` (Stage B)
    Verifies the frozen subset artifact and its hash first, fails closed if
    anything changed, and only then loads the held-out reader to score the
    frozen subsets. It never re-runs the selector or rebuilds a subset.

``--mode both`` runs A then B (convenience; the code paths stay separate).

Evidence identity: predictor artifacts are keyed by the canonical
``dataset|event|cutoff|node`` string. This runner parses that key back into its
components and hands the selector a per-snapshot ``{node_id: score}`` map, so
predictor output, single-reader arms (S3a/S3b), the shared arm (S4) and the
robust arm (S5) all speak the same contract.

S6 (legacy TC-DSCR Utility-TM) is PHEME-only and must be requested explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
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
from cr_tser.models.legacy_utility import (build_legacy_scorer,  # noqa: E402
                                           legacy_arm_enabled)
from cr_tser.models.robust_selector import all_arms  # noqa: E402
from cr_tser.readers.base_reader import (build_messages,  # noqa: E402
                                         build_reader_prompt, ReaderSpec,
                                         build_reader)
from cr_tser.readers.sequence_scorer import ab_scores  # noqa: E402
from cr_tser.evaluation.unseen_reader import (rotation_delta,  # noqa: E402
                                              selection_metrics)


class FrozenSubsetMissing(RuntimeError):
    """Raised when Stage B cannot find or verify the frozen subsets."""


class FrozenSubsetChanged(RuntimeError):
    """Raised when a frozen subset artifact no longer matches its hash."""


class FrozenSubsetAlreadyExists(RuntimeError):
    """Raised when Stage A would overwrite an already frozen subset."""


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


def _load_single_predictions(out_root, dataset, reader):
    """S3a/S3b source: the single-reader-trained selector artifact (plan §22)."""
    path = os.path.join(out_root, "predictor", dataset, f"single_{reader}",
                        "predictions.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    training_readers = payload.get("training_readers", [])
    if training_readers != [reader]:
        raise ValueError(
            f"single-reader artifact for {reader!r} was trained on "
            f"{training_readers!r}; S3a/S3b must be single-reader trained")
    return payload.get("predictions", {})


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


def _legacy_item(event, snapshot, sem_rows):
    """Light TC-DSCR item for one snapshot; S6 and B2 share it."""
    from tcdscr_run_e2 import build_light_item
    return build_light_item(event, snapshot, sem_rows)


def _legacy_scores(legacy_scorer, item):
    """Frozen Static Utility ``u_i`` for every S6 candidate (PHEME only)."""
    if legacy_scorer is None:
        return {}
    if not hasattr(legacy_scorer, "score_items"):
        return dict(legacy_scorer(item))
    return legacy_scorer.score_items([item], device=legacy_scorer.device)[0]


def _canonical_sha(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def frozen_dir(out_root, dataset):
    return os.path.join(out_root, "unseen_reader", dataset, "frozen")


def _frozen_path(out_root, dataset, held):
    return os.path.join(frozen_dir(out_root, dataset), f"rotation_{held}.json")


def existing_frozen_rotations(out_root, dataset):
    """Held-out readers whose frozen subset artifact is already on disk."""
    return [rotation[2] for rotation in LORO_ROTATIONS
            if os.path.exists(_frozen_path(out_root, dataset, rotation[2]))]


def assert_unfrozen(dataset, out_root):
    """Stage A is write-once (plan §33).

    The frozen evidence subset is the audit anchor of Stage B, so rewriting the
    JSON together with its ``.sha256`` would silently change what the held-out
    reader is scored on. There is deliberately no ``--force`` escape hatch: a
    restart is an explicit researcher action that removes the whole unused
    artifact set, never an automatic overwrite. The check runs before any
    expensive work so a second Stage A fails immediately and leaves the
    existing artifacts byte-identical.
    """
    existing = existing_frozen_rotations(out_root, dataset)
    if existing:
        raise FrozenSubsetAlreadyExists(
            f"frozen evidence subsets already exist for {dataset}: {existing}; "
            "Stage A is write-once (plan §33). Remove the unused artifact set "
            "explicitly before re-freezing.")


# --------------------------------------------------------------------------
# Stage A — subset construction (no held-out reader anywhere in this path)
# --------------------------------------------------------------------------
def _write_b2_diagnostic(dataset, out_root, legacy_scorer, items, selections,
                         frozen_records):
    """PHEME-only B2 continuity diagnostic artifact (plan §17 B2).

    The surface comes from the frozen TC-DSCR Proxy through
    ``LegacyPHEMEUtility.b2_surface`` and is recorded for the pilot report
    only. It is excluded from P0–P4, the P3 baseline competition and the P4
    primary comparison by construction — the aggregator reads it but never
    scores it.
    """
    from cr_tser.evaluation.unseen_reader import reader_metrics
    surfaces = legacy_scorer.b2_items(items, selections)
    golds = [int(item["label"]) for item in items]
    preds = [int(surface["predicted_index"]) for surface in surfaces]
    rows = [{"event_id": item["event_id"],
             "cutoff": int(item["cutoff_minutes"]),
             "gold": int(item["label"]),
             "pred": int(surface["predicted_index"]),
             "p_rumor": surface["p_rumor"],
             "p_nonrumor": surface["p_nonrumor"],
             "n_selected": surface["n_selected"],
             "selected_node_ids": surface["selected_node_ids"]}
            for item, surface in zip(items, surfaces)]
    metrics_by_cutoff = {}
    for cutoff in sorted({row["cutoff"] for row in rows}):
        picked = [row for row in rows if row["cutoff"] == cutoff]
        metrics_by_cutoff[str(cutoff)] = reader_metrics(
            [row["gold"] for row in picked], [row["pred"] for row in picked])
    payload = {
        "stage": "C_b2_legacy_diagnostic",
        "dataset": dataset,
        "diagnostic": "B2_legacy_tcdscr",
        "diagnostic_only": True,
        "participates_in_primary_gate": False,
        "excluded_from_gates": ["P0", "P1", "P2", "P3", "P4"],
        "never_in_p3_baseline_competition": True,
        "never_in_p4_primary_comparison": True,
        "selection_arm": "S6_legacy_utility_tm",
        "utility_definition": "StaticUtilitySelector(h_i, event_repr, sem_i, "
                              "sem_src, struct3_i)",
        "classification_contract": "classify_selected(proxy, h_source, "
                                   "mean(selected h_i))",
        "frozen_fingerprint": legacy_scorer.fingerprint(),
        "frozen_subset_sha256": {
            record["held_out_reader"]: record["sha256"]
            for record in frozen_records},
        "n_snapshots": len(rows),
        "metrics": reader_metrics(golds, preds) if rows else {},
        "metrics_by_cutoff": metrics_by_cutoff,
        "rows": rows,
        "note": "PHEME-only continuity diagnostic; it can never change the "
                "GO/NO-GO recommendation (plan §17 B2, §25)",
    }
    common.write_json(os.path.join(out_root, "unseen_reader", dataset,
                                   "b2_legacy_diagnostic.json"), payload)
    return payload


def freeze_subsets(dataset, paths, out_root, max_snapshots=None, legacy=False,
                   legacy_scorer=None, single_predictions=None):
    assert_unfrozen(dataset, out_root)
    common.assert_frozen_source(dataset, paths, out_root, "freeze_subsets")
    split = json.loads((Path(out_root) / "manifests" / dataset /
                        "event_split.json").read_text(encoding="utf-8"))
    eval_ids = set(split["utility_eval"])
    events = {e["event_id"]: e for e in
              common.load_dataset_events(dataset, paths)
              if e["event_id"] in eval_ids}
    encoder = common.CrSemanticEncoder(paths.semantic_model, dataset)
    tokenizer = common.canonical_tokenizer(paths.canonical_tokenizer)
    if legacy and not legacy_arm_enabled(dataset):
        raise ValueError(
            f"S6 legacy Utility-TM is PHEME-only; refusing dataset {dataset!r}")
    if legacy and legacy_scorer is None:
        legacy_scorer = build_legacy_scorer(dataset, "cpu")

    arms_used = [a for a in SELECTION_ARMS if a != "S6_legacy_utility_tm"] + \
        (["S6_legacy_utility_tm"] if legacy else [])
    os.makedirs(frozen_dir(out_root, dataset), exist_ok=True)
    legacy_items, legacy_selections = [], []
    out = []
    for rotation_index, rotation in enumerate(LORO_ROTATIONS):
        train_readers, held = list(rotation[:2]), rotation[2]
        payload = _load_predictor(out_root, dataset, train_readers)
        predictions_artifact = payload["predictions"]
        singles = single_predictions if single_predictions is not None else {
            r: _load_single_predictions(out_root, dataset, r)
            for r in train_readers}
        rows = []
        n_snap = 0
        for event_id in sorted(events):
            for cutoff in CUTOFFS_MIN:
                art = common.snapshot_artifacts(events[event_id], cutoff,
                                                encoder, tokenizer)
                if art["zero_reply"]:
                    continue
                units, src = art["units"], art["src"]
                snapshot_predictions = {
                    key: predictions_for_snapshot(predictions_artifact, dataset,
                                                  event_id, cutoff, key)
                    for key in list(train_readers) + ["shared"]}
                for index, reader_key in enumerate(train_readers):
                    slot = "S3a_source" if index == 0 else "S3b_source"
                    snapshot_predictions[slot] = predictions_for_snapshot(
                        {"single": singles.get(reader_key, {})}, dataset,
                        event_id, cutoff, "single")
                item = None
                if legacy:
                    sem_rows = {nid: art["semantic"][i] for i, nid in
                                enumerate(art["snapshot"]["node_ids"])}
                    item = _legacy_item(events[event_id], art["snapshot"],
                                        sem_rows)
                    snapshot_predictions["legacy"] = _legacy_scores(
                        legacy_scorer, item)
                arms = all_arms(units, src, snapshot_predictions, tokenizer,
                                train_readers, PARTITION_SEED,
                                include_legacy=legacy)
                if legacy and rotation_index == 0:
                    # B2 is dataset-level, not rotation-level: the frozen
                    # selector sees the same candidates in every rotation, so
                    # the diagnostic is collected once from the first pass.
                    legacy_items.append(item)
                    legacy_selections.append(
                        list(arms["S6_legacy_utility_tm"]["selected_node_ids"]))
                for arm_name, arm in arms.items():
                    rows.append({
                        "event_id": event_id, "cutoff": cutoff,
                        "gold": int(events[event_id]["label"]),
                        "arm": arm_name,
                        "selected_node_ids": list(arm["selected_node_ids"]),
                        "total_tokens": arm["total_tokens"],
                        "target_tokens": arm["target_tokens"],
                        "content_hash": arm["content_hash"],
                    })
                n_snap += 1
                if max_snapshots and n_snap >= max_snapshots:
                    break
            if max_snapshots and n_snap >= max_snapshots:
                break
        record = {
            "stage": "A_freeze_subsets",
            "dataset": dataset,
            "train_readers": train_readers,
            "held_out_reader": held,
            "arms": arms_used,
            "n_snapshots": n_snap,
            "legacy_s6_enabled": bool(legacy),
            "evidence_key_contract": "dataset|event|cutoff|node",
            "source": common.source_fingerprint_for(dataset, paths),
            "predictor_fingerprint": payload.get("prediction_coverage", {}),
            "subsets": rows,
        }
        record["sha256"] = _canonical_sha(record)
        path = _frozen_path(out_root, dataset, held)
        common.write_json(path, record)
        with open(path + ".sha256", "w", encoding="utf-8") as fh:
            fh.write(record["sha256"])
        out.append(record)
    if legacy and legacy_items:
        _write_b2_diagnostic(dataset, out_root, legacy_scorer, legacy_items,
                             legacy_selections, out)
    return out


def load_frozen_subsets(out_root, dataset, held):
    """Stage B entry: read and hash-verify a frozen subset artifact."""
    path = _frozen_path(out_root, dataset, held)
    if not os.path.exists(path):
        raise FrozenSubsetMissing(
            f"frozen subset artifact missing for {dataset}/{held}: {path}; "
            "run --mode freeze-subsets first (plan §33)")
    with open(path, encoding="utf-8") as fh:
        record = json.load(fh)
    expected = record.get("sha256")
    actual = _canonical_sha(record)
    digest_file = path + ".sha256"
    on_disk = open(digest_file, encoding="utf-8").read().strip() \
        if os.path.exists(digest_file) else None
    if expected != actual or (on_disk is not None and on_disk != expected):
        raise FrozenSubsetChanged(
            f"frozen subset artifact changed for {dataset}/{held}: "
            f"recorded={expected} actual={actual} file={on_disk}; refusing to "
            "score a modified subset (plan §33)")
    return record


# --------------------------------------------------------------------------
# Stage B — held-out scoring of the already frozen subsets
# --------------------------------------------------------------------------
def score_heldout(dataset, paths, out_root, device, mock=False,
                  max_snapshots=None, subset_hashes_override=None):
    common.assert_frozen_source(dataset, paths, out_root, "score_heldout")
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

    # verify every frozen subset BEFORE any held-out reader exists
    frozen = {}
    for rotation in LORO_ROTATIONS:
        held = rotation[2]
        frozen[held] = load_frozen_subsets(out_root, dataset, held)

    rotations = []
    for rotation in LORO_ROTATIONS:
        train_readers, held = list(rotation[:2]), rotation[2]
        record = frozen[held]
        expected_sha = record["sha256"]
        if subset_hashes_override is not None and \
                subset_hashes_override.get(held) != expected_sha:
            raise FrozenSubsetChanged(
                f"caller-provided hash for {held} does not match the frozen "
                "artifact")
        reader = build_reader(held, ReaderSpec(held, paths.reader_path(held)),
                              mock=mock)
        collected = {}
        for row in record["subsets"]:
            arm = row["arm"]
            bucket = collected.setdefault(
                arm, {"golds": [], "preds": [], "tokens": [],
                      "reference_ids": set()})
            art = common.snapshot_artifacts(events[row["event_id"]],
                                            row["cutoff"], encoder, tokenizer)
            units = art["units"]
            selected = set(row["selected_node_ids"])
            ctx = [u for u in units if u["node_id"] in selected]
            text = render_units_for_budget(units, row["selected_node_ids"])
            prompt = build_reader_prompt(_source_text(events[row["event_id"]]),
                                         row["cutoff"], ctx, text)
            out = ab_scores(reader.candidate_logprobs(prompt))
            bucket["golds"].append(row["gold"])
            bucket["preds"].append(1 if out["prediction"] == "A" else 0)
            bucket["tokens"].append(row["total_tokens"])
            bucket["reference_ids"] |= set(art["src"]["selected_node_ids"])
        reader.unload()

        metrics, subsets = {}, {}
        for arm, bucket in collected.items():
            if not bucket["golds"]:
                continue
            metrics[arm] = selection_metrics(bucket["golds"], bucket["preds"],
                                             bucket["tokens"])
            subsets[arm] = {
                "reference_ids": sorted(bucket["reference_ids"]),
                "selected_node_ids": [r["selected_node_ids"] for r in
                                      record["subsets"] if r["arm"] == arm][:1],
                "total_tokens": max((r["total_tokens"] for r in
                                     record["subsets"] if r["arm"] == arm),
                                    default=0),
                "target_tokens": max((r["target_tokens"] for r in
                                      record["subsets"] if r["arm"] == arm),
                                     default=0),
                "frozen_subsets": [r for r in record["subsets"]
                                   if r["arm"] == arm],
            }
        delta = rotation_delta(metrics) if metrics else {"delta": float("nan")}
        token_ok = all(r["total_tokens"] <= r["target_tokens"]
                       for r in record["subsets"])
        delta["token_target_ok"] = token_ok
        result = {
            "stage": "B_score_heldout",
            "dataset": dataset, "train_readers": train_readers,
            "held_out_reader": held,
            "n_snapshots": record["n_snapshots"],
            "legacy_s6_enabled": record["legacy_s6_enabled"],
            "frozen_subset_sha256": expected_sha,
            "evidence_key_contract": record["evidence_key_contract"],
            "arms": subsets, "metrics": metrics, "delta": delta,
            "reader_identity": reader.identity() if not mock else {"mock": True},
        }
        rotations.append(result)
        common.write_json(os.path.join(out_dir, f"rotation_{held}.json"),
                          result)
    return rotations


def run(dataset, paths, out_root, device, mock=False, max_snapshots=None,
        legacy=False, legacy_scorer=None, single_predictions=None):
    """Stage A then Stage B (the code paths stay distinct)."""
    freeze_subsets(dataset, paths, out_root, max_snapshots=max_snapshots,
                   legacy=legacy, legacy_scorer=legacy_scorer,
                   single_predictions=single_predictions)
    return score_heldout(dataset, paths, out_root, device, mock=mock,
                         max_snapshots=max_snapshots)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("maweibo", "pheme"), required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-snapshots", type=int, default=None)
    ap.add_argument("--legacy", action="store_true",
                    help="enable the PHEME-only S6 legacy arm")
    ap.add_argument("--mode", choices=("freeze-subsets", "score-heldout",
                                       "both"), default="both")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    out_root = common.default_out_root(paths, args.out_root)
    if args.smoke:
        out_root = common.smoke_root(out_root)
    if args.mode in ("freeze-subsets", "both"):
        frozen = freeze_subsets(args.dataset, paths, out_root,
                                max_snapshots=args.max_snapshots,
                                legacy=args.legacy)
        print(json.dumps([{"held_out": r["held_out_reader"],
                           "sha256": r["sha256"],
                           "subsets": len(r["subsets"])}
                          for r in frozen], indent=1))
    if args.mode in ("score-heldout", "both"):
        rotations = score_heldout(args.dataset, paths, out_root, args.device,
                                  mock=args.smoke,
                                  max_snapshots=args.max_snapshots)
        print(json.dumps([{"held_out": r["held_out_reader"],
                           "delta": r["delta"].get("delta"),
                           "arms": len(r["metrics"])}
                          for r in rotations], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
