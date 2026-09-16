#!/usr/bin/env python
"""CR-TSER V2R1 P1/P2 formal gate evaluation (plan §18, §25).

A deliberately thin stage entrypoint. It does exactly three things:

1. loads the frozen v2r1 manifests and utility-label cache, and refuses to run
   if their identity is not the frozen one;
2. calls the **existing frozen** evaluation code —
   :func:`cr_tser.evaluation.heterogeneity.heterogeneity_report` /
   :func:`~cr_tser.evaluation.heterogeneity.gate_p1` and
   :func:`cr_tser.evaluation.structural_interaction.
   structural_interaction_report` / ``gate_p2`` — together with the frozen
   data-reading semantics of :mod:`cr_tser_run_pilot` (``load_unit_table``,
   ``load_interaction_records``);
3. serialises the artifacts.

No statistic is reimplemented, no intervention group is re-matched, no label is
regenerated, and no threshold is read from anywhere but the frozen config. The
only arithmetic done here is descriptive (per-slot means and coverage counts),
and it never feeds a gate.

P1 and P2 are decided on **Ma-Weibo only**. PHEME produces the same reports but
with ``diagnostic_only = true`` and never decides a primary gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for _p in (REPO / "scripts", REPO / "project"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cr_tser_run_pilot as pilot  # noqa: E402

from cr_tser.config.pilot_config import (BOOTSTRAP_ITERATIONS,  # noqa: E402
                                         BOOTSTRAP_SEED, CUTOFFS_MIN,
                                         P1_MEAN_DISAGREEMENT_MIN,
                                         P1_PAIR_DISAGREEMENT_MIN,
                                         P1_PAIRS_REQUIRED, P2_EDGE_DELTA_MIN,
                                         PRIMARY_DATASET, READER_KEYS,
                                         SECONDARY_DATASET, UTILITY_THRESHOLD,
                                         V2R1_RESULTS_ROOT)
from cr_tser.evaluation.bootstrap import mean  # noqa: E402
from cr_tser.evaluation.heterogeneity import (  # noqa: E402
    gate_p1, heterogeneity_report, is_active)
from cr_tser.evaluation.structural_interaction import (  # noqa: E402
    build_event_payloads, gate_p2, structural_interaction_report)

V2R1 = "results/cr_tser_v2r1"
GATES_DIR = os.path.join(V2R1, "gates")

#: The frozen v2r1 utility-label caches this stage is allowed to read. A cache
#: that does not match is a STOP: the gates would otherwise be computed on
#: evidence the research approval did not cover.
FROZEN_LABELS_SHA256 = {
    "maweibo":
        "6c6591eaa76451c45917e97bdedf493cddcc06261a98ec8c8ff1eaab065646c3",
    "pheme":
        "773bee3e98d8d0f0ffc521bb9024839beeb64d2d8c2572f9f8a07dbcfff4ec15",
}

#: The frozen v2r1 manifests (identical to the verifier's pins).
FROZEN_MANIFEST_SHA256 = {
    "maweibo/source.json":
        "80b07954ce3199c57cb25e7ca11b07115dd0e20a84787b97effc1cfed291b783",
    "maweibo/event_split.json":
        "24e18ed10954e8387c49d9a119e78934e2c8d33ccb582aa62e4f2cf201b8dde2",
    "maweibo/hashes.json":
        "f106af7a60b5cb76fd338a4bea53c56a9de79ee700c43a2525c848384d7d7045",
    "maweibo/snapshot_manifest.jsonl":
        "611cb9c6ca43afbaec8a0eba10d71c1fcc4dcb9af3ebbd7cd90a2660a2f434dc",
    "maweibo/intervention_manifest.jsonl":
        "f37c3ffcb07e8e0142a082326ec405417c8f99399a1b09fd907cfaba9b680782",
    "pheme/source.json":
        "1a85a6d9e1f1da9ff7404d3231664463a38b5db444a57bb0f0292876b5b0e363",
    "pheme/event_split.json":
        "f4a2a1cb5a2d81c84eb394fec45373b261c436cc43822a79d5c1a41dcbada385",
    "pheme/hashes.json":
        "a3c46403d9173bdcacc00604706ecd4586f8e80b5a97e5d8992f91a373fe9cd9",
    "pheme/snapshot_manifest.jsonl":
        "8c2ec0462a1fbf9ff681d237fb9e57df09afd9af5c7e402779c2defe8812315f",
    "pheme/intervention_manifest.jsonl":
        "a0c0ad7abb5c8132ecb9483143ad8c70bc6836568a183e32797a8af73201d2e3",
}

RETIRED_READERS = ("glm",)


class StageRefused(RuntimeError):
    """Raised when the frozen inputs are not the ones this stage may use."""


def _canonical_sha256(path: str) -> str:
    """LF-normalized content digest (CRLF and LF checkouts agree)."""
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")
                          ).hexdigest()


def _assert_frozen_inputs(out_root: str) -> dict:
    """Fail closed unless every manifest and label cache is the frozen one."""
    manifest_hashes, drifted = {}, []
    for rel, expected in FROZEN_MANIFEST_SHA256.items():
        path = os.path.join(out_root, "manifests", rel)
        if not os.path.exists(path):
            drifted.append(f"{rel}: missing")
            continue
        got = _canonical_sha256(path)
        manifest_hashes[rel] = got
        if got != expected:
            drifted.append(f"{rel}: {got} != {expected}")

    label_hashes, n_rows = {}, {}
    for dataset, expected in FROZEN_LABELS_SHA256.items():
        path = pilot.labels_path(out_root, dataset)
        if not os.path.exists(path):
            drifted.append(f"{dataset}: label cache missing")
            continue
        got = _canonical_sha256(path)
        label_hashes[dataset] = got
        n_rows[dataset] = sum(1 for line in open(path, encoding="utf-8")
                              if line.strip())
        if got != expected:
            drifted.append(f"{dataset}: labels {got} != {expected}")

    if drifted:
        raise StageRefused(
            "frozen v2r1 inputs drifted; refusing to evaluate a gate on "
            "unapproved evidence: " + "; ".join(drifted))
    return {"manifest_sha256": manifest_hashes,
            "labels_sha256": label_hashes, "labels_rows": n_rows}


def _write_json(path: str, payload) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return path


def _coverage(out_root: str, dataset: str, rows) -> dict:
    """Event/cutoff coverage of the labelled pool (descriptive only)."""
    split_path = os.path.join(out_root, "manifests", dataset,
                              "event_split.json")
    split = json.load(open(split_path, encoding="utf-8"))
    labelled_pool = sorted(set(split["utility_train"]) | set(split["utility_dev"])
                           | set(split["utility_eval"]))
    labelled = {row["event_id"] for row in rows}
    return {
        "labelled_pool_events": len(labelled_pool),
        "events_with_labels": len(labelled),
        "events_missing_labels": len(set(labelled_pool) - labelled),
        "cutoffs": sorted({int(row["cutoff"]) for row in rows}),
        "event_cutoff_pairs": len({(row["event_id"], int(row["cutoff"]))
                                   for row in rows}),
        "split_sizes": split.get("sizes"),
        "split_seed": split.get("split_seed"),
        "source_sha256": split.get("source", {}).get("sha256"),
    }


def _atomic_rows(rows):
    return [row for row in rows if row["intervention_type"] == "I1_atomic"]


def evaluate_p1(out_root: str, dataset: str, rows, frozen: dict) -> dict:
    """P1 heterogeneity — frozen report + frozen gate (no reimplementation)."""
    unit_table = pilot.load_unit_table(out_root, dataset)
    readers = sorted({r for entry in unit_table.values() for r in entry})
    retired = [r for r in readers if r in RETIRED_READERS]

    report = heterogeneity_report(unit_table, READER_KEYS)
    gate = gate_p1(report)

    atomic = _atomic_rows(rows)
    active_counts = {}
    for key in READER_KEYS:
        subset = [row for row in atomic if row["reader"] == key]
        active_counts[key] = {
            "atomic_rows": len(subset),
            "active": sum(1 for row in subset
                          if is_active(row["utility"],
                                       row.get("correctness_before"),
                                       row.get("correctness_after"))),
        }
    active_counts["utility_threshold"] = UTILITY_THRESHOLD
    pair_active = {f"{p['reader_a']}+{p['reader_b']}": p["n_active"]
                   for p in report["pairs"]}

    primary = dataset == PRIMARY_DATASET
    ok_readers = sorted(readers) == sorted(READER_KEYS)
    payload = {
        "protocol": "v2r1",
        "stage": "P1_reader_heterogeneity",
        "dataset": dataset,
        "role": "primary_decision" if primary else "diagnostic_only",
        "diagnostic_only": not primary,
        "decides_primary_gate": primary,
        "reader_keys": list(READER_KEYS),
        "readers_present": readers,
        "reader_set_exact": ok_readers,
        "retired_readers_present": retired,
        "n_atomic_evidence_keys": len(unit_table),
        "active_counts": active_counts,
        "jointly_active_pairs": pair_active,
        "coverage": _coverage(out_root, dataset, rows),
        "frozen_inputs": frozen,
        "frozen_statistics": {
            "utility_threshold": UTILITY_THRESHOLD,
            "p1_mean_disagreement_min": P1_MEAN_DISAGREEMENT_MIN,
            "p1_pair_disagreement_min": P1_PAIR_DISAGREEMENT_MIN,
            "p1_pairs_required": P1_PAIRS_REQUIRED,
        },
        "report": report,
        "gate": gate,
    }
    if not ok_readers or retired:
        raise StageRefused(
            f"{dataset}: reader set {readers} is not exactly "
            f"{list(READER_KEYS)} (retired present: {retired})")
    if primary:
        payload["verdict"] = "P1_PASS" if gate["pass"] else "P1_FAIL"
    else:
        payload["verdict"] = "DIAGNOSTIC_ONLY"
        payload["diagnostic_gate"] = gate
    return payload


def _slot_means(records) -> dict:
    """Mean |I| per intervention slot, in the frozen (reader, snapshot) unit.

    Descriptive only: the gate value stays ``Delta_edge`` / ``Delta_subtree``
    from the frozen implementation.
    """
    payloads = build_event_payloads(records)
    out = {}
    for slot in ("pc", "na", "sub", "disc"):
        values = [value for payload in payloads.values()
                  for (_reader, _snap, value) in payload[slot]]
        out[slot] = {"n_pairs": len(values),
                     "mean_abs_interaction": mean(values) if values else
                     float("nan")}
    return out


def evaluate_p2(out_root: str, dataset: str, rows, frozen: dict) -> dict:
    """P2 structured interaction — frozen report + frozen gate."""
    records = pilot.load_interaction_records(out_root, dataset)
    report = structural_interaction_report(records)
    gate = gate_p2(report)

    by_type = Counter(row["intervention_type"] for row in rows)
    reader_counts = Counter(record["reader"] for record in records)
    retired = [r for r in sorted(reader_counts) if r in RETIRED_READERS]
    readers = sorted(reader_counts)
    ok_readers = readers == sorted(READER_KEYS)

    primary = dataset == PRIMARY_DATASET
    payload = {
        "protocol": "v2r1",
        "stage": "P2_structured_interaction",
        "dataset": dataset,
        "role": "primary_decision" if primary else "diagnostic_only",
        "diagnostic_only": not primary,
        "decides_primary_gate": primary,
        "reader_keys": list(READER_KEYS),
        "readers_present": readers,
        "reader_set_exact": ok_readers,
        "retired_readers_present": retired,
        "record_counts": {
            "structured_records": len(records),
            "structured_records_by_reader": dict(reader_counts),
            "cache_rows_by_intervention_type": dict(by_type),
        },
        "slot_means": _slot_means(records),
        "coverage": _coverage(out_root, dataset, rows),
        "frozen_inputs": frozen,
        "frozen_statistics": {
            "utility_threshold": UTILITY_THRESHOLD,
            "p2_edge_delta_min": P2_EDGE_DELTA_MIN,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "matching_unit": report.get("matching_unit"),
            "bootstrap_unit": report.get("bootstrap_unit"),
            "cutoffs": list(CUTOFFS_MIN),
        },
        "report": report,
        "gate": gate,
    }
    if not ok_readers or retired:
        raise StageRefused(
            f"{dataset}: structured records cover readers {readers}, expected "
            f"{list(READER_KEYS)} (retired present: {retired})")
    if primary:
        payload["verdict"] = "P2_PASS" if gate["pass"] else "P2_FAIL"
    else:
        payload["verdict"] = "DIAGNOSTIC_ONLY"
        payload["diagnostic_gate"] = gate
    return payload


def evaluate(out_root: str = V2R1) -> dict:
    frozen = _assert_frozen_inputs(out_root)
    gates_dir = os.path.join(out_root, "gates")
    out = {"frozen_inputs": frozen, "datasets": {}, "gates_dir": gates_dir}
    written = {}
    for dataset in (PRIMARY_DATASET, SECONDARY_DATASET):
        rows = pilot._read_jsonl(pilot.labels_path(out_root, dataset))
        if not rows:
            raise StageRefused(f"{dataset}: no utility labels under {out_root}")
        p1 = evaluate_p1(out_root, dataset, rows, frozen)
        p2 = evaluate_p2(out_root, dataset, rows, frozen)
        suffix = "" if dataset == PRIMARY_DATASET else "_diagnostic"
        written[f"p1_{dataset}{suffix}"] = _write_json(
            os.path.join(gates_dir, f"p1_{dataset}{suffix}.json"), p1)
        written[f"p2_{dataset}{suffix}"] = _write_json(
            os.path.join(gates_dir, f"p2_{dataset}{suffix}.json"), p2)
        out["datasets"][dataset] = {"P1": p1, "P2": p2}
        print(f"[{dataset}] P1 {p1['verdict']} "
              f"mean_disagreement={p1['report']['macro_mean_disagreement']:.4f}"
              f" pairs={[(p['reader_a'], p['reader_b'], round(p['disagreement'], 4)) for p in p1['report']['pairs']]}")
        print(f"[{dataset}] P2 {p2['verdict']} "
              f"delta_edge={p2['report']['edge']['delta']} "
              f"ci=({p2['report']['edge'].get('ci_low')}, "
              f"{p2['report']['edge'].get('ci_high')}) "
              f"delta_subtree={p2['report']['subtree']['delta']}")
    out["written"] = written
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_root = args.out_root or os.environ.get("CRTSER_OUT_ROOT", V2R1)
    try:
        result = evaluate(out_root)
    except StageRefused as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}, indent=1))
        return 3
    print(json.dumps({
        "protocol": "v2r1",
        "P1_maweibo": result["datasets"][PRIMARY_DATASET]["P1"]["verdict"],
        "P2_maweibo": result["datasets"][PRIMARY_DATASET]["P2"]["verdict"],
        "pheme": "diagnostic_only",
        "files": sorted(result["written"].values()),
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
