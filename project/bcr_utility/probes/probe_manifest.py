"""Frozen behavioral probe manifest (plan §6.1, §11.7).

The probe battery is what a *new* reader can be measured on without any target
utility label. It is therefore built once, from ``foundation_train`` only, and
frozen before any reader is called:

* 36 events per dataset, 12 assigned to each of the 15m / 1h / 6h cutoffs;
* an event assigned to a cutoff must have at least one valid visible evidence
  unit *and* at least one SRC-selected unit there, otherwise the P1–P3
  contexts cannot be constructed;
* the three cutoff groups are event-disjoint, so one event never contributes
  two probe items;
* gold labels are used **only** to balance the manifest; the manifest records
  the balance as an aggregate and deliberately stores no per-item label, so a
  gold label cannot leak into the fingerprint feature vector (plan §19).

Selection is deterministic: the two label classes are shuffled independently
with ``random.Random(seed)`` and dealt cutoff by cutoff.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

from ..config import protocol as P
from ..data.historical_import import sha256_bytes


class ProbeManifestRefused(RuntimeError):
    """Raised when the frozen probe design cannot be satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeManifestRefused(message)


def _availability_entry(availability: dict, event_id: str, cutoff) -> dict:
    per_event = availability.get(event_id) or {}
    return per_event.get(str(cutoff)) or per_event.get(int(cutoff)) or {}


def _shuffled_by_label(event_ids, labels, seed: int) -> dict:
    by_label = defaultdict(list)
    for eid in sorted(event_ids):
        by_label[int(labels[eid])].append(eid)
    rng = random.Random(seed)
    for label in sorted(by_label):
        rng.shuffle(by_label[label])
    return by_label


def _pick_cutoff_group(candidates_by_label, target: int) -> list:
    """Deal ``target`` events, preferring an even split across the classes."""
    half = target // 2
    picked = []
    for label in sorted(candidates_by_label):
        picked.extend(candidates_by_label[label][:half])
    remaining = target - len(picked)
    if remaining > 0:
        leftovers = []
        for label in sorted(candidates_by_label):
            leftovers.extend(candidates_by_label[label][half:])
        picked.extend(leftovers[:remaining])
    return picked


def foundation_labels(split: dict, dataset: str, labels_by_event: dict):
    """Gold labels of the foundation events — for sampling balance only."""
    foundation = [str(e) for e in split.get(P.PROBE_SOURCE_SPLIT, [])]
    missing = [e for e in foundation if e not in labels_by_event]
    _require(not missing,
             f"{dataset}: gold labels missing for {len(missing)} foundation "
             f"events, e.g. {missing[:3]}")
    return {e: int(labels_by_event[e]) for e in foundation}


def build_probe_manifest(split: dict, dataset: str, availability: dict,
                         labels_by_event: dict,
                         seed: int = P.PROBE_SEED) -> dict:
    """Freeze the 36-item probe manifest of one dataset.

    ``availability`` maps ``event_id -> {cutoff: {"n_units": int,
    "src_selected": int, "n_visible_nodes": int}}`` for every
    ``foundation_train`` event, measured on the raw data.
    """
    _require(dataset in P.DATASETS, f"unknown dataset {dataset!r}")
    foundation = [str(e) for e in split.get(P.PROBE_SOURCE_SPLIT, [])]
    _require(bool(foundation), f"{dataset}: {P.PROBE_SOURCE_SPLIT} is empty")
    labels = foundation_labels(split, dataset, labels_by_event)
    missing = [e for e in foundation if e not in availability]
    _require(not missing,
             f"{dataset}: availability missing for {len(missing)} foundation "
             f"events, e.g. {missing[:3]}")

    used = set()
    items = []
    per_cutoff = {}
    balance = {}
    candidate_pool = {}
    for cutoff in P.CUTOFFS_MIN:
        eligible = [
            eid for eid in foundation
            if int(_availability_entry(availability, eid, cutoff)
                   .get("n_units") or 0) >= P.PROBE_MIN_VISIBLE_UNITS
            and int(_availability_entry(availability, eid, cutoff)
                    .get("src_selected") or 0) >= P.PROBE_MIN_SRC_SELECTED]
        candidate_pool[str(cutoff)] = len(eligible)
        pool = [e for e in eligible if e not in used]
        _require(len(pool) >= P.PROBE_EVENTS_PER_CUTOFF,
                 f"{dataset}: cutoff {cutoff}m has only {len(pool)} eligible "
                 f"unused events, need {P.PROBE_EVENTS_PER_CUTOFF}")
        picked = _pick_cutoff_group(
            _shuffled_by_label(pool, labels, seed + int(cutoff)),
            P.PROBE_EVENTS_PER_CUTOFF)
        _require(len(picked) == P.PROBE_EVENTS_PER_CUTOFF,
                 f"{dataset}: cutoff {cutoff}m could only place "
                 f"{len(picked)} events")
        counts = defaultdict(int)
        for eid in picked:
            used.add(eid)
            counts[labels[eid]] += 1
            entry = _availability_entry(availability, eid, cutoff)
            items.append({
                "event_id": str(eid),
                "cutoff": int(cutoff),
                "n_visible_units": int(entry.get("n_units") or 0),
                "src_selected": int(entry.get("src_selected") or 0),
                "n_visible_nodes": int(entry.get("n_visible_nodes") or 0),
            })
        per_cutoff[str(cutoff)] = len(picked)
        balance[str(cutoff)] = {str(label): counts.get(label, 0)
                                for label in (0, 1)}

    _require(len(items) == P.PROBE_EVENTS_PER_DATASET,
             f"{dataset}: {len(items)} probe items, expected "
             f"{P.PROBE_EVENTS_PER_DATASET}")
    _require(len({i["event_id"] for i in items}) == len(items),
             f"{dataset}: probe items are not event-disjoint")
    items.sort(key=lambda i: (i["cutoff"], i["event_id"]))
    all_counts = defaultdict(int)
    for item in items:
        all_counts[labels[item["event_id"]]] += 1
    return {
        "dataset": dataset,
        "n_items": len(items),
        "n_events": len({i["event_id"] for i in items}),
        "per_cutoff_counts": per_cutoff,
        "label_balance": balance,
        "total_label_balance": {str(label): all_counts.get(label, 0)
                                for label in (0, 1)},
        "candidate_pool": candidate_pool,
        "items": items,
    }


def audit_overlap(manifest: dict, split: dict) -> dict:
    """Probe events must not intersect utility_train/dev/eval (plan §11.7)."""
    probe_events = {i["event_id"] for i in manifest["items"]}
    return {name: len(probe_events & {str(e) for e in split.get(name, [])})
            for name in P.NON_PROBE_SPLITS}


def validate_probe_manifest(manifest: dict, dataset: str = None) -> None:
    """Structural validation of a frozen manifest (used by the verifier)."""
    dataset = dataset or manifest.get("dataset")
    _require(dataset in P.DATASETS, f"unknown dataset {dataset!r}")
    items = manifest.get("items") or []
    _require(len(items) == P.PROBE_EVENTS_PER_DATASET,
             f"{dataset}: {len(items)} probe items != "
             f"{P.PROBE_EVENTS_PER_DATASET}")
    _require(len({i["event_id"] for i in items}) == len(items),
             f"{dataset}: probe events are not distinct")
    counts = defaultdict(int)
    for item in items:
        _require(not ({"label", "gold", "gold_label"} & set(item)),
                 f"{dataset}: probe item carries a gold label")
        counts[int(item["cutoff"])] += 1
        _require(int(item["n_visible_units"]) >= P.PROBE_MIN_VISIBLE_UNITS,
                 f"{dataset}: probe item has no visible evidence unit")
        _require(int(item["src_selected"]) >= P.PROBE_MIN_SRC_SELECTED,
                 f"{dataset}: probe item has no SRC-selected unit")
    for cutoff in P.CUTOFFS_MIN:
        _require(counts[int(cutoff)] == P.PROBE_EVENTS_PER_CUTOFF,
                 f"{dataset}: cutoff {cutoff}m has {counts[int(cutoff)]} items "
                 f"!= {P.PROBE_EVENTS_PER_CUTOFF}")


def availability_digest(availability: dict) -> str:
    """Canonical digest of a raw availability payload (provenance)."""
    payload = json.dumps(availability, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))
