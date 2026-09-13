#!/usr/bin/env python
"""CR-TSER P0 — data + reader readiness (plan §30).

Runs only what Checkpoint 1 allows:

* Weibo22 raw-field audit (``plan §30`` field list),
* Weibo22 normalized-adapter smoke,
* three-reader load / hash audit,
* A/B sequence-score sanity (≥20 fixed examples per model, scored twice with
  100% prediction identity required).

Writes ``results/cr_tser/p0/`` exactly as the plan specifies and never
fabricates a timestamp, a text field or a reader score: missing inputs produce
an explicit readiness verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
PROJECT = REPO / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from cr_tser.config.pilot_config import (CUTOFFS_MIN, READER_KEYS,  # noqa: E402
                                         READER_MODEL_IDS,
                                         V2_MIN_VIABLE_EVENTS,
                                         V2_RESULTS_ROOT, paths_from_env)
from cr_tser.data import weibo22_adapter
from cr_tser.readers.base_reader import (ReaderSpec, build_reader,
                                         dump_reader_audit, reader_identity)
from cr_tser.readers.sequence_scorer import ab_scores


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return path


def weibo22_temporal_check(paths, raw_dir=None):
    """Temporal validation of the frozen Weibo22 source of record.

    The normalized export (``CRTSER_WEIBO22_NORMALIZED``) is validated field by
    field when present; otherwise the raw KPG release is only audited for
    structure and can never yield READY (plan §4.1, §30).
    """
    normalized = getattr(paths, "weibo22_normalized", "")
    if normalized and os.path.exists(normalized):
        report = weibo22_adapter.validate_normalized_export(normalized)
        report["source_of_record"] = "normalized_export"
        return report
    raw = raw_dir or getattr(paths, "weibo22_raw", "")
    if raw and os.path.isdir(raw):
        report = weibo22_adapter.audit_release(raw)
        report["source_of_record"] = "raw_release"
        return report
    return {"dataset": "Weibo22", "verdict": weibo22_adapter.
            VERDICT_UNAVAILABLE, "source_of_record": "none",
            "verdict_reason": "no Weibo22 temporal source configured"}


def evaluate_readiness(audit, readers, sanity, readers_checked,
                       sanity_requested):
    """P0 decision. Fail closed on every prerequisite (plan §25, §30).

    A/B sanity passes only when each reader produced *identical predictions*
    on the repeated scoring **and** its continuation tokenization/boundary
    validation succeeded; a recorded-but-failing boundary is a P0 failure, not
    a warning.
    """
    readers_ready = all(
        r.get("model_path_exists") and (r.get("loaded") or not readers_checked)
        for r in readers.values())
    if sanity_requested:
        sanity_ok = bool(sanity) and all(
            r.get("identical_predictions") and r.get("boundaries_ok")
            for r in sanity.values())
    else:
        sanity_ok = False
    data_ready = audit.get("verdict") == weibo22_adapter.VERDICT_READY
    p0_pass = bool(data_ready and readers_ready and sanity_ok)
    return {
        "P0": "P0_PASS" if p0_pass else "P0_FAIL",
        "weibo22_temporal": audit.get("verdict"),
        "weibo22_source_of_record": audit.get("source_of_record"),
        "weibo22_reason": audit.get("verdict_reason") or (
            "" if data_ready else f"field coverage: {audit.get('errors')}"),
        "readers_ready": readers_ready,
        "readers_checked": readers_checked,
        "ab_sanity_ok": sanity_ok if sanity_requested else None,
        "ab_boundaries_ok": bool(sanity) and all(
            r.get("boundaries_ok") for r in sanity.values())
        if sanity_requested else None,
        "next_step": ("COMMIT PUSH STOP — research approval required before "
                      "expensive labels (plan §30)"
                      if p0_pass else
                      "STOP: a P0 prerequisite is unmet (plan §25); do not "
                      "generate interventions"),
    }


def weibo22_smoke(raw_dir: str):  # pragma: no cover - needs real data
    """Normalized-adapter smoke: load the export if one exists."""
    try:
        events = weibo22_adapter.load_events(raw_dir)
    except weibo22_adapter.Weibo22TemporalUnavailable as exc:
        return {"status": "UNAVAILABLE", "reason": str(exc),
                "audit_verdict": exc.audit.get("verdict")}
    return {"status": "OK", "n_events": len(events),
            "first_ids": [e["event_id"] for e in events[:5]]}


def reader_audit(paths, out_dir, dtype="bfloat16", device="cuda",
                 load: bool = False):
    """Hash audit for the three frozen readers (plan §8, §30)."""
    specs = {key: ReaderSpec(key, paths.reader_path(key), dtype=dtype,
                             device=device) for key in READER_KEYS}
    audit = dump_reader_audit(specs, out_dir)
    for key, spec in specs.items():
        entry = audit[key]
        entry["model_path_exists"] = bool(spec.model_path and
                                          os.path.isdir(spec.model_path))
        entry["expected_model_id"] = READER_MODEL_IDS[key]
        entry["loaded"] = False
        if load and entry["model_path_exists"]:
            try:
                reader = build_reader(key, spec)
                entry["loaded"] = True
                entry["identity_after_load"] = reader.identity()
                reader.unload()
            except Exception as exc:  # pragma: no cover - env dependent
                entry["load_error"] = f"{type(exc).__name__}: {exc}"
    _write_json(os.path.join(out_dir, "reader_audit.json"), audit)
    return audit


def label_scoring_sanity(paths, n_examples: int = 20, device="cuda"):
    """Score fixed prompts twice per reader; prediction identity must be 100%.

    The artifact records the **explicit A/B tokenization** (prompt token count,
    each candidate's token ids, and the continuation boundary check) so it is
    auditable that scoring is teacher-forced candidate scoring and not
    generated text or generated confidence (plan §9, §30).
    """
    prompts = [f"Sanity example {i}: source post number {i}." for i in range(n_examples)]
    report = {}
    for key in READER_KEYS:
        spec = ReaderSpec(key, paths.reader_path(key), device=device)
        entry = {"model_id": READER_MODEL_IDS[key], "n_examples": n_examples,
                 "identical_predictions": False, "loaded": False,
                 "scoring_mode": "teacher_forced_logprob_sum",
                 "generated_text_used": False,
                 "generated_confidence_used": False}
        if not spec.model_path or not os.path.isdir(spec.model_path):
            entry["status"] = "MODEL_PATH_MISSING"
            report[key] = entry
            continue
        try:
            reader = build_reader(key, spec)
            entry["loaded"] = True
            entry["identity"] = reader.identity()
            first, second = [], []
            tokenization = None
            for i, prompt in enumerate(prompts):
                out1 = reader.score_ab(prompt)
                out2 = reader.score_ab(prompt)
                first.append(out1["prediction"])
                second.append(out2["prediction"])
                if i == 0:
                    tokenization = reader.ab_token_report(prompt)
            entry["tokenization_example"] = tokenization
            entry["boundaries_ok"] = bool(
                tokenization and tokenization.get("all_boundaries_ok"))
            entry["identical_predictions"] = first == second
            entry["identity_rate"] = (sum(1 for a, b in zip(first, second)
                                          if a == b) / len(prompts))
            entry["label_distribution"] = {
                "A": first.count("A"), "B": first.count("B")}
            reader.unload()
        except Exception as exc:  # pragma: no cover - env dependent
            entry["status"] = f"ERROR: {type(exc).__name__}: {exc}"
        report[key] = entry
    return report


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=("v1", "v2"), default="v2",
                    help="v2 (default) audits Ma-Weibo + PHEME; v1 keeps the "
                         "historical Weibo22 preflight reachable")
    ap.add_argument("--weibo22-raw", default=None)
    ap.add_argument("--maweibo-raw", default=None)
    ap.add_argument("--maweibo-labels", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--readers", action="store_true",
                    help="also load+hash the three readers (needs GPU)")
    ap.add_argument("--sanity", action="store_true",
                    help="also run the A/B sequence-score sanity check")
    ap.add_argument("--n-sanity", type=int, default=20)
    return ap


# --------------------------------------------------------------------------
# Amendment V2 — P0 = Ma-Weibo integrity + PHEME smoke + reader readiness
# --------------------------------------------------------------------------
def maweibo_integrity_ok(audit: dict) -> bool:
    """Structural integrity required before Ma-Weibo can be the primary source.

    Coverage ratios must be non-zero and the structural defects the amendment
    lists (duplicate ids, cycles, multi-root events) must be absent.
    """
    return bool(
        audit.get("raw_event_count", 0) > 0
        and audit.get("duplicate_ids") == 0
        and audit.get("cycle_count") == 0
        and audit.get("multi_root_event_count") == 0
        and audit.get("source_text_coverage", 0) > 0
        and audit.get("source_timestamp_coverage", 0) > 0
        and audit.get("timestamp_coverage", 0) > 0)


def maweibo_integrity_check(paths, raw=None, labels=None) -> dict:
    """P0-A: Ma-Weibo composite source audit (amendment §9)."""
    raw = raw or getattr(paths, "maweibo_raw", "")
    labels = labels or getattr(paths, "maweibo_labels", "")
    if not raw or not labels or not os.path.isdir(raw) \
            or not os.path.exists(labels):
        return {
            "dataset": "maweibo",
            "source_of_record": "maweibo_composite",
            "raw_dir": raw, "label_file": labels,
            "verdict": "MAWEIBO_SOURCE_MISSING",
            "verdict_reason": "CRTSER_MAWEIBO_RAW / CRTSER_MAWEIBO_LABELS "
                              "are unset or missing on this machine",
            "total_viable_events": 0,
        }
    from cr_tser.data import maweibo_bridge
    audit = maweibo_bridge.audit_maweibo(raw, labels)
    audit["source_of_record"] = "maweibo_composite"
    audit["verdict"] = ("MAWEIBO_READY" if maweibo_integrity_ok(audit)
                        else "MAWEIBO_INTEGRITY_FAIL")
    return audit


def pheme_smoke(paths) -> dict:
    """P0-B: PHEME entry smoke — load, text, timestamps, parents, snapshots."""
    raw = getattr(paths, "pheme_raw", "")
    out = {"dataset": "pheme", "raw_dir": raw, "labels_generated": False}
    if not raw or not os.path.isdir(raw):
        out.update({"status": "PHEME_RAW_MISSING",
                    "reason": "CRTSER_PHEME_RAW is unset or not a directory"})
        return out
    try:
        from cr_tser.data.snapshot_bridge import build_causal_snapshot
        from tcdscr.data import pheme_adapter

        ids = pheme_adapter.event_ids(raw)
        out["n_events"] = len(ids)
        if not ids:
            out["status"] = "PHEME_RAW_EMPTY"
            return out
        _eid, topic, label, folder = ids[0]
        event = pheme_adapter.load_event(topic, label, folder)
        out["sample_event_id"] = _eid
        out["source_text_available"] = bool(
            str(event["nodes"][0].get("text") or "").strip())
        out["reply_text_available"] = any(
            str(n.get("text") or "").strip() for n in event["nodes"][1:])
        out["timestamps_available"] = all(
            n.get("timestamp") is not None for n in event["nodes"])
        out["parent_relation_available"] = all(
            n.get("parent_id") is not None for n in event["nodes"][1:])
        built, leakage = [], []
        t0 = event["source_timestamp"]
        for cutoff in CUTOFFS_MIN:
            snap = build_causal_snapshot(event, cutoff)
            built.append({"cutoff_minutes": cutoff,
                          "n_nodes": len(snap["node_ids"])})
            limit = t0 + int(cutoff) * 60
            late = [nid for nid, ts in zip(snap["node_ids"],
                                           snap["timestamps"]) if ts > limit]
            if late:
                leakage.append({"cutoff_minutes": cutoff, "nodes": late[:5]})
        out["snapshots_built"] = built
        out["future_leakage"] = leakage
        out["status"] = "OK" if not leakage else "FUTURE_LEAKAGE"
    except Exception as exc:  # pragma: no cover - environment dependent
        out["status"] = f"ERROR: {type(exc).__name__}: {exc}"
    return out


def evaluate_readiness_v2(audit, pheme, readers, sanity, readers_checked,
                          sanity_requested):
    """Amendment §10: every prerequisite must hold for P0 PASS."""
    integrity_ok = maweibo_integrity_ok(audit)
    viable = int(audit.get("total_viable_events", 0) or 0)
    viable_ok = viable >= V2_MIN_VIABLE_EVENTS
    pheme_ok = pheme.get("status") == "OK"
    readers_ready = all(
        r.get("model_path_exists") and (r.get("loaded") or not readers_checked)
        for r in readers.values())
    if sanity_requested:
        sanity_ok = bool(sanity) and all(
            r.get("identical_predictions") and r.get("boundaries_ok")
            for r in sanity.values())
    else:
        sanity_ok = False
    p0_pass = bool(integrity_ok and viable_ok and pheme_ok and readers_ready
                   and sanity_ok)
    return {
        "protocol": "v2",
        "P0": "P0_PASS" if p0_pass else "P0_FAIL",
        "primary_dataset": "maweibo",
        "secondary_dataset": "pheme",
        "weibo22_status": "REJECTED_PRIMARY_CANDIDATE",
        "maweibo_source_integrity": integrity_ok,
        "maweibo_source_verdict": audit.get("verdict"),
        "maweibo_viable_events": viable,
        "maweibo_viable_required": V2_MIN_VIABLE_EVENTS,
        "maweibo_viable_ok": viable_ok,
        "pheme_smoke": pheme.get("status"),
        "pheme_smoke_ok": pheme_ok,
        "readers_ready": readers_ready,
        "readers_checked": readers_checked,
        "ab_sanity_ok": sanity_ok if sanity_requested else None,
        "ab_boundaries_ok": bool(sanity) and all(
            r.get("boundaries_ok") for r in sanity.values())
        if sanity_requested else None,
        "next_step": (
            "COMMIT PUSH STOP — research approval required before expensive "
            "labels (amendment V2 §26)"
            if p0_pass else
            "STOP: a V2 P0 prerequisite is unmet (amendment V2 §10)"),
    }


def _v2_out_root(args, paths):
    base = args.out_root or getattr(paths, "out_root", "") or \
        str(REPO / V2_RESULTS_ROOT)
    if not os.path.isabs(base):
        base = str(REPO / base)
    return base if args.out_root else os.path.join(base, "p0")


def main_v2(args, paths):
    out_root = _v2_out_root(args, paths)
    os.makedirs(out_root, exist_ok=True)

    audit = maweibo_integrity_check(paths, args.maweibo_raw,
                                    args.maweibo_labels)
    _write_json(os.path.join(out_root, "maweibo_audit.json"), audit)
    pheme = pheme_smoke(paths)
    _write_json(os.path.join(out_root, "pheme_smoke.json"), pheme)

    readers = reader_audit(paths, out_root, dtype=args.dtype,
                           device=args.device, load=args.readers)
    sanity = {}
    if args.sanity:
        sanity = label_scoring_sanity(paths, n_examples=args.n_sanity,
                                      device=args.device)
    _write_json(os.path.join(out_root, "label_scoring_sanity.json"), sanity)

    readiness = evaluate_readiness_v2(audit, pheme, readers, sanity,
                                      args.readers, args.sanity)
    _write_json(os.path.join(out_root, "p0_readiness.json"), readiness)
    lines = ["# CR-TSER V2 P0 readiness (amendment V2 §9–§10)", "",
             f"- **verdict**: {readiness['P0']}",
             f"- primary dataset: `maweibo` "
             f"(viable events {readiness['maweibo_viable_events']} / "
             f"{readiness['maweibo_viable_required']})",
             f"- Ma-Weibo source integrity: "
             f"`{readiness['maweibo_source_integrity']}` "
             f"({readiness['maweibo_source_verdict']})",
             f"- PHEME smoke: `{readiness['pheme_smoke']}`",
             f"- readers ready: `{readiness['readers_ready']}` "
             f"(loaded={args.readers})",
             f"- A/B sanity: `{readiness['ab_sanity_ok']}`",
             f"- Weibo22: `{readiness['weibo22_status']}` (V1 evidence only)",
             f"- next: {readiness['next_step']}"]
    if audit.get("verdict_reason"):
        lines += ["", f"> {audit['verdict_reason']}"]
    with open(os.path.join(out_root, "P0_READINESS.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps(readiness, indent=1, ensure_ascii=False))
    return 0 if readiness["P0"] == "P0_PASS" else 2


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = paths_from_env()
    if args.protocol == "v2":
        return main_v2(args, paths)
    return main_v1(args, paths)


def main_v1(args):
    """Historical V1 preflight (Weibo22). Kept reachable via ``--protocol v1``.

    It always writes into the V1 namespace so the frozen V1 evidence can never
    be mixed with a V2 run.
    """
    from cr_tser.config.pilot_config import V1_RESULTS_ROOT
    paths = paths_from_env()
    raw = args.weibo22_raw or paths.weibo22_raw
    out_root = args.out_root or os.path.join(str(REPO / V1_RESULTS_ROOT), "p0")
    os.makedirs(out_root, exist_ok=True)

    audit = weibo22_temporal_check(paths, raw)
    _write_json(os.path.join(out_root, "weibo22_audit.json"), audit)
    if audit.get("source_of_record") == "raw_release":
        smoke = weibo22_smoke(raw)
    else:
        smoke = {"status": "OK" if audit.get("valid") else "UNAVAILABLE",
                 "source_of_record": audit.get("source_of_record"),
                 "n_events": audit.get("n_events"),
                 "coverage": audit.get("coverage")}
    _write_json(os.path.join(out_root, "weibo22_smoke.json"), smoke)

    readers = reader_audit(paths, out_root, dtype=args.dtype,
                           device=args.device, load=args.readers)
    sanity = {}
    if args.sanity:
        sanity = label_scoring_sanity(paths, n_examples=args.n_sanity,
                                      device=args.device)
    _write_json(os.path.join(out_root, "label_scoring_sanity.json"), sanity)

    readiness = evaluate_readiness(audit, readers, sanity, args.readers,
                                   args.sanity)
    p0_pass = readiness["P0"] == "P0_PASS"
    _write_json(os.path.join(out_root, "p0_readiness.json"), readiness)
    lines = ["# CR-TSER P0 readiness", "",
             f"- **verdict**: {readiness['P0']}",
             f"- Weibo22 temporal: `{readiness['weibo22_temporal']}`",
             f"- readers ready: `{readiness['readers_ready']}` "
             f"(loaded={args.readers})",
             f"- A/B sanity: `{readiness['ab_sanity_ok']}`",
             f"- next: {readiness['next_step']}"]
    if audit.get("verdict_reason"):
        lines += ["", f"> {audit['verdict_reason']}"]
    with open(os.path.join(out_root, "P0_READINESS.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps(readiness, indent=1, ensure_ascii=False))
    return 0 if p0_pass else 2


if __name__ == "__main__":
    sys.exit(main())
