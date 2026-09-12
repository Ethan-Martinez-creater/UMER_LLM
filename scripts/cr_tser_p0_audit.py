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

from cr_tser.config.pilot_config import (READER_KEYS, READER_MODEL_IDS,
                                         VERDICT_READY, paths_from_env)
from cr_tser.data import weibo22_adapter
from cr_tser.readers.base_reader import (ReaderSpec, build_reader,
                                         dump_reader_audit, reader_identity)
from cr_tser.readers.sequence_scorer import ab_scores


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return path


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
    """Score fixed prompts twice per reader; prediction identity must be 100%."""
    from cr_tser.readers.base_reader import build_messages
    from cr_tser.readers.sequence_scorer import apply_chat
    prompts = [f"Sanity example {i}: source post number {i}." for i in range(n_examples)]
    report = {}
    for key in READER_KEYS:
        spec = ReaderSpec(key, paths.reader_path(key), device=device)
        entry = {"model_id": READER_MODEL_IDS[key], "n_examples": n_examples,
                 "identical_predictions": False, "loaded": False}
        if not spec.model_path or not os.path.isdir(spec.model_path):
            entry["status"] = "MODEL_PATH_MISSING"
            report[key] = entry
            continue
        try:
            reader = build_reader(key, spec)
            entry["loaded"] = True
            first, second = [], []
            for prompt in prompts:
                first.append(reader.score_ab(prompt)["prediction"])
                second.append(reader.score_ab(prompt)["prediction"])
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
    ap.add_argument("--weibo22-raw", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--readers", action="store_true",
                    help="also load+hash the three readers (needs GPU)")
    ap.add_argument("--sanity", action="store_true",
                    help="also run the A/B sequence-score sanity check")
    ap.add_argument("--n-sanity", type=int, default=20)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = paths_from_env()
    raw = args.weibo22_raw or paths.weibo22_raw
    out_root = args.out_root or os.path.join(paths.out_root or
                                             str(REPO / "results" / "cr_tser"),
                                             "p0")
    os.makedirs(out_root, exist_ok=True)

    if raw and os.path.isdir(raw):
        audit = weibo22_adapter.write_audit(raw, out_root)
        smoke = weibo22_smoke(raw)
    else:
        audit = {"dataset": "Weibo22", "verdict": weibo22_adapter.
                 VERDICT_UNAVAILABLE,
                 "verdict_reason": f"raw dir not found: {raw!r}"}
        _write_json(os.path.join(out_root, "weibo22_audit.json"), audit)
        smoke = {"status": "UNAVAILABLE", "reason": "raw dir not found"}
    _write_json(os.path.join(out_root, "weibo22_smoke.json"), smoke)

    readers = reader_audit(paths, out_root, dtype=args.dtype,
                           device=args.device, load=args.readers)
    sanity = {}
    if args.sanity:
        sanity = label_scoring_sanity(paths, n_examples=args.n_sanity,
                                      device=args.device)
    _write_json(os.path.join(out_root, "label_scoring_sanity.json"), sanity)

    readers_ready = all(r.get("model_path_exists") and
                        (r.get("loaded") or not args.readers)
                        for r in readers.values())
    sanity_ok = bool(sanity) and all(r.get("identical_predictions")
                                     for r in sanity.values())
    data_ready = audit.get("verdict") == VERDICT_READY
    p0_pass = bool(data_ready and readers_ready and sanity_ok)
    readiness = {
        "P0": "P0_PASS" if p0_pass else "P0_FAIL",
        "weibo22_temporal": audit.get("verdict"),
        "weibo22_reason": audit.get("verdict_reason"),
        "readers_ready": readers_ready,
        "readers_checked": args.readers,
        "ab_sanity_ok": sanity_ok if args.sanity else None,
        "next_step": ("COMMIT PUSH STOP — research approval required before "
                      "expensive labels (plan §30)"
                      if p0_pass else
                      "STOP: a P0 prerequisite is unmet (plan §25); do not "
                      "generate interventions"),
    }
    _write_json(os.path.join(out_root, "p0_readiness.json"), readiness)
    lines = ["# CR-TSER P0 readiness", "",
             f"- **verdict**: {readiness['P0']}",
             f"- Weibo22 temporal: `{readiness['weibo22_temporal']}`",
             f"- readers ready: `{readers_ready}` (loaded={args.readers})",
             f"- A/B sanity: `{sanity_ok if args.sanity else 'not run'}`",
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
