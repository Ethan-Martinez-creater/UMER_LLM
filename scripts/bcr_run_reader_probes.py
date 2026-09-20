#!/usr/bin/env python
"""M1-A — execute the frozen behavioral probes (M1 plan §7).

Reads the frozen M0 probe manifest (never rebuilds it), constructs the four
P0/P1/P2/P3 contexts through the M0 context builder and scores each one with
the existing teacher-forced A/B scorer. One shard per (dataset, reader) is
written under ``results/bcr_utility_v1/m1/probe_responses/``; rerunning a
shard resumes after the rows already present (with prompt-hash verification,
never a silent reuse).

Probe responses contain no utility labels, no gold labels and no reader-ID
embeddings (M1 plan §7).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "project"))
sys.path.insert(0, str(REPO / "scripts"))

import cr_tser_common as common  # noqa: E402

from bcr_utility.config import protocol as P  # noqa: E402
from bcr_utility.probes import probe_contexts  # noqa: E402
from bcr_utility.probes.fingerprint import response_metrics  # noqa: E402
from cr_tser.readers.base_reader import (ReaderSpec, build_messages,  # noqa: E402
                                         build_reader,
                                         build_reader_prompt)
from cr_tser.readers.sequence_scorer import (ab_scores, apply_chat,  # noqa: E402
                                             tokenize_prompt)

SHARD_TEMPLATE = "probe_responses_{dataset}_{reader}.jsonl"


class ProbeRunRefused(RuntimeError):
    """Raised when a probe execution contract is violated."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _source_text(event):
    return next(n["text"] for n in event["nodes"]
                if n["node_id"] == event["source_id"])


def _load_manifest(repo_root):
    path = os.path.join(P.bootstrap_dir(repo_root), P.PROBE_MANIFEST_FILENAME)
    if not os.path.exists(path):
        raise ProbeRunRefused(f"frozen probe manifest missing: {path}")
    with open(path, "rb") as fh:
        raw = fh.read()
    return json.loads(P.canonical_bytes(raw).decode("utf-8")), \
        hashlib.sha256(raw).hexdigest()


def _load_existing(path):
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    rows[(row["event_id"], row["cutoff"], row["context"])] = row
    return rows


def _prompt_identity(tokenizer, user_prompt):
    chat = apply_chat(tokenizer, build_messages(user_prompt))
    ids = tokenize_prompt(tokenizer, chat)
    return {"prompt_hash": _sha(chat),
            "prompt_ids_hash": _sha(",".join(str(i) for i in ids)),
            "prompt_tokens": len(ids)}


def run(dataset: str, reader_key: str, paths, repo_root, mock=False,
        limit=None):
    manifest, manifest_sha = _load_manifest(repo_root)
    items = (manifest.get("datasets") or {}).get(dataset, {}).get("items")
    if not items:
        raise ProbeRunRefused(f"{dataset}: no items in the frozen manifest")
    wanted = {str(i["event_id"]) for i in items}
    events = {str(e["event_id"]): e
              for e in common.load_dataset_events(dataset, paths)
              if str(e["event_id"]) in wanted}
    missing = sorted(wanted - set(events))
    if missing:
        raise ProbeRunRefused(f"{dataset}: {len(missing)} probe events not "
                              f"loadable: {missing[:3]}")
    encoder = common.CrSemanticEncoder(paths.semantic_model, dataset)
    canonical = common.canonical_tokenizer(paths.canonical_tokenizer)
    spec = ReaderSpec(reader_key, paths.reader_path(reader_key))
    reader = build_reader(reader_key, spec, mock=mock)
    identity = reader.identity()
    print(f"[m1a] reader {reader_key} identity "
          f"{str(identity.get('reader_identity_hash'))[:16]}...")
    # Mock readers carry no tokenizer; smoke runs identify prompts with the
    # canonical tokenizer instead (scoring itself is mocked anyway).
    prompt_tokenizer = reader.tokenizer if not mock else canonical

    out_dir = P.m1_path(repo_root, "probe_responses" if not mock
                                     else "probe_responses_smoke")
    os.makedirs(out_dir, exist_ok=True)
    shard = os.path.join(out_dir, SHARD_TEMPLATE.format(
        dataset=dataset, reader=reader_key))
    existing = _load_existing(shard)
    if existing:
        print(f"[m1a] resume: {len(existing)} rows already in {shard}")

    n_done = n_skip = 0
    t0 = time.time()
    for pos, item in enumerate(items):
        if limit and pos >= limit:
            break
        event = events[str(item["event_id"])]
        cutoff = int(item["cutoff"])
        art = common.snapshot_artifacts(event, cutoff, encoder, canonical)
        if art["zero_reply"]:
            raise ProbeRunRefused(
                f"{event['event_id']}/{cutoff}: zero-reply snapshot reached "
                "the probe loop; the manifest promised visible units")
        contexts = probe_contexts.build_probe_contexts(art["units"],
                                                       art["src"])
        source_text = _source_text(event)
        for context_name in P.PROBE_CONTEXTS:
            key = (str(event["event_id"]), cutoff, context_name)
            ctx = contexts[context_name]
            prompt = build_reader_prompt(source_text, cutoff, art["units"],
                                         ctx["evidence_block"])
            identity_prompt = _prompt_identity(prompt_tokenizer, prompt)
            canonical_ids = tokenize_prompt(
                canonical, apply_chat(canonical, build_messages(prompt)))
            if key in existing:
                row = existing[key]
                if row.get("prompt_hash") != identity_prompt["prompt_hash"]:
                    raise ProbeRunRefused(
                        f"{key}: cached row prompt hash drift "
                        f"({row.get('prompt_hash')} vs "
                        f"{identity_prompt['prompt_hash']})")
                n_skip += 1
                continue
            out = ab_scores(reader.candidate_logprobs(prompt))
            metrics = response_metrics(out["score_A"], out["score_B"],
                                       out["p_rumor"])
            row = {
                "dataset": dataset,
                "event_id": str(event["event_id"]),
                "cutoff": cutoff,
                "context": context_name,
                "n_units": ctx["n_units"],
                "reader": reader_key,
                "model_id": identity.get("model_id"),
                "reader_identity_hash": identity.get(
                    "reader_identity_hash"),
                "weight_hash": identity.get("weight_hash"),
                "tokenizer_hash": identity.get("tokenizer_hash"),
                "chat_template_hash": identity.get("chat_template_hash"),
                "dtype": identity.get("dtype"),
                "score_A": out["score_A"],
                "score_B": out["score_B"],
                "p_rumor": out["p_rumor"],
                "p_nonrumor": out["p_nonrumor"],
                "prediction": out["prediction"],
                **metrics,
                "reader_prompt_tokens": identity_prompt["prompt_tokens"],
                "canonical_qwen_tokens": len(canonical_ids),
                "prompt_hash": identity_prompt["prompt_hash"],
                "prompt_ids_hash": identity_prompt["prompt_ids_hash"],
            }
            with open(shard, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing[key] = row
            n_done += 1
        if (pos + 1) % 6 == 0:
            print(f"[m1a] {dataset}/{reader_key} {pos + 1}/{len(items)} "
                  f"events, {n_done} scored, {n_skip} resumed, "
                  f"{time.time() - t0:.1f}s")
    reader.unload()
    return {"dataset": dataset, "reader": reader_key, "shard": shard,
            "rows_written": n_done, "rows_resumed": n_skip,
            "rows_total": len(existing),
            "expected_rows": len(items) * len(P.PROBE_CONTEXTS),
            "manifest_sha256": manifest_sha,
            "reader_identity_hash": identity.get("reader_identity_hash"),
            "seconds": time.time() - t0}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=P.DATASETS, required=True)
    ap.add_argument("--reader", choices=P.READER_KEYS, required=True)
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--smoke", action="store_true",
                    help="mock reader; smoke namespace only")
    ap.add_argument("--limit", type=int, default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    result = run(args.dataset, args.reader, paths, args.repo_root,
                 mock=args.smoke, limit=args.limit)
    print(json.dumps(result, indent=1))
    complete = result["rows_total"] == result["expected_rows"]
    print(f"[m1a] shard complete: {complete}")
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
