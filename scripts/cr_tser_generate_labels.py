#!/usr/bin/env python
"""Generate frozen reader utility labels (plan §32).

For every (dataset, event, cutoff, reader, intervention) the script scores the
base context ``C_ref`` and the intervened context ``C_ref \\ A`` with
teacher-forced A/B sequence scoring and stores one immutable cache row with
every §32 field.

Identity rules enforced by the review rounds:

* ``prompt_hash`` covers the complete chat-formatted prompt and
  ``prompt_ids_hash`` covers the **actual tokenized prompt ids**, so a
  tokenizer or chat-template substitution that leaves the text unchanged still
  invalidates the cache;
* ``reader_identity_hash`` folds weight, tokenizer, chat template, dtype and
  model id into one digest;
* a cache hit is reused only when *every* fingerprint matches, otherwise
  :class:`CacheIdentityMismatch` is raised — never a silent reuse.
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

from cr_tser.config.pilot_config import CUTOFFS_MIN, READER_KEYS  # noqa: E402
from cr_tser.intervention.evidence_units import (  # noqa: E402
    render_units_for_budget, structured_group_key)
from cr_tser.models.utility_heads import utility_record  # noqa: E402
from cr_tser.readers.base_reader import (build_messages,  # noqa: E402
                                         build_reader_prompt, ReaderSpec,
                                         build_reader)
from cr_tser.readers.sequence_scorer import (ab_scores, apply_chat,  # noqa: E402
                                             tokenize_prompt)

FINGERPRINT_FIELDS = ("base_context_hash", "intervened_context_hash",
                      "reader_hash", "reader_identity_hash", "tokenizer_hash",
                      "chat_template_hash", "prompt_hash", "prompt_ids_hash")


class CacheIdentityMismatch(RuntimeError):
    """Raised when a cached label does not match the current fingerprints."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _prompt_identity(tokenizer, user_prompt: str) -> dict:
    """Identity of the exact input the reader saw (chat text + token ids)."""
    chat = apply_chat(tokenizer, build_messages(user_prompt))
    ids = tokenize_prompt(tokenizer, chat)
    return {
        "prompt_hash": _sha(chat),
        "prompt_ids_hash": _sha(",".join(str(i) for i in ids)),
        "prompt_tokens": len(ids),
    }


def _cache_key(dataset, event_id, cutoff, reader, intervention_id):
    return structured_group_key(dataset, event_id, cutoff,
                                f"{reader}@{intervention_id}")


def _load_existing(path):
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    rows[_cache_key(row["dataset"], row["event_id"],
                                    row["cutoff"], row["reader"],
                                    row["intervention_id"])] = row
    return rows


def _verify_cached(row, expected, key):
    """Fail closed when a cache hit disagrees with the current fingerprints."""
    mismatched = {f: (row.get(f), expected[f]) for f in FINGERPRINT_FIELDS
                  if row.get(f) != expected[f]}
    if mismatched:
        raise CacheIdentityMismatch(
            f"cache identity mismatch for {key}: {mismatched}; refusing to "
            "reuse the cached label (plan §32)")


def _context_units(units, src, remove_ids):
    remove = set(remove_ids)
    keep = {nid for nid in src["selected_node_ids"] if nid not in remove}
    return [u for u in units if u["node_id"] in keep]


def _row(dataset, event, cutoff, reader, intervention_id, intervention_type,
         affected, base_text, ctx_text, base_prompt_identity, prompt_identity,
         reader_ident, base_out, gold, out):
    rec = utility_record(gold, base_out, out)
    return {
        "dataset": dataset, "event_id": event["event_id"], "cutoff": cutoff,
        "reader": reader, "intervention_id": intervention_id,
        "base_context_hash": _sha(base_text),
        "intervened_context_hash": _sha(ctx_text),
        "reader_hash": reader_ident.get("weight_hash", ""),
        "reader_identity_hash": reader_ident.get("reader_identity_hash", ""),
        "tokenizer_hash": reader_ident.get("tokenizer_hash", ""),
        "chat_template_hash": reader_ident.get("chat_template_hash", ""),
        "model_id": reader_ident.get("model_id", ""),
        "dtype": reader_ident.get("dtype", ""),
        "base_prompt_hash": base_prompt_identity["prompt_hash"],
        "base_prompt_ids_hash": base_prompt_identity["prompt_ids_hash"],
        "prompt_hash": prompt_identity["prompt_hash"],
        "prompt_ids_hash": prompt_identity["prompt_ids_hash"],
        "prompt_tokens": prompt_identity["prompt_tokens"],
        "score_A": out["score_A"], "score_B": out["score_B"],
        "p_rumor": out["p_rumor"], "p_nonrumor": out["p_nonrumor"],
        "gold": gold, "utility": rec["utility"],
        "prediction_before": rec["prediction_before"],
        "prediction_after": rec["prediction_after"],
        "correctness_before": rec["correct_before"],
        "correctness_after": rec["correct_after"],
        "gold_probability_before": rec["gold_probability_before"],
        "gold_probability_after": rec["gold_probability_after"],
        "label_flip": rec["label_flip"], "sign": rec["sign"],
        "intervention_type": intervention_type,
        "affected_reply_ids": list(affected),
        "score_A_before": rec["score_A_before"],
        "score_B_before": rec["score_B_before"],
    }


def _append(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_text(event):
    return next(n["text"] for n in event["nodes"]
                if n["node_id"] == event["source_id"])


def _expected_identity(reader_ident, base_ctx_hash, ctx_hash, prompt_identity):
    return {
        "base_context_hash": base_ctx_hash,
        "intervened_context_hash": ctx_hash,
        "reader_hash": reader_ident.get("weight_hash", ""),
        "reader_identity_hash": reader_ident.get("reader_identity_hash", ""),
        "tokenizer_hash": reader_ident.get("tokenizer_hash", ""),
        "chat_template_hash": reader_ident.get("chat_template_hash", ""),
        "prompt_hash": prompt_identity["prompt_hash"],
        "prompt_ids_hash": prompt_identity["prompt_ids_hash"],
    }


def generate(dataset, paths, out_root, split_segments=("utility_train",
                                                       "utility_dev",
                                                       "utility_eval"),
             mock=False, only_reader=None, limit_snapshots=None):
    common.assert_frozen_source(dataset, paths, out_root, "generate_labels")
    split = json.loads((Path(out_root) / "manifests" / dataset /
                        "event_split.json").read_text(encoding="utf-8"))
    wanted = set()
    for seg in split_segments:
        wanted |= set(split[seg])
    events = [e for e in common.load_dataset_events(dataset, paths)
              if e["event_id"] in wanted]
    encoder = common.CrSemanticEncoder(paths.semantic_model, dataset)
    tokenizer = common.canonical_tokenizer(paths.canonical_tokenizer)
    label_path = os.path.join(out_root, "utility_labels", dataset,
                              "labels.jsonl")
    existing = _load_existing(label_path)
    if existing:
        print(f"resume: {len(existing)} cached rows")

    readers = {}
    for key in READER_KEYS:
        if only_reader and key != only_reader:
            continue
        spec = ReaderSpec(key, paths.reader_path(key))
        readers[key] = build_reader(key, spec, mock=mock)
        print(f"reader {key} identity: {readers[key].identity().get('weight_hash')}")

    n_done = n_skip = 0
    for event in events:
        gold = int(event["label"])
        for cutoff in CUTOFFS_MIN:
            art = common.snapshot_artifacts(event, cutoff, encoder, tokenizer)
            if art["zero_reply"]:
                continue
            src, units = art["src"], art["units"]
            base_text = render_units_for_budget(units, src["selected_node_ids"])
            base_prompt = build_reader_prompt(_source_text(event), cutoff,
                                              units, base_text)
            for key, reader in readers.items():
                reader_ident = reader.identity()
                base_ctx_hash = _sha(base_text)
                base_prompt_identity = _prompt_identity(reader.tokenizer,
                                                        base_prompt)
                base_key = _cache_key(dataset, event["event_id"], cutoff, key,
                                      "I0")
                expected_base = _expected_identity(
                    reader_ident, base_ctx_hash, base_ctx_hash,
                    base_prompt_identity)
                if base_key in existing:
                    _verify_cached(existing[base_key], expected_base, base_key)
                    base_out = _out_from_row(existing[base_key])
                    n_skip += 1
                else:
                    base_out = ab_scores(reader.candidate_logprobs(base_prompt))
                    row = _row(dataset, event, cutoff, key, "I0", "I0_base", [],
                               base_text, base_text, base_prompt_identity,
                               base_prompt_identity, reader_ident, base_out,
                               gold, base_out)
                    existing[base_key] = row
                    _append(label_path, row)
                    n_done += 1
                for iv in art["interventions"]:
                    if iv["intervention_id"] == "I0":
                        continue
                    if iv["status"] != "OK":
                        continue
                    row_key = _cache_key(dataset, event["event_id"], cutoff,
                                         key, iv["intervention_id"])
                    ctx_units = _context_units(units, src,
                                               iv["remove_node_ids"])
                    ctx_text = render_units_for_budget(
                        units, [u["node_id"] for u in ctx_units])
                    prompt = build_reader_prompt(_source_text(event), cutoff,
                                                 ctx_units, ctx_text)
                    prompt_identity = _prompt_identity(reader.tokenizer, prompt)
                    expected = _expected_identity(
                        reader_ident, base_ctx_hash, _sha(ctx_text),
                        prompt_identity)
                    if row_key in existing:
                        _verify_cached(existing[row_key], expected, row_key)
                        n_skip += 1
                        continue
                    out = ab_scores(reader.candidate_logprobs(prompt))
                    row = _row(dataset, event, cutoff, key,
                               iv["intervention_id"], iv["type"],
                               iv["remove_node_ids"], base_text, ctx_text,
                               base_prompt_identity, prompt_identity,
                               reader_ident, base_out, gold, out)
                    existing[row_key] = row
                    _append(label_path, row)
                    n_done += 1
            if limit_snapshots and n_done >= limit_snapshots:
                break
    for reader in readers.values():
        reader.unload()
    return {"dataset": dataset, "rows_written": n_done, "rows_skipped": n_skip,
            "label_file": label_path}


def _out_from_row(row):
    return {"score_A": row["score_A"], "score_B": row["score_B"],
            "p_rumor": row["p_rumor"], "p_nonrumor": row["p_nonrumor"],
            "prediction": row["prediction_before"]}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pheme", "weibo22"), required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--reader", choices=READER_KEYS, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="use mock readers and the smoke namespace only")
    ap.add_argument("--limit-snapshots", type=int, default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = common.paths_or_exit()
    out_root = args.out_root or os.path.join(paths.out_root or
                                             str(common.REPO / "results" /
                                                 "cr_tser"))
    if args.smoke:
        out_root = common.smoke_root(out_root)
    result = generate(args.dataset, paths, out_root, mock=args.smoke,
                      only_reader=args.reader,
                      limit_snapshots=args.limit_snapshots)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
