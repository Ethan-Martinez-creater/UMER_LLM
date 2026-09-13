"""Shared orchestration helpers for the CR-TSER entry scripts.

Script layer only. Keeps dataset loading, snapshot iteration, semantic
embeddings and canonical tokenization in one place so the pilot entry points
cannot disagree about what the readers saw.

The frozen TC-DSCR snapshot builder and text cleaning are reused unchanged
(plan §29). The MiniLM wrapper is local because TC-DSCR's encoder only knows
``pheme`` / ``maweibo`` and CR-TSER needs ``weibo22``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
PROJECT = REPO / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from cr_tser.config.pilot_config import (CUTOFFS_MIN, SEMANTIC_DIM,  # noqa: E402
                                         paths_from_env)
from cr_tser.data.snapshot_bridge import build_causal_snapshot  # noqa: E402


class CrSemanticEncoder:
    """Frozen 384D MiniLM with dataset-specific cleaning (plan §7, §29)."""

    def __init__(self, model_path: str, dataset: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        from tcdscr.data.text_cleaning import clean_text_weibo, clean_tweet_pheme
        self.dataset = dataset
        self.clean = clean_tweet_pheme if dataset == "pheme" else clean_text_weibo
        self.model = SentenceTransformer(model_path, device=device)
        if self.model.get_sentence_embedding_dimension() != SEMANTIC_DIM:
            raise ValueError("semantic model must output 384 dims")
        self.model.eval()

    def encode(self, texts):
        import torch
        cleaned = [self.clean(t) for t in texts]
        with torch.no_grad():
            emb = self.model.encode(cleaned, batch_size=64,
                                    show_progress_bar=False,
                                    convert_to_numpy=True)
        return torch.tensor(emb, dtype=torch.float32)


def load_dataset_events(dataset: str, paths, normalized_paths=None):
    """Load the pilot event pool for a dataset.

    * ``maweibo`` (amendment V2 primary) — the audited TC-DSCR adapter through
      :mod:`cr_tser.data.maweibo_bridge`, so timestamps keep coming only from
      the raw ``t`` field and the V2 eligibility statuses apply;
    * ``pheme`` (secondary) — the raw thread directory;
    * ``weibo22`` — historical V1 candidate; only the normalized export can
      reach a snapshot, and it fails closed otherwise.
    """
    if dataset == "maweibo":
        from cr_tser.data import maweibo_bridge
        return maweibo_bridge.load_events(paths.maweibo_raw,
                                          paths.maweibo_labels)
    if dataset == "pheme":
        from tcdscr.data import pheme_adapter
        return [pheme_adapter.load_event(topic, label, folder)
                for _eid, topic, label, folder
                in pheme_adapter.event_ids(paths.pheme_raw)]
    if dataset == "weibo22":
        from cr_tser.data import weibo22_adapter
        normalized = normalized_paths or paths.weibo22_normalized or None
        return weibo22_adapter.load_events(paths.weibo22_raw, normalized)
    raise ValueError(f"unknown dataset {dataset!r}; the V2 protocol accepts "
                     "only 'maweibo' (primary) and 'pheme' (secondary)")


def dataset_registry(dataset: str, paths):
    """``{event_id: label}`` for the pool the split will actually sample from.

    The Ma-Weibo registry comes from the composite source of record (raw JSON
    directory + label file); Weibo22 is retained only for the historical V1
    path and can never reach a V2 split.
    """
    if dataset == "maweibo":
        if not (paths.maweibo_raw and paths.maweibo_labels):
            return {}
        from tcdscr.data import maweibo_adapter
        return {eid: int(label) for eid, label
                in maweibo_adapter.event_ids(paths.maweibo_raw,
                                             paths.maweibo_labels)}
    if dataset == "pheme":
        from tcdscr.data import pheme_adapter
        return {eid: label for eid, _topic, label, _folder
                in pheme_adapter.event_ids(paths.pheme_raw)}
    if dataset == "weibo22":
        if paths.weibo22_normalized:
            return {e["event_id"]: int(e["label"])
                    for e in load_dataset_events("weibo22", paths)}
        from cr_tser.data import weibo22_adapter
        return dict(weibo22_adapter.dataset_event_ids(paths.weibo22_raw))
    raise ValueError(dataset)


def canonical_tokenizer(path: str):
    from transformers import AutoTokenizer
    if not path or not os.path.isdir(path):
        raise FileNotFoundError(f"canonical tokenizer path missing: {path!r}")
    return AutoTokenizer.from_pretrained(path, trust_remote_code=False,
                                         local_files_only=True)


def snapshot_artifacts(event, cutoff, encoder, tokenizer):
    """Everything one (event, cutoff) contributes to the pilot."""
    from cr_tser.data.structural_stats import structural_scalars
    from cr_tser.intervention.evidence_units import build_evidence_units
    from cr_tser.intervention.intervention_generator import \
        generate_interventions
    from cr_tser.intervention.semantic_reference import build_src

    snap = build_causal_snapshot(event, cutoff)
    units = build_evidence_units(snap)
    if not units:
        return {"snapshot": snap, "units": [], "src": None,
                "interventions": [], "zero_reply": True}
    emb = encoder.encode(snap["texts"])
    pos = {nid: i for i, nid in enumerate(snap["node_ids"])}
    reply_emb = {nid: emb[pos[nid]] for nid in snap["node_ids"]}
    src = build_src(units, emb[pos[snap["source_id"]]], reply_emb, tokenizer)
    interventions = generate_interventions(snap, units, src, tokenizer)
    scalars = structural_scalars(snap, cutoff)
    return {"snapshot": snap, "units": units, "src": src,
            "interventions": interventions, "zero_reply": False,
            "scalars": scalars, "semantic": emb, "reply_emb": reply_emb}


def iter_events_cutoffs(events, cutoffs=CUTOFFS_MIN):
    for event in events:
        for cutoff in cutoffs:
            yield event, cutoff


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return path


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def paths_or_exit():
    return paths_from_env()


def default_out_root(paths, explicit=None):
    """Formal V2 artifact root (amendment §22).

    V1 ``results/cr_tser`` stays historical and read-only; a formal V2 run
    writes into ``results/cr_tser_v2`` (``CRTSER_OUT_ROOT`` still overrides).
    A relative root is anchored to the repository so a stage cannot silently
    write next to whatever the current working directory happens to be.
    """
    from cr_tser.config.pilot_config import V2_RESULTS_ROOT
    if explicit:
        return explicit
    root = getattr(paths, "out_root", "") or ""
    if not root:
        return str(REPO / V2_RESULTS_ROOT)
    return root if os.path.isabs(root) else str(REPO / root)


def assert_frozen_source(dataset: str, paths, out_root: str,
                         context: str) -> dict:
    """Verify this stage is reading the data source the manifest froze.

    Any stage after manifest creation calls this before touching data, so a
    swapped or re-generated dataset cannot silently produce labels or
    predictions against a different source (plan §31).
    """
    from cr_tser.data.source_manifest import (assert_same_source,
                                              load_frozen_source,
                                              source_fingerprint)
    manifest_dir = os.path.join(out_root, "manifests", dataset)
    frozen = load_frozen_source(manifest_dir)
    if not frozen:
        return {}
    current = source_fingerprint(dataset, paths)
    assert_same_source(frozen, current, context=context)
    return frozen


def smoke_root(out_root: str) -> str:
    """Smoke/test namespace, never the formal artifact namespace.

    A ``--smoke`` run must not be able to freeze the formal cache or be
    mistaken for a formal result, so all smoke writes are redirected under
    ``<out_root>/smoke``.
    """
    return os.path.join(out_root, "smoke")


def source_fingerprint_for(dataset: str, paths) -> dict:
    """Frozen source identity of the dataset this stage is reading."""
    from cr_tser.data.source_manifest import source_fingerprint
    return source_fingerprint(dataset, paths)
