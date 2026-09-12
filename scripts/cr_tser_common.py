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
    """Load the pilot event pool for a dataset; Weibo22 fails closed."""
    if dataset == "pheme":
        from tcdscr.data import pheme_adapter
        return [pheme_adapter.load_event(topic, label, folder)
                for _eid, topic, label, folder
                in pheme_adapter.event_ids(paths.pheme_raw)]
    if dataset == "weibo22":
        from cr_tser.data import weibo22_adapter
        return weibo22_adapter.load_events(paths.weibo22_raw, normalized_paths)
    raise ValueError(f"unknown dataset {dataset!r}")


def dataset_registry(dataset: str, paths):
    """``{event_id: label}`` without parsing every event body."""
    if dataset == "pheme":
        from tcdscr.data import pheme_adapter
        return {eid: label for eid, _topic, label, _folder
                in pheme_adapter.event_ids(paths.pheme_raw)}
    if dataset == "weibo22":
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
