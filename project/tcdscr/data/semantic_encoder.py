"""384D semantic features with snapshot-level caching (plan §7, §15).

Both datasets use sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384D, self-normalizing). Text preprocessing is the historically verified one:
PHEME -> clean_tweet_pheme, Ma-Weibo -> clean_text_weibo(original_text) — the
step the server pipeline lost to version drift and that the D6 parity run
proved mandatory.

Cache key (§15): dataset / event_id / cutoff / preprocess_version /
semantic_model_hash. No test-set-specific refitting exists anywhere: the
encoder is a frozen external model and cleaning is a pure function.
"""
from __future__ import annotations

import hashlib
import json
import os

import torch

from ..config.schema import PREPROCESS_VERSION
from .text_cleaning import clean_text_weibo, clean_tweet_pheme

SEMANTIC_DIM = 384


class SemanticEncoder:
    """Frozen MiniLM encoder with dataset-specific text cleaning."""

    def __init__(self, model_path: str, dataset: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        if dataset not in ("pheme", "maweibo"):
            raise ValueError(f"unknown dataset {dataset!r}")
        self.dataset = dataset
        self.model = SentenceTransformer(model_path, device=device)
        if self.model.get_sentence_embedding_dimension() != SEMANTIC_DIM:
            raise ValueError("semantic model must output 384 dims")
        self.model.eval()

    def clean(self, text: str) -> str:
        if self.dataset == "pheme":
            return clean_tweet_pheme(text)
        return clean_text_weibo(text)

    @torch.no_grad()
    def encode(self, texts) -> torch.Tensor:
        cleaned = [self.clean(t) for t in texts]
        emb = self.model.encode(
            cleaned, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True)
        return torch.tensor(emb, dtype=torch.float32)


def semantic_model_hash(model_path: str) -> str:
    """Stable hash over the model's identity files (no weights reading)."""
    h = hashlib.sha256()
    for name in ("config.json", "tokenizer_config.json",
                 "sentence_bert_config.json", "modules.json"):
        p = os.path.join(model_path, name)
        h.update(name.encode())
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
        h.update(b"\x00")
    return h.hexdigest()[:16]


class SnapshotFeatureCache:
    """Disk cache for per-snapshot semantic features (§15 key)."""

    def __init__(self, cache_dir: str, dataset: str, model_path: str):
        self.root = os.path.join(
            cache_dir, dataset, f"{PREPROCESS_VERSION}_{semantic_model_hash(model_path)}")
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "cache_identity.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({
                "dataset": dataset,
                "preprocess_version": PREPROCESS_VERSION,
                "semantic_model_hash": semantic_model_hash(model_path),
                "semantic_model_path": model_path,
            }, fh, indent=1)

    def _path(self, event_id: str, cutoff) -> str:
        return os.path.join(self.root, f"{event_id}_{cutoff}.pt")

    def get(self, event_id: str, cutoff):
        p = self._path(event_id, cutoff)
        if os.path.exists(p):
            return torch.load(p, map_location="cpu", weights_only=True)
        return None

    def put(self, event_id: str, cutoff, semantic: torch.Tensor):
        torch.save(semantic, self._path(event_id, cutoff))
