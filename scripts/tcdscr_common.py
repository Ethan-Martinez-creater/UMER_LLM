"""Shared helpers for TC-DSCR entry scripts (script layer only).

Pipeline glue shared by the tcdscr_*.py entry points: deterministic event
sampling, snapshot assembly, feature construction with the §15 disk cache.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tcdscr.config.schema import SOURCE_ONLY, config_from_env  # noqa: E402
from tcdscr.data import maweibo_adapter, pheme_adapter  # noqa: E402
from tcdscr.data.semantic_encoder import (SemanticEncoder,  # noqa: E402
                                          SnapshotFeatureCache)
from tcdscr.data.snapshot_builder import (build_snapshot,  # noqa: E402
                                          build_source_only)
from tcdscr.data.structural_features import build_snapshot_features  # noqa: E402


def event_label_registry(dataset: str, cfg) -> dict:
    """Lightweight {event_id: label} registry — no event JSON is parsed."""
    if dataset == "pheme":
        return {eid: label for eid, _topic, label, _folder
                in pheme_adapter.event_ids(cfg.raw_dir)}
    if dataset == "maweibo":
        return {eid: label for eid, label
                in maweibo_adapter.event_ids(cfg.raw_dir, cfg.label_file)}
    raise ValueError(dataset)


def resolve_train_events(dataset: str, cfg, split, limit=None, seed=3090):
    """Load only the train-split events (test/validation are never read).

    ``limit`` caps the number of train events for smoke runs (deterministic
    sample from the sorted train ids); None loads every train event.
    """
    train_ids = sorted(split["train"])
    if limit is not None and limit < len(train_ids):
        train_ids = sorted(random.Random(seed).sample(train_ids, limit))
    if dataset == "pheme":
        by_id = {eid: (topic, label, folder)
                 for eid, topic, label, folder
                 in pheme_adapter.event_ids(cfg.raw_dir)}
        return [pheme_adapter.load_event(*by_id[eid]) for eid in train_ids]
    if dataset == "maweibo":
        labels = dict(maweibo_adapter.event_ids(cfg.raw_dir, cfg.label_file))
        return [maweibo_adapter.load_event(
            eid, labels[eid],
            f"{cfg.raw_dir.rstrip('/')}/{eid}.json") for eid in train_ids]
    raise ValueError(dataset)


def label_counts(ids, registry) -> dict:
    counts = {0: 0, 1: 0}
    for eid in ids:
        counts[registry[eid]] += 1
    return counts


class EventSemanticStore:
    """Per-event 384D text embedding cache: {event_id: {node_id: tensor}}.

    One MiniLM encode per unique text per event; snapshots gather rows from
    the store instead of re-encoding text at every cutoff. Cache files live
    under cache_dir/event_texts/<dataset>/<preprocess_version>_<model_hash>/.
    """

    def __init__(self, cfg, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder = None
        self._device = device
        self.cfg = cfg
        from tcdscr.data.semantic_encoder import (PREPROCESS_VERSION,
                                                  semantic_model_hash)
        self.root = os.path.join(
            cfg.cache_dir, "event_texts", cfg.dataset,
            f"{PREPROCESS_VERSION}_{semantic_model_hash(cfg.semantic_model_path)}")
        os.makedirs(self.root, exist_ok=True)

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = SemanticEncoder(self.cfg.semantic_model_path,
                                            self.cfg.dataset,
                                            device=self._device)
        return self._encoder

    def _path(self, event_id):
        return os.path.join(self.root, f"{event_id}.pt")

    def get_store(self, event, cache=None):
        """Return {node_id: float32 tensor(384)} for the event, encoding any
        missing nodes. ``cache`` is an optional in-memory
        {event_id: store} dict updated in place."""
        eid = event["event_id"]
        if cache is not None and eid in cache:
            return cache[eid]
        path = self._path(eid)
        store = None
        if os.path.exists(path):
            try:
                store = torch.load(path, map_location="cpu",
                                   weights_only=True)
            except Exception:
                store = None
        if store is None:
            store = {}
        missing = [n["node_id"] for n in event["nodes"]
                   if n["node_id"] not in store]
        if missing:
            by_id = {n["node_id"]: n["text"] for n in event["nodes"]}
            unique_texts = list({by_id[nid] for nid in missing})
            emb = self.encoder.encode(unique_texts)
            text_to_vec = {t: emb[i] for i, t in enumerate(unique_texts)}
            for nid in missing:
                store[nid] = text_to_vec[by_id[nid]].clone()
            tmp = path + ".tmp"
            torch.save(store, tmp)
            os.replace(tmp, path)
        if cache is not None:
            cache[eid] = store
        return store


def snapshot_features_from_store(event, snapshot, store):
    """Online Module-3 features: semantics gathered from the store, 1021D
    signature and 3D summary built fresh for the current snapshot."""
    sem = torch.stack([store[nid] for nid in snapshot["node_ids"]])
    return build_snapshot_features(snapshot, sem)


def load_split_events(dataset: str, cfg, split):
    """Load train/validation/test events of a resolved Protocol A split.

    Each event is loaded exactly once; the three id sets are disjoint.
    """
    if dataset == "pheme":
        by_id = {eid: (topic, label, folder)
                 for eid, topic, label, folder
                 in pheme_adapter.event_ids(cfg.raw_dir)}
        load_one = lambda eid: pheme_adapter.load_event(*by_id[eid])
    elif dataset == "maweibo":
        labels = dict(maweibo_adapter.event_ids(cfg.raw_dir, cfg.label_file))
        load_one = lambda eid: maweibo_adapter.load_event(
            eid, labels[eid], f"{cfg.raw_dir.rstrip('/')}/{eid}.json")
    else:
        raise ValueError(dataset)
    out = {}
    for seg in ("train", "validation", "test"):
        out[seg] = [load_one(eid) for eid in sorted(split[seg])]
    return out


def load_events(dataset: str, cfg, limit=None, seed=3090):
    """Deterministically sample up to ``limit`` events (sorted ids + seed)."""
    if dataset == "pheme":
        registry = pheme_adapter.event_ids(cfg.raw_dir)
    elif dataset == "maweibo":
        registry = maweibo_adapter.event_ids(cfg.raw_dir, cfg.label_file)
    else:
        raise ValueError(dataset)
    registry = sorted(registry)
    if limit is not None and limit < len(registry):
        chosen = sorted(random.Random(seed).sample(registry, limit))
    else:
        chosen = registry
    events = []
    for item in chosen:
        if dataset == "pheme":
            _eid, topic, label, folder = item
            events.append(pheme_adapter.load_event(topic, label, folder))
        else:
            eid, label = item
            path = f"{cfg.raw_dir.rstrip('/')}/{eid}.json"
            events.append(maweibo_adapter.load_event(eid, label, path))
    return events


def build_event_snapshots(event, cutoffs_min):
    """{cutoff -> snapshot} with the SOURCE_ONLY snapshot included."""
    snaps = {SOURCE_ONLY: build_source_only(event)}
    for cut in cutoffs_min:
        snaps[int(cut)] = build_snapshot(event, int(cut))
    return snaps


class SemanticBackend:
    """Lazily constructed MiniLM encoder + disk cache for one dataset."""

    def __init__(self, cfg, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SemanticEncoder(cfg.semantic_model_path, cfg.dataset,
                                       device=device)
        self.cache = SnapshotFeatureCache(cfg.cache_dir, cfg.dataset,
                                          cfg.semantic_model_path)

    def snapshot_semantics(self, event, snapshot):
        cached = self.cache.get(event["event_id"], snapshot["cutoff_minutes"])
        if cached is not None and cached.shape[0] == len(snapshot["node_ids"]):
            return cached
        emb = self.encoder.encode(snapshot["texts"])
        self.cache.put(event["event_id"], snapshot["cutoff_minutes"], emb)
        return emb

    def snapshot_features(self, event, snapshot):
        sem = self.snapshot_semantics(event, snapshot)
        return build_snapshot_features(snapshot, sem)
