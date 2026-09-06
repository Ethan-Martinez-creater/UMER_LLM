"""Snapshot sampling (plan §36).

Each epoch, every train event contributes exactly one snapshot, chosen
uniformly at random from its available dynamic snapshots. SOURCE_ONLY is
never used as an encoder training snapshot.
"""
from __future__ import annotations

import random


class SnapshotSampler:

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def sample_one(self, event_features: dict):
        """``event_features``: {cutoff -> feature dict}, dynamic cutoffs only."""
        dynamic = sorted(k for k in event_features
                         if k != "SOURCE_ONLY")
        if not dynamic:
            raise ValueError("event has no dynamic snapshot for training")
        return event_features[self.rng.choice(dynamic)]

    def sample_batch(self, per_event_features: dict):
        """One sampled snapshot per event; returns list of feature dicts."""
        out = []
        for eid in sorted(per_event_features):
            out.append(self.sample_one(per_event_features[eid]))
        return out
