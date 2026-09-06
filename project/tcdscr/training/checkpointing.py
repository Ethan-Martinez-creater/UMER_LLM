"""Checkpoint persistence and hashing (for LLM cache keys, plan §24)."""
from __future__ import annotations

import hashlib
import os

import torch


def save_checkpoint(payload: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=True)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
