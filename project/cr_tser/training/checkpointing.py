"""Checkpoint save/load with identity hashes (plan §31, §33).

Checkpoints are frozen after training; the verifier compares their hashes
before and after the held-out reader is exposed.
"""
from __future__ import annotations

import hashlib
import json
import os

import torch


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path: str, bitte, model, meta: dict | None = None) -> dict:
    """Persist BiTTE + utility heads; returns the checkpoint identity."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "bitte": bitte.state_dict(),
        "utility": model.state_dict(),
        "meta": meta or {},
    }, path)
    return {"path": path, "sha256": file_sha256(path),
            "bytes": os.path.getsize(path), "meta": meta or {}}


def load_checkpoint(path: str, bitte, model, device="cpu") -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    bitte.load_state_dict(payload["bitte"])
    model.load_state_dict(payload["utility"])
    return payload.get("meta", {})


def save_baseline_checkpoint(path: str, model, meta: dict | None = None) -> dict:
    """Persist a B0/B1 predictor (no BiTTE component)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"baseline": model.state_dict(), "meta": meta or {}}, path)
    return {"path": path, "sha256": file_sha256(path),
            "bytes": os.path.getsize(path), "meta": meta or {}}


def load_baseline_checkpoint(path: str, model, device="cpu") -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["baseline"])
    return payload.get("meta", {})


def freeze_manifest(entries, path: str) -> dict:
    """Write ``{name: {sha256, bytes}}`` and the manifest's own hash (§31)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = {}
    for name, file_path in entries.items():
        out[name] = {"path": file_path, "sha256": file_sha256(file_path),
                     "bytes": os.path.getsize(file_path)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    out["_manifest_sha256"] = file_sha256(path)
    return out
