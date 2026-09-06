"""LLM response cache (plan §24).

Key fields (frozen): model_id, model_revision, prompt_sha256,
generation_config_sha256, dataset, event_id, cutoff, selector_checkpoint_hash,
budget. The same key is never inferred twice. Storage is an append-only JSONL
file plus an in-memory index; the file lives outside the git tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

KEY_FIELDS = ("model_id", "model_revision", "prompt_sha256",
              "generation_config_sha256", "dataset", "event_id", "cutoff",
              "selector_checkpoint_hash", "budget")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_cache_key(model_id, model_revision, prompt, generation_config,
                    dataset, event_id, cutoff, selector_checkpoint_hash,
                    budget) -> str:
    payload = {
        "model_id": model_id,
        "model_revision": model_revision,
        "prompt_sha256": sha256_text(prompt),
        "generation_config_sha256": sha256_text(
            json.dumps(generation_config, sort_keys=True)),
        "dataset": dataset,
        "event_id": event_id,
        "cutoff": cutoff,
        "selector_checkpoint_hash": selector_checkpoint_hash,
        "budget": budget,
    }
    if set(payload) != set(KEY_FIELDS):
        raise ValueError("cache key fields deviate from the frozen list")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class LLMResponseCache:

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._index = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    self._index[row["key"]] = row["response"]

    def get(self, key: str):
        with self._lock:
            return self._index.get(key)

    def put(self, key: str, response: str):
        with self._lock:
            if key in self._index:
                return  # same key is never inferred twice
            self._index[key] = response
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "response": response}) + "\n")
