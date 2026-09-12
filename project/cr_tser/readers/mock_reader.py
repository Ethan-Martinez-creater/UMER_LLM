"""Deterministic reader test doubles.

Used by the unit tests and by the P0 A/B sanity harness. **Never** used for
formal pilot results: the runner refuses mock readers outside ``--smoke``.
"""
from __future__ import annotations

import hashlib

from ..config.pilot_config import CANDIDATES, READER_MODEL_IDS
from .base_reader import BaseReader, ReaderSpec


class MockReader(BaseReader):
    """Deterministic pseudo-scores derived from the prompt, or a script.

    ``script`` may be a callable ``prompt -> {"A": float, "B": float}`` or a
    dict keyed by prompt. A default deterministic hash keeps the scores stable
    across processes while still varying with the context.
    """

    def __init__(self, key: str = "qwen", script=None, spec: ReaderSpec | None = None):
        spec = spec or ReaderSpec(key, model_path=f"/mock/{key}",
                                  dtype="float32", device="cpu")
        super().__init__(spec)
        self.script = script
        self.calls = 0

    def load(self):
        return self

    def candidate_logprobs(self, user_prompt, candidates=CANDIDATES):
        self.calls += 1
        if callable(self.script):
            out = self.script(user_prompt)
        elif isinstance(self.script, dict):
            out = self.script[user_prompt]
        else:
            digest = hashlib.sha256(user_prompt.encode("utf-8")).digest()
            a = (digest[0] / 255.0) - 0.5
            b = (digest[1] / 255.0) - 0.5
            out = {"A": a, "B": b}
        return {c: float(out[c]) for c in candidates}

    def identity(self) -> dict:
        return {"key": self.spec.key, "model_id": READER_MODEL_IDS.get(self.spec.key),
                "mock": True}


class RecordingReader(BaseReader):
    """Wraps another reader and records every prompt it is asked to score.

    The leave-one-reader-out verifier uses it to prove the held-out reader's
    prompts/labels never enter training, early stopping or selection.
    """

    def __init__(self, inner: BaseReader, label: str = "heldout"):
        super().__init__(inner.spec)
        self.inner = inner
        self.label = label
        self.prompts: list = []

    def candidate_logprobs(self, user_prompt, candidates=CANDIDATES):
        self.prompts.append(user_prompt)
        return self.inner.candidate_logprobs(user_prompt, candidates)

    def identity(self) -> dict:
        return self.inner.identity()
