"""Frozen reader interface and prompt (plan §8, §9, §29).

All three readers share exactly one prompt builder and one identity-hash
scheme, so a reader can never be quietly swapped or formatted differently.
The task is the plan §9 A/B choice task; confidence is **never** the utility
signal — utility comes from teacher-forced sequence scores computed by
:mod:`cr_tser.readers.sequence_scorer`.

Evidence text is rendered with the frozen TC-DSCR reader prompt (plan §29), so
the intervention contexts and the token accounting stay identical to the
verified TC-DSCR implementation.
"""
from __future__ import annotations

import hashlib
import json
import os

from tcdscr.llm import reader_prompt

from ..config.pilot_config import CANDIDATES, READER_MODEL_IDS, READER_KEYS

SYSTEM_PROMPT = (
    "You are evaluating whether a social-media source post is a rumor.\n"
    "Use only the source post and the supplied observed social evidence.\n"
    "Do not add external facts.\n"
    "Answer with exactly one letter: A or B."
)

USER_TEMPLATE = """Source Post:
{source_text}

Observed Social Evidence up to {cutoff_label}:
{evidence_block}

Task:
Use only the source post and supplied observed social evidence.
Do not add external facts.

Choose:
A = RUMOR
B = NON_RUMOR

Answer with a single letter: A or B."""

CUTOFF_LABELS = {15: "15m", 60: "1h", 360: "6h"}


def cutoff_label(cutoff) -> str:
    cutoff = int(cutoff)
    return CUTOFF_LABELS.get(cutoff, f"{cutoff}m")


def evidence_block_from_units(units) -> str:
    return reader_prompt.render_evidence_block(units)


def build_reader_prompt(source_text: str, cutoff, units, evidence_block=None):
    """The single reader prompt builder (plan §9)."""
    block = evidence_block if evidence_block is not None else \
        evidence_block_from_units(units)
    return USER_TEMPLATE.format(source_text=source_text,
                                cutoff_label=cutoff_label(cutoff),
                                evidence_block=block)


def build_messages(user_prompt: str):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}]


class ReaderSpec:
    """Identity of one frozen reader (plan §8). No substitution after freeze."""

    def __init__(self, key: str, model_path: str, dtype: str = "bfloat16",
                 device: str = "cuda", thinking: bool = False):
        if key not in READER_KEYS:
            raise ValueError(f"unknown reader key {key!r}")
        self.key = key
        self.model_id = READER_MODEL_IDS[key]
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.thinking = thinking

    def to_dict(self) -> dict:
        return {"key": self.key, "model_id": self.model_id,
                "model_path": self.model_path, "dtype": self.dtype,
                "device": self.device, "thinking": self.thinking}


def _file_sha256(path: str, head_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if head_bytes is None:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(fh.read(head_bytes))
    return h.hexdigest()


def model_weight_hash(model_path: str, head_bytes: int = 1 << 20) -> str:
    """Hash of the weight files' identity (name, size, first MiB each).

    Reading whole multi-GB checkpoints is unnecessary for detecting
    substitution; the name/size/prefix triple is stable and cheap.
    """
    if not os.path.isdir(model_path):
        return ""
    h = hashlib.sha256()
    for name in sorted(os.listdir(model_path)):
        if not (name.endswith(".safetensors") or name.endswith(".bin")):
            continue
        p = os.path.join(model_path, name)
        h.update(f"{name}:{os.path.getsize(p)}:".encode())
        h.update(_file_sha256(p, head_bytes=head_bytes).encode())
    return h.hexdigest()


def tokenizer_hash(model_path: str) -> str:
    if not os.path.isdir(model_path):
        return ""
    h = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "special_tokens_map.json", "tokenizer.model"):
        p = os.path.join(model_path, name)
        h.update(name.encode())
        if os.path.exists(p):
            h.update(_file_sha256(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


def chat_template_hash(tokenizer) -> str:
    template = getattr(tokenizer, "chat_template", None) or ""
    return hashlib.sha256(template.encode()).hexdigest()


def transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return ""


def reader_identity(spec: ReaderSpec, tokenizer=None) -> dict:
    """Full §8 identity record, written before any experiment."""
    return {
        **spec.to_dict(),
        "weight_hash": model_weight_hash(spec.model_path),
        "tokenizer_hash": tokenizer_hash(spec.model_path),
        "chat_template_hash": chat_template_hash(tokenizer)
        if tokenizer is not None else "",
        "transformers_version": transformers_version(),
        "device_config": spec.device,
    }


class BaseReader:
    """Interface every frozen reader implements (plan §9).

    Subclasses only supply the model plumbing; the scoring math and the prompt
    live in shared code so the three readers cannot diverge.
    """

    def __init__(self, spec: ReaderSpec, tokenizer=None, model=None):
        self.spec = spec
        self.tokenizer = tokenizer
        self.model = model

    # -- lifecycle ---------------------------------------------------------
    def load(self):  # pragma: no cover - requires real weights
        raise NotImplementedError

    def unload(self):
        self.model = None

    # -- what subclasses must provide --------------------------------------
    def candidate_logprobs(self, user_prompt: str, candidates=CANDIDATES):
        """``{candidate: summed teacher-forced log prob}``."""
        raise NotImplementedError

    def identity(self) -> dict:
        return reader_identity(self.spec, self.tokenizer)

    # -- shared behaviour --------------------------------------------------
    def score_ab(self, user_prompt: str) -> dict:
        from .sequence_scorer import ab_scores
        return ab_scores(self.candidate_logprobs(user_prompt))


def reader_specs(paths, dtype="bfloat16", device="cuda") -> dict:
    """Build the three frozen specs from resolved paths (plan §8)."""
    return {key: ReaderSpec(key, paths.reader_path(key), dtype=dtype,
                            device=device, thinking=False)
            for key in READER_KEYS}


def dump_reader_audit(specs, out_dir: str) -> dict:
    """JSON-safe reader audit (weights/tokenizer/template hashes)."""
    os.makedirs(out_dir, exist_ok=True)
    audit = {key: reader_identity(spec) for key, spec in specs.items()}
    with open(os.path.join(out_dir, "reader_audit.json"), "w",
              encoding="utf-8") as fh:
        json.dump(audit, fh, indent=1)
    return audit


def build_reader(key: str, spec: ReaderSpec, mock: bool = False):
    """Construct the frozen reader for ``key`` (plan §8).

    ``mock`` is a smoke/test-only escape hatch; formal pilot runs must pass
    ``mock=False`` and the runner enforces that.
    """
    if mock:
        from .mock_reader import MockReader
        return MockReader(key, spec=spec).load()
    if key == "qwen":
        from .qwen_reader import QwenReader
        return QwenReader(spec).load()
    if key == "glm":
        from .glm_reader import GLMReader
        return GLMReader(spec).load()
    if key == "internlm":
        from .internlm_reader import InternLMReader
        return InternLMReader(spec).load()
    raise ValueError(f"unknown reader key {key!r}")
