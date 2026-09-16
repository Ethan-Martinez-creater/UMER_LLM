"""Mistral-7B-Instruct-v0.3 frozen reader (amendment R1 §2, §11).

Reader Protocol Amendment R1 replaced the retired GLM reader with the official
Mistral instruction checkpoint. The reader is deliberately identical to the
other two in every scientific respect: it reuses the shared chat formatter, the
shared teacher-forced candidate scorer and the shared reader-identity hashing,
and it never generates text or reads generated confidence (plan §9).
"""
from __future__ import annotations

from ..config.pilot_config import CANDIDATES
from .base_reader import BaseReader


class MistralReader(BaseReader):
    """R2 of the R1 set = mistralai/Mistral-7B-Instruct-v0.3, BF16, CUDA."""

    trust_remote_code = False

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = self.spec.model_path
        if not path:
            raise ValueError("Mistral reader path is empty (amendment R1 §2)")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=self.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=getattr(torch, self.spec.dtype),
            device_map=self.spec.device,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        return self

    def candidate_logprobs(self, user_prompt, candidates=CANDIDATES):
        from .sequence_scorer import candidate_logprobs_hf
        return candidate_logprobs_hf(
            self.model, self.tokenizer, user_prompt, candidates,
            thinking=False, device=self.spec.device)
