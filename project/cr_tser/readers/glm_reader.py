"""GLM-4-9B-Chat frozen reader (plan §8, R2)."""
from __future__ import annotations

from ..config.pilot_config import CANDIDATES
from .base_reader import BaseReader


class GLMReader(BaseReader):
    """R2 = zai-org/glm-4-9b-chat-hf."""

    trust_remote_code = False

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = self.spec.model_path
        if not path:
            raise ValueError("GLM reader path is empty (plan §8)")
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
