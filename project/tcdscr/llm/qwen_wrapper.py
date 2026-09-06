"""Frozen Qwen3-8B wrapper (plan §23).

Frozen decoding: greedy (do_sample=False), thinking disabled when the
generation stack supports it. The wrapper returns only the raw generated
text; parsing is delegated to ``parser.parse_label``. A deterministic
``MockRumorLLM`` exists for tests only — never for formal results.
"""
from __future__ import annotations

import torch

from ..config.schema import INVALID_OUTPUT
from .parser import NON_RUMOR, RUMOR

GENERATION_CONFIG = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 8,
}


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """The single chat-formatting implementation (single user message,
    add_generation_prompt=True, thinking disabled when supported).

    ``QwenRumorLLM`` and all-current context accounting both call this, so
    the context-limit check can never drift from real inference inputs.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        # older chat templates without the enable_thinking kwarg
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def count_chat_input_tokens(tokenizer, prompt: str) -> int:
    """Token count of the chat-formatted input the model actually sees."""
    text = format_chat_prompt(tokenizer, prompt)
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


class QwenRumorLLM:

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=False,
        )
        self.model.eval()

    def _chat_text(self, prompt: str) -> str:
        return format_chat_prompt(self.tokenizer, prompt)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        text = self._chat_text(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(
            self.model.device)
        out = self.model.generate(
            **inputs,
            do_sample=GENERATION_CONFIG["do_sample"],
            num_beams=GENERATION_CONFIG["num_beams"],
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            pad_token_id=self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            new_tokens, skip_special_tokens=True)

    def classify(self, prompt: str) -> str:
        from .parser import parse_label
        return parse_label(self.generate(prompt))


class MockRumorLLM:
    """Deterministic test double — tests only, never formal results."""

    def __init__(self, fixed_label: str = RUMOR):
        if fixed_label not in (RUMOR, NON_RUMOR):
            raise ValueError("mock must return a valid exact label")
        self.fixed_label = fixed_label
        self.calls = 0

    def classify(self, prompt: str) -> str:
        self.calls += 1
        return self.fixed_label

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.fixed_label


__all__ = ["QwenRumorLLM", "MockRumorLLM", "GENERATION_CONFIG",
           "INVALID_OUTPUT"]
