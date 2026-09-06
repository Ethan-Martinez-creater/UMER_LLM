"""Exact LLM output parser (plan §23).

Only the exact strings RUMOR / NON_RUMOR are accepted after whitespace
stripping; anything else is INVALID_OUTPUT. No fuzzy matching is permitted.
"""
from __future__ import annotations

from ..config.schema import INVALID_OUTPUT

RUMOR = "RUMOR"
NON_RUMOR = "NON_RUMOR"


def parse_label(raw_output: str) -> str:
    if not isinstance(raw_output, str):
        return INVALID_OUTPUT
    text = raw_output.strip()
    if text == RUMOR:
        return RUMOR
    if text == NON_RUMOR:
        return NON_RUMOR
    return INVALID_OUTPUT


def label_to_int(label: str):
    """RUMOR=1, NON_RUMOR=0, INVALID_OUTPUT=None (excluded from metrics)."""
    if label == RUMOR:
        return 1
    if label == NON_RUMOR:
        return 0
    return None
