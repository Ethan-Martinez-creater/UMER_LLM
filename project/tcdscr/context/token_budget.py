"""Token budget selection (plan §21).

Budgets {512, 1024, 2048} are counted with the Qwen3-8B tokenizer. Units are
added whole in dynamic-score descending order; a Reply-Parent Pair is never
truncated. Source text and instructions do not count against the evidence
budget; total prompt tokens are reported separately. A unit that does not fit
is skipped and the walk continues (deterministic greedy, no label awareness).
"""
from __future__ import annotations

from ..config.schema import TOKEN_BUDGETS
from .evidence_unit import render_evidence


def format_cutoff(cutoff) -> str:
    """Deterministic human label for a cutoff value."""
    if cutoff == "SOURCE_ONLY":
        return "SOURCE_ONLY"
    minutes = int(cutoff)
    return {60: "1h", 180: "3h", 360: "6h", 1440: "24h"}.get(
        minutes, f"{minutes}m")


class EvidenceBudgetSelector:

    def __init__(self, tokenizer, budget: int):
        if budget not in TOKEN_BUDGETS:
            raise ValueError(
                f"budget must be one of {TOKEN_BUDGETS}, got {budget}")
        self.tokenizer = tokenizer
        self.budget = budget

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def select(self, units, scores):
        """Greedy whole-unit selection; returns (accepted, evidence_tokens).

        ``units`` is the snapshot-ordered unit list; ``scores`` the aligned
        dynamic (or baseline) scores. Ties keep snapshot order (earlier first).
        """
        order = sorted(range(len(units)),
                       key=lambda i: (-scores[i], units[i]["order"]))
        accepted, total = [], 0
        for idx in order:
            candidate = render_evidence(units[idx], len(accepted) + 1)
            cost = self.count_tokens(candidate)
            if total + cost > self.budget:
                continue
            accepted.append(units[idx])
            total += cost
        return accepted, total
