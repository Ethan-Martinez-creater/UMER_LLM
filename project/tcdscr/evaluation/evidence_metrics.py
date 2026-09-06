"""Efficiency metrics (plan §42): evidence and prompt token statistics."""
from __future__ import annotations


def _p90(values):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * 0.9
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def efficiency_metrics(rows) -> dict:
    """``rows``: records with evidence_tokens / total_prompt_tokens."""
    ev = [r["evidence_tokens"] for r in rows if r.get("evidence_tokens") is not None]
    tot = [r["total_prompt_tokens"] for r in rows
           if r.get("total_prompt_tokens") is not None]
    return {
        "mean_evidence_tokens": sum(ev) / max(len(ev), 1),
        "median_evidence_tokens": (
            sorted(ev)[len(ev) // 2] if ev else None),
        "mean_total_prompt_tokens": sum(tot) / max(len(tot), 1),
        "p90_total_prompt_tokens": _p90(tot),
        "n_rows": len(rows),
    }
