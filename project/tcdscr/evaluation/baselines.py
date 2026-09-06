"""Frozen baseline selectors (plan §25, §26).

Eight evaluation arms share the same evidence-unit pipeline; only the score
differs. The structural budget is frozen as
score_i = z(normDegree_i) - 0.25 * z(normDepth_i) with z=0 when std=0 and
earlier-timestamp-first tie-breaking. Randomness is seeded per
(dataset, event, cutoff) so runs are reproducible.
"""
from __future__ import annotations

import hashlib
import random

import torch

BASELINE_NAMES = (
    "source_only", "all_current", "random_budget", "recent_budget",
    "semantic_budget", "structural_budget", "static_selector",
    "tcdscr_dynamic",
)


def _seeded_rng(dataset, event_id, cutoff) -> random.Random:
    digest = hashlib.sha256(
        f"{dataset}:{event_id}:{cutoff}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _z(values):
    n = len(values)
    if n == 0:
        return [0.0] * n
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    if std == 0:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def rank_candidates(name, snapshot, units, ctx):
    """Return scores aligned to ``units`` for a baseline arm.

    ``ctx`` keys: sem (N,384 tensor incl. source row 0 aligned to
    snapshot node order), source_pos, selector_scores (N,), dynamic_scores
    (N,). all_current / source_only return None scores (handled below).
    """
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}")
    if name in ("source_only", "all_current"):
        return None
    n_units = len(units)
    if name == "random_budget":
        rng = _seeded_rng(ctx["dataset"], snapshot["event_id"],
                          snapshot["cutoff_minutes"])
        return [rng.random() for _ in range(n_units)]
    if name == "recent_budget":
        return [-u["elapsed_seconds"] for u in units]
    if name == "semantic_budget":
        src_idx = ctx["source_pos"]
        src_vec = ctx["sem"][src_idx]
        rows = []
        index = {nid: i for i, nid in enumerate(snapshot["node_ids"])}
        for u in units:
            rows.append(index[u["node_id"]])
        sem = ctx["sem"][rows]
        cos = sem @ src_vec
        return cos.tolist()
    if name == "structural_budget":
        index = {nid: i for i, nid in enumerate(snapshot["node_ids"])}
        deg, dep = [], []
        for u in units:
            i = index[u["node_id"]]
            deg.append(ctx["struct3"][i][0])
            dep.append(ctx["struct3"][i][1])
        z_deg, z_dep = _z(deg), _z(dep)
        return [z_deg[k] - 0.25 * z_dep[k] for k in range(n_units)]
    if name == "static_selector":
        index = {nid: i for i, nid in enumerate(snapshot["node_ids"])}
        return [float(ctx["selector_scores"][index[u["node_id"]]])
                for u in units]
    if name == "tcdscr_dynamic":
        index = {nid: i for i, nid in enumerate(snapshot["node_ids"])}
        return [float(ctx["dynamic_scores"][index[u["node_id"]]])
                for u in units]
    raise ValueError(f"unhandled baseline {name!r}")


def select_evidence(name, snapshot, units, ctx, budget_selector):
    """Return (selected_units, evidence_tokens, scores) for one arm."""
    scores = rank_candidates(name, snapshot, units, ctx)
    if name == "source_only":
        return [], 0, scores
    if name == "all_current":
        # the whole current snapshot, deterministic snapshot order; tokens
        # are counted but the budget does not bind this baseline
        total = 0
        rendered = []
        for i, u in enumerate(units):
            from ..context.evidence_unit import render_evidence
            text = render_evidence(u, i + 1)
            total += budget_selector.count_tokens(text)
            rendered.append(u)
        return rendered, total, scores
    order = sorted(range(len(units)),
                   key=lambda i: (-scores[i], units[i]["order"]))
    accepted, total = [], 0
    for idx in order:
        from ..context.evidence_unit import render_evidence
        candidate = render_evidence(units[idx], len(accepted) + 1)
        cost = budget_selector.count_tokens(candidate)
        if total + cost > budget_selector.budget:
            continue
        accepted.append(units[idx])
        total += cost
    return accepted, total, scores
