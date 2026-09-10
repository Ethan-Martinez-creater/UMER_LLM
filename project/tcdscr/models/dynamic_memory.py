"""Dynamic Evidence Memory (Formal E3): frozen-scorer dynamic selection.

The E3 dynamic score is exactly

    d_i^t = u_i^t + lambda_n * novelty_i^t + lambda_p * persistence_i^t

where u_i^t is the frozen Static Utility Selector score, novelty is
1 - max cosine(e_i, e_j) over the previous memory M_{t-1} (1.0 when the
memory is empty), and persistence is 1 iff node_id_i in M_{t-1}.

Memory stores only (node_id, semantic embedding); it is updated with the
current cutoff's selected node ids. This module has no trainable parameters
and never performs a gradient update (E3 order §4, §9-§13).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NODE_EMBED_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2, self-normalized


def novelty_scores(node_embs: torch.Tensor,
                   memory_embs: torch.Tensor | None) -> torch.Tensor:
    """novelty_i = 1 - max cosine(e_i, e_j) for j in M_{t-1}.

    Empty memory yields a constant 1.0 for every candidate. Cosine is the
    true normalized cosine, clamped to [-1, 1] before subtraction.
    """
    if memory_embs is None or memory_embs.shape[0] == 0:
        return torch.ones(node_embs.shape[0], dtype=node_embs.dtype,
                          device=node_embs.device)
    cos = F.cosine_similarity(node_embs.unsqueeze(1),
                              memory_embs.unsqueeze(0), dim=-1)
    cos = cos.clamp(-1.0, 1.0)
    return 1.0 - cos.max(dim=-1).values


def persistence_flags(node_ids, memory_ids) -> torch.Tensor:
    """persistence_i = 1 iff node_id_i in M_{t-1}, else 0 (E3 order §10)."""
    if memory_ids is None or len(memory_ids) == 0:
        return torch.zeros(len(node_ids), dtype=torch.float32)
    mem = set(memory_ids)
    return torch.tensor([1.0 if nid in mem else 0.0 for nid in node_ids],
                        dtype=torch.float32)


def dynamic_scores(u: torch.Tensor, novelty: torch.Tensor,
                   persistence: torch.Tensor,
                   lambda_n: float, lambda_p: float) -> torch.Tensor:
    """d_i = u_i + lambda_n * novelty_i + lambda_p * persistence_i (§11)."""
    return u + lambda_n * novelty + lambda_p * persistence


def select_with_costs(units, scores, costs, budget):
    """Greedy whole-unit selection with precomputed token costs.

    Identical semantics to EvidenceBudgetSelector.select: units are tried in
    (-score, order) order, a unit that does not fit is skipped and the walk
    continues (never truncated), ties keep snapshot order. ``costs`` is the
    aligned per-unit evidence-token count. Returns (accepted, total_tokens).
    """
    order = sorted(range(len(units)),
                   key=lambda i: (-scores[i], units[i]["order"]))
    accepted, total = [], 0
    for idx in order:
        cost = costs[idx]
        if total + cost > budget:
            continue
        accepted.append(units[idx])
        total += cost
    return accepted, total


def memory_embeddings(selected_units, sem_matrix, node_ids):
    """Stacked 384D semantic rows for the selected units (memory payload)."""
    if not selected_units:
        return None
    index = {nid: i for i, nid in enumerate(node_ids)}
    rows = [sem_matrix[index[u["node_id"]]] for u in selected_units]
    return torch.stack(rows) if rows else None
