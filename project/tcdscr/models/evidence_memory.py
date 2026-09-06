"""Dynamic evidence memory (plan §19).

M_{t-1} is the evidence selected at the previous snapshot of the same event
(single-step rolling memory). For a candidate i at time t:

  novelty_i = 1                                if M_{t-1} is empty
            = 1 - max_{j in M_{t-1}} cos(e_i, e_j)   otherwise
  persist_i = 1[i in M_{t-1} by node id]
  d_i = u_i + lambda_n * novelty_i + lambda_p * persist_i

The memory only ever contains past selections; scoring at time t never reads
anything from time >= t (unit-tested).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class DynamicEvidenceMemory:

    def __init__(self, lambda_n: float, lambda_p: float):
        if lambda_n not in (0.0, 0.25, 0.5, 1.0):
            raise ValueError("lambda_n must be in {0, 0.25, 0.5, 1.0}")
        if lambda_p not in (0.0, 0.1, 0.25):
            raise ValueError("lambda_p must be in {0, 0.1, 0.25}")
        self.lambda_n = lambda_n
        self.lambda_p = lambda_p
        self.prev_ids = set()
        self.prev_embeddings = None  # (K, 384) tensor or None

    def reset(self):
        self.prev_ids = set()
        self.prev_embeddings = None

    def load_previous(self, node_ids, embeddings):
        """Set M_{t-1} from the previous snapshot's selected evidence."""
        if len(node_ids) != embeddings.shape[0]:
            raise ValueError("node_ids/embeddings length mismatch")
        self.prev_ids = set(node_ids)
        self.prev_embeddings = embeddings

    def score(self, node_ids, u, sem_nodes):
        """Return (d, novelty, persistence) as float lists aligned to inputs."""
        n = len(node_ids)
        if sem_nodes.shape[0] != n or u.shape[0] != n:
            raise ValueError("candidate tensors must align on length")

        # cos(e_i, e_j): normalize defensively — the MiniLM encoder already
        # emits L2-normalized rows, so this is identity on real features.
        sem = F.normalize(sem_nodes, dim=1)
        if self.prev_embeddings is None or self.prev_embeddings.shape[0] == 0:
            novelty = [1.0] * n
        else:
            prev = F.normalize(self.prev_embeddings, dim=1)
            cos = sem @ prev.t()  # (N, K)
            novelty = (1.0 - cos.max(dim=1).values).tolist()

        persistence = [1.0 if nid in self.prev_ids else 0.0
                       for nid in node_ids]
        dynamic = [float(u[i]) + self.lambda_n * novelty[i]
                   + self.lambda_p * persistence[i] for i in range(n)]
        return dynamic, novelty, persistence

    def current_selection(self):
        return {"node_ids": sorted(self.prev_ids),
                "embeddings": self.prev_embeddings}
