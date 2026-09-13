"""Utility training rows and grouped batching (plan §11, §14, §16, §20).

Only **atomic (I1) removals** train the utility model (plan §11). Structured
interventions exist to measure interaction and are handled by the evaluation
modules, not by the training target.

The unit of batching is the snapshot: BiTTE encodes a whole snapshot once, and
every labeled unit in it becomes a training row. A batch therefore groups
complete snapshots until at least ``batch_size`` reader-intervention rows are
collected, which keeps ``batch = 128 reader-intervention rows`` (plan §16) true
while never re-encoding a snapshot twice inside one step.

Leave-one-reader-out isolation is enforced **here**, at the entrance: a
rotation builder filters the label rows down to the two training readers, and
:func:`assert_only_readers` fails closed if a held-out reader ever survives
into a dataset object.
"""
from __future__ import annotations

import random

import torch

from ..intervention.evidence_units import evidence_key
from ..models.utility_heads import SIGN_TO_INDEX


def make_group(snapshot, sem, structural, q, unit_rows, cutoff, ctx_indices,
               dataset: str = ""):
    """One snapshot's model input plus its labeled atomic unit rows.

    ``sem`` (N,384), ``structural`` (N,10), ``q`` (N,6). ``unit_rows`` entries
    carry ``unit_index`` (snapshot position), ``reader``, ``target`` (utility),
    ``sign`` (HELPFUL/NEUTRAL/HARMFUL) and correctness transitions.

    Every row receives the canonical ``unit_key`` (dataset/event/cutoff/node)
    so predictors, selectors and the verifier share one identity contract.
    """
    node_ids = snapshot["node_ids"]
    pos = {nid: i for i, nid in enumerate(node_ids)}
    parent_idx = []
    for pid in snapshot["parent_ids"]:
        parent_idx.append(pos[pid] if pid is not None and pid in pos else -1)
    edges = torch.tensor(snapshot["edge_index"], dtype=torch.long) \
        if snapshot["edge_index"] else torch.zeros((0, 2), dtype=torch.long)
    rows = []
    for row in unit_rows:
        index = int(row["unit_index"])
        rows.append({**row,
                     "unit_key": evidence_key(dataset, snapshot["event_id"],
                                              cutoff, node_ids[index])})
    return {
        "dataset": dataset,
        "event_id": snapshot["event_id"],
        "cutoff": snapshot["cutoff_minutes"],
        "sem": sem,
        "struct": structural,
        "q": q,
        "parent_idx": torch.tensor(parent_idx, dtype=torch.long),
        "edges": edges,
        "source_idx": pos[snapshot["source_id"]],
        "ctx_indices": sorted(set(ctx_indices)),
        "unit_rows": rows,
    }


def group_as_graph(group):
    """The subset :func:`bitte.pack_graph_batch` needs."""
    return {"sem": group["sem"], "struct": group["struct"],
            "parent_idx": group["parent_idx"], "edges": group["edges"],
            "source_idx": group["source_idx"]}


def row_sign_index(sign: str) -> int:
    return SIGN_TO_INDEX[sign]


# --------------------------------------------------------------------------
# Leave-one-reader-out isolation (plan §20, §33)
# --------------------------------------------------------------------------
def filter_rows_for_readers(rows, reader_keys, context: str = "rotation"):
    """Split rows into ``(kept, dropped)`` by reader membership.

    Rotation construction uses this to *explicitly* read only the two training
    readers out of a three-reader cache; the returned ``dropped`` count is
    reported so a silent, unintended drop would be visible.
    """
    allowed = set(reader_keys)
    kept, dropped = [], []
    for row in rows:
        (kept if row.get("reader") in allowed else dropped).append(row)
    return kept, dropped


def assert_only_readers(rows, reader_keys, context: str = "dataset"):
    """Fail closed if any row belongs to a reader outside ``reader_keys``."""
    allowed = set(reader_keys)
    offenders = sorted({row.get("reader") for row in rows
                        if row.get("reader") not in allowed})
    if offenders:
        raise ValueError(
            f"{context}: held-out reader labels present: {offenders}; "
            "training data must contain only the rotation's training readers")


def assert_groups_only_readers(groups, reader_keys, context="dataset"):
    """Same check one level up, over every group's unit rows."""
    rows = [row for g in groups for row in g["unit_rows"]]
    assert_only_readers(rows, reader_keys, context=context)


class UtilityDataset:
    """Snapshot-grouped utility rows for one reader-holdout rotation."""

    def __init__(self, groups, reader_keys, reader_index_map):
        self.groups = list(groups)
        self.reader_keys = list(reader_keys)
        self.reader_index_map = dict(reader_index_map)
        self.n_rows = sum(len(g["unit_rows"]) for g in self.groups)
        assert_groups_only_readers(self.groups, self.reader_keys,
                                   context="UtilityDataset")

    def __len__(self):
        return len(self.groups)

    def iter_batches(self, batch_size: int, seed: int = 0, shuffle: bool = True):
        """Yield batches of complete groups totalling ≥ ``batch_size`` rows."""
        order = list(range(len(self.groups)))
        if shuffle:
            random.Random(seed).shuffle(order)
        batch, rows = [], 0
        for idx in order:
            batch.append(self.groups[idx])
            rows += len(self.groups[idx]["unit_rows"])
            if rows >= batch_size:
                yield batch
                batch, rows = [], 0
        if batch:
            yield batch

    def reader_rows(self, groups):
        """Flatten groups into model-ready rows for the training readers."""
        out = []
        for g in groups:
            for row in g["unit_rows"]:
                if row["reader"] not in self.reader_keys:
                    raise ValueError(
                        f"reader {row['reader']!r} is not a training reader; "
                        "held-out labels must never reach the training batch")
                out.append({**row, "group": g})
        return out
