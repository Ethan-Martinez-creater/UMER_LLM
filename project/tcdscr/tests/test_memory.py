"""Unit tests: dynamic evidence memory (plan §28 —
test_novelty_memory_only_past, test_persistence_flag, test_dynamic_score)."""
import pytest
import torch
import torch.nn.functional as F

from ..models.evidence_memory import DynamicEvidenceMemory


def _sem(rows):
    gen = torch.Generator().manual_seed(11)
    return F.normalize(torch.rand(rows, 384, generator=gen), dim=1)


def test_novelty_memory_only_past():
    memory = DynamicEvidenceMemory(lambda_n=1.0, lambda_p=0.0)
    node_ids = ["a", "b"]
    u = torch.tensor([0.0, 0.0])
    sem = _sem(2)
    # empty memory -> novelty exactly 1 regardless of current semantics
    _dynamic, novelty, _persist = memory.score(node_ids, u, sem)
    assert novelty == [1.0, 1.0]

    # after loading M_{t-1}, novelty is 1 - max cos vs the PAST selection only
    past_sem = _sem(1)
    memory.load_previous(["p0"], past_sem)
    _dynamic, novelty, _ = memory.score(node_ids, u, sem)
    cos_a = float(sem[0] @ past_sem[0])
    assert abs(novelty[0] - (1.0 - cos_a)) < 1e-5
    # identical text to a past item -> novelty 0; different text -> the
    # exact 1 - cos value, strictly positive
    memory.load_previous(["a"], sem[0:1].clone())
    _, novelty2, _ = memory.score(node_ids, u, sem)
    assert abs(novelty2[0]) < 1e-5
    expected_b = 1.0 - float(sem[1] @ sem[0])
    assert abs(novelty2[1] - expected_b) < 1e-5
    assert novelty2[1] > 0.0


def test_persistence_flag():
    memory = DynamicEvidenceMemory(lambda_n=0.0, lambda_p=0.25)
    sem = _sem(2)
    u = torch.tensor([0.3, 0.9])
    memory.load_previous(["b"], sem[1:2])
    _dynamic, _novelty, persistence = memory.score(["a", "b"], u, sem)
    assert persistence == [0.0, 1.0]


def test_dynamic_score():
    memory = DynamicEvidenceMemory(lambda_n=0.25, lambda_p=0.1)
    sem = _sem(2)
    u = torch.tensor([1.0, 2.0])
    memory.load_previous(["b"], sem[1:2])
    dynamic, novelty, persistence = memory.score(["a", "b"], u, sem)
    # novelty for b is 0 (identical to past self), persistence for b is 1
    assert abs(novelty[1]) < 1e-5
    assert abs(dynamic[1] - (2.0 + 0.25 * novelty[1] + 0.1 * 1.0)) < 1e-9
    expected_a = 1.0 + 0.25 * novelty[0] + 0.0
    assert abs(dynamic[0] - expected_a) < 1e-9


def test_memory_grid_frozen():
    with pytest.raises(ValueError):
        DynamicEvidenceMemory(lambda_n=0.3, lambda_p=0.0)
    with pytest.raises(ValueError):
        DynamicEvidenceMemory(lambda_n=0.0, lambda_p=0.5)
