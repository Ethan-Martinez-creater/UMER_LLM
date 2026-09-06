"""Unit tests: static utility selector (plan §28 — test_selector_shape,
test_no_source_in_candidate_pool)."""
import torch

from ..models.selector import StaticUtilitySelector


def _inputs(n=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return {
        "h_nodes": torch.randn(n, 768, generator=gen),
        "event_repr": torch.randn(768, generator=gen),
        "semantic_nodes": torch.randn(n, 384, generator=gen),
        "semantic_source": torch.randn(384, generator=gen),
        "struct3": torch.rand(n, 3, generator=gen),
    }


def test_selector_shape():
    selector = StaticUtilitySelector()
    ctx = _inputs(6)
    u = selector(**ctx)
    assert u.shape == (6,)
    assert torch.isfinite(u).all()


def test_no_source_in_candidate_pool():
    # rows [0..4] are replies; row 5 is the source, masked out. Even when the
    # source row is physically present in the tensors, a masked selector can
    # never rank it as a candidate (§17: source never participates).
    selector = StaticUtilitySelector()
    torch.manual_seed(1)
    n = 6
    inputs = _inputs(n, seed=3)
    mask = torch.ones(n, dtype=torch.bool)
    mask[5] = False
    u = selector(candidate_mask=mask, **inputs)
    assert u[5].item() == float("-inf")
    assert torch.isfinite(u[:5]).all()
    # softmax over u gives the source exactly zero probability
    alpha = torch.softmax(u, dim=0)
    assert alpha[5].item() == 0.0
    assert abs(alpha[:5].sum().item() - 1.0) < 1e-6


def test_selector_dims_frozen():
    selector = StaticUtilitySelector()
    ln = selector.scorer[0]
    assert ln.normalized_shape == (1540,)
    assert selector.scorer[1].in_features == 1540
    assert selector.scorer[1].out_features == 256
    assert selector.scorer[4].out_features == 1
