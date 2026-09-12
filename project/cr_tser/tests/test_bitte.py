"""Plan §35 — BiTTE structure and padding tests."""
from __future__ import annotations

import torch

from ..config.pilot_config import BITTE_HIDDEN, SEMANTIC_DIM, STRUCT_SCALAR_DIM
from ..models.bitte import BiTTE, pack_graph_batch


def _graph(n, pairs, seed=0):
    g = torch.Generator().manual_seed(seed)
    parent_idx = torch.full((n,), -1, dtype=torch.long)
    for child, parent in pairs:
        parent_idx[child] = parent
    edges = torch.tensor(pairs, dtype=torch.long) if pairs else \
        torch.zeros((0, 2), dtype=torch.long)
    return {
        "sem": torch.randn(n, SEMANTIC_DIM, generator=g),
        "struct": torch.rand(n, STRUCT_SCALAR_DIM, generator=g),
        "parent_idx": parent_idx,
        "edges": edges,
        "source_idx": 0,
    }


def test_bitte_parent_message():
    torch.manual_seed(0)
    model = BiTTE().eval()
    g = _graph(3, [(1, 0), (2, 1)], seed=1)
    altered = {**g, "sem": g["sem"].clone(), "struct": g["struct"].clone()}
    # change ONLY the parent's (node 0) semantic vector
    altered["sem"][0] = torch.randn(
        SEMANTIC_DIM, generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        h, _ = model(**pack_graph_batch([g]))
        h_alt, _ = model(**pack_graph_batch([altered]))
    # the child (node 1) has identical own features but a different parent, so
    # its state must change -> the parent message is genuinely used
    assert not torch.allclose(h[0, 1], h_alt[0, 1], atol=1e-7)
    # and the change propagates one more hop to node 2
    assert not torch.allclose(h[0, 2], h_alt[0, 2], atol=1e-7)


def test_bitte_child_mean_message():
    torch.manual_seed(0)
    model = BiTTE().eval()
    small = _graph(3, [(1, 0), (2, 1)], seed=3)
    big = _graph(5, [(1, 0), (2, 1), (3, 1), (4, 1)], seed=3)
    # align the shared nodes explicitly: the per-node generators advance
    # differently once the row count changes, so compare identical inputs
    big["sem"][:3] = small["sem"][:3].clone()
    big["struct"][:3] = small["struct"][:3].clone()
    # duplicate node 2 twice so mean(child states) is unchanged
    big["sem"][3] = big["sem"][2].clone()
    big["sem"][4] = big["sem"][2].clone()
    big["struct"][3] = big["struct"][2].clone()
    big["struct"][4] = big["struct"][2].clone()
    with torch.no_grad():
        h_small, _ = model(**pack_graph_batch([small]))
        h_big, _ = model(**pack_graph_batch([big]))
    # mean(child states) is the same in both graphs, so node 1 must agree; a
    # sum would triple the child message and break this
    assert torch.allclose(h_small[0, 1], h_big[0, 1], atol=1e-6)


def test_bitte_padding_independent():
    torch.manual_seed(0)
    model = BiTTE().eval()
    g1 = _graph(3, [(1, 0), (2, 1)], seed=4)
    g2a = _graph(3, [(1, 0), (2, 1)], seed=5)
    g2b = _graph(7, [(i, i - 1) for i in range(1, 7)], seed=5)
    with torch.no_grad():
        ha, _ = model(**pack_graph_batch([g1, g2a]))
        hb, _ = model(**pack_graph_batch([g1, g2b]))
    # g1's output must not depend on how much padding g2 introduces
    assert torch.allclose(ha[0][:3], hb[0][:3], atol=1e-6)


def test_bitte_handles_large_tree_without_1021_signature():
    torch.manual_seed(0)
    model = BiTTE().eval()
    n = 1500
    g = _graph(n, [(i, i - 1) for i in range(1, n)], seed=6)
    with torch.no_grad():
        h, h_src = model(**pack_graph_batch([g]))
    assert h.shape == (1, n, BITTE_HIDDEN)
    assert h_src.shape == (1, BITTE_HIDDEN)
    assert 1021 not in (h.shape[-1], g["sem"].shape[-1])
    dims = {int(p.shape[-1]) for p in model.parameters() if p.dim() > 0}
    assert 1021 not in dims


def test_bitte_is_deterministic_in_eval():
    torch.manual_seed(0)
    model = BiTTE().eval()
    g = _graph(5, [(1, 0), (2, 1), (3, 1), (4, 3)], seed=7)
    with torch.no_grad():
        h1, _ = model(**pack_graph_batch([g]))
        h2, _ = model(**pack_graph_batch([g]))
    assert torch.equal(h1, h2)
