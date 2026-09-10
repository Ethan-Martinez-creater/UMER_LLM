"""Contract tests for the corrected E2 proxy classification (E2 fix §7–§8).

Pins the frozen proxy contract so the train/eval drift cannot regress:
- training must feed h_source (the source node representation), never
  event_repr, as the proxy's second input;
- the classifier order is [h_source ; z_sel] (classify), never
  [z_sel ; h_source] or [z_sel ; event_repr];
- soft (training) and hard (validation/baseline) paths share exactly one
  classifier entry point (proxy.classify / classify_selected);
- the semantic baseline uses true cosine similarity;
- with identical z_sel the soft and hard paths produce identical logits.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tcdscr_run_e2 as e2  # noqa: E402

from ..models.selector_proxy import SelectorProxy, classify_selected  # noqa: E402


def _setup():
    torch.manual_seed(0)
    proxy = SelectorProxy().eval()
    n = 5
    h_nodes = torch.randn(n, 768)
    h_source = torch.randn(768)
    event_repr = h_source + 100.0  # guaranteed != h_source
    u = torch.randn(n)
    return proxy, h_nodes, h_source, event_repr, u


def test_proxy_training_uses_source_repr_not_event_repr():
    proxy, h_nodes, h_source, event_repr, u = _setup()
    logits_src = proxy(h_nodes, h_source, u)[2]
    logits_evt = proxy(h_nodes, event_repr, u)[2]
    # substituting event_repr must change the output (the old bug)
    assert not torch.allclose(logits_src, logits_evt)


def test_proxy_classify_order_is_source_then_selected():
    proxy, h_nodes, h_source, _event_repr, u = _setup()
    _alpha, z_sel, logits = proxy(h_nodes, h_source, u)
    expected = proxy.head(torch.cat([h_source, z_sel], dim=-1))
    assert torch.allclose(logits, expected)
    wrong = proxy.head(torch.cat([z_sel, h_source], dim=-1))
    assert not torch.allclose(logits, wrong)  # order is frozen


def test_proxy_soft_and_hard_paths_share_same_classifier():
    proxy, h_nodes, h_source, _event_repr, u = _setup()
    alpha, z_sel, logits_soft = proxy(h_nodes, h_source, u)
    # soft path via classify_selected with alpha equals forward's logits
    logits_soft2 = classify_selected(proxy, h_source, h_nodes, alpha=alpha)
    assert torch.allclose(logits_soft, logits_soft2)
    # hard path (uniform mean) equals classify(mean)
    logits_hard = classify_selected(proxy, h_source, h_nodes)
    assert torch.allclose(logits_hard, proxy.classify(h_source,
                                                      h_nodes.mean(0)))
    # both end in the same head entry
    assert torch.allclose(logits_soft, proxy.classify(h_source, z_sel))


def test_validation_does_not_call_proxy_head_directly():
    src = inspect.getsource(e2.evaluate_arms_on_items)
    assert "proxy.head(" not in src
    assert "classify_selected(" in src
    assert "proxy.classify(" in src


def test_semantic_budget_uses_true_cosine():
    src = inspect.getsource(e2.evaluate_arms_on_items)
    assert "F.cosine_similarity(" in src
    assert "sem[cand_idx] @ sem[src]" not in src


def test_proxy_contract_detects_event_repr_substitution():
    proxy, h_nodes, h_source, event_repr, _u = _setup()
    logits_correct = proxy.classify(h_source, h_nodes.mean(0))
    logits_wrong = proxy.classify(event_repr, h_nodes.mean(0))
    # the audit itself must be able to catch the substitution
    assert not torch.allclose(logits_correct, logits_wrong)


def test_soft_hard_logits_agree_when_z_sel_equal():
    """Regression: with identical z_sel, soft and hard paths give the same
    logits (max abs diff <= 1e-7), so train/eval contract drift is caught."""
    torch.manual_seed(3)
    proxy = SelectorProxy().eval()
    h_nodes = torch.randn(5, 768)
    h_source = torch.randn(768)
    alpha_oh = torch.zeros(5)
    alpha_oh[2] = 1.0
    z_soft = (alpha_oh.unsqueeze(1) * h_nodes).sum(0)
    sel_repr = h_nodes[2:3]  # mean over one row == that row
    z_hard = sel_repr.mean(0)
    assert torch.allclose(z_soft, z_hard)
    l_soft = proxy.classify(h_source, z_soft)
    l_hard = classify_selected(proxy, h_source, sel_repr)
    assert (l_soft - l_hard).abs().max().item() <= 1e-7