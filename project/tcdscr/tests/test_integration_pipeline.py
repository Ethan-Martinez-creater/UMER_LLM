"""Integration tests (plan §29).

Synthetic events walk the full chain — raw event -> source-only/15m/1h ->
features -> encoder -> selector -> packer — and the checks required by §29:
finite tensors, correct dimensions, no future leakage, deterministic
selection, deterministic prompt. No real dataset and no LLM weights are
needed; the frozen Qwen path is exercised by the tiny smoke run.
"""
import torch

from ..context.evidence_unit import build_evidence_units
from ..context.packer import pack_context
from ..context.token_budget import EvidenceBudgetSelector
from ..data.snapshot_builder import build_snapshot, build_source_only
from ..data.structural_features import build_snapshot_features
from ..evaluation.leakage_scanner import scan_prompt
from ..models.causal_social_encoder import CausalSocialEncoder
from ..models.evidence_memory import DynamicEvidenceMemory
from ..models.selector import StaticUtilitySelector
from ..models.selector_proxy import SelectorProxy
from .conftest import make_event


def _event():
    return make_event([
        ("n0", None, 1000, "source claim text", 0),
        ("n1", "n0", 1100, "first reply", 1),
        ("n2", "n1", 1400, "second reply", 2),
        ("n3", "n0", 2400, "third reply", 3),
    ], label=1)


def _sem_for(snap, seed=5):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(len(snap["node_ids"]), 384, generator=gen)


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _run_chain(event, cutoffs=("SOURCE_ONLY", 15, 60)):
    """raw -> snapshots -> features -> encoder -> selector -> packer."""
    torch.manual_seed(0)
    encoder = CausalSocialEncoder().eval()
    selector = StaticUtilitySelector().eval()
    memory = DynamicEvidenceMemory(lambda_n=0.5, lambda_p=0.1)
    budget = EvidenceBudgetSelector(_Tok(), budget=512)

    outputs = []
    for cutoff in cutoffs:
        snap = build_source_only(event) if cutoff == "SOURCE_ONLY" \
            else build_snapshot(event, cutoff)
        feats = build_snapshot_features(snap, _sem_for(snap))
        with torch.no_grad():
            node_repr, event_repr, logits = encoder(
                feats["node_feat"], feats["struct_feat"],
                feats["node_feat"].size(0))
        src_pos = feats["node_ids"].index(feats["source_id"])
        cand = [i for i in range(len(feats["node_ids"])) if i != src_pos]
        selected_ids, evidence_tokens = [], 0
        u_cand = None
        if cand:
            mask = torch.zeros(len(cand), dtype=torch.bool)
            mask[:] = True
            u = selector(
                node_repr[cand], event_repr,
                feats["node_feat"][cand], feats["node_feat"][src_pos],
                feats["struct_feat"][cand, -3:])
            u_cand = u
            # dynamic memory refinement over the candidate pool
            cand_ids = [feats["node_ids"][i] for i in cand]
            d, _novelty, _persist = memory.score(
                cand_ids, u, feats["node_feat"][cand])
            units = build_evidence_units(snap)
            d_by_id = dict(zip(cand_ids, d))
            scores = [d_by_id[u_["node_id"]] for u_ in units]
            accepted, tokens = budget.select(units, scores)
            selected_ids = [u_["node_id"] for u_ in accepted]
            evidence_tokens = tokens
            memory.load_previous(
                selected_ids,
                feats["node_feat"][[feats["node_ids"].index(s)
                                    for s in selected_ids]]
                if selected_ids else torch.empty(0, 384))
        unit_by_id = {u_["node_id"]: u_ for u_ in build_evidence_units(snap)}
        selected_units = [unit_by_id[sid] for sid in selected_ids]
        # source text bound by source_id within the snapshot's own ordering
        packed = pack_context(snap["texts"][src_pos], snap, selected_units)
        leak = scan_prompt(packed["prompt"], event, snap)
        outputs.append({
            "cutoff": cutoff, "snapshot": snap, "feats": feats,
            "node_repr": node_repr, "event_repr": event_repr,
            "logits": logits, "u_cand": u_cand, "packed": packed,
            "leakage": leak, "selected_ids": selected_ids,
            "evidence_tokens": evidence_tokens,
        })
    return outputs


def test_integration_pheme_style_event():
    outputs = _run_chain(_event())
    # finite tensors + correct dimensions at every step
    for out in outputs:
        for name in ("node_repr", "event_repr", "logits"):
            assert torch.isfinite(out[name]).all(), name
        assert out["node_repr"].size(-1) == 768
        assert out["event_repr"].shape == (768,)
        assert out["logits"].shape == (2,)
        assert out["feats"]["node_feat"].shape[1] == 384
        assert out["feats"]["struct_feat"].shape[1] == 1024
        assert out["leakage"]["pass"], out["leakage"]
    # SOURCE_ONLY has a single node and no evidence
    assert outputs[0]["snapshot"]["num_nodes_after_cap"] == 1
    assert outputs[0]["selected_ids"] == []


def test_integration_deterministic_selection_and_prompt():
    a = _run_chain(_event())
    b = _run_chain(_event())
    for x, y in zip(a, b):
        assert x["selected_ids"] == y["selected_ids"]
        assert x["packed"]["prompt"] == y["packed"]["prompt"]
        assert x["evidence_tokens"] == y["evidence_tokens"]
        assert torch.equal(x["logits"], y["logits"])


def test_integration_memory_evolution_across_cutoffs():
    # novelty/persistence operate only on past selections and the selected
    # set evolves causally across the trajectory
    outputs = _run_chain(_event())
    assert [o["cutoff"] for o in outputs] == ["SOURCE_ONLY", 15, 60]
    assert len(outputs[1]["selected_ids"]) >= 1
    assert len(outputs[2]["selected_ids"]) >= 1
