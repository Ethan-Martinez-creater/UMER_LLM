"""Integration test on a deliberately non-ideal synthetic event (delta-fix
§37): raw order != timestamp order, source is not raw index 0, two replies
share a timestamp, one parent is unresolved, and one node is in the future.
The full chain raw -> snapshot -> feature -> encoder -> selector -> memory ->
evidence -> prompt must behave: correct source text, no future leakage, fold
consistency, correct recent baseline, bounded all-current.
"""
import torch

from ..context.evidence_unit import build_evidence_units
from ..context.packer import pack_context
from ..context.token_budget import EvidenceBudgetSelector
from ..data.snapshot_builder import build_snapshot, build_source_only
from ..data.structural_features import build_snapshot_features
from ..data.temporal_split import (event_folds_to_snapshot_folds,
                                   stratified_event_folds)
from ..evaluation.baselines import select_all_current, select_evidence
from ..evaluation.leakage_scanner import scan_prompt
from ..models.causal_social_encoder import CausalSocialEncoder
from ..models.evidence_memory import DynamicEvidenceMemory
from ..models.selector import StaticUtilitySelector
from .conftest import make_event


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _adversarial_event():
    # raw order:  future_node, twin_1, twin_2, reply_ok, orphan, SOURCE
    # timestamps: source 1000 < twin_1 = twin_2 (1100, same ts) < reply_ok
    #             1400 < orphan 1500 << future_node 99999
    return make_event([
        ("future_node", "src", 1000 + 98999, "future text must never appear", 0),
        ("twin_1", "src", 1100, "twin first", 1),
        ("twin_2", "src", 1100, "twin second", 2),
        ("reply_ok", "src", 1400, "normal reply", 3),
        ("orphan", None, 1500, "orphan with no parent field", 4),
        ("src", None, 1000, "the true source claim text", 5),
    ], source_id="src")


def _sem_for(snap, seed=17):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(len(snap["node_ids"]), 384, generator=gen)


def test_adversarial_event_full_chain():
    torch.manual_seed(0)
    event = _adversarial_event()
    encoder = CausalSocialEncoder().eval()
    selector = StaticUtilitySelector().eval()
    memory = DynamicEvidenceMemory(lambda_n=0.5, lambda_p=0.1)
    budget = EvidenceBudgetSelector(_Tok(), budget=2048)

    snapshots = {}
    for cutoff in ("SOURCE_ONLY", 5, 15, 60):
        snapshots[cutoff] = build_source_only(event) \
            if cutoff == "SOURCE_ONLY" else build_snapshot(event, cutoff)

    # fold consistency through the real helpers (delta-fix §5/§6)
    labels = {event["event_id"]: event["label"]}
    event_folds = stratified_event_folds(labels, k=5, seed=3090)
    snap_keys = {(event["event_id"], c): event["event_id"]
                 for c in snapshots}
    snap_folds = event_folds_to_snapshot_folds(snap_keys, event_folds)
    assert len(set(snap_folds.values())) == 1  # every snapshot, one fold

    for cutoff, snap in snapshots.items():
        feats = build_snapshot_features(snap, _sem_for(snap))
        src_pos = feats["node_ids"].index(feats["source_id"])
        # 1. source text is bound by source_id in the snapshot's own ordering
        source_text = snap["texts"][src_pos]
        assert source_text == "the true source claim text"
        with torch.no_grad():
            node_repr, event_repr, logits = encoder(
                feats["node_feat"], feats["struct_feat"],
                feats["node_feat"].size(0))
        assert torch.isfinite(node_repr).all() and torch.isfinite(logits).all()

        if cutoff != "SOURCE_ONLY":
            units = build_evidence_units(snap)
            # 2. no future text anywhere in the chain
            leak = scan_prompt(
                pack_context(source_text, snap, units)["prompt"],
                event, snap)
            assert leak["pass"], leak
            assert all("future text" not in u["reply_text"] for u in units)

            # 3. recent baseline must rank the latest VISIBLE reply first
            _, tokens, scores = select_evidence(
                "recent_budget", snap, units,
                {"dataset": "pheme"}, budget)
            order = [units[i]["node_id"] for i in
                     sorted(range(len(units)),
                            key=lambda i: (-scores[i], units[i]["order"]))]
            max_elapsed = max(u["elapsed_seconds"] for u in units)
            first_elapsed = next(u["elapsed_seconds"] for u in units
                                 if u["node_id"] == order[0])
            assert first_elapsed == max_elapsed
            if cutoff == 60:
                assert order[0] == "orphan"  # 1500s is the latest at 1h

            # 4. all-current stays inside a tight context budget
            result = select_all_current(
                snap, units, budget, context_length=120, max_new_tokens=8,
                prompt_builder=lambda s, acc: pack_context(
                    source_text, s, acc)["prompt"],
                chat_counter=lambda p: len(p.split()))
            assert result["input_tokens"] + 8 <= 120

            # 5. same-timestamp replies keep deterministic snapshot order:
            # twin_1 (original_order 1) must precede twin_2 (original_order 2)
            ids = snap["node_ids"]
            assert ids.index("twin_1") < ids.index("twin_2")
