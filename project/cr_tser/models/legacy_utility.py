"""PHEME-only legacy TC-DSCR diagnostic (plan §17 B2, §22 S6).

The plan keeps the frozen TC-DSCR Static Utility as a **continuity diagnostic
for PHEME only**. This module is the single sanctioned interface to it and
enforces every boundary the plan states:

* inference only — the frozen checkpoint is loaded read-only and no parameter
  is ever trained or updated (plan §3.3: "a clean checkpoint already exists");
* PHEME only — a non-PHEME dataset raises immediately, so a Ma-Weibo
  checkpoint can never be zero-shot applied to Weibo22 (plan §3.3);
* never a primary gate input — B2/S6 are excluded from the Weibo22 gates by
  the pilot aggregator, not by this module.

The heavy lifting (checkpoint loading, feature construction) stays in the
TC-DSCR scripts; this module only owns the interface and the guards.
"""
from __future__ import annotations

LEGACY_DATASET = "pheme"


class LegacyNotPermitted(RuntimeError):
    """Raised when the legacy TC-DSCR utility would be used off-label."""


def assert_legacy_dataset(dataset: str) -> None:
    """The legacy Static Utility is PHEME-only (plan §3.3, §17 B2)."""
    if dataset != LEGACY_DATASET:
        raise LegacyNotPermitted(
            f"legacy TC-DSCR Static Utility is PHEME-only; refusing "
            f"dataset {dataset!r} (plan §3.3 forbids zero-shot reuse of a "
            "Ma-Weibo checkpoint on Weibo22)")


class LegacyPHEMEUtility:
    """Read-only wrapper around the frozen TC-DSCR Static Utility.

    ``encoder`` / ``selector`` / ``proxy`` are the frozen TC-DSCR modules
    (e.g. loaded with ``tcdscr_run_e3.load_frozen_components``). Nothing here
    trains them; :meth:`score` runs under ``torch.no_grad`` and the modules
    are put in eval mode at construction.
    """

    def __init__(self, dataset: str, encoder, proxy, device: str = "cpu"):
        assert_legacy_dataset(dataset)
        self.dataset = dataset
        self.encoder = encoder
        self.proxy = proxy
        self.device = device
        for module in (self.encoder, self.proxy):
            if module is not None and hasattr(module, "eval"):
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)

    @classmethod
    def from_tcdscr(cls, dataset, e2_root, fold, seed, device="cpu",
                    e1_root=None):
        """Load the frozen PHEME static components from TC-DSCR artifacts.

        Import is local so this module stays importable without the TC-DSCR
        scripts on the path (tests use the guard, not the loader).
        """
        assert_legacy_dataset(dataset)
        from tcdscr_run_e3 import load_frozen_components  # noqa: PLC0415
        e1_root = e1_root or "/data/jyz/next/llm/results/tcdscr/formal_e1"
        encoder, _selector, proxy, checksums = load_frozen_components(
            dataset, fold, seed, e1_root, e2_root, device)
        instance = cls(dataset, encoder, proxy, device=device)
        instance.checksums = checksums
        return instance

    def score(self, h_source, sel_repr):
        """Static utility score for one snapshot feature pair (no training)."""
        import torch
        from tcdscr.models.selector_proxy import classify_selected
        with torch.no_grad():
            if sel_repr is None:
                z_sel = torch.zeros(768, device=self.device)
                return self.proxy.classify(h_source, z_sel)
            return classify_selected(self.proxy, h_source, sel_repr)

    def per_unit_scores(self, node_repr, source_pos, candidate_indices):
        """Legacy static-utility score for **every** S6 candidate (plan §22).

        Each candidate is classified on its own through the real frozen path
        ``encoder -> [h_source ; h_i] -> proxy.head`` and scored by the
        RUMOR-vs-NON_RUMOR logit difference, which gives a reader-independent,
        gold-independent legacy utility for the S6 packing.
        """
        import torch
        from tcdscr.models.selector_proxy import classify_selected
        h_source = node_repr[source_pos]
        out = {}
        with torch.no_grad():
            for i in candidate_indices:
                if int(i) == int(source_pos):
                    continue
                logits = classify_selected(self.proxy, h_source,
                                           node_repr[i:i + 1])
                out[int(i)] = float(logits[0, 0] - logits[0, 1])
        return out

    def score_items(self, items, device=None):
        """Run the frozen encoder+proxy over light items (real feature path).

        Returns ``[{node_id: legacy_score}]`` — one dict per item, covering all
        non-source nodes of the snapshot. Requires the TC-DSCR scripts on the
        path; never touches gold labels and never trains.
        """
        device = device or self.device
        from tcdscr_run_e2 import encoder_forward_batch
        outs = encoder_forward_batch(self.encoder, list(items), device)
        results = []
        for item, (node_repr, _event_repr, _logits) in zip(items, outs):
            candidates = [i for i in range(len(item["node_ids"]))
                          if i != item["source_pos"]]
            by_index = self.per_unit_scores(node_repr, item["source_pos"],
                                            candidates)
            results.append({item["node_ids"][i]: value
                            for i, value in by_index.items()})
        return results

    def b2_surface(self, h_source, sel_repr):
        """B2 continuity diagnostic: the frozen static classification surface."""
        logits = self.score(h_source, sel_repr)
        import torch
        probs = torch.softmax(logits.float(), dim=-1).tolist()
        return {"static_logits": logits.tolist()[0] if logits.dim() == 2
                else logits.tolist(),
                "p_rumor": probs[0][0] if isinstance(probs[0], list) else probs[0],
                "diagnostic_only": True,
                "participates_in_primary_gate": False}

    def fingerprint(self) -> dict:
        """Identity of the frozen components, for the verifier."""
        checksums = getattr(self, "checksums", {}) or {}
        return {
            "dataset": self.dataset,
            "mode": "inference_only",
            "trainable_parameters": 0,
            "selector_checkpoint_sha": checksums.get("selector_checkpoint_sha"),
            "encoder_checkpoint_sha": checksums.get("e2_manifest_encoder_sha"),
        }


def legacy_arm_enabled(dataset: str) -> bool:
    """S6 / B2 are enabled only for PHEME (plan §22 S6, §25)."""
    return dataset == LEGACY_DATASET


class LegacyUnavailable(RuntimeError):
    """Raised when the frozen PHEME legacy checkpoint cannot be located."""


def build_legacy_scorer(dataset: str, device: str = "cpu"):
    """Load the frozen PHEME static utility from TC-DSCR artifacts.

    Configuration comes from ``CRTSER_LEGACY_E2_ROOT`` /
    ``CRTSER_LEGACY_E1_ROOT`` (plus optional ``CRTSER_LEGACY_FOLD`` /
    ``CRTSER_LEGACY_SEED``). Missing configuration raises rather than silently
    running S6 with an empty score set.
    """
    import os
    assert_legacy_dataset(dataset)
    e2_root = os.environ.get("CRTSER_LEGACY_E2_ROOT", "")
    if not e2_root:
        raise LegacyUnavailable(
            "CRTSER_LEGACY_E2_ROOT is not set; the PHEME legacy Static Utility "
            "cannot be loaded, so S6/B2 must not run (plan §22)")
    e1_root = os.environ.get("CRTSER_LEGACY_E1_ROOT", "") or None
    fold = int(os.environ.get("CRTSER_LEGACY_FOLD", "0"))
    seed = int(os.environ.get("CRTSER_LEGACY_SEED", "2000"))
    return LegacyPHEMEUtility.from_tcdscr(dataset, e2_root, fold, seed,
                                          device, e1_root)
