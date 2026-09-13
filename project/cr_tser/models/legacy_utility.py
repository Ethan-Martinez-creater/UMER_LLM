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

S6 uses the **frozen StaticUtilitySelector** exactly as the TC-DSCR E2/E3
pipeline does::

    u_i = selector(h_i, event_repr, semantic_i, semantic_source, struct3_i)

The Proxy classifier is the separate legacy *classification* diagnostic used
by B2; it must never substitute for the selector's utility ranking.
"""
from __future__ import annotations

LEGACY_DATASET = "pheme"


class LegacyNotPermitted(RuntimeError):
    """Raised when the legacy TC-DSCR utility would be used off-label."""


class LegacyUnavailable(RuntimeError):
    """Raised when the frozen PHEME legacy checkpoint cannot be located."""


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
    (loaded with ``tcdscr_run_e3.load_frozen_components``). Nothing here
    trains them: every entry point runs under ``torch.no_grad`` and the
    modules are put in eval mode at construction.
    """

    def __init__(self, dataset: str, encoder, selector, proxy=None,
                 device: str = "cpu"):
        assert_legacy_dataset(dataset)
        if selector is None:
            raise LegacyUnavailable(
                "the frozen TC-DSCR StaticUtilitySelector is required for S6; "
                "it cannot be replaced by the Proxy classifier (plan §22)")
        self.dataset = dataset
        self.encoder = encoder
        self.selector = selector
        self.proxy = proxy
        self.device = device
        for module in (self.encoder, self.selector, self.proxy):
            if module is not None and hasattr(module, "eval"):
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)

    @classmethod
    def from_tcdscr(cls, dataset, e2_root, fold, seed, device="cpu",
                    e1_root=None):
        """Load the frozen PHEME static components from TC-DSCR artifacts."""
        assert_legacy_dataset(dataset)
        from tcdscr_run_e3 import load_frozen_components  # noqa: PLC0415
        e1_root = e1_root or "/data/jyz/next/llm/results/tcdscr/formal_e1"
        encoder, selector, proxy, checksums = load_frozen_components(
            dataset, fold, seed, e1_root, e2_root, device)
        instance = cls(dataset, encoder, selector, proxy, device=device)
        instance.checksums = checksums
        return instance

    # -- S6: frozen Static Utility -----------------------------------------
    def per_unit_scores(self, node_repr, event_repr, sem, struct3, source_pos,
                        candidate_indices):
        """Frozen ``u_i`` for every candidate, via the real selector path.

        Mirrors TC-DSCR's E2/E3 call
        ``selector(h_cand, event_repr, sem[cand], sem[src], struct3[cand])``.
        """
        import torch
        indices = [int(i) for i in candidate_indices
                   if int(i) != int(source_pos)]
        if not indices:
            return {}
        idx = torch.tensor(indices, dtype=torch.long,
                           device=getattr(node_repr, "device", self.device))
        with torch.no_grad():
            utility = self.selector(node_repr[idx], event_repr, sem[idx],
                                    sem[int(source_pos)], struct3[idx])
        values = utility.tolist() if hasattr(utility, "tolist") else list(utility)
        return {index: float(value) for index, value in zip(indices, values)}

    def score_items(self, items, device=None):
        """Run the frozen encoder + selector over light items (S6 scores).

        Returns ``[{node_id: u_i}]`` covering every non-source node.
        """
        device = device or self.device
        from tcdscr_run_e2 import encoder_forward_batch
        outs = encoder_forward_batch(self.encoder, list(items), device)
        results = []
        for item, (node_repr, event_repr, _logits) in zip(items, outs):
            candidates = [i for i in range(len(item["node_ids"]))
                          if i != item["source_pos"]]
            by_index = self.per_unit_scores(
                node_repr, event_repr, item["sem"].to(node_repr.device),
                item["summary"].to(node_repr.device), item["source_pos"],
                candidates)
            results.append({item["node_ids"][i]: value
                            for i, value in by_index.items()})
        return results

    # -- B2: legacy classification diagnostic (Proxy, never used for S6) ----
    def static_classification(self, h_source, sel_repr):
        """Frozen Proxy classification — B2 continuity diagnostic only."""
        import torch
        from tcdscr.models.selector_proxy import classify_selected
        if self.proxy is None:
            raise LegacyUnavailable("Proxy classifier not loaded (B2 only)")
        with torch.no_grad():
            if sel_repr is None:
                z_sel = torch.zeros(768, device=self.device)
                return self.proxy.classify(h_source, z_sel)
            return classify_selected(self.proxy, h_source, sel_repr)

    def b2_surface(self, h_source, sel_repr):
        """B2 artifact surface: static classification, explicitly diagnostic."""
        import torch
        logits = self.static_classification(h_source, sel_repr)
        probs = torch.softmax(logits.float(), dim=-1)
        return {
            "static_logits": logits.tolist(),
            "p_rumor": float(probs.reshape(-1)[0]),
            "p_nonrumor": float(probs.reshape(-1)[1]),
            "diagnostic_only": True,
            "participates_in_primary_gate": False,
            "utility_definition": "StaticUtilitySelector(h, event_repr, sem, "
                                  "sem_src, struct3)",
        }

    def fingerprint(self) -> dict:
        """Identity of the frozen components, for the verifier."""
        checksums = getattr(self, "checksums", {}) or {}
        return {
            "dataset": self.dataset,
            "mode": "inference_only",
            "trainable_parameters": 0,
            "selector_loaded": self.selector is not None,
            "proxy_loaded": self.proxy is not None,
            "s6_score_source": "StaticUtilitySelector",
            "selector_checkpoint_sha": checksums.get("selector_checkpoint_sha"),
            "encoder_checkpoint_sha": checksums.get("e2_manifest_encoder_sha"),
        }


def legacy_arm_enabled(dataset: str) -> bool:
    """S6 / B2 are enabled only for PHEME (plan §22 S6, §25)."""
    return dataset == LEGACY_DATASET


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
