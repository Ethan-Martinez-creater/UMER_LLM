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
