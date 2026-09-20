"""Frozen ``bcr_v1`` protocol constants (plan §11, §19, §20, §21, §22, §30).

Every number the M0 round freezes lives here exactly once, so a drifted value
fails the verifier instead of silently changing the protocol.

Two kinds of constants are kept apart on purpose:

* **BCR-owned** values (probe sizes, seed, namespace) — new protocol surface;
* **inherited** values (utility threshold, cutoffs, reader panel) — copied from
  the frozen CR-TSER pilot because BCR's utility target is defined to be the
  same atomic removal utility (plan §5). They are re-declared here rather than
  imported so a future CR-TSER edit cannot move a BCR constant, and the
  verifier asserts the two declarations still agree.

BCR does **not** inherit CR-TSER's structured-interaction hypothesis: the
I2–I5 groups are explicitly named as non-supervision (plan §5, §12).

Nothing in this module reads data, models or the network.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
PROTOCOL_VERSION = "bcr_v1"
#: The frozen CR-TSER commit this research line starts from.
BASELINE_COMMIT = "518a821d8d839fb7857be99acb91fde71b80c846"

PRIMARY_DATASET = "maweibo"
SECONDARY_DATASET = "pheme"
DATASETS = (PRIMARY_DATASET, SECONDARY_DATASET)

# --------------------------------------------------------------------------
# Inherited utility semantics (plan §5). Same definition, same numbers.
# --------------------------------------------------------------------------
UTILITY_THRESHOLD = 0.05
SIGN_CLASSES = ("HELPFUL", "NEUTRAL", "HARMFUL")
CUTOFFS_MIN = (15, 60, 360)
ATOMIC_INTERVENTION_TYPE = "I1_atomic"
BASE_INTERVENTION_TYPE = "I0_base"

#: Structured CR-TSER interventions. They are readable as historical evidence
#: but are **never** BCR supervision (plan §5, §12).
NON_SUPERVISION_INTERVENTION_TYPES = (
    "I2_parent_child", "I3_matched_nonadjacent", "I4_subtree",
    "I5_matched_disconnected",
)

READER_KEYS = ("qwen", "mistral", "internlm")
READER_MODEL_IDS = {
    "qwen": "Qwen/Qwen3-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "internlm": "internlm/internlm3-8b-instruct",
}

# --------------------------------------------------------------------------
# Statistics (plan §10.4)
# --------------------------------------------------------------------------
SEED = 7319
BOOTSTRAP_SEED = 7319
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_UNIT = "event"

# --------------------------------------------------------------------------
# Namespaces (plan §2, §3)
# --------------------------------------------------------------------------
HISTORICAL_RESULTS_ROOT = "results/cr_tser_v2r1"
#: Permanently read-only. No BCR code may write below these roots.
HISTORICAL_NAMESPACES = ("results/cr_tser", "results/cr_tser_v2",
                         "results/cr_tser_v2r1")
RESULTS_ROOT = "results/bcr_utility_v1"
BOOTSTRAP_DIRNAME = "bootstrap"

#: M0 artifact names (plan §11 "M0 outputs").
HISTORICAL_IDENTITY_FILENAME = "historical_identity.json"
ATOMIC_INDEX_FILENAME = "atomic_index.json"
GEOMETRY_DIAGNOSTIC_FILENAME = "geometry_diagnostic.json"
PROBE_MANIFEST_FILENAME = "probe_manifest.json"
#: Data-side input to the frozen probe manifest (one entry per foundation
#: event and cutoff; produced on SERVER because it needs the raw datasets).
PROBE_AVAILABILITY_FILENAME = "probe_availability.json"
REPORT_FILENAME = "M0_REPORT.md"

# --------------------------------------------------------------------------
# Frozen historical inputs (plan §2)
# --------------------------------------------------------------------------
#: ``utility_labels/<dataset>/labels.jsonl`` digests as re-verified on SERVER
#: for this round. Raw bytes (the caches are not git-tracked, so no line-ending
#: normalisation applies).
FROZEN_HISTORICAL_LABELS = {
    "maweibo": {
        "rel_path": "utility_labels/maweibo/labels.jsonl",
        "sha256": "6c6591eaa76451c45917e97bdedf493cddcc06261a98ec8c8ff1eaab065646c3",
        "bytes": 16015182,
        "rows": 9606,
    },
    "pheme": {
        "rel_path": "utility_labels/pheme/labels.jsonl",
        "sha256": "773bee3e98d8d0f0ffc521bb9024839beeb64d2d8c2572f9f8a07dbcfff4ec15",
        "bytes": 14457787,
        "rows": 8637,
    },
}

#: Rows per reader in the same caches. Recorded so the importer can check the
#: reader panel instead of trusting a total.
FROZEN_HISTORICAL_ROWS_PER_READER = {
    "maweibo": {"qwen": 3202, "mistral": 3202, "internlm": 3202},
    "pheme": {"qwen": 2879, "mistral": 2879, "internlm": 2879},
}

#: Numbers the M0 completion gate pins (plan §30). ``atomic_keys`` counts the
#: distinct ``(event, cutoff, reply node)`` identities carried by
#: ``I1_atomic`` rows; with the frozen reader panel that equals the number of
#: I1 rows per reader.
FROZEN_ATOMIC_KEYS = {"maweibo": 2549, "pheme": 2107}

#: Canonical (LF-normalised) digests of the frozen v2r1 manifests, so the
#: historical bootstrap can prove it imported the same manifests the CR-TSER
#: line closed on.
FROZEN_HISTORICAL_MANIFEST_SHA256 = {
    "manifests/maweibo/source.json":
        "80b07954ce3199c57cb25e7ca11b07115dd0e20a84787b97effc1cfed291b783",
    "manifests/maweibo/event_split.json":
        "24e18ed10954e8387c49d9a119e78934e2c8d33ccb582aa62e4f2cf201b8dde2",
    "manifests/maweibo/hashes.json":
        "f106af7a60b5cb76fd338a4bea53c56a9de79ee700c43a2525c848384d7d7045",
    "manifests/maweibo/snapshot_manifest.jsonl":
        "611cb9c6ca43afbaec8a0eba10d71c1fcc4dcb9af3ebbd7cd90a2660a2f434dc",
    "manifests/maweibo/intervention_manifest.jsonl":
        "f37c3ffcb07e8e0142a082326ec405417c8f99399a1b09fd907cfaba9b680782",
    "manifests/pheme/source.json":
        "1a85a6d9e1f1da9ff7404d3231664463a38b5db444a57bb0f0292876b5b0e363",
    "manifests/pheme/event_split.json":
        "f4a2a1cb5a2d81c84eb394fec45373b261c436cc43822a79d5c1a41dcbada385",
    "manifests/pheme/hashes.json":
        "a3c46403d9173bdcacc00604706ecd4586f8e80b5a97e5d8992f91a373fe9cd9",
    "manifests/pheme/snapshot_manifest.jsonl":
        "8c2ec0462a1fbf9ff681d237fb9e57df09afd9af5c7e402779c2defe8812315f",
    "manifests/pheme/intervention_manifest.jsonl":
        "a0c0ad7abb5c8132ecb9483143ad8c70bc6836568a183e32797a8af73201d2e3",
}

#: Historical split sizes the probe manifest must stay out of (plan §6.1).
HISTORICAL_SPLIT_NAMES = ("foundation_train", "utility_train", "utility_dev",
                          "utility_eval")
PROBE_SOURCE_SPLIT = "foundation_train"
NON_PROBE_SPLITS = ("utility_train", "utility_dev", "utility_eval")

# --------------------------------------------------------------------------
# Probe protocol (plan §6.1, §6.2)
# --------------------------------------------------------------------------
PROBE_SEED = 7319
PROBE_EVENTS_PER_DATASET = 36
PROBE_EVENTS_PER_CUTOFF = 12
PROBE_CONTEXTS = ("P0", "P1", "P2", "P3")
PROBE_ITEMS_TOTAL = 2 * PROBE_EVENTS_PER_DATASET          # 72
#: An event assigned to a cutoff must have at least one valid visible evidence
#: unit there, and the SRC-driven contexts P1–P3 need at least one
#: SRC-selected unit (plan §6.1, §6.2).
PROBE_MIN_VISIBLE_UNITS = 1
PROBE_MIN_SRC_SELECTED = 1
#: ``P1`` is the globally highest-relevance visible unit; ``P2`` is the lowest
#: relevance unit among the SRC-selected ones; ``P3`` is the full SRC.
PROBE_CONTEXT_REQUIREMENTS = {
    "P0": "source only",
    "P1": "source + highest semantic-relevance valid evidence unit",
    "P2": "source + lowest semantic-relevance unit among SRC-selected units",
    "P3": "source + full SRC under the frozen 1024 canonical-token budget",
}

# --------------------------------------------------------------------------
# Reused CR-TSER dependencies (plan §3)
# --------------------------------------------------------------------------
#: Read-only imports. The historical-identity artifact records each file's
#: canonical digest so a later CR-TSER edit cannot silently change the shared
#: scientific contract.
REUSED_CR_TSER_MODULES = (
    "project/cr_tser/config/pilot_config.py",
    "project/cr_tser/data/snapshot_bridge.py",
    "project/cr_tser/intervention/evidence_units.py",
    "project/cr_tser/intervention/semantic_reference.py",
    "project/cr_tser/evaluation/bootstrap.py",
    "project/cr_tser/readers/sequence_scorer.py",
    "project/cr_tser/readers/base_reader.py",
    "project/cr_tser/models/utility_heads.py",
    "scripts/cr_tser_common.py",
)

#: Values BCR inherits from the frozen pilot; the verifier asserts the
#: declarations still agree (plan §3: reuse never means silently diverging).
INHERITED_FROM_CR_TSER = {
    "utility_threshold": UTILITY_THRESHOLD,
    "cutoffs_min": list(CUTOFFS_MIN),
    "reader_keys": list(READER_KEYS),
    "reader_model_ids": dict(READER_MODEL_IDS),
    "atomic_cap": 20,
    "budget_ref": 1024,
    "semantic_dim": 384,
    "canonical_tokenizer_id": "Qwen/Qwen3-8B",
}

#: Stage names that M0 must not have executed (plan §11 stop rule, §30).
FORBIDDEN_M0_ARTIFACT_DIRS = ("probe_responses", "reader_probes",
                              "utility_labels", "fingerprints", "models",
                              "predictions", "transfer")

# --------------------------------------------------------------------------
# M1 — three-reader mechanism pilot (M1 plan §5, §7–§17)
# --------------------------------------------------------------------------
M1_DIRNAME = "m1"
M1_PROBE_RESPONSES_FILENAME = "probe_responses.jsonl"
M1_PROBE_AUDIT_FILENAME = "probe_audit.json"
M1_RAW_VECTORS_FILENAME = "raw_vectors.json"
M1_FINGERPRINTS_FILENAME = "fingerprints.json"
M1_FINGERPRINT_AUDIT_FILENAME = "fingerprint_audit.json"
M1_FEATURE_AUDIT_FILENAME = "feature_audit.json"
M1_EVALUATION_FILENAME = "evaluation.json"
M1_PREDICTIONS_FILENAME = "predictions.jsonl"
M1_GATE_FILENAME = "gate.json"
M1_VERDICT_FILENAME = "M1_VERDICT.json"
M1_REPORT_FILENAME = "M1_REPORT.md"

#: The one frozen NLI extractor (M1 plan §5). No substitution is allowed.
NLI_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NLI_SERVER_PATH = "/data/jyz/next/llm/model/mdeberta-v3-base-mnli-xnli"
NLI_LABELS = ("entailment", "neutral", "contradiction")

#: Probe responses (M1 plan §7). Exactly 72 items x 4 contexts per reader.
PROBE_CONTEXT_EVALS_PER_READER = PROBE_ITEMS_TOTAL * len(PROBE_CONTEXTS)  # 288
PROBE_TOTAL_EVALS = len(READER_KEYS) * PROBE_CONTEXT_EVALS_PER_READER     # 864

#: Compact fingerprint layout (M1 plan §8): one group per
#: dataset x cutoff x context; P1/P2/P3 groups carry the P0-relative block.
FINGERPRINT_BASE_METRICS = (
    "mean_signed_margin", "mean_abs_margin", "std_abs_margin",
    "mean_entropy", "mean_log1p_reader_prompt_tokens",
    "mean_log1p_canonical_qwen_tokens",
)
FINGERPRINT_DELTA_METRICS = (
    "mean_delta_margin", "mean_abs_delta_margin", "flip_rate",
    "mean_delta_entropy",
)
FINGERPRINT_DIM = (
    len(DATASETS) * len(CUTOFFS_MIN) * len(PROBE_CONTEXTS)
    * len(FINGERPRINT_BASE_METRICS)
    + len(DATASETS) * len(CUTOFFS_MIN) * (len(PROBE_CONTEXTS) - 1)
    * len(FINGERPRINT_DELTA_METRICS)
)  # 24*6 + 18*4 = 216

#: E0 reader-agnostic evidence features (M1 plan §9).
E0_FEATURE_NAMES = (
    "cos_reply_source", "cos_parent_source", "cos_reply_parent",
    "src_relevance", "src_rank_percentile", "canonical_token_cost",
    "reply_chars", "parent_chars", "combined_chars",
    "elapsed_seconds", "cutoff_minutes",
)
#: E1 frozen NLI features: three text pairs x three NLI labels.
E1_PAIRS = ("source_reply", "source_parent", "reply_parent")
E1_FEATURE_NAMES = tuple(f"nli_{pair}_{label}"
                         for pair in E1_PAIRS for label in NLI_LABELS)
#: E2 tokenizer-only reader compatibility features (per reader).
E2_FEATURE_NAMES = (
    "reply_tokens", "parent_tokens", "unit_tokens",
    "canonical_unit_tokens", "reply_frac", "unit_vs_canonical",
)
#: E3 LIGHT-TOUCH reader-forward compatibility features (per key x reader;
#: M1 plan §17 — generated only after a ZERO-TOUCH failure). The two
#: source-only entries are defined per (event, cutoff, reader) and broadcast
#: to that snapshot's evidence keys. ``nll_gap`` is
#: ``evidence_nll_per_token - conditional_evidence_nll_per_token``: how much
#: conditioning on the source lowers the evidence NLL.
E3_FEATURE_NAMES = (
    "source_only_margin", "source_only_entropy", "source_nll",
    "evidence_nll_per_token", "conditional_evidence_nll_per_token",
    "nll_gap",
)
#: B1 control: the ten frozen CR-TSER structural/time scalars (reused).
B1_STRUCT_NAMES = (
    "depth_norm", "child_count_norm", "degree_norm", "subtree_size_norm",
    "sibling_count_norm", "is_source_child", "is_leaf", "elapsed_norm",
    "parent_lag_norm", "arrival_rank",
)

# --------------------------------------------------------------------------
# M1 models (M1 plan §10–§11)
# --------------------------------------------------------------------------
MODEL_B0 = "B0_evidence_only"
MODEL_B1 = "B1_structure_time_control"
MODEL_B2 = "B2_reader_id_diagnostic"
MODEL_B3 = "B3_nearest_reader_transfer"
MODEL_B4 = "B4_zero_touch_bcr"
MODEL_B5 = "B5_light_touch_bcr"
ZERO_TOUCH_MODELS = (MODEL_B0, MODEL_B1, MODEL_B3, MODEL_B4)
PRIMARY_COMPARISON = (MODEL_B4, MODEL_B0)

B4_EMBED_DIM = 32
B4_MAX_PARAMETERS = 250000

HUBER_DELTA = 0.05
LAMBDA_SIGN = 1.0
MODEL_SEEDS = (7319, 17319, 27319)
MODEL_GRID = {
    "hidden": (32,),
    "dropout": (0.0, 0.1),
    "lr": (1e-3, 3e-4),
    "weight_decay": (0.0, 1e-4),
}
#: Engineering defaults (not protocol surface): small-MLP training loop.
TRAIN_BATCH_SIZE = 128
TRAIN_MAX_EPOCHS = 100
TRAIN_PATIENCE = 10
TRAIN_GRAD_CLIP = 1.0

#: Leave-one-reader-out rotations (M1 plan §12). ``(train, train, hold)`` and
#: the hold order follows the approved sequence qwen, mistral, internlm.
LORO_ROTATIONS = (
    ("mistral", "internlm", "qwen"),
    ("qwen", "internlm", "mistral"),
    ("qwen", "mistral", "internlm"),
)

# --------------------------------------------------------------------------
# M1 primary gate (M1 plan §15). Ma-Weibo only; PHEME is diagnostic-only.
# --------------------------------------------------------------------------
GATE_MEAN_DELTA_MIN = 0.03
GATE_POSITIVE_READERS_MIN = 2
GATE_WORST_READER_MIN = -0.05
GATE_CI_ALPHA = 0.05
SECONDARY_HARMFUL_AUPRC_MIN = -0.02

M1_OUTCOMES = ("M1_FULL_GO", "M1_CONDITIONAL_GO", "M1_NO_GO",
               "IMPLEMENTATION_BLOCKED", "INFRASTRUCTURE_PAUSE")

#: Fields that must never appear in a probe response or fingerprint row —
#: utility supervision, gold labels and reader identity are all forbidden
#: fingerprint inputs (M1 plan §7, §8).
PROBE_FORBIDDEN_FIELDS = ("utility", "sign", "gold", "label", "helpful",
                          "harmful", "correct", "reader_id_embedding")


def bootstrap_dir(repo_root) -> str:
    return os.path.join(str(repo_root), RESULTS_ROOT, BOOTSTRAP_DIRNAME)


def historical_root(repo_root) -> str:
    return os.path.join(str(repo_root), HISTORICAL_RESULTS_ROOT)


def result_path(repo_root, filename: str) -> str:
    return os.path.join(bootstrap_dir(repo_root), filename)


def m1_dir(repo_root) -> str:
    return os.path.join(str(repo_root), RESULTS_ROOT, M1_DIRNAME)


def m1_path(repo_root, *parts) -> str:
    return os.path.join(m1_dir(repo_root), *parts)


# --------------------------------------------------------------------------
# M1-E — Conditional-GO attribution audit (post-hoc diagnostic only)
# --------------------------------------------------------------------------
M1E_DIRNAME = "m1e"
M1E_ATTRIBUTION_DIRNAME = "attribution"
M1E_EVIDENCE_PINS_FILENAME = "m1_evidence_pins.json"
M1E_SHIFT_FILENAME = "dataset_shift.json"
M1E_CONCENTRATION_FILENAME = "concentration.json"
M1E_B3_CONTRACT_FILENAME = "b3_contract.json"
M1E_VERDICT_FILENAME = "M1E_VERDICT.json"
M1E_REPORT_FILENAME = "M1E_REPORT.md"

#: The frozen M1 verdict this audit must find unchanged.
M1E_EXPECTED_M1_VERDICT = "M1_CONDITIONAL_GO"

#: Frozen feature groups of the B5 input (M1 plan §17 / M1-E Task B).
#: ``Z`` = ZERO-TOUCH evidence compatibility (E0+E1+E2),
#: ``F`` = behavioral fingerprint, ``S`` = source-state E3,
#: ``C`` = evidence-familiarity E3.
E3_SOURCE_STATE_NAMES = ("source_only_margin", "source_only_entropy",
                         "source_nll")
E3_EVIDENCE_FAMILIARITY_NAMES = ("evidence_nll_per_token",
                                 "conditional_evidence_nll_per_token",
                                 "nll_gap")

#: The pre-registered diagnostic variants. ``D1``/``D4`` are the frozen
#: B4/B5 themselves and are reused, never redefined.
ATTRIBUTION_VARIANTS = ("D0_Z", "D1_Z_F", "D2_Z_F_S", "D3_Z_F_C",
                        "D4_Z_F_S_C", "D5_Z_S_C")
ATTRIBUTION_FINGERPRINT_VARIANTS = ("D1_Z_F", "D2_Z_F_S", "D3_Z_F_C",
                                    "D4_Z_F_S_C")
ATTRIBUTION_NO_FINGERPRINT_VARIANTS = ("D0_Z", "D5_Z_S_C")
#: Variants whose result is the already-frozen M1 model output.
ATTRIBUTION_REUSED = {"D1_Z_F": MODEL_B4, "D4_Z_F_S_C": MODEL_B5}
#: Variants this audit must train itself.
ATTRIBUTION_TRAINED = ("D0_Z", "D2_Z_F_S", "D3_Z_F_C", "D5_Z_S_C")

#: Artifact names that must not appear under ``m1e/`` (no reader inference,
#: no labels, no next-stage artifacts).
M1E_FORBIDDEN_ARTIFACT_DIRS = ("probe_responses", "reader_probes",
                               "utility_labels", "fingerprints", "models",
                               "m2", "m2_pilot", "phi", "gemma")


def m1e_dir(repo_root) -> str:
    return os.path.join(str(repo_root), RESULTS_ROOT, M1E_DIRNAME)


def m1e_path(repo_root, *parts) -> str:
    return os.path.join(m1e_dir(repo_root), *parts)


def frozen_constants() -> dict:
    """The numbers the M0 verifier checks for drift."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "baseline_commit": BASELINE_COMMIT,
        "primary_dataset": PRIMARY_DATASET,
        "secondary_dataset": SECONDARY_DATASET,
        "datasets": list(DATASETS),
        "utility_threshold": UTILITY_THRESHOLD,
        "cutoffs_min": list(CUTOFFS_MIN),
        "sign_classes": list(SIGN_CLASSES),
        "seed": SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_unit": BOOTSTRAP_UNIT,
        "reader_keys": list(READER_KEYS),
        "reader_model_ids": dict(READER_MODEL_IDS),
        "atomic_intervention_type": ATOMIC_INTERVENTION_TYPE,
        "non_supervision_intervention_types":
            list(NON_SUPERVISION_INTERVENTION_TYPES),
        "frozen_atomic_keys": dict(FROZEN_ATOMIC_KEYS),
        "probe_seed": PROBE_SEED,
        "probe_events_per_dataset": PROBE_EVENTS_PER_DATASET,
        "probe_events_per_cutoff": PROBE_EVENTS_PER_CUTOFF,
        "probe_items_total": PROBE_ITEMS_TOTAL,
        "probe_contexts": list(PROBE_CONTEXTS),
        "probe_source_split": PROBE_SOURCE_SPLIT,
        "non_probe_splits": list(NON_PROBE_SPLITS),
        "results_root": RESULTS_ROOT,
        "historical_results_root": HISTORICAL_RESULTS_ROOT,
    }


def canonical_bytes(data: bytes) -> bytes:
    """CRLF -> LF so identity does not depend on the checkout's line endings."""
    return data.replace(b"\r\n", b"\n")
