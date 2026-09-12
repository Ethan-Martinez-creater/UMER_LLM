"""Frozen CR-TSER pilot configuration (plan §4–§7, §13–§16, §21, §24–§26).

Every constant that the plan freezes lives here exactly once. Research code
imports these names instead of re-typing numbers, so a drifted value fails the
verifier instead of silently changing the protocol.

Nothing in this module reads data, models or the network.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Plan §5 — event split (seed frozen by the plan)
# --------------------------------------------------------------------------
PARTITION_SEED = 7319
TRAIN_SEEDS = (7319, 7320, 7321)
BOOTSTRAP_SEED = 7319
BOOTSTRAP_ITERATIONS = 10000

SPLIT_SIZES = {
    "foundation_train": 80,
    "utility_train": 50,
    "utility_dev": 15,
    "utility_eval": 25,
}
SPLIT_TOTAL = sum(SPLIT_SIZES.values())  # 170 events per dataset

# --------------------------------------------------------------------------
# Plan §6 — temporal cutoffs (exactly three)
# --------------------------------------------------------------------------
CUTOFFS_MIN = (15, 60, 360)

# --------------------------------------------------------------------------
# Plan §7 — Semantic Reference Context (SRC)
# --------------------------------------------------------------------------
BUDGET_REF = 1024                      # canonical social-evidence tokens
SRC_COSINE_METRIC = "cosine"
CANONICAL_TOKENIZER_ID = "Qwen/Qwen3-8B"

# --------------------------------------------------------------------------
# Plan §8 — frozen readers (exactly three, no substitution)
# --------------------------------------------------------------------------
READER_KEYS = ("qwen", "glm", "internlm")
READER_MODEL_IDS = {
    "qwen": "Qwen/Qwen3-8B",
    "glm": "zai-org/glm-4-9b-chat-hf",
    "internlm": "internlm/internlm3-8b-instruct",
}
# Plan §20 — exactly three leave-one-reader-out rotations.
LORO_ROTATIONS = (
    ("qwen", "glm", "internlm"),      # train qwen+glm, hold internlm
    ("qwen", "internlm", "glm"),      # train qwen+internlm, hold glm
    ("glm", "internlm", "qwen"),      # train glm+internlm, hold qwen
)

# --------------------------------------------------------------------------
# Plan §9 — reader task and label scoring
# --------------------------------------------------------------------------
CANDIDATES = ("A", "B")
LABEL_TOKEN = {"A": "RUMOR", "B": "NON_RUMOR"}
ANSWER_MAX_NEW_TOKENS = 2              # "A"/"B" only

# --------------------------------------------------------------------------
# Plan §10 — intervention utility
# --------------------------------------------------------------------------
UTILITY_THRESHOLD = 0.05               # frozen ±0.05 neutral band
SIGN_CLASSES = ("HELPFUL", "NEUTRAL", "HARMFUL")

# --------------------------------------------------------------------------
# Plan §11 — intervention family
# --------------------------------------------------------------------------
ATOMIC_CAP = 20                        # only top-20 SRC units get labels
MAX_STRUCTURED_INTERVENTIONS = 4       # parent-child, control, subtree, control
MATCH_TOKEN_TOLERANCE = 0.20           # ±20% matched-group token cost
MATCH_DEPTH_TOLERANCE = 1              # mean depth difference ≤ 1
SUBTREE_SIZE_CAP = 5                   # cap subtree removal at earliest 5 units

# --------------------------------------------------------------------------
# Plan §13 — BiTTE (bidirectional temporal tree encoder)
# --------------------------------------------------------------------------
SEMANTIC_DIM = 384
STRUCT_SCALAR_DIM = 10
DEPTH_NORM_CAP = 20                    # depth_norm = min(depth,20)/20
BITTE_SEMANTIC_HIDDEN = 256
BITTE_STRUCT_HIDDEN = 64
BITTE_HIDDEN = 256
BITTE_LAYERS = 2
BITTE_DROPOUT = 0.1

# --------------------------------------------------------------------------
# Plan §14–§15 — atomic utility representation and shared/residual heads
# --------------------------------------------------------------------------
UTILITY_Q_DIM = 6
UTILITY_Z_DIM = BITTE_HIDDEN * 5 + UTILITY_Q_DIM      # 1286
UTILITY_PROJ_HIDDEN = 256
UTILITY_EMBED_DIM = 256
READER_EMBED_DIM = 16
SHARED_HEAD_HIDDEN = 128
RESIDUAL_HEAD_HIDDEN = 128

# --------------------------------------------------------------------------
# Plan §16 — loss and training
# --------------------------------------------------------------------------
LOSS_W_READER = 1.0
LOSS_W_SHARED = 0.5
LOSS_W_SIGN = 0.5
LOSS_W_RESID = 0.01

ADAMW_LR = 1e-3
ADAMW_WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 10
GRAD_CLIP = 1.0

# --------------------------------------------------------------------------
# Plan §21 — robust selection
# --------------------------------------------------------------------------
PILOT_BUDGET_FRACTION = 0.50           # B_pilot = floor(0.50 * Tokens(C_ref))

# --------------------------------------------------------------------------
# Plan §17 — predictor baselines
# --------------------------------------------------------------------------
# B0 text-only input: reply semantic 384 + source semantic 384 + cosine +
# token cost + context size + cutoff scalar.
TEXT_FEATURE_DIM = 2 * SEMANTIC_DIM + 4        # 772
# B1 adds the ten fixed structural/time scalars (no message passing).
SCALAR_STRUCTURE_FEATURE_DIM = TEXT_FEATURE_DIM + STRUCT_SCALAR_DIM  # 782
BASELINE_HIDDEN = 256

# --------------------------------------------------------------------------
# Plan §22 — selection arms
# --------------------------------------------------------------------------
SELECTION_ARMS = ("S0_src_full", "S1_random_tm", "S2_semantic_tm",
                  "S3a_single_a", "S3b_single_b", "S4_shared",
                  "S5_cross_reader_robust", "S6_legacy_utility_tm")
PRIMARY_ARM = "S5_cross_reader_robust"
SIMPLE_BASELINE_ARMS = ("S1_random_tm", "S2_semantic_tm")

# --------------------------------------------------------------------------
# Plan §23 — held-out reader evaluation metrics
# --------------------------------------------------------------------------
READER_EVAL_METRICS = ("Accuracy", "Macro-F1", "Rumor-F1", "Non-rumor-F1",
                       "mean_social_tokens", "median_social_tokens")

# --------------------------------------------------------------------------
# Plan §25 — pre-registered feasibility gates
# --------------------------------------------------------------------------
P1_MEAN_DISAGREEMENT_MIN = 0.10
P1_PAIR_DISAGREEMENT_MIN = 0.05
P1_PAIRS_REQUIRED = 2                  # at least 2/3 pairs

P2_EDGE_DELTA_MIN = 0.02

P3_MACRO_F1_DELTA_MIN = 0.02

P4_MEAN_DELTA_MIN = 0.01
P4_ROTATIONS_POSITIVE_REQUIRED = 2     # at least 2/3 rotations > 0
P4_WORST_ROTATION_MIN = -0.005

PHEME_MEAN_DELTA_MIN = -0.005

GATE_RECOMMENDATIONS = (
    "START_FULL_CR_TSER_METHOD_DEVELOPMENT",
    "STOP_FOR_RESEARCH_REVIEW",
    "STOP_CR_TSER",
)

VERDICT_READY = "WEIBO22_TEMPORAL_READY"
VERDICT_UNAVAILABLE = "WEIBO22_TEMPORAL_UNAVAILABLE"


@dataclass
class PilotPaths:
    """Resolved filesystem layout for one pilot run.

    Paths are supplied by the environment/CLI. The module never guesses a
    dataset location: a missing path must fail loudly rather than silently
    pick a different dataset.
    """

    weibo22_raw: str = ""
    weibo22_labels: str = ""
    pheme_raw: str = ""
    semantic_model: str = ""
    qwen_model: str = ""
    glm_model: str = ""
    internlm_model: str = ""
    canonical_tokenizer: str = ""
    out_root: str = "results/cr_tser"
    extra: dict = field(default_factory=dict)

    def reader_path(self, key: str) -> str:
        if key not in READER_KEYS:
            raise ValueError(f"unknown reader key {key!r}")
        return getattr(self, f"{key}_model")

    def reader_paths(self) -> dict:
        return {k: self.reader_path(k) for k in READER_KEYS}


_ENV_PREFIX = "CRTSER_"


def paths_from_env(env=None) -> PilotPaths:
    """Build paths from ``CRTSER_<NAME>`` environment variables.

    Unset variables stay empty strings so callers can decide whether a given
    stage needs them; stages that do need a path fail on the empty value.
    """
    env = os.environ if env is None else env
    kw = {}
    for name in ("weibo22_raw", "weibo22_labels", "pheme_raw",
                 "semantic_model", "qwen_model", "glm_model",
                 "internlm_model", "canonical_tokenizer", "out_root"):
        val = env.get(_ENV_PREFIX + name.upper())
        if val:
            kw[name] = val
    return PilotPaths(**kw)


def frozen_constants() -> dict:
    """The plan-frozen numbers the verifier checks for drift (plan §34)."""
    return {
        "partition_seed": PARTITION_SEED,
        "train_seeds": list(TRAIN_SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "split_sizes": dict(SPLIT_SIZES),
        "cutoffs_min": list(CUTOFFS_MIN),
        "budget_ref": BUDGET_REF,
        "reader_keys": list(READER_KEYS),
        "reader_model_ids": dict(READER_MODEL_IDS),
        "loro_rotations": [list(r) for r in LORO_ROTATIONS],
        "utility_threshold": UTILITY_THRESHOLD,
        "atomic_cap": ATOMIC_CAP,
        "pilot_budget_fraction": PILOT_BUDGET_FRACTION,
        "p1_mean_disagreement_min": P1_MEAN_DISAGREEMENT_MIN,
        "p1_pair_disagreement_min": P1_PAIR_DISAGREEMENT_MIN,
        "p2_edge_delta_min": P2_EDGE_DELTA_MIN,
        "p3_macro_f1_delta_min": P3_MACRO_F1_DELTA_MIN,
        "p4_mean_delta_min": P4_MEAN_DELTA_MIN,
        "p4_worst_rotation_min": P4_WORST_ROTATION_MIN,
        "pheme_mean_delta_min": PHEME_MEAN_DELTA_MIN,
    }
