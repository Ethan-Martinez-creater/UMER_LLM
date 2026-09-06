"""TC-DSCR V2 frozen configuration schema.

Every frozen constant of the V2 formal execution plan lives here. YAML files
may only supply paths and dataset-level parameters; the frozen rules are
validated so a wrong value fails fast instead of silently changing protocol.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

# ---- Frozen protocol constants (V2 plan sections 4/6/9/11/17/18/19/21) ----
PRIMARY_CUTOFFS_MIN = (5, 15, 30, 60, 180, 360)
LATE_DIAGNOSTIC_CUTOFFS_MIN = (1440,)  # 24h: late diagnostic only
SOURCE_ONLY = "SOURCE_ONLY"

MAX_NODES = 1021           # 1021-node cap (section 11)
DEPTH_NORM_CONST = 19      # norm_depth = depth / 19, never clipped (9.2)
BIN_SECONDS = 1800         # 30-minute bins (9.3)
MAX_BINS = 480             # norm_time = time_bin / 480

SEMANTIC_DIM = 384
STRUCT_ADJ_DIM = 1021
STRUCT_SUMMARY_DIM = 3
STRUCT_FEAT_DIM = STRUCT_ADJ_DIM + STRUCT_SUMMARY_DIM  # 1024

NODE_REPR_DIM = 768        # hidden_dim 256 * 3, OriginalGraphBranch d_model
SELECTOR_INPUT_DIM = NODE_REPR_DIM * 2 + 1 + 3  # [h_i; g_t; r_i; struct3] = 1540
PROXY_INPUT_DIM = NODE_REPR_DIM * 2             # [z_sel; h_source] = 1536

TAU = 0.5                              # selector softmax temperature (18)
LOSS_WEIGHT_FID = 1.0                  # L = L_cls + 1.0 L_fid + 0.05 L_div (18.1)
LOSS_WEIGHT_DIV = 0.05

LAMBDA_N_GRID = (0.0, 0.25, 0.5, 1.0)  # dynamic memory grids (19)
LAMBDA_P_GRID = (0.0, 0.1, 0.25)

TOKEN_BUDGETS = (512, 1024, 2048)      # evidence token budgets (21)

FOLD_SEED = 3090                       # Protocol A event-level partition seed
TRAIN_SEEDS = (2000, 2001, 2002)       # Stage B formal seeds (36)

PREPROCESS_VERSION = "tcdscr_v2.0"

ALLOWED_NODE_STATUS = ("VALID", "MISSING_PARENT", "EXTERNAL_PARENT",
                       "TEMPORAL_INVALID_NODE")

LLM_LABELS = ("RUMOR", "NON_RUMOR")
INVALID_OUTPUT = "INVALID_OUTPUT"


@dataclass
class TCConfig:
    """Resolved configuration for one dataset run."""

    dataset: str                          # "pheme" | "maweibo"
    raw_dir: str
    label_file: str = ""                  # Ma-Weibo only
    semantic_model_path: str = ""
    qwen_model_path: str = ""
    cache_dir: str = ""
    output_dir: str = ""
    primary_cutoffs_min: tuple = PRIMARY_CUTOFFS_MIN
    late_diagnostic_cutoffs_min: tuple = LATE_DIAGNOSTIC_CUTOFFS_MIN
    max_nodes: int = MAX_NODES
    depth_norm_const: int = DEPTH_NORM_CONST
    bin_seconds: int = BIN_SECONDS
    max_bins: int = MAX_BINS
    fold_seed: int = FOLD_SEED
    preprocess_version: str = PREPROCESS_VERSION
    extra: dict = field(default_factory=dict)

    @property
    def all_cutoffs_min(self):
        return tuple(self.primary_cutoffs_min) + tuple(self.late_diagnostic_cutoffs_min)

    def validate(self):
        if self.dataset not in ("pheme", "maweibo"):
            raise ValueError(f"unknown dataset {self.dataset!r}")
        frozen = {
            "max_nodes": MAX_NODES,
            "depth_norm_const": DEPTH_NORM_CONST,
            "bin_seconds": BIN_SECONDS,
            "max_bins": MAX_BINS,
            "fold_seed": FOLD_SEED,
            "preprocess_version": PREPROCESS_VERSION,
        }
        for name, want in frozen.items():
            got = getattr(self, name)
            if got != want:
                raise ValueError(
                    f"frozen config {name} must be {want}, got {got}")
        if tuple(self.primary_cutoffs_min) != PRIMARY_CUTOFFS_MIN:
            raise ValueError("primary cutoffs are frozen by the V2 plan")
        if tuple(self.late_diagnostic_cutoffs_min) != LATE_DIAGNOSTIC_CUTOFFS_MIN:
            raise ValueError("late diagnostic cutoffs are frozen by the V2 plan")
        return self


def load_config(path: str) -> TCConfig:
    """Load a per-dataset YAML and validate frozen constants."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw.pop("frozen_note", None)
    cfg = TCConfig(**raw)
    return cfg.validate()


def config_from_env(dataset: str) -> TCConfig:
    """Build config from TCDSRC_<DATASET>_YAML environment override."""
    env = os.environ.get(f"TCDSRC_{dataset.upper()}_YAML")
    if env:
        return load_config(env)
    base = os.path.join(os.path.dirname(__file__), f"{dataset}.yaml")
    return load_config(base)
