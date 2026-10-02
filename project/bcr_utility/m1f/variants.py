"""M1-F strong-baseline variant definitions (the fixed S0–S4 ladder).

The ladder is pre-registered in the M1-F plan §6 and never re-derived from
``utility_eval``:

``S0`` all-NEUTRAL constant prediction (no fitting);
``S1`` training-fold class prior, majority class (no held-out information);
``S2`` source-state only — the three E3 source-state features + cutoff, the
       only reader-specific inputs that are constant inside a snapshot;
``S3`` S2 + E0 + E1 + E2 (the ZERO-TOUCH block), still without the
       evidence-familiarity C block;
``S4`` the frozen M1-E ``D5_Z_S_C`` prediction, reused verbatim.

``S2``/``S3`` are plain evidence networks from the frozen M1 grid; they add no
architecture, no head and no tuning surface.
"""
from __future__ import annotations

from ..attribution import feature_ablation as fa
from ..config import protocol as P
from ..evaluation import utility_metrics as um


class VariantRefused(RuntimeError):
    """Raised when a baseline variant violates the frozen ladder."""


_S_INDEX = [P.E3_FEATURE_NAMES.index(n) for n in P.E3_SOURCE_STATE_NAMES]


def is_trained(variant: str) -> bool:
    if variant not in P.M1F_VARIANTS:
        raise VariantRefused(f"unknown M1-F variant {variant!r}")
    return variant in P.M1F_TRAINED_VARIANTS


def variant_dim(variant: str) -> int:
    """Input width of one variant (before scaling)."""
    if variant == P.M1F_S2:
        return len(P.E3_SOURCE_STATE_NAMES) + 1
    if variant == P.M1F_S3:
        return (len(P.E0_FEATURE_NAMES) + len(P.E1_FEATURE_NAMES)
                + len(P.E2_FEATURE_NAMES) + len(P.E3_SOURCE_STATE_NAMES) + 1)
    raise VariantRefused(f"{variant} has no learned input vector")


def variant_vector(table_row: dict, reader: str, variant: str) -> list:
    """The frozen evidence vector of one ``(key, reader)`` under one variant."""
    if not is_trained(variant):
        raise VariantRefused(f"{variant} is not a trained variant")
    e3 = table_row.get("e3")
    if not e3 or reader not in e3:
        raise VariantRefused(f"{variant} needs E3 rows; none present")
    values = e3[reader]
    source_state = [values[i] for i in _S_INDEX]
    cutoff = float(table_row["cutoff"])
    if variant == P.M1F_S2:
        vector = list(source_state) + [cutoff]
    else:
        vector = list(table_row["e0e1"]) + list(table_row["e2"][reader]) \
            + list(source_state) + [cutoff]
    if len(vector) != variant_dim(variant):
        raise VariantRefused(f"{variant}: dim {len(vector)} != "
                             f"{variant_dim(variant)}")
    return vector


def build_rows(variant: str, atomic_entries, table, readers, event_ids) -> list:
    """Supervised rows of one trained variant; only ``readers`` x ``event_ids``."""
    wanted = {str(e) for e in event_ids}
    rows = []
    for entry in atomic_entries:
        if str(entry["event_id"]) not in wanted:
            continue
        features = table[entry["key"]]
        for reader in readers:
            if reader not in entry["utility"] or reader not in entry["sign"]:
                raise VariantRefused(f"{entry['key']}: reader {reader} missing "
                                     "from the atomic index")
            utility = float(entry["utility"][reader])
            rows.append({
                "key": entry["key"],
                "event_id": str(entry["event_id"]),
                "cutoff": int(features["cutoff"]),
                "reader": reader,
                "x": variant_vector(features, reader, variant),
                "u": utility,
                "s": um.SIGN_TO_INDEX[entry["sign"][reader]],
                "sign": entry["sign"][reader],
                "utility": utility,
            })
    return rows


def prior_sign(train_rows) -> str:
    """Majority gold sign of the *training* rows only (Task B, S1)."""
    if not train_rows:
        raise VariantRefused("empty training fold for the class prior")
    counts = {c: 0 for c in P.SIGN_CLASSES}
    for row in train_rows:
        if row["sign"] not in counts:
            raise VariantRefused(f"unknown sign {row['sign']!r}")
        counts[row["sign"]] += 1
    return max(P.SIGN_CLASSES, key=lambda c: (counts[c], c))


def constant_prediction(rows, sign: str, utility: float = 0.0) -> dict:
    """A constant sign prediction (S0/S1): one-hot class, constant utility."""
    if sign not in P.SIGN_CLASSES:
        raise VariantRefused(f"unknown sign {sign!r}")
    index = um.SIGN_TO_INDEX[sign]
    probs = []
    for _ in rows:
        row = [0.0] * len(P.SIGN_CLASSES)
        row[index] = 1.0
        probs.append(row)
    return {"utility": [float(utility)] * len(rows), "sign_probs": probs}


def prior_utility(train_rows) -> float:
    """Constant utility S1 predicts: the training-fold mean gold utility."""
    if not train_rows:
        raise VariantRefused("empty training fold for the utility prior")
    return sum(float(r["utility"]) for r in train_rows) / len(train_rows)


def s0_prediction(rows) -> dict:
    return constant_prediction(rows, "NEUTRAL", 0.0)


def s1_prediction(rows, train_rows) -> dict:
    return constant_prediction(rows, prior_sign(train_rows),
                               prior_utility(train_rows))


def frozen_s4_predictions(repo_root, dataset: str) -> dict:
    """The frozen M1-E ``D5_Z_S_C`` predictions, reused verbatim (plan §6)."""
    return fa.load_variant_predictions(repo_root, dataset, "D5_Z_S_C")
