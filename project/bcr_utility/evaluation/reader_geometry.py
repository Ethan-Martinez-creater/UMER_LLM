"""Reader-utility geometry diagnostics (plan §11.6).

M0 asks a descriptive question: what is the shape of the imported utility
records across the three frozen readers? Following the literature boundary in
plan §1.2 this module keeps **four things apart** and refuses to let one stand
in for another:

``activity``
    how often is an evidence unit active for a reader at all (``|u| >= 0.05``
    or a correctness transition — the frozen CR-TSER definition, imported
    read-only);
``ordinal_geometry``
    do the readers *rank* evidence by magnitude the same way;
``signed_direction_geometry``
    do they agree on the *sign* (help / neutral / harm) — reported separately
    and with the full 3x3 contingency, because stable ordinal geometry does
    **not** imply cross-reader help/harm transfer;
``reader_evidence_interaction``
    a two-way sum-of-squares decomposition of the utility matrix
    (keys x readers) into reader, evidence and interaction shares, with an
    event-level bootstrap on the interaction share.

All intervals are event-level bootstraps with the frozen
``iterations``/``seed``/unit. The resampling draws follow the same
``random.Random(seed).randrange(n)`` sequence as
``cr_tser.evaluation.bootstrap.paired_event_bootstrap``, so a BCR interval and
a CR-TSER interval over the same payload resample the same events.

These results are **diagnostic only**: M0 produces no GO/NO-GO.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

from ..config import protocol as P

try:  # reused read-only from the frozen CR-TSER implementation
    from cr_tser.evaluation.bootstrap import mean, spearman_rank
    from cr_tser.evaluation.heterogeneity import is_active
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "BCR reuses cr_tser.evaluation.bootstrap / .heterogeneity read-only"
    ) from exc

SIGN_INDEX = {"HARMFUL": -1, "NEUTRAL": 0, "HELPFUL": 1}
SIGN_NAMES = {-1: "HARMFUL", 0: "NEUTRAL", 1: "HELPFUL"}

ORDINAL_CAVEAT = (
    "Ordinal reader geometry describes how similarly two readers rank "
    "evidence magnitude. A stable ordinal geometry does not by itself "
    "establish that a signed help/harm intervention transfers between "
    "readers, so the signed direction is reported separately (plan §1.2)."
)
SIGNED_CAVEAT = (
    "Signed direction is the quantity the BCR research question is about; it "
    "is reported independently of the ordinal geometry and with the full 3x3 "
    "sign contingency, not as a single correlation."
)


def _json_number(value):
    """JSON-safe float (``None`` instead of NaN/Inf)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _pearson(xs, ys) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return float("nan")
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = math.sqrt(float((xs ** 2).sum()) * float((ys ** 2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xs * ys).sum() / denom)


# --------------------------------------------------------------------------
# bootstrap machinery
# --------------------------------------------------------------------------
def bootstrap_draws(n_events: int, iterations: int, seed: int):
    """The frozen event resampling sequence.

    Identical to ``random.Random(seed)`` + ``randrange(n)`` repeated ``n``
    times, which is what ``paired_event_bootstrap`` does; kept here so the
    vectorised statistics below resample exactly the same events.
    """
    rng = random.Random(seed)
    return [[rng.randrange(n_events) for _ in range(n_events)]
            for _ in range(iterations)]


class Rows:
    """Global row indices of one statistic, grouped by sampling event.

    ``per_event`` holds, for every event, the rows that belong to it; events
    with no qualifying row carry an empty array so a resample still moves them
    (they contribute nothing, exactly as in a payload bootstrap).
    """

    def __init__(self, per_event):
        counts = [len(idx) for idx in per_event]
        flat = [np.asarray(idx, dtype=np.int64) for idx in per_event]
        self.counts = np.array(counts, dtype=np.int64)
        self.starts = (np.concatenate([[0], np.cumsum(self.counts)[:-1]])
                       if counts else np.zeros(0, dtype=np.int64))
        self.flat = (np.concatenate(flat) if flat and sum(counts)
                     else np.zeros(0, dtype=np.int64))
        self.n_events = len(counts)

    def expand(self, draw) -> np.ndarray:
        """Rows of a resampled event multiset (multiplicity preserved)."""
        draw = np.asarray(draw, dtype=np.int64)
        starts = self.starts[draw]
        counts = self.counts[draw]
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        base = np.repeat(starts, counts)
        offset = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts)
        return self.flat[base + offset]


def _interval(observed, draws, statistic, n_events: int) -> dict:
    """Percentile interval over event-level bootstrap replicates."""
    reps = []
    for draw in draws:
        value = statistic(draw)
        if value is not None and value == value:
            reps.append(float(value))
    out = {
        "observed": _json_number(observed),
        "ci_low": None,
        "ci_high": None,
        "n_events": n_events,
        "iterations": len(draws),
        "seed": P.BOOTSTRAP_SEED,
        "n_replicates_used": len(reps),
    }
    if reps:
        reps.sort()
        out["ci_low"] = reps[int(0.025 * (len(reps) - 1))]
        out["ci_high"] = reps[int(0.975 * (len(reps) - 1))]
    return out


# --------------------------------------------------------------------------
# activity
# --------------------------------------------------------------------------
def _activity(entries, readers) -> dict:
    per_reader = {}
    active = {}
    for reader in readers:
        flags = [bool(is_active(e["utility"][reader],
                                e["correct_before"][reader],
                                e["correct_after"][reader]))
                 for e in entries]
        values = [float(e["utility"][reader]) for e in entries]
        active[reader] = flags
        counts = defaultdict(int)
        for entry in entries:
            counts[entry["sign"][reader]] += 1
        per_reader[reader] = {
            "n_keys": len(entries),
            "n_active": int(sum(flags)),
            "active_rate": float(sum(flags) / len(entries)),
            "mean_utility": _json_number(mean(values)),
            "mean_abs_utility": _json_number(
                mean([abs(v) for v in values])),
            "sign_counts": {c: counts.get(c, 0) for c in P.SIGN_CLASSES},
        }
    per_pair = {}
    for i in range(len(readers)):
        for j in range(i + 1, len(readers)):
            a, b = readers[i], readers[j]
            joint = [x and y for x, y in zip(active[a], active[b])]
            per_pair[f"{a}|{b}"] = {
                "n_jointly_active": int(sum(joint)),
                "jointly_active_rate": float(sum(joint) / len(entries)),
            }
    return {"per_reader": per_reader, "per_pair": per_pair}


# --------------------------------------------------------------------------
# ordinal + signed geometry
# --------------------------------------------------------------------------
def _contingency(sa: np.ndarray, sb: np.ndarray) -> dict:
    out = {}
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            out[f"{SIGN_NAMES[x]}:{SIGN_NAMES[y]}"] = int(
                np.sum((sa == x) & (sb == y)))
    return out


def _conditional(sa: np.ndarray, sb: np.ndarray) -> dict:
    """``P(sign_b | sign_a)`` rows — the signed-direction surface."""
    out = {}
    for x in (-1, 0, 1):
        mask = sa == x
        total = int(mask.sum())
        row = {SIGN_NAMES[y]: (float(np.sum(mask & (sb == y)) / total)
                               if total else None)
               for y in (-1, 0, 1)}
        out[SIGN_NAMES[x]] = {"n": total, "p_sign_b": row}
    return out


def _pair_geometry(entries, event_slices, readers, draws) -> tuple:
    ordinal, signed = {}, {}
    utilities = {r: np.array([float(e["utility"][r]) for e in entries])
                 for r in readers}
    signs = {r: np.array([SIGN_INDEX[e["sign"][r]] for e in entries])
             for r in readers}
    active = {r: np.array([bool(is_active(e["utility"][r],
                                         e["correct_before"][r],
                                         e["correct_after"][r]))
                           for e in entries])
              for r in readers}
    n_events = len(event_slices)

    for i in range(len(readers)):
        for j in range(i + 1, len(readers)):
            a, b = readers[i], readers[j]
            pair = f"{a}|{b}"
            joint = active[a] & active[b]
            rows = np.nonzero(joint)[0]
            ua, ub = utilities[a][rows], utilities[b][rows]
            sa, sb = signs[a][rows], signs[b][rows]
            row_index = Rows([rows[(rows >= lo) & (rows < hi)]
                              for lo, hi in event_slices])

            agreement = float(np.mean(sa == sb)) if rows.size else float("nan")
            directional = (sa != 0) & (sb != 0) if rows.size else \
                np.zeros(0, dtype=bool)
            n_directional = int(directional.sum())
            directional_rate = (
                float(np.mean(sa[directional] == sb[directional]))
                if n_directional else float("nan"))

            spearman_abs = (spearman_rank(np.abs(ua), np.abs(ub))
                            if rows.size else float("nan"))
            pearson_abs = _pearson(np.abs(ua), np.abs(ub)) if rows.size \
                else float("nan")
            spearman_signed = (spearman_rank(ua, ub) if rows.size
                               else float("nan"))
            pearson_signed = _pearson(ua, ub) if rows.size else float("nan")

            ordinal[pair] = {
                "n_jointly_active": int(rows.size),
                "spearman_abs_utility": _json_number(spearman_abs),
                "pearson_abs_utility": _json_number(pearson_abs),
                "spearman_signed_utility": _json_number(spearman_signed),
                "abs_utility_pearson_event_ci": _interval(
                    pearson_abs, draws,
                    lambda draw, ri=row_index: _rs_pearson_abs(
                        utilities[a], utilities[b], ri, draw), n_events),
            }
            signed[pair] = {
                "n_jointly_active": int(rows.size),
                "n_directional": n_directional,
                "sign_agreement_rate": _json_number(agreement),
                "directional_agreement_rate": _json_number(directional_rate),
                "sign_disagreement_rate": _json_number(1.0 - agreement),
                "signed_pearson": _json_number(pearson_signed),
                "sign_contingency": _contingency(sa, sb),
                "conditional_direction": _conditional(sa, sb),
                "sign_agreement_event_ci": _interval(
                    agreement, draws,
                    lambda draw, ri=row_index: _rs_agreement(
                        signs[a], signs[b], ri, draw), n_events),
                "signed_pearson_event_ci": _interval(
                    pearson_signed, draws,
                    lambda draw, ri=row_index: _rs_pearson(
                        utilities[a], utilities[b], ri, draw), n_events),
            }
    return ordinal, signed


def _rs_agreement(sa, sb, rows: Rows, draw) -> float:
    idx = rows.expand(draw)
    if idx.size == 0:
        return float("nan")
    return float(np.mean(sa[idx] == sb[idx]))


def _rs_pearson(ua, ub, rows: Rows, draw) -> float:
    idx = rows.expand(draw)
    if idx.size < 2:
        return float("nan")
    return _pearson(ua[idx], ub[idx])


def _rs_pearson_abs(ua, ub, rows: Rows, draw) -> float:
    idx = rows.expand(draw)
    if idx.size < 2:
        return float("nan")
    return _pearson(np.abs(ua[idx]), np.abs(ub[idx]))


# --------------------------------------------------------------------------
# reader x evidence interaction
# --------------------------------------------------------------------------
def interaction_shares(matrix: np.ndarray) -> dict:
    """Two-way SS decomposition of a keys x readers matrix (no replication)."""
    matrix = np.asarray(matrix, dtype=float)
    n_keys, n_readers = matrix.shape
    total = float(matrix.sum())
    n = float(matrix.size)
    ss_total = float((matrix ** 2).sum()) - total ** 2 / n
    ss_reader = float((matrix.sum(axis=0) ** 2).sum() / n_keys) - total ** 2 / n
    ss_evidence = (float((matrix.sum(axis=1) ** 2).sum() / n_readers)
                   - total ** 2 / n)
    ss_residual = ss_total - ss_reader - ss_evidence
    denom = ss_total if ss_total else float("nan")
    return {
        "ss_total": ss_total,
        "ss_reader": ss_reader,
        "ss_evidence": ss_evidence,
        "ss_residual": ss_residual,
        "reader_share": ss_reader / denom if denom else float("nan"),
        "evidence_share": ss_evidence / denom if denom else float("nan"),
        "interaction_share": ss_residual / denom if denom else float("nan"),
    }


def _interaction(matrix: np.ndarray, event_slices, draws, iterations, seed,
                 n_events: int) -> dict:
    observed = interaction_shares(matrix)
    rows = Rows([np.arange(lo, hi) for lo, hi in event_slices])

    def statistic(draw):
        idx = rows.expand(draw)
        if idx.size == 0:
            return float("nan")
        return interaction_shares(matrix[idx])["interaction_share"]

    ci = _interval(observed["interaction_share"], draws, statistic, n_events)
    return {
        "n_keys": int(matrix.shape[0]),
        "n_readers": int(matrix.shape[1]),
        "ss_total": _json_number(observed["ss_total"]),
        "ss_reader": _json_number(observed["ss_reader"]),
        "ss_evidence": _json_number(observed["ss_evidence"]),
        "ss_residual": _json_number(observed["ss_residual"]),
        "reader_share": _json_number(observed["reader_share"]),
        "evidence_share": _json_number(observed["evidence_share"]),
        "interaction_share": _json_number(observed["interaction_share"]),
        "interaction_share_event_ci": dict(ci, iterations=iterations,
                                           seed=seed),
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _event_slices(entries) -> list:
    """``[(start, end), ...]`` for the event-grouped entries, contiguity checked."""
    slices = []
    start = 0
    for i in range(1, len(entries) + 1):
        if i == len(entries) or entries[i]["event_id"] != entries[start]["event_id"]:
            if i > start:
                slices.append((start, i))
            start = i
    if len(slices) != len({e["event_id"] for e in entries}):
        raise ValueError("atomic index is not grouped by event")
    return slices


def geometry_report(index: dict, iterations: int = P.BOOTSTRAP_ITERATIONS,
                    seed: int = P.BOOTSTRAP_SEED) -> dict:
    """The M0 ``geometry_diagnostic.json`` payload for one dataset."""
    entries = index["entries"]
    readers = list(index.get("reader_keys") or P.READER_KEYS)
    if sorted(readers) != sorted(P.READER_KEYS):
        raise ValueError(f"reader panel {readers} is not the frozen bcr_v1 set")

    slices = _event_slices(entries)
    draws = bootstrap_draws(len(slices), iterations, seed)
    matrix = np.array([[float(e["utility"][r]) for r in readers]
                       for e in entries], dtype=float) if entries else \
        np.zeros((0, len(readers)), dtype=float)

    ordinal, signed = _pair_geometry(entries, slices, readers, draws)
    return {
        "protocol": P.PROTOCOL_VERSION,
        "dataset": index.get("dataset"),
        "scope": "diagnostic_only",
        "decides_gate": False,
        "reader_keys": readers,
        "utility_threshold": P.UTILITY_THRESHOLD,
        "bootstrap": {"unit": P.BOOTSTRAP_UNIT, "iterations": iterations,
                      "seed": seed},
        "n_evidence_keys": len(entries),
        "n_events": len(slices),
        "cutoffs": sorted({e["cutoff"] for e in entries}),
        "source_index_keys_sha256": index.get("keys_sha256"),
        "activity": _activity(entries, readers),
        "ordinal_geometry": {"per_pair": ordinal, "caveat": ORDINAL_CAVEAT},
        "signed_direction_geometry": {"per_pair": signed,
                                      "caveat": SIGNED_CAVEAT},
        "reader_evidence_interaction": _interaction(
            matrix, slices, draws, iterations, seed, len(slices)),
        "interpretation": {
            "ordinal_is_not_transfer": ORDINAL_CAVEAT,
            "signed_direction_reported_independently": SIGNED_CAVEAT,
            "m0_scope": (
                "M0 reports geometry only; it does not establish, weaken or "
                "decide BCR feasibility and must not be read as a GO/NO-GO."),
        },
    }
