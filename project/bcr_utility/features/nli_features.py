"""E1 frozen NLI evidence features (M1 plan §5, §9).

The exact frozen extractor is ``MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`` —
no substitution is permitted (M1 plan §5). For every atomic evidence key the
three text pairs (source->reply, source->parent, reply->parent) are scored
and the three class probabilities per pair are recorded in the frozen order
``(entailment, neutral, contradiction)``.

Texts are cleaned with the same dataset-specific cleaner the frozen MiniLM
pipeline uses, so E0/E1 describe the same textual content.

A missing parent yields three exact 0.0 probabilities (the row's
``parent_available`` flag lives in the E0 cache); the NLI model is never
queried with an invented parent text.
"""
from __future__ import annotations

import math

from ..config import protocol as P


class NLIRefused(RuntimeError):
    """Raised when the frozen NLI extractor cannot be used exactly."""


def load_nli(model_path: str, device: str = "cpu"):
    """Load the frozen NLI extractor and pin its label order."""
    if not model_path:
        raise NLIRefused("no NLI model path configured; the frozen extractor "
                         f"is {P.NLI_MODEL_ID}")
    try:
        from transformers import AutoModelForSequenceClassification, \
        AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise NLIRefused(f"transformers unavailable: {exc}") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True)
    id2label = {int(k): str(v).lower()
                for k, v in (model.config.id2label or {}).items()}
    label_index = {}
    for idx, name in id2label.items():
        for wanted in P.NLI_LABELS:
            if wanted in name:
                label_index[wanted] = idx
    if sorted(label_index) != sorted(P.NLI_LABELS):
        raise NLIRefused(
            f"NLI label set {sorted(id2label.values())} does not contain "
            f"{list(P.NLI_LABELS)}; refusing to substitute the frozen "
            "extractor")
    model.to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model,
            "label_index": label_index, "id2label": id2label,
            "device": device}


def nli_probs(nli: dict, pairs, batch_size: int = 32) -> list:
    """``[(p_entailment, p_neutral, p_contradiction), ...]`` for text pairs."""
    import torch
    tokenizer, model = nli["tokenizer"], nli["model"]
    label_index = nli["label_index"]
    out = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            premises = [p for p, _h in batch]
            hypotheses = [h for _p, h in batch]
            enc = tokenizer(premises, hypotheses, truncation=True,
                            max_length=512, padding=True,
                            return_tensors="pt")
            enc = {k: v.to(nli["device"]) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            for row in probs:
                out.append([float(row[label_index[label]])
                            for label in P.NLI_LABELS])
    return out


def e1_feature_dict(probs_by_pair: dict) -> dict:
    """Flatten ``{pair: [p_e, p_n, p_c]}`` into the frozen E1 feature order."""
    features = {}
    for pair in P.E1_PAIRS:
        probs = probs_by_pair.get(pair)
        if probs is None or len(probs) != len(P.NLI_LABELS):
            raise NLIRefused(f"pair {pair!r}: missing/invalid probabilities")
        for label, value in zip(P.NLI_LABELS, probs):
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise NLIRefused(f"pair {pair!r}/{label}: {value!r}")
            features[f"nli_{pair}_{label}"] = float(value)
    if tuple(features) != P.E1_FEATURE_NAMES:
        raise NLIRefused("E1 feature order drift")
    return features


def e1_rows(text_rows, nli: dict, clean, batch_size: int = 32) -> list:
    """E1 rows for ``[{key, source_text, reply_text, parent_text}, ...]``.

    ``clean`` is the dataset-specific text cleaner shared with the MiniLM
    pipeline. Missing parents are emitted as three 0.0 probabilities without
    querying the model.
    """
    text_rows = list(text_rows)
    tasks = []   # (row_pos, pair_name, premise, hypothesis)
    for pos, row in enumerate(text_rows):
        source = clean(row["source_text"])
        reply = clean(row["reply_text"])
        parent = clean(row["parent_text"]) if row.get("parent_text") else None
        tasks.append((pos, "source_reply", source, reply))
        if parent is not None:
            tasks.append((pos, "source_parent", source, parent))
            tasks.append((pos, "reply_parent", reply, parent))
    probs = nli_probs(nli, [(p, h) for _i, _n, p, h in tasks],
                      batch_size=batch_size)
    per_row = {}
    for (pos, pair, _p, _h), prob in zip(tasks, probs):
        per_row.setdefault(pos, {})[pair] = prob
    rows = []
    for pos, row in enumerate(text_rows):
        by_pair = per_row.get(pos, {})
        if row.get("parent_text"):
            missing = [p for p in P.E1_PAIRS if p not in by_pair]
            if missing:
                raise NLIRefused(f"{row['key']}: unscored pairs {missing}")
        else:
            for pair in ("source_parent", "reply_parent"):
                by_pair[pair] = [0.0, 0.0, 0.0]
        rows.append({"key": row["key"],
                     "e1": e1_feature_dict(by_pair)})
    return rows


def validate_e1_rows(rows, expected_keys) -> dict:
    keys = [r["key"] for r in rows]
    problems = []
    if len(set(keys)) != len(keys):
        problems.append("duplicate keys")
    missing = sorted(set(expected_keys) - set(keys))
    extra = sorted(set(keys) - set(expected_keys))
    if missing:
        problems.append(f"{len(missing)} missing: {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} unexpected: {extra[:3]}")
    for row in rows:
        if tuple(row["e1"]) != P.E1_FEATURE_NAMES:
            problems.append(f"{row['key']}: feature order drift")
            break
        for name, value in row["e1"].items():
            if not math.isfinite(value):
                problems.append(f"{row['key']}: {name} not finite")
                break
    if problems:
        raise NLIRefused("; ".join(problems[:8]))
    return {"rows": len(rows), "ok": True}
