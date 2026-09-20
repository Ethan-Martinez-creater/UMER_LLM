"""B0/B1/B2 baseline networks and the shared small-MLP trainer (M1 plan §10).

All trainable M1 models share one deliberately tiny architecture family so
the B4-vs-B0 comparison isolates the *reader-conditioning* increment rather
than a capacity change:

* ``EvidenceNet`` — evidence features -> Linear->GELU->Linear (d=32) ->
  utility head + 3-class sign head. B0 uses E0+E1; B1 adds the frozen
  structural/time scalars; B2 appends a reader one-hot as an in-domain
  diagnostic input (never an unseen-reader comparator).

The trainer is deterministic given (config, seed): AdamW, Huber(0.05) +
1.0 * CE(class weights from the *training fold only*), batch 128, at most
100 epochs, patience 10 on dev Macro-F1, grad clip 1.0. Dev never sees a
held-out reader row.
"""
from __future__ import annotations

import copy
import math

from ..config import protocol as P


class TrainRefused(RuntimeError):
    """Raised when a training/evaluation contract is violated."""


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover
        raise TrainRefused(f"torch unavailable: {exc}") from exc


class EvidenceNet:  # constructed lazily so the module imports without torch
    pass


def build_evidence_net(in_dim: int, hidden: int = 32, dropout: float = 0.0,
                       extra_dim: int = 0):
    """``features (+ extra) -> encoder -> (utility, sign_logits)``."""
    torch = _torch()
    nn = torch.nn

    class _EvidenceNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim + extra_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.utility_head = nn.Linear(hidden, 1)
            self.sign_head = nn.Linear(hidden, len(P.SIGN_CLASSES))

        def forward(self, x, extra=None):
            if extra is not None:
                x = torch.cat([x, extra], dim=-1)
            h = self.encoder(x)
            return self.utility_head(h).squeeze(-1), self.sign_head(h)

    return _EvidenceNet()


def count_parameters(model) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def set_seed(seed: int) -> None:
    torch = _torch()
    import random
    import numpy as np
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 31))
    torch.manual_seed(int(seed))


def sign_class_weights(signs, classes=P.SIGN_CLASSES) -> list:
    """Inverse-frequency weights of the *training* fold, mean-normalised."""
    counts = {c: 0 for c in classes}
    for s in signs:
        if s not in counts:
            raise TrainRefused(f"unknown sign {s!r}")
        counts[s] += 1
    total = sum(counts.values())
    if total == 0:
        raise TrainRefused("no training rows")
    weights = []
    for c in classes:
        # a class absent from the training fold gets weight 0: it can never be
        # a target there, and its probability mass still receives gradient.
        weights.append((total / (len(classes) * counts[c]))
                       if counts[c] else 0.0)
    return weights


def macro_f1_torch(logits, targets, n_classes=len(P.SIGN_CLASSES)):
    """Dev Macro-F1 straight from tensors (model selection metric)."""
    torch = _torch()
    pred = logits.argmax(dim=-1)
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (targets == c)).sum())
        fp = int(((pred == c) & (targets != c)).sum())
        fn = int(((pred != c) & (targets == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)


def _as_tensors(rows):
    torch = _torch()
    return {
        "x": torch.tensor([r["x"] for r in rows], dtype=torch.float32),
        "u": torch.tensor([r["u"] for r in rows], dtype=torch.float32),
        "s": torch.tensor([r["s"] for r in rows], dtype=torch.long),
    }


def train_model(model, train_rows, dev_rows, config: dict, seed: int,
                class_weights, max_epochs=P.TRAIN_MAX_EPOCHS,
                patience=P.TRAIN_PATIENCE,
                batch_size=P.TRAIN_BATCH_SIZE,
                grad_clip=P.TRAIN_GRAD_CLIP,
                forward_extra=None) -> dict:
    """Deterministic small-MLP training with dev-Macro-F1 early stopping.

    ``forward_extra`` lets B4 pass a second (fingerprint) input: it maps a
    list of rows to the extra tensor aligned with ``x``.
    """
    torch = _torch()
    if not train_rows:
        raise TrainRefused("empty training fold")
    if not dev_rows:
        raise TrainRefused("empty dev fold")
    set_seed(seed)
    train = _as_tensors(train_rows)
    dev = _as_tensors(dev_rows)
    train_extra = torch.tensor(forward_extra(train_rows),
                               dtype=torch.float32) if forward_extra else None
    dev_extra = torch.tensor(forward_extra(dev_rows),
                             dtype=torch.float32) if forward_extra else None
    weights = torch.tensor(class_weights, dtype=torch.float32)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    huber = torch.nn.HuberLoss(delta=P.HUBER_DELTA)
    ce = torch.nn.CrossEntropyLoss(weight=weights)
    n = train["x"].shape[0]
    generator = torch.Generator().manual_seed(int(seed))
    best = {"macro_f1": -1.0, "state": None, "epoch": -1, "dev_loss": None}
    stalls = 0
    history = []
    for epoch in range(int(max_epochs)):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            optimiser.zero_grad()
            extra = train_extra[idx] if train_extra is not None else None
            u_hat, logits = model(train["x"][idx], extra)
            loss = (huber(u_hat, train["u"][idx])
                    + P.LAMBDA_SIGN * ce(logits, train["s"][idx]))
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            du_hat, dlogits = model(dev["x"], dev_extra)
            dev_loss = float(huber(du_hat, dev["u"])
                             + P.LAMBDA_SIGN * ce(dlogits, dev["s"]))
            dev_f1 = macro_f1_torch(dlogits, dev["s"])
        history.append({"epoch": epoch, "dev_macro_f1": dev_f1,
                        "dev_loss": dev_loss})
        if dev_f1 > best["macro_f1"] + 1e-12:
            best = {"macro_f1": dev_f1,
                    "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch, "dev_loss": dev_loss}
            stalls = 0
        else:
            stalls += 1
            if stalls >= patience:
                break
    if best["state"] is None:
        raise TrainRefused("no improving epoch")
    model.load_state_dict(best["state"])
    return {"best_dev_macro_f1": best["macro_f1"],
            "best_dev_loss": best["dev_loss"],
            "best_epoch": best["epoch"],
            "epochs_ran": len(history),
            "history": history,
            "n_parameters": count_parameters(model)}


def predict(model, rows, forward_extra=None, batch_size=1024) -> dict:
    """``{"utility": [...], "sign_probs": [[...]...]}`` in row order."""
    torch = _torch()
    data = _as_tensors(rows)
    extra = torch.tensor(forward_extra(rows),
                         dtype=torch.float32) if forward_extra else None
    model.eval()
    utilities, probs = [], []
    with torch.no_grad():
        for start in range(0, data["x"].shape[0], batch_size):
            xb = data["x"][start:start + batch_size]
            xtra = extra[start:start + batch_size] if extra is not None \
                else None
            u_hat, logits = model(xb, xtra)
            utilities += [float(v) for v in u_hat]
            probs += torch.softmax(logits, dim=-1).cpu().tolist()
    return {"utility": utilities, "sign_probs": probs}


def average_predictions(runs) -> dict:
    """Seed-averaged prediction: mean utility, mean sign probabilities."""
    if not runs:
        raise TrainRefused("no runs to average")
    n = len(runs[0]["utility"])
    for run in runs:
        if len(run["utility"]) != n:
            raise TrainRefused("prediction length mismatch across seeds")
    utilities = [sum(run["utility"][i] for run in runs) / len(runs)
                 for i in range(n)]
    probs = []
    for i in range(n):
        row = [sum(run["sign_probs"][i][c] for run in runs) / len(runs)
               for c in range(len(P.SIGN_CLASSES))]
        total = sum(row)
        if not math.isfinite(total) or total <= 0:
            raise TrainRefused("degenerate averaged sign probabilities")
        probs.append([v / total for v in row])
    return {"utility": utilities, "sign_probs": probs}
