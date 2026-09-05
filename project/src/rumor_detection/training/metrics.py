"""Training metrics and validation."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


def compute_metrics(labels, preds, probs=None, calibration_bins=10):
    """Compute classification and probability calibration metrics."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    if probs is not None and np.asarray(probs).ndim == 2:
        metric_labels = np.arange(np.asarray(probs).shape[1])
    else:
        metric_labels = np.unique(np.concatenate([labels, preds]))

    weighted_precision = precision_score(
        labels, preds, labels=metric_labels, average='weighted', zero_division=0
    )
    weighted_recall = recall_score(
        labels, preds, labels=metric_labels, average='weighted', zero_division=0
    )
    weighted_f1 = f1_score(
        labels, preds, labels=metric_labels, average='weighted', zero_division=0
    )

    result = {
        'accuracy': accuracy_score(labels, preds),
        # Backwards-compatible aliases used by existing trainer/evaluate code.
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'macro_precision': precision_score(
            labels, preds, labels=metric_labels, average='macro', zero_division=0
        ),
        'macro_recall': recall_score(
            labels, preds, labels=metric_labels, average='macro', zero_division=0
        ),
        'macro_f1': f1_score(
            labels, preds, labels=metric_labels, average='macro', zero_division=0
        ),
        'per_class_precision': precision_score(
            labels, preds, labels=metric_labels, average=None, zero_division=0
        ).tolist(),
        'per_class_recall': recall_score(
            labels, preds, labels=metric_labels, average=None, zero_division=0
        ).tolist(),
        'per_class_f1': f1_score(
            labels, preds, labels=metric_labels, average=None, zero_division=0
        ).tolist(),
        'confusion_matrix': confusion_matrix(labels, preds, labels=metric_labels),
    }

    if probs is not None and len(labels) > 0:
        result.update(_compute_calibration_metrics(labels, np.asarray(probs),
                                                   calibration_bins))

    return result


def predict_with_binary_threshold(probs, threshold=0.5, positive_class=1):
    """Return binary predictions using a positive-class probability threshold."""
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("Binary thresholding requires probs with shape [N, 2].")
    if positive_class not in (0, 1):
        raise ValueError("positive_class must be 0 or 1.")

    negative_class = 1 - positive_class
    positive_preds = probs[:, positive_class] >= float(threshold)
    return np.where(positive_preds, positive_class, negative_class)


def find_best_binary_threshold(labels, probs, thresholds=None,
                               metric_name="macro_f1",
                               positive_class=1):
    """Choose a binary threshold on validation data and return its metrics."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best = None
    for threshold in thresholds:
        preds = predict_with_binary_threshold(
            probs,
            threshold=threshold,
            positive_class=positive_class,
        )
        metrics = compute_metrics(labels, preds, probs=probs)
        score = metrics[metric_name]
        tie_breakers = (
            score,
            metrics["accuracy"],
            metrics["weighted_f1"],
            -abs(float(threshold) - 0.5),
        )
        if best is None or tie_breakers > best["tie_breakers"]:
            best = {
                "threshold": float(threshold),
                "metric_name": metric_name,
                "metrics": metrics,
                "tie_breakers": tie_breakers,
            }

    best.pop("tie_breakers")
    return best


def _compute_calibration_metrics(labels, probs, calibration_bins):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return {}

    pred_classes = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correctness = (pred_classes == labels).astype(np.float64)

    one_hot = np.zeros_like(probs)
    valid = (labels >= 0) & (labels < probs.shape[1])
    one_hot[np.arange(len(labels))[valid], labels[valid]] = 1.0
    brier_score = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    for idx in range(calibration_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        bin_acc = correctness[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.mean()) * abs(bin_acc - bin_conf)

    true_class_probs = probs[np.arange(len(labels))[valid], labels[valid]]
    mean_true_class_prob = (
        float(true_class_probs.mean()) if true_class_probs.size > 0 else 0.0
    )

    return {
        'avg_confidence': float(confidences.mean()),
        'mean_true_class_probability': mean_true_class_prob,
        'brier_score': float(brier_score),
        'ece': float(ece),
    }


def _loss_sum(criterion, outputs, labels):
    """Return summed loss over samples for common PyTorch criteria."""
    loss = criterion(outputs, labels)
    if loss.ndim > 0:
        return loss.sum().item()

    reduction = getattr(criterion, 'reduction', 'mean')
    if reduction == 'sum':
        return loss.item()
    if reduction == 'mean':
        return loss.item() * labels.size(0)
    return loss.item()


def _clone_cross_entropy_with_reduction(criterion, reduction):
    if not isinstance(criterion, nn.CrossEntropyLoss):
        return None
    return nn.CrossEntropyLoss(
        weight=criterion.weight,
        ignore_index=criterion.ignore_index,
        reduction=reduction,
        label_smoothing=criterion.label_smoothing,
    )


def validate(model, loader, criterion, device, report_criterion=None,
             calibration_bins=10):
    """Run validation on a DataLoader."""
    model.eval()
    objective_loss_sum = 0.0
    report_loss_sum = 0.0
    total_samples = 0
    all_preds, all_labels = [], []
    all_probs = []

    objective_criterion = _clone_cross_entropy_with_reduction(criterion, 'sum')
    if objective_criterion is None:
        objective_criterion = criterion
    if report_criterion is None:
        report_criterion = criterion
    report_sum_criterion = _clone_cross_entropy_with_reduction(
        report_criterion, 'sum'
    )
    if report_sum_criterion is None:
        report_sum_criterion = report_criterion

    with torch.no_grad():
        for batch in loader:
            node_feats, struct_feats, num_nodes, labels = batch
            node_feats = node_feats.to(device)
            struct_feats = struct_feats.to(device)
            num_nodes = num_nodes.to(device)
            labels = labels.to(device)

            outputs = model(
                node_feats=node_feats,
                struct_feats=struct_feats,
                num_nodes=num_nodes,
            )
            batch_size = labels.size(0)
            objective_loss_sum += _loss_sum(objective_criterion, outputs, labels)
            report_loss_sum += _loss_sum(report_sum_criterion, outputs, labels)
            total_samples += batch_size

            probs = torch.softmax(outputs.float(), dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    if total_samples == 0:
        raise ValueError("Cannot validate on an empty loader.")

    report_loss = report_loss_sum / total_samples
    objective_loss = objective_loss_sum / total_samples
    metrics = compute_metrics(
        all_labels, all_preds, probs=np.asarray(all_probs),
        calibration_bins=calibration_bins
    )
    metrics['loss'] = report_loss
    metrics['report_loss'] = report_loss
    metrics['objective_loss'] = objective_loss
    return report_loss, metrics
