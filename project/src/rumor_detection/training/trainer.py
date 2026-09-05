"""Trainer for the single original rumor detector."""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

from rumor_detection.training.metrics import (
    validate,
    compute_metrics,
)
from rumor_detection.training.logger import init_train_logger
from rumor_detection.utils.io import ensure_dir


def checkpoint_scores(val_loss, val_metrics):
    """Scores used for deciding which diagnostic checkpoints to keep."""
    val_f1 = float(val_metrics['f1'])
    val_macro_f1 = float(val_metrics['macro_f1'])
    val_loss = float(val_loss)
    return {
        'f1': val_f1,
        'macro_f1': val_macro_f1,
        'loss': -val_loss,
        'composite': val_f1 - val_loss,
    }


def runtime_training_options(cfg, device):
    """Resolve optional runtime training controls from config.

    Defaults intentionally preserve the original trainer behavior:
    no gradient accumulation and no AMP.
    """
    accumulation_steps = int(
        cfg.training.get('gradient_accumulation_steps', 1)
    )
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    requested_amp = bool(cfg.training.get('use_amp', False))
    device_type = getattr(device, "type", str(device).split(":")[0])
    use_amp = requested_amp and device_type == "cuda"
    batch_size = int(cfg.training.get('batch_size', 1))

    return {
        'gradient_accumulation_steps': accumulation_steps,
        'use_amp': use_amp,
        'requested_amp': requested_amp,
        'effective_batch_size': batch_size * accumulation_steps,
    }


def should_step_optimizer(batch_index, total_batches, accumulation_steps):
    """Return True when the optimizer should step for an accumulated batch."""
    return (
        (batch_index + 1) % accumulation_steps == 0
        or (batch_index + 1) == total_batches
    )


def clone_model_state(model):
    """Return a detached clone of a model state_dict."""
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


@torch.no_grad()
def update_ema_state(ema_state, model, decay):
    """Update EMA tensors from a model state_dict in place."""
    model_state = model.state_dict()
    for name, current in model_state.items():
        target = ema_state[name]
        if torch.is_floating_point(target):
            target.mul_(decay).add_(current.detach(), alpha=1.0 - decay)
        else:
            target.copy_(current.detach())


@torch.no_grad()
def update_swa_state(swa_state, model, count):
    """Update an equal-weight running average over model states."""
    model_state = model.state_dict()
    for name, current in model_state.items():
        target = swa_state[name]
        if torch.is_floating_point(target):
            target.mul_(float(count) / float(count + 1)).add_(
                current.detach(),
                alpha=1.0 / float(count + 1),
            )
        else:
            target.copy_(current.detach())


def class_weights_from_counts(counts, power=0.0):
    """Return normalized inverse-frequency class weights or None."""
    power = float(power or 0.0)
    if power <= 0.0:
        return None
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if counts.numel() == 0:
        return None
    safe_counts = counts.clamp_min(1.0)
    weights = safe_counts.min() / safe_counts
    weights = weights.pow(power)
    weights = weights / weights.max().clamp_min(1e-12)
    return weights


def dataset_labels(dataset):
    """Return labels from a dataset or Subset without loading samples."""
    if isinstance(dataset, Subset):
        base_labels = dataset_labels(dataset.dataset)
        return [base_labels[int(idx)] for idx in dataset.indices]
    if hasattr(dataset, "samples"):
        return [int(sample["label"]) for sample in dataset.samples]
    if hasattr(dataset, "labels"):
        return [int(label) for label in dataset.labels]
    return None


def resolve_train_class_counts(train_loader, num_classes, device):
    labels = dataset_labels(train_loader.dataset)
    if labels is None:
        return None
    counts = torch.bincount(
        torch.as_tensor(labels, dtype=torch.long),
        minlength=int(num_classes),
    )
    return counts.to(device=device, dtype=torch.float32)


def resolve_train_class_weights(train_loader, num_classes, power, device):
    counts = resolve_train_class_counts(train_loader, num_classes, device)
    if counts is None:
        return None
    weights = class_weights_from_counts(counts, power=power)
    if weights is None:
        return None
    return weights.to(device)


class FocalCrossEntropyLoss(nn.Module):
    """Cross entropy with focal weighting for hard-example training."""

    def __init__(self, weight=None, gamma=2.0, reduction='mean',
                 label_smoothing=0.0):
        super().__init__()
        self.register_buffer(
            "weight",
            weight.detach().clone() if weight is not None else None,
        )
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = target_log_probs.exp()
        loss = ((1.0 - pt).clamp_min(0.0).pow(self.gamma)) * ce_loss

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()


class BalancedSoftmaxLoss(nn.Module):
    """Balanced softmax loss using training-set class counts as priors."""

    def __init__(self, class_counts, label_smoothing=0.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        self.register_buffer("log_counts", counts.clamp_min(1.0).log())
        self.label_smoothing = float(label_smoothing)

    def forward(self, inputs, targets):
        adjusted = inputs + self.log_counts.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        return nn.functional.cross_entropy(
            adjusted,
            targets,
            label_smoothing=self.label_smoothing,
        )


def symmetric_kl_loss(logits_a, logits_b):
    """Symmetric KL divergence between two prediction distributions."""
    log_probs_a = nn.functional.log_softmax(logits_a.float(), dim=1)
    log_probs_b = nn.functional.log_softmax(logits_b.float(), dim=1)
    probs_a = log_probs_a.exp()
    probs_b = log_probs_b.exp()
    kl_ab = nn.functional.kl_div(log_probs_a, probs_b, reduction='batchmean')
    kl_ba = nn.functional.kl_div(log_probs_b, probs_a, reduction='batchmean')
    return (0.5 * (kl_ab + kl_ba)).clamp_min(0.0)


def build_training_criterion(cfg, class_weights=None, class_counts=None):
    gamma = float(cfg.training.get('focal_gamma', 0.0) or 0.0)
    label_smoothing = cfg.training.label_smoothing
    if bool(cfg.training.get('balanced_softmax', False)):
        if class_counts is None:
            raise ValueError(
                "training.balanced_softmax requires training class counts"
            )
        return BalancedSoftmaxLoss(
            class_counts=class_counts,
            label_smoothing=label_smoothing,
        )
    if gamma <= 0.0:
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    return FocalCrossEntropyLoss(
        weight=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, cfg, device,
                 experiment_dir=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device

        # experiment_dir isolates each run's outputs
        #   outputs/{dataset}/experiments/{timestamp}/
        # If None, fall back to output_dir root (backwards compat).
        self.experiment_dir = experiment_dir or cfg.data.output_dir
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.metrics_dir = os.path.join(self.experiment_dir, "metrics")
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)
        ensure_dir(self.metrics_dir)

        self.best_val_f1 = 0.0
        self.best_scores = {
            'f1': float('-inf'),
            'macro_f1': float('-inf'),
            'loss': float('-inf'),
            'composite': float('-inf'),
        }
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.best_checkpoint_paths = {
            'f1': self.best_model_path,
            'macro_f1': os.path.join(
                self.checkpoint_dir, "best_macro_f1_model.pth"
            ),
            'loss': os.path.join(self.checkpoint_dir, "best_val_loss_model.pth"),
            'composite': os.path.join(self.checkpoint_dir, "best_composite_model.pth"),
        }
        self.ema_decay = float(cfg.training.get('ema_decay', 0.0) or 0.0)
        self.ema_state = None
        if self.ema_decay > 0.0:
            if self.ema_decay >= 1.0:
                raise ValueError("training.ema_decay must be in [0, 1)")
            self.ema_state = clone_model_state(self.model)
            self._enable_ema_checkpoints()
        self.swa_start_epoch = int(cfg.training.get('swa_start_epoch', 0) or 0)
        if self.swa_start_epoch < 0:
            raise ValueError("training.swa_start_epoch must be >= 0")
        self.swa_state = None
        self.swa_count = 0
        if self.swa_start_epoch > 0:
            self._enable_swa_checkpoint()
        self.rdrop_alpha = float(cfg.training.get('rdrop_alpha', 0.0) or 0.0)
        if self.rdrop_alpha < 0.0:
            raise ValueError("training.rdrop_alpha must be >= 0")
        self.runtime_options = runtime_training_options(cfg, device)
        self.gradient_accumulation_steps = self.runtime_options[
            'gradient_accumulation_steps'
        ]
        self.use_amp = self.runtime_options['use_amp']
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.class_weights = resolve_train_class_weights(
            self.train_loader,
            cfg.model.get('num_classes', 2),
            cfg.training.get('class_weight_power', 0.0),
            self.device,
        )
        self.class_counts = resolve_train_class_counts(
            self.train_loader,
            cfg.model.get('num_classes', 2),
            self.device,
        )
        self.patience_counter = 0
        self.patience = cfg.training.patience

        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_macro_f1': [],
            'val_loss': [], 'val_objective_loss': [], 'val_acc': [],
            'val_f1': [], 'val_macro_f1': [], 'val_ece': [], 'val_brier_score': [],
        }

    def run(self, max_epochs=None):
        if max_epochs is None:
            max_epochs = self.cfg.training.get('max_epochs', 100)
        stage1_epochs = min(self.cfg.training.stage1_epochs, max_epochs)

        # Initialize logger
        logger = init_train_logger(self.log_dir)
        logger.info(f"Device: {self.device}")
        logger.info(f"Train: {len(self.train_loader.dataset)}, "
                    f"Val: {len(self.val_loader.dataset)}, "
                    f"Test: {len(self.test_loader.dataset)}")
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Max epochs: {max_epochs}, Stage1: {stage1_epochs}")
        logger.info(
            "Runtime training options: "
            f"micro_batch={self.cfg.training.batch_size}, "
            f"grad_accum={self.gradient_accumulation_steps}, "
            f"effective_batch={self.runtime_options['effective_batch_size']}, "
            f"use_amp={self.use_amp}"
        )
        if self.class_weights is not None:
            logger.info(
                "Class weights: "
                + ", ".join(f"{w:.4f}" for w in self.class_weights.tolist())
            )
        focal_gamma = float(self.cfg.training.get('focal_gamma', 0.0) or 0.0)
        if focal_gamma > 0.0:
            logger.info(f"Focal gamma: {focal_gamma:.4f}")
        if bool(self.cfg.training.get('balanced_softmax', False)):
            if self.class_counts is None:
                raise ValueError(
                    "training.balanced_softmax requires dataset labels"
                )
            logger.info(
                "Balanced softmax class counts: "
                + ", ".join(f"{c:.0f}" for c in self.class_counts.tolist())
            )
        if self.ema_state is not None:
            logger.info(f"EMA decay: {self.ema_decay:.4f}")
        if self.swa_start_epoch > 0:
            logger.info(f"SWA start epoch: {self.swa_start_epoch}")
        if self.rdrop_alpha > 0.0:
            logger.info(f"R-Drop alpha: {self.rdrop_alpha:.4f}")

        logger.info("=== Single-model training stage ===")
        self._set_all_trainable()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.cfg.training.stage1_lr),
            weight_decay=float(
                self.cfg.training.get("stage1_weight_decay", 3e-4)
            ),
            betas=(0.9, 0.999), eps=1e-8
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        criterion = build_training_criterion(
            self.cfg,
            class_weights=self.class_weights,
            class_counts=self.class_counts,
        )
        eval_criterion = nn.CrossEntropyLoss()

        for epoch in range(stage1_epochs):
            improved = self._train_epoch(
                epoch, optimizer, criterion, eval_criterion, max_epochs, logger
            )
            val_f1 = self.history['val_f1'][-1]
            scheduler.step(val_f1)
            if self.patience_counter >= self.patience:
                logger.info("Early stopping in Stage 1!")
                break

        # Stage 2: unfreeze all, fine-tune
        remaining = max_epochs - stage1_epochs
        if remaining > 0:
            logger.info("=== Stage 2: Unfreeze all, fine-tune ===")
            self._set_all_trainable()
            optimizer = optim.AdamW(
                self.model.parameters(), lr=float(self.cfg.training.stage2_lr),
                weight_decay=float(self.cfg.training.weight_decay),
                betas=(0.9, 0.999), eps=1e-8
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3
            )
            self.patience_counter = 0

            for epoch in range(stage1_epochs, max_epochs):
                improved = self._train_epoch(
                    epoch, optimizer, criterion, eval_criterion, max_epochs, logger
                )
                val_f1 = self.history['val_f1'][-1]
                scheduler.step(val_f1)
                if self.patience_counter >= self.patience:
                    logger.info("Early stopping in Stage 2!")
                    break

        # Save training history as CSV
        self._save_history()

        # Final test (skip if test loader is empty)
        if len(self.test_loader) > 0:
            logger.info("=== Final Test ===")
            checkpoint_results = self._evaluate_test_checkpoints(
                criterion, eval_criterion, logger
            )
            self._save_checkpoint_test_metrics(checkpoint_results)
            if "f1" in checkpoint_results:
                self._save_test_metrics(
                    checkpoint_results["f1"]["test_loss"],
                    checkpoint_results["f1"]["metrics"],
                )
        else:
            logger.info("=== Final Test skipped (empty test set) ===")

        return self.history

    def _train_epoch(self, epoch, optimizer, criterion, eval_criterion,
                     total_epochs, logger):
        self.model.train()
        train_loss_sum = 0.0
        train_samples = 0
        train_preds, train_labels = [], []
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(self.train_loader):
            node_feats, struct_feats, num_nodes, labels = batch
            node_feats = node_feats.to(self.device)
            struct_feats = struct_feats.to(self.device)
            num_nodes = num_nodes.to(self.device)
            labels = labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                model_kwargs = {
                    "node_feats": node_feats,
                    "struct_feats": struct_feats,
                    "num_nodes": num_nodes,
                }
                if getattr(self.model, "uses_auxiliary_loss", False):
                    model_kwargs["labels"] = labels
                model_output = self.model(**model_kwargs)
                if isinstance(model_output, dict):
                    outputs = model_output["logits"]
                    loss = criterion(outputs, labels) + model_output.get(
                        "aux_loss", outputs.new_tensor(0.0)
                    )
                else:
                    outputs = model_output
                    loss = criterion(outputs, labels)
                if self.rdrop_alpha > 0.0:
                    outputs_2 = self.model(
                        node_feats=node_feats,
                        struct_feats=struct_feats,
                        num_nodes=num_nodes,
                    )
                    loss = 0.5 * (loss + criterion(outputs_2, labels))
                    loss = loss + self.rdrop_alpha * symmetric_kl_loss(
                        outputs, outputs_2
                    )

            backward_loss = loss / self.gradient_accumulation_steps
            if self.use_amp:
                self.scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            if should_step_optimizer(
                i, len(self.train_loader), self.gradient_accumulation_steps
            ):
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                if self.ema_state is not None:
                    update_ema_state(self.ema_state, self.model, self.ema_decay)
                optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            if i % max(1, len(self.train_loader) // 3) == 0:
                logger.info(f"  [Batch {i+1}/{len(self.train_loader)}] Loss: {loss.item():.4f}")

        train_loss = train_loss_sum / max(1, train_samples)
        train_metrics = compute_metrics(train_labels, train_preds)

        # Validate
        val_loss, val_metrics = validate(
            self.model, self.val_loader, criterion, self.device,
            report_criterion=eval_criterion
        )

        # Record
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['train_f1'].append(train_metrics['f1'])
        self.history['train_macro_f1'].append(train_metrics['macro_f1'])
        self.history['val_loss'].append(val_loss)
        self.history['val_objective_loss'].append(val_metrics['objective_loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['val_macro_f1'].append(val_metrics['macro_f1'])
        self.history['val_ece'].append(val_metrics.get('ece', 0.0))
        self.history['val_brier_score'].append(val_metrics.get('brier_score', 0.0))

        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} "
            f"F1: {train_metrics['f1']:.4f} MacroF1: {train_metrics['macro_f1']:.4f} | "
            f"Val CE Loss: {val_loss:.4f} Obj Loss: {val_metrics['objective_loss']:.4f} "
            f"Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} "
            f"MacroF1: {val_metrics['macro_f1']:.4f} ECE: {val_metrics.get('ece', 0.0):.4f}"
        )

        improvements = self._update_checkpoints(epoch, val_loss, val_metrics)
        if self.ema_state is not None:
            ema_val_loss, ema_val_metrics = self._validate_with_state(
                self.ema_state, criterion, eval_criterion
            )
            logger.info(
                f"Epoch {epoch+1}/{total_epochs} | "
                f"EMA Val CE Loss: {ema_val_loss:.4f} "
                f"Obj Loss: {ema_val_metrics['objective_loss']:.4f} "
                f"Acc: {ema_val_metrics['accuracy']:.4f} "
                f"F1: {ema_val_metrics['f1']:.4f} "
                f"MacroF1: {ema_val_metrics['macro_f1']:.4f} "
                f"ECE: {ema_val_metrics.get('ece', 0.0):.4f}"
            )
            self._update_checkpoints(
                epoch,
                ema_val_loss,
                ema_val_metrics,
                prefix="ema_",
                state_dict=self.ema_state,
            )
        if self.swa_start_epoch > 0 and (epoch + 1) >= self.swa_start_epoch:
            if self.swa_state is None:
                self.swa_state = clone_model_state(self.model)
                self.swa_count = 1
            else:
                update_swa_state(self.swa_state, self.model, self.swa_count)
                self.swa_count += 1
            swa_val_loss, swa_val_metrics = self._validate_with_state(
                self.swa_state, criterion, eval_criterion
            )
            logger.info(
                f"Epoch {epoch+1}/{total_epochs} | "
                f"SWA Val CE Loss: {swa_val_loss:.4f} "
                f"Obj Loss: {swa_val_metrics['objective_loss']:.4f} "
                f"Acc: {swa_val_metrics['accuracy']:.4f} "
                f"F1: {swa_val_metrics['f1']:.4f} "
                f"MacroF1: {swa_val_metrics['macro_f1']:.4f} "
                f"ECE: {swa_val_metrics.get('ece', 0.0):.4f}"
            )
            self._save_checkpoint(
                self.best_checkpoint_paths["swa"],
                epoch,
                "swa",
                swa_val_loss,
                swa_val_metrics,
                state_dict=self.swa_state,
            )
        if improvements['f1']:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return improvements['f1']

    def _save_history(self):
        """Save training history as CSV."""
        import pandas as pd
        df = pd.DataFrame(self.history)
        df.index.name = 'epoch'
        path = os.path.join(self.metrics_dir, "history.csv")
        df.to_csv(path, index=True)
        print(f"Training history saved to {path}")

    def _save_test_metrics(self, test_loss, metrics):
        """Save final test metrics as JSON."""
        path = os.path.join(self.metrics_dir, "final_test_metrics.json")
        result = {
            'test_loss': test_loss,
            **metrics,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_to_jsonable(result), f, indent=2)
        print(f"Test metrics saved to {path}")

    def _evaluate_test_checkpoints(self, criterion, eval_criterion, logger):
        """Evaluate every diagnostic best checkpoint on the test set."""
        results = {}
        for name, path in self.best_checkpoint_paths.items():
            if not os.path.exists(path):
                logger.info(f"Skipping missing checkpoint '{name}': {path}")
                continue
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            test_loss, test_metrics = validate(
                self.model, self.test_loader, criterion, self.device,
                report_criterion=eval_criterion
            )
            logger.info(
                f"Test checkpoint={name} "
                f"selection={checkpoint.get('selection_metric', name)} "
                f"epoch={checkpoint.get('epoch')} "
                f"val_f1={float(checkpoint.get('val_f1', 0.0)):.4f} "
                f"val_loss={float(checkpoint.get('val_loss', 0.0)):.4f} "
                f"test_weighted_f1={test_metrics['weighted_f1']:.4f} "
                f"test_macro_f1={test_metrics['macro_f1']:.4f} "
                f"test_acc={test_metrics['accuracy']:.4f}"
            )
            results[name] = {
                "checkpoint_path": path,
                "selection_metric": checkpoint.get("selection_metric", name),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_val_loss": checkpoint.get("val_loss"),
                "checkpoint_val_f1": checkpoint.get("val_f1"),
                "checkpoint_val_macro_f1": checkpoint.get("val_macro_f1"),
                "checkpoint_val_objective_loss": checkpoint.get(
                    "val_objective_loss"
                ),
                "checkpoint_val_ece": checkpoint.get("val_ece"),
                "test_loss": test_loss,
                "metrics": test_metrics,
            }
        if not results:
            logger.info("No test checkpoints were available for evaluation.")
        return results

    def _save_checkpoint_test_metrics(self, checkpoint_results):
        """Save aggregate and per-checkpoint test metrics."""
        aggregate_path = os.path.join(
            self.metrics_dir, "final_test_metrics_by_checkpoint.json"
        )
        with open(aggregate_path, 'w', encoding='utf-8') as f:
            json.dump(_to_jsonable(checkpoint_results), f, indent=2)

        for name, result in checkpoint_results.items():
            path = os.path.join(
                self.metrics_dir, f"final_test_metrics_best_{name}.json"
            )
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(_to_jsonable(result), f, indent=2)
        print(f"Checkpoint test metrics saved to {aggregate_path}")

    def _set_all_trainable(self):
        """Keep every parameter of OriginalRumorDetector trainable."""
        for param in self.model.parameters():
            param.requires_grad = True

    def _enable_ema_checkpoints(self):
        self.best_checkpoint_paths.update({
            'ema_f1': os.path.join(self.checkpoint_dir, "best_ema_model.pth"),
            'ema_macro_f1': os.path.join(
                self.checkpoint_dir, "best_ema_macro_f1_model.pth"
            ),
            'ema_loss': os.path.join(
                self.checkpoint_dir, "best_ema_val_loss_model.pth"
            ),
            'ema_composite': os.path.join(
                self.checkpoint_dir, "best_ema_composite_model.pth"
            ),
        })
        self.best_scores.update({
            'ema_f1': float('-inf'),
            'ema_macro_f1': float('-inf'),
            'ema_loss': float('-inf'),
            'ema_composite': float('-inf'),
        })

    def _enable_swa_checkpoint(self):
        self.best_checkpoint_paths["swa"] = os.path.join(
            self.checkpoint_dir, "swa_model.pth"
        )

    def _validate_with_state(self, state_dict, criterion, eval_criterion):
        current_state = clone_model_state(self.model)
        try:
            self.model.load_state_dict(state_dict)
            return validate(
                self.model, self.val_loader, criterion, self.device,
                report_criterion=eval_criterion
            )
        finally:
            self.model.load_state_dict(current_state)

    def _update_checkpoints(
        self, epoch, val_loss, val_metrics, prefix="", state_dict=None
    ):
        scores = checkpoint_scores(val_loss, val_metrics)
        improvements = {}
        for name, score in scores.items():
            key = f"{prefix}{name}"
            improved = score > self.best_scores[key]
            improvements[key] = improved
            if improved:
                self.best_scores[key] = score
                self._save_checkpoint(
                    self.best_checkpoint_paths[key],
                    epoch,
                    key,
                    val_loss,
                    val_metrics,
                    state_dict=state_dict,
                )
        if not prefix and improvements['f1']:
            self.best_val_f1 = val_metrics['f1']
        return improvements

    def _save_checkpoint(
        self, path, epoch, selection_metric, val_loss, val_metrics,
        state_dict=None
    ):
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict or self.model.state_dict(),
            'selection_metric': selection_metric,
            'val_loss': float(val_loss),
            'val_f1': float(val_metrics['f1']),
            'val_macro_f1': float(val_metrics['macro_f1']),
            'val_objective_loss': float(val_metrics.get('objective_loss', val_loss)),
            'val_ece': float(val_metrics.get('ece', 0.0)),
        }, path)

    def _load_best_model(self):
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model (val_f1={checkpoint['val_f1']:.4f})")


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    return value
