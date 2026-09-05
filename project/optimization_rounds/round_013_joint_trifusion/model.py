"""One-checkpoint joint graph, text, and retrieval rumor detector."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_view_subset(view_logits: torch.Tensor, selected_count: int):
    """Uniformly sample distinct views per event; zero means use every view."""
    if view_logits.ndim != 3:
        raise ValueError("view_logits must have shape [batch, views, classes]")
    view_count = int(view_logits.size(1))
    selected_count = int(selected_count)
    if selected_count == 0 or selected_count == view_count:
        indices = torch.arange(view_count, device=view_logits.device)
        indices = indices.unsqueeze(0).expand(view_logits.size(0), -1)
        return view_logits, indices
    if selected_count < 1 or selected_count > view_count:
        raise ValueError("selected_count must be zero or within available views")
    random_scores = torch.rand(
        view_logits.size(0), view_count, device=view_logits.device
    )
    indices = random_scores.topk(selected_count, dim=1, largest=False).indices
    gather_indices = indices.unsqueeze(-1).expand(-1, -1, view_logits.size(-1))
    return view_logits.gather(1, gather_indices), indices


class ReliabilityGatedFusion(nn.Module):
    """Conditionally weight graph, text, and retrieval rumor margins."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.log_scales = nn.Parameter(torch.zeros(3))
        self.bias = nn.Parameter(torch.zeros(()))
        self.gate = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
        )
        # Start from an equal-weight mixture, then learn conditional reliability.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, graph_logits, text_logits, retrieval_logits, num_nodes):
        margins = torch.stack(
            [
                graph_logits[:, 1] - graph_logits[:, 0],
                text_logits[:, 1] - text_logits[:, 0],
                retrieval_logits.reshape(-1),
            ],
            dim=1,
        )
        bounded = torch.tanh(margins / 4.0)
        node_scale = torch.log1p(num_nodes.float()).unsqueeze(1) / 5.0
        gate_features = torch.cat([bounded, bounded.abs(), node_scale], dim=1)
        weights = torch.softmax(self.gate(gate_features), dim=1)
        calibrated = margins * self.log_scales.exp().unsqueeze(0)
        rumor_margin = (weights * calibrated).sum(dim=1) + self.bias
        logits = torch.stack([-0.5 * rumor_margin, 0.5 * rumor_margin], dim=1)
        return logits, weights


class ConditionalResidualFusion(nn.Module):
    """Retain linear fusion and add a bounded event-conditional correction."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.base = nn.Linear(5, 2)
        self.residual = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        # Exact Round 018 behavior at initialization; learn correction only
        # when the validation-supported training signal warrants it.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, graph_logits, text_logits, retrieval_logits, num_nodes):
        fusion_input = torch.cat(
            [graph_logits, text_logits, retrieval_logits.reshape(-1, 1)], dim=1
        )
        base_logits = self.base(fusion_input)
        margins = torch.stack(
            [
                graph_logits[:, 1] - graph_logits[:, 0],
                text_logits[:, 1] - text_logits[:, 0],
                retrieval_logits.reshape(-1),
            ],
            dim=1,
        )
        bounded = torch.tanh(margins / 4.0)
        disagreement = torch.stack(
            [
                (bounded[:, 0] - bounded[:, 1]).abs(),
                (bounded[:, 0] - bounded[:, 2]).abs(),
                (bounded[:, 1] - bounded[:, 2]).abs(),
            ],
            dim=1,
        )
        node_scale = torch.log1p(num_nodes.float()).unsqueeze(1) / 5.0
        base_margin = torch.tanh(
            (base_logits[:, 1] - base_logits[:, 0]).unsqueeze(1) / 4.0
        )
        features = torch.cat(
            [bounded, bounded.abs(), disagreement, node_scale, base_margin], dim=1
        )
        correction = 0.5 * torch.tanh(self.residual(features).squeeze(1))
        correction_logits = torch.stack([-0.5 * correction, 0.5 * correction], dim=1)
        return base_logits + correction_logits, correction


class JointTriFusionModel(nn.Module):
    """Jointly fine-tune both neural branches and a train-fold memory fusion head."""

    def __init__(
        self,
        graph_model: nn.Module,
        text_model: nn.Module,
        memory_features: torch.Tensor,
        memory_labels: torch.Tensor,
        retrieval_text_dim: int = 384,
        retrieval_k: int = 31,
        retrieval_temperature: float = 0.05,
        fusion_initialization: str = "late_fusion",
        fusion_type: str = "linear",
        ablate_evidence: str = "none",
        cognitive_target_dim: int = 0,
        node_cognitive_target_dim: int = 0,
    ):
        super().__init__()
        self.graph_model = graph_model
        self.text_model = text_model
        self.retrieval_text_dim = int(retrieval_text_dim)
        self.retrieval_k = int(retrieval_k)
        self.retrieval_temperature = float(retrieval_temperature)
        self.register_buffer(
            "memory_features", F.normalize(memory_features.float(), dim=1)
        )
        self.register_buffer("memory_labels", memory_labels.float().reshape(-1))
        if self.memory_features.size(0) != self.memory_labels.numel():
            raise ValueError("Retrieval memory features and labels differ in length")
        self.fusion_type = str(fusion_type)
        self.ablate_evidence = str(ablate_evidence)
        self.cognitive_target_dim = int(cognitive_target_dim)
        if self.cognitive_target_dim < 0:
            raise ValueError("cognitive_target_dim cannot be negative")
        self.cognitive_head = (
            nn.Sequential(
                nn.LayerNorm(int(self.graph_model.d_model)),
                nn.Linear(int(self.graph_model.d_model), self.cognitive_target_dim),
                nn.Sigmoid(),
            )
            if self.cognitive_target_dim > 0
            else None
        )
        self.node_cognitive_target_dim = int(node_cognitive_target_dim)
        if self.node_cognitive_target_dim < 0:
            raise ValueError("node_cognitive_target_dim cannot be negative")
        self.node_cognitive_head = (
            nn.Sequential(
                nn.LayerNorm(int(self.graph_model.d_model)),
                nn.Linear(
                    int(self.graph_model.d_model),
                    self.node_cognitive_target_dim,
                ),
            )
            if self.node_cognitive_target_dim > 0
            else None
        )
        if self.ablate_evidence not in {"none", "graph", "text", "memory"}:
            raise ValueError(
                f"Unknown evidence ablation: {self.ablate_evidence}"
            )
        if self.fusion_type == "linear":
            self.fusion = nn.Linear(5, 2)
            if fusion_initialization == "late_fusion":
                self._initialize_from_late_fusion()
            elif fusion_initialization == "random":
                self.fusion.reset_parameters()
            else:
                raise ValueError(
                    f"Unknown fusion initialization: {fusion_initialization}"
                )
        elif self.fusion_type == "reliability_gate":
            if fusion_initialization != "random":
                raise ValueError("Reliability gate supports clean random initialization only")
            self.fusion = ReliabilityGatedFusion()
        elif self.fusion_type == "conditional_residual":
            if fusion_initialization != "random":
                raise ValueError(
                    "Conditional residual supports clean random initialization only"
                )
            self.fusion = ConditionalResidualFusion()
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def event_semantic_features(self, node_feats, num_nodes):
        text = node_feats[..., : self.retrieval_text_dim].float()
        max_nodes = text.size(1)
        mask = (
            torch.arange(max_nodes, device=text.device)[None, :]
            < num_nodes[:, None].clamp(min=1, max=max_nodes)
        )
        mask_float = mask.unsqueeze(-1).to(text.dtype)
        source = text[:, 0]
        mean = (text * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        maximum = text.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        blocks = [source, mean, maximum, mean - source]
        blocks = [F.normalize(block, dim=1) for block in blocks]
        return F.normalize(torch.cat(blocks, dim=1), dim=1)

    def retrieve(self, query_features, memory_positions=None):
        similarities = query_features @ self.memory_features.T
        if memory_positions is not None:
            positions = memory_positions.to(similarities.device).long()
            rows = torch.arange(similarities.size(0), device=similarities.device)
            valid = (positions >= 0) & (positions < similarities.size(1))
            similarities[rows[valid], positions[valid]] = -torch.inf
        k = min(self.retrieval_k, similarities.size(1) - 1)
        if k < 1:
            raise ValueError("Retrieval memory must contain at least two events")
        values, indices = torch.topk(similarities, k=k, dim=1)
        weights = torch.softmax(values / self.retrieval_temperature, dim=1)
        probability = (weights * self.memory_labels[indices]).sum(dim=1)
        probability = probability.clamp(1e-5, 1.0 - 1e-5)
        return torch.log(probability / (1.0 - probability))

    @torch.no_grad()
    def retrieve_with_evidence(self, query_features, memory_positions=None):
        """Expose unchanged train-memory retrieval details for explanation audits.

        Keeping this method separate from ``retrieve`` ensures that explanation
        instrumentation cannot alter the detector's existing forward path.
        """
        similarities = query_features @ self.memory_features.T
        if memory_positions is not None:
            positions = memory_positions.to(similarities.device).long()
            rows = torch.arange(similarities.size(0), device=similarities.device)
            valid = (positions >= 0) & (positions < similarities.size(1))
            similarities[rows[valid], positions[valid]] = -torch.inf
        k = min(self.retrieval_k, similarities.size(1) - 1)
        if k < 1:
            raise ValueError("Retrieval memory must contain at least two events")
        values, indices = torch.topk(similarities, k=k, dim=1)
        weights = torch.softmax(values / self.retrieval_temperature, dim=1)
        labels = self.memory_labels[indices]
        probability = (weights * labels).sum(dim=1).clamp(1e-5, 1.0 - 1e-5)
        return {
            "logit": torch.log(probability / (1.0 - probability)),
            "probability": probability,
            "indices": indices,
            "similarities": values,
            "weights": weights,
            "labels": labels,
        }

    def _initialize_from_late_fusion(self):
        # Symmetric initialization reproduces the useful validation-selected
        # score 0.5*g + 0.5*d + 1.0*r + 0.8 before joint fine-tuning.
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.copy_(torch.tensor([-0.4, 0.4]))
            self.fusion.weight[0].copy_(
                torch.tensor([0.25, -0.25, 0.25, -0.25, -0.5])
            )
            self.fusion.weight[1].copy_(
                torch.tensor([-0.25, 0.25, -0.25, 0.25, 0.5])
            )

    def forward(
        self,
        node_feats,
        struct_feats,
        num_nodes,
        text_inputs,
        memory_positions=None,
        view_count: int = 1,
        classification_view_count: int = 0,
        teacher_view_count: int = 0,
        compute_node_cognitive: bool = False,
    ):
        view_count = int(view_count)
        teacher_view_count = int(teacher_view_count)
        if teacher_view_count < 0 or teacher_view_count >= view_count:
            if teacher_view_count != 0 or view_count < 1:
                raise ValueError("teacher_view_count must be within [0, view_count)")
        graph_features = None
        graph_node_features = None
        if self.ablate_evidence == "graph":
            graph_logits = node_feats.new_zeros((node_feats.size(0), 2))
        elif compute_node_cognitive:
            if self.node_cognitive_head is None:
                raise ValueError(
                    "node cognitive outputs requested without node cognitive head"
                )
            graph_logits, graph_features, graph_node_features = self.graph_model(
                node_feats,
                struct_feats,
                num_nodes,
                return_features=True,
                return_node_features=True,
            )
        elif self.cognitive_head is not None:
            graph_logits, graph_features = self.graph_model(
                node_feats, struct_feats, num_nodes, return_features=True
            )
        else:
            graph_logits = self.graph_model(node_feats, struct_feats, num_nodes)
        if self.ablate_evidence == "text":
            raw_text_logits = graph_logits.new_zeros(
                (graph_logits.size(0) * view_count, 2)
            )
        else:
            raw_text_logits = self.text_model(**text_inputs).logits
        if view_count > 1:
            expected = graph_logits.size(0) * view_count
            if raw_text_logits.size(0) != expected:
                raise ValueError("Text view batch does not match graph batch")
            text_view_logits = raw_text_logits.reshape(
                graph_logits.size(0), view_count, 2
            )
            ordinary_view_count = view_count - teacher_view_count
            ordinary_view_logits = text_view_logits[:, :ordinary_view_count]
            supervised_view_logits, selected_view_indices = sample_view_subset(
                ordinary_view_logits, classification_view_count
            )
            text_logits = supervised_view_logits.mean(dim=1)
        else:
            text_view_logits = raw_text_logits.reshape(graph_logits.size(0), 1, 2)
            ordinary_view_logits = text_view_logits
            supervised_view_logits = text_view_logits
            selected_view_indices = torch.zeros(
                graph_logits.size(0), 1, dtype=torch.long,
                device=graph_logits.device,
            )
            text_logits = raw_text_logits
        if self.ablate_evidence == "memory":
            retrieval = graph_logits.new_zeros((graph_logits.size(0), 1))
        else:
            query = self.event_semantic_features(node_feats, num_nodes)
            retrieval = self.retrieve(query, memory_positions).reshape(-1, 1)
            retrieval = retrieval.to(graph_logits.dtype)
        gate_weights = None
        fusion_residual = None
        if self.fusion_type == "linear":
            fusion_input = torch.cat([graph_logits, text_logits, retrieval], dim=1)
            logits = self.fusion(fusion_input)
        elif self.fusion_type == "reliability_gate":
            logits, gate_weights = self.fusion(
                graph_logits, text_logits, retrieval, num_nodes
            )
        else:
            logits, fusion_residual = self.fusion(
                graph_logits, text_logits, retrieval, num_nodes
            )
        return {
            "logits": logits,
            "graph_logits": graph_logits,
            "text_logits": text_logits,
            "text_view_logits": text_view_logits,
            "ordinary_text_view_logits": ordinary_view_logits,
            "teacher_text_view_logits": (
                text_view_logits[:, -teacher_view_count:]
                if teacher_view_count > 0
                else None
            ),
            "supervised_text_view_logits": supervised_view_logits,
            "selected_view_indices": selected_view_indices,
            "retrieval_logits": retrieval,
            "gate_weights": gate_weights,
            "fusion_residual": fusion_residual,
            "cognitive_predictions": (
                self.cognitive_head(graph_features)
                if self.cognitive_head is not None and graph_features is not None
                else None
            ),
            "node_cognitive_predictions": (
                self.node_cognitive_head(graph_node_features)
                if compute_node_cognitive
                and self.node_cognitive_head is not None
                and graph_node_features is not None
                else None
            ),
        }

    @staticmethod
    def loss(
        outputs, labels, graph_weight=0.15, text_weight=0.15,
        class_weights=None, cognitive_targets=None, cognitive_mask=None,
        cognitive_weight=0.0,
        node_cognitive_targets=None, node_cognitive_weight=0.0,
    ):
        final_loss = F.cross_entropy(
            outputs["logits"], labels, weight=class_weights
        )
        graph_loss = F.cross_entropy(
            outputs["graph_logits"], labels, weight=class_weights
        )
        view_logits = outputs.get(
            "supervised_text_view_logits", outputs.get("text_view_logits")
        )
        if view_logits is not None and view_logits.size(1) > 1:
            repeated = labels[:, None].expand(-1, view_logits.size(1)).reshape(-1)
            text_loss = F.cross_entropy(
                view_logits.reshape(-1, 2), repeated, weight=class_weights
            )
        else:
            text_loss = F.cross_entropy(
                outputs["text_logits"], labels, weight=class_weights
            )
        total = final_loss + float(graph_weight) * graph_loss
        total = total + float(text_weight) * text_loss
        cognitive_loss = total.new_zeros(())
        if float(cognitive_weight) > 0.0:
            predictions = outputs.get("cognitive_predictions")
            if predictions is None:
                raise ValueError("cognitive loss requested without cognitive head")
            if cognitive_targets is None or cognitive_mask is None:
                raise ValueError("cognitive targets and mask are required")
            if predictions.shape != cognitive_targets.shape:
                raise ValueError("cognitive prediction/target shape mismatch")
            if cognitive_mask.shape != cognitive_targets.shape:
                raise ValueError("cognitive mask/target shape mismatch")
            squared_error = (predictions.float() - cognitive_targets.float()).square()
            active = cognitive_mask.float()
            cognitive_loss = (squared_error * active).sum() / active.sum().clamp_min(1.0)
            total = total + float(cognitive_weight) * cognitive_loss
        node_cognitive_loss = total.new_zeros(())
        if float(node_cognitive_weight) > 0.0:
            predictions = outputs.get("node_cognitive_predictions")
            if predictions is None:
                raise ValueError(
                    "node cognitive loss requested without node predictions"
                )
            if predictions.ndim != 3 or predictions.size(-1) != 14:
                raise ValueError("node cognitive predictions must be [batch,nodes,14]")
            if node_cognitive_targets is None:
                raise ValueError("node cognitive targets are required")
            required = {
                "stance", "stance_mask", "role", "role_mask",
                "salience", "salience_mask",
            }
            if set(node_cognitive_targets) != required:
                raise ValueError("unexpected node cognitive target fields")
            target_shape = predictions.shape[:2]
            for name in required:
                if node_cognitive_targets[name].shape != target_shape:
                    raise ValueError(
                        f"node cognitive {name} shape does not match predictions"
                    )
            components = []
            stance_mask = node_cognitive_targets["stance_mask"].bool()
            if bool(stance_mask.any()):
                components.append(
                    F.cross_entropy(
                        predictions[..., :6][stance_mask],
                        node_cognitive_targets["stance"][stance_mask].long(),
                    )
                )
            role_mask = node_cognitive_targets["role_mask"].bool()
            if bool(role_mask.any()):
                components.append(
                    F.cross_entropy(
                        predictions[..., 6:13][role_mask],
                        node_cognitive_targets["role"][role_mask].long(),
                    )
                )
            salience_mask = node_cognitive_targets["salience_mask"].bool()
            if bool(salience_mask.any()):
                salience_prediction = torch.sigmoid(predictions[..., 13])
                components.append(
                    F.mse_loss(
                        salience_prediction[salience_mask].float(),
                        node_cognitive_targets["salience"][salience_mask].float(),
                    )
                )
            if components:
                node_cognitive_loss = torch.stack(components).mean()
                total = total + float(node_cognitive_weight) * node_cognitive_loss
        return {
            "total": total,
            "final": final_loss,
            "graph": graph_loss,
            "text": text_loss,
            "cognitive": cognitive_loss,
            "node_cognitive": node_cognitive_loss,
        }
