"""Layer-wise learning-rate groups for the DeBERTa text branch."""


def build_text_optimizer_groups(text_model, base_lr, layerwise_decay):
    """Assign progressively smaller learning rates to lower encoder layers."""
    decay = float(layerwise_decay)
    if not 0.0 < decay <= 1.0:
        raise ValueError("text layerwise LR decay must be within (0, 1]")
    if decay == 1.0:
        return [{"params": list(text_model.parameters()), "lr": float(base_lr)}]

    num_layers = int(text_model.config.num_hidden_layers)
    top_depth = num_layers + 1
    grouped = {}
    for name, parameter in text_model.named_parameters():
        if not parameter.requires_grad:
            continue
        marker = "encoder.layer."
        if marker in name:
            suffix = name.split(marker, 1)[1]
            layer_index = int(suffix.split(".", 1)[0])
            depth = layer_index + 1
        elif name.startswith(("classifier.", "pooler.")):
            depth = top_depth
        else:
            depth = 0
        learning_rate = float(base_lr) * (decay ** (top_depth - depth))
        grouped.setdefault(learning_rate, []).append(parameter)
    return [
        {"params": parameters, "lr": learning_rate}
        for learning_rate, parameters in sorted(grouped.items())
    ]
