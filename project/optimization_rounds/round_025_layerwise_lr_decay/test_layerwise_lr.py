import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization_rounds.round_025_layerwise_lr_decay.optimizer import (  # noqa: E402
    build_text_optimizer_groups,
)


class TinyEncoder(torch.nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers)
        self.deberta = torch.nn.Module()
        self.deberta.embeddings = torch.nn.Linear(2, 2)
        self.deberta.encoder = torch.nn.Module()
        self.deberta.encoder.layer = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(layers)]
        )
        self.pooler = torch.nn.Linear(2, 2)
        self.classifier = torch.nn.Linear(2, 2)


def parameter_lrs(model, groups):
    mapping = {}
    for group in groups:
        for parameter in group["params"]:
            mapping[id(parameter)] = group["lr"]
    return {name: mapping[id(parameter)] for name, parameter in model.named_parameters()}


def test_decay_one_preserves_single_original_group():
    model = TinyEncoder()
    groups = build_text_optimizer_groups(model, 2e-5, 1.0)
    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(2e-5)
    assert len(groups[0]["params"]) == len(list(model.parameters()))


def test_lower_layers_receive_monotonically_smaller_learning_rates():
    model = TinyEncoder(layers=3)
    lrs = parameter_lrs(model, build_text_optimizer_groups(model, 2e-5, 0.95))
    embedding_lr = lrs["deberta.embeddings.weight"]
    layer0_lr = lrs["deberta.encoder.layer.0.weight"]
    layer1_lr = lrs["deberta.encoder.layer.1.weight"]
    layer2_lr = lrs["deberta.encoder.layer.2.weight"]
    classifier_lr = lrs["classifier.weight"]
    assert embedding_lr < layer0_lr < layer1_lr < layer2_lr < classifier_lr
    assert classifier_lr == pytest.approx(2e-5)


@pytest.mark.parametrize("decay", [0.0, -0.1, 1.01])
def test_invalid_decay_is_rejected(decay):
    with pytest.raises(ValueError):
        build_text_optimizer_groups(TinyEncoder(), 2e-5, decay)
