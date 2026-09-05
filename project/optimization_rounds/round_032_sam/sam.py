from dataclasses import dataclass
from typing import Iterable, List

import torch


@dataclass(frozen=True)
class TorchRNGState:
    cpu: torch.Tensor
    cuda: tuple


def capture_torch_rng_state() -> TorchRNGState:
    cuda = tuple(
        state.clone() for state in torch.cuda.get_rng_state_all()
    ) if torch.cuda.is_available() else ()
    return TorchRNGState(cpu=torch.get_rng_state().clone(), cuda=cuda)


def restore_torch_rng_state(state: TorchRNGState) -> None:
    torch.set_rng_state(state.cpu)
    if state.cuda:
        torch.cuda.set_rng_state_all(list(state.cuda))


def effective_batch_windows(iterable: Iterable, accumulation_steps: int):
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    window: List[object] = []
    for item in iterable:
        window.append(item)
        if len(window) == accumulation_steps:
            yield window
            window = []
    if window:
        yield window


def manually_unscale_gradients_(optimizer, scale: float) -> bool:
    """Unscale first-pass AMP gradients without mutating GradScaler state."""

    scale = float(scale)
    if not scale > 0.0:
        raise ValueError("gradient scale must be positive")
    all_finite = True
    inverse = 1.0 / scale
    with torch.no_grad():
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.grad.mul_(inverse)
                if not bool(torch.isfinite(parameter.grad).all()):
                    all_finite = False
    return all_finite


class SAMController:
    """Apply and restore the SAM neighborhood perturbation around AdamW."""

    def __init__(self, optimizer, rho: float = 0.05):
        if rho <= 0.0:
            raise ValueError("rho must be positive")
        self.optimizer = optimizer
        self.rho = float(rho)
        self._originals = {}

    def _parameters_with_grad(self):
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    yield parameter

    def grad_norm(self):
        parameters = list(self._parameters_with_grad())
        if not parameters:
            raise RuntimeError("SAM requires at least one gradient")
        device = parameters[0].grad.device
        norms = [
            torch.linalg.vector_norm(parameter.grad.detach()).to(device)
            for parameter in parameters
        ]
        return torch.linalg.vector_norm(torch.stack(norms))

    @torch.no_grad()
    def perturb(self):
        if self._originals:
            raise RuntimeError("SAM parameters are already perturbed")
        norm = self.grad_norm()
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            raise FloatingPointError("SAM gradient norm is not finite and positive")
        scale = self.rho / (norm + 1e-12)
        for parameter in self._parameters_with_grad():
            self._originals[parameter] = parameter.detach().clone()
            perturbation = parameter.grad.detach().mul(scale.to(parameter))
            parameter.add_(perturbation)
        return float(norm)

    @torch.no_grad()
    def restore(self):
        if not self._originals:
            raise RuntimeError("SAM restore called without a perturbation")
        for parameter, original in self._originals.items():
            parameter.copy_(original)
        self._originals.clear()

    @property
    def is_perturbed(self):
        return bool(self._originals)
