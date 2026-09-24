"""Non-adaptive sharpness-aware minimization utilities."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def ascent_step(parameters, rho: float) -> list[tuple[nn.Parameter, torch.Tensor]]:
    """Perturb parameters by rho in the normalized gradient direction."""

    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return []
    norm = torch.linalg.vector_norm(
        torch.stack([parameter.grad.norm(p=2) for parameter in parameters]), ord=2
    )
    scale = float(rho) / (norm + 1e-12)
    perturbations = []
    for parameter in parameters:
        perturbation = parameter.grad * scale
        parameter.add_(perturbation)
        perturbations.append((parameter, perturbation))
    return perturbations


@torch.no_grad()
def restore_parameters(perturbations: list[tuple[nn.Parameter, torch.Tensor]]) -> None:
    for parameter, perturbation in perturbations:
        parameter.sub_(perturbation)
