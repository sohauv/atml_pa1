"""Domain discriminator and gradient-reversal layer for DANN and CDAN."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def gradient_reverse(inputs: torch.Tensor, strength: float) -> torch.Tensor:
    """Identity in the forward pass and -strength times the backward gradient."""

    return _GradientReversalFunction.apply(inputs, strength)


def gradient_reversal_schedule(
    progress: float,
    maximum_strength: float = 1.0,
    gamma: float = 10.0,
) -> float:
    """Standard DANN schedule for progress in [0, 1]."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"Training progress must be in [0, 1], received {progress}.")
    return maximum_strength * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


class DomainDiscriminator(nn.Module):
    """Two-class source/target discriminator shared by DANN and CDAN."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 2),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        reversal_strength: float,
    ) -> torch.Tensor:
        reversed_inputs = gradient_reverse(inputs, reversal_strength)
        return self.network(reversed_inputs)
