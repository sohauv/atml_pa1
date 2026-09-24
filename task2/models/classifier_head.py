"""Seven-class linear classifier used by every Task 2 method."""

from __future__ import annotations

import torch
from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, feature_dimension: int = 512, number_of_classes: int = 7) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dimension, number_of_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)
