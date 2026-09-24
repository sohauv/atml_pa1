"""ResNet-18 feature extractor for PACS adaptation experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


FEATURE_DIMENSION = 512


def freeze_batchnorm_running_statistics(module: nn.Module) -> None:
    """Put BatchNorm layers in eval mode without freezing affine parameters."""

    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


class ResNet18Backbone(nn.Module):
    """ImageNet-pretrained ResNet-18 returning its 512-dimensional feature."""

    feature_dimension = FEATURE_DIMENSION

    def __init__(
        self,
        pretrained: bool = True,
        freeze_batchnorm_statistics: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        network = resnet18(weights=weights)
        network.fc = nn.Identity()
        self.network = network
        self.freeze_batchnorm_statistics = freeze_batchnorm_statistics

        if self.freeze_batchnorm_statistics:
            freeze_batchnorm_running_statistics(self.network)

    def train(self, mode: bool = True):
        """Preserve fixed BatchNorm statistics whenever training mode is set."""

        super().train(mode)
        if mode and self.freeze_batchnorm_statistics:
            freeze_batchnorm_running_statistics(self.network)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.network(images)
        if features.ndim != 2 or features.shape[1] != self.feature_dimension:
            raise RuntimeError(
                f"Expected ResNet-18 features shaped [batch, {self.feature_dimension}], "
                f"received {tuple(features.shape)}."
            )
        return features


def batchnorm_state(backbone: nn.Module) -> list[dict[str, object]]:
    """Return BatchNorm state for smoke tests and reproducibility checks."""

    state = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            state.append(
                {
                    "name": name,
                    "training": module.training,
                    "weight_trainable": module.weight is not None
                    and module.weight.requires_grad,
                    "bias_trainable": module.bias is not None and module.bias.requires_grad,
                }
            )
    return state
