"""Source-only ERM objective."""

from __future__ import annotations

import torch
from torch import nn


def source_only_objective(
    backbone: nn.Module,
    classifier: nn.Module,
    source_images: torch.Tensor,
    source_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute cross-entropy over a domain-balanced pooled source batch."""

    source_features = backbone(source_images)
    source_logits = classifier(source_features)
    classification_loss = nn.functional.cross_entropy(source_logits, source_labels)
    return {
        "total_loss": classification_loss,
        "classification_loss": classification_loss,
        "source_features": source_features,
        "source_logits": source_logits,
    }
