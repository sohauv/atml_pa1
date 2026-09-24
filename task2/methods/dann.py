"""DANN marginal adversarial alignment objective."""

from __future__ import annotations

import torch
from torch import nn


def domain_labels(
    source_count: int,
    target_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Use class 0 for source and class 1 for target."""

    return torch.cat(
        [
            torch.zeros(source_count, dtype=torch.long, device=device),
            torch.ones(target_count, dtype=torch.long, device=device),
        ],
        dim=0,
    )


def dann_objective(
    backbone: nn.Module,
    classifier: nn.Module,
    discriminator: nn.Module,
    source_images: torch.Tensor,
    source_labels: torch.Tensor,
    target_images: torch.Tensor,
    reversal_strength: float,
    domain_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute source classification and binary adversarial domain loss."""

    source_features = backbone(source_images)
    target_features = backbone(target_images)
    source_logits = classifier(source_features)
    classification_loss = nn.functional.cross_entropy(source_logits, source_labels)

    combined_features = torch.cat([source_features, target_features], dim=0)
    combined_features = nn.functional.normalize(
        combined_features,
        p=2,
        dim=1,
    )
    labels = domain_labels(
        source_features.shape[0],
        target_features.shape[0],
        combined_features.device,
    )
    domain_logits = discriminator(combined_features, reversal_strength)
    domain_loss = nn.functional.cross_entropy(domain_logits, labels)
    domain_accuracy = (domain_logits.argmax(dim=1) == labels).float().mean()
    total_loss = classification_loss + float(domain_loss_weight) * domain_loss

    return {
        "total_loss": total_loss,
        "classification_loss": classification_loss,
        "domain_loss": domain_loss,
        "domain_accuracy": domain_accuracy,
        "source_features": source_features,
        "target_features": target_features,
        "source_logits": source_logits,
        "domain_logits": domain_logits,
    }
