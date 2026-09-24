"""CDAN class-conditional adversarial alignment objective."""

from __future__ import annotations

import torch
from torch import nn

from task2.methods.dann import domain_labels


def multilinear_conditioning(
    features: torch.Tensor,
    class_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Flatten the per-example outer product f tensor p without detaching either."""

    if features.shape[0] != class_probabilities.shape[0]:
        raise ValueError("Feature and probability batch sizes must match.")
    outer_products = torch.bmm(
        features.unsqueeze(2),
        class_probabilities.unsqueeze(1),
    )
    return outer_products.flatten(start_dim=1)


def cdan_objective(
    backbone: nn.Module,
    classifier: nn.Module,
    discriminator: nn.Module,
    source_images: torch.Tensor,
    source_labels: torch.Tensor,
    target_images: torch.Tensor,
    reversal_strength: float,
    domain_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute source classification and conditional adversarial domain loss."""

    source_features = backbone(source_images)
    target_features = backbone(target_images)
    combined_features = torch.cat([source_features, target_features], dim=0)

    combined_logits = classifier(combined_features)
    source_count = source_features.shape[0]
    source_logits = combined_logits[:source_count]
    classification_loss = nn.functional.cross_entropy(source_logits, source_labels)

    probabilities = combined_logits.softmax(dim=1)
    normalized_features = nn.functional.normalize(
        combined_features,
        p=2,
        dim=1,
    )
    conditioned_features = multilinear_conditioning(
        normalized_features,
        probabilities,
    )
    labels = domain_labels(
        source_count,
        target_features.shape[0],
        combined_features.device,
    )
    domain_logits = discriminator(conditioned_features, reversal_strength)
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
