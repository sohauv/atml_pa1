"""PROSER classifier-placeholder and data-placeholder objectives."""

from __future__ import annotations

import torch
from torch import nn


def strongest_dummy_logits(dummy_logits: torch.Tensor) -> torch.Tensor:
    """Collapse multiple dummy classifiers to the strongest response."""

    return dummy_logits.max(dim=1, keepdim=True).values


def different_class_partners(labels: torch.Tensor) -> torch.Tensor:
    """Sample one partner with a different label for every batch example."""

    partners = []
    for index in range(labels.shape[0]):
        candidates = torch.nonzero(labels != labels[index], as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("Manifold mixup requires at least two classes per half-batch.")
        choice = torch.randint(candidates.numel(), (1,), device=labels.device)
        partners.append(candidates[choice])
    return torch.cat(partners)


def classifier_placeholder_loss(
    known_logits: torch.Tensor,
    dummy_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equation 5: preserve the target and make a dummy the runner-up."""

    strongest_dummy = strongest_dummy_logits(dummy_logits)
    augmented_logits = torch.cat([known_logits, strongest_dummy], dim=1)
    classification_loss = nn.functional.cross_entropy(augmented_logits, labels)

    masked_known_logits = known_logits.clone()
    masked_known_logits.scatter_(1, labels[:, None], float("-inf"))
    masked_logits = torch.cat([masked_known_logits, strongest_dummy], dim=1)
    dummy_targets = torch.full_like(labels, known_logits.shape[1])
    placeholder_loss = nn.functional.cross_entropy(masked_logits, dummy_targets)
    combined = classification_loss + float(beta) * placeholder_loss
    return combined, classification_loss, placeholder_loss


def data_placeholder_loss(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation 7: train different-class layer-2 mixups as dummy examples."""

    hidden = model.forward_to_layer2(images)
    partner_indices = different_class_partners(labels)
    beta_distribution = torch.distributions.Beta(alpha, alpha)
    mixing = beta_distribution.sample((hidden.shape[0],)).to(hidden.device)
    mixing = mixing.view(-1, 1, 1, 1)
    mixed_hidden = mixing * hidden + (1.0 - mixing) * hidden[partner_indices]

    mixed_features = model.forward_from_layer2(mixed_hidden)
    known_logits, dummy_logits = model.classify_features(mixed_features)
    augmented_logits = torch.cat(
        [known_logits, strongest_dummy_logits(dummy_logits)], dim=1
    )
    dummy_targets = torch.full_like(labels, known_logits.shape[1])
    loss = nn.functional.cross_entropy(augmented_logits, dummy_targets)
    return loss, mixing.flatten()


def proser_objective(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 0.1,
    mixup_alpha: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Split a batch equally and compute the paper's combined objective."""

    if images.shape[0] < 4:
        raise ValueError("PROSER requires a batch containing at least four examples.")
    split = images.shape[0] // 2
    ordinary_images, ordinary_labels = images[:split], labels[:split]
    mixup_images, mixup_labels = images[split:], labels[split:]

    known_logits, dummy_logits, _ = model.forward_all(ordinary_images)
    classifier_loss, classification_loss, placeholder_loss = (
        classifier_placeholder_loss(
            known_logits, dummy_logits, ordinary_labels, beta=beta
        )
    )
    mixup_loss, mixing = data_placeholder_loss(
        model, mixup_images, mixup_labels, alpha=mixup_alpha
    )
    total_loss = classifier_loss + float(gamma) * mixup_loss
    return {
        "total_loss": total_loss,
        "classifier_placeholder_loss": classifier_loss,
        "classification_loss": classification_loss,
        "runner_up_dummy_loss": placeholder_loss,
        "data_placeholder_loss": mixup_loss,
        "mean_mixup_lambda": mixing.mean(),
    }


def placeholder_unknownness(
    known_logits: torch.Tensor,
    dummy_logits: torch.Tensor,
    temperature: float = 1024.0,
) -> torch.Tensor:
    """Reference DeltaP score: strongest dummy probability minus strongest known."""

    reduced_logits = torch.cat(
        [known_logits, strongest_dummy_logits(dummy_logits)], dim=1
    )
    probabilities = (reduced_logits / float(temperature)).softmax(dim=1)
    return probabilities[:, -1] - probabilities[:, :-1].max(dim=1).values
