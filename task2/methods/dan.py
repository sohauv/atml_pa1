"""DAN-style multi-kernel MMD alignment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def squared_pairwise_distances(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable squared Euclidean distances."""

    distances = torch.cdist(first, second, p=2).square()
    return distances.clamp_min(0.0)


def median_squared_distance(features: torch.Tensor) -> torch.Tensor:
    """Median non-diagonal squared distance, detached from optimization."""

    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("At least two feature vectors are required for bandwidth selection.")
    with torch.no_grad():
        distances = torch.pdist(features, p=2).square()
        positive_distances = distances[distances > 0]
        if positive_distances.numel() == 0:
            return features.new_tensor(1.0)
        return positive_distances.median().clamp_min(1e-6)


def multi_kernel_rbf(
    first: torch.Tensor,
    second: torch.Tensor,
    base_bandwidth: torch.Tensor,
    bandwidth_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Sum RBF kernels with bandwidths relative to the current batch median."""

    squared_distances = squared_pairwise_distances(first, second)
    kernels = []
    for multiplier in bandwidth_multipliers:
        bandwidth = (base_bandwidth * float(multiplier)).clamp_min(1e-6)
        kernels.append(torch.exp(-squared_distances / (2.0 * bandwidth)))
    return torch.stack(kernels, dim=0).sum(dim=0)


def multi_kernel_mmd(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    bandwidth_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Biased empirical squared MMD and its batch-derived base bandwidth."""

    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("MMD inputs must be [batch, feature] matrices.")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Source and target feature dimensions must match.")

    combined = torch.cat([source_features, target_features], dim=0)
    base_bandwidth = median_squared_distance(combined)
    source_source = multi_kernel_rbf(
        source_features,
        source_features,
        base_bandwidth,
        bandwidth_multipliers,
    )
    target_target = multi_kernel_rbf(
        target_features,
        target_features,
        base_bandwidth,
        bandwidth_multipliers,
    )
    source_target = multi_kernel_rbf(
        source_features,
        target_features,
        base_bandwidth,
        bandwidth_multipliers,
    )
    mmd = source_source.mean() + target_target.mean() - 2.0 * source_target.mean()
    return mmd.clamp_min(0.0), base_bandwidth


def dan_objective(
    backbone: nn.Module,
    classifier: nn.Module,
    source_images: torch.Tensor,
    source_labels: torch.Tensor,
    target_images: torch.Tensor,
    mmd_weight: float = 1.0,
    bandwidth_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
) -> dict[str, torch.Tensor]:
    """Compute source classification plus source-target MMD alignment."""

    source_features = backbone(source_images)
    target_features = backbone(target_images)
    source_logits = classifier(source_features)
    classification_loss = nn.functional.cross_entropy(source_logits, source_labels)
    alignment_loss, bandwidth = multi_kernel_mmd(
        source_features,
        target_features,
        bandwidth_multipliers,
    )
    total_loss = classification_loss + float(mmd_weight) * alignment_loss
    return {
        "total_loss": total_loss,
        "classification_loss": classification_loss,
        "alignment_loss": alignment_loss,
        "mmd_bandwidth": bandwidth,
        "source_features": source_features,
        "target_features": target_features,
        "source_logits": source_logits,
    }
