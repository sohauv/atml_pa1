"""Source-only DAN-DG objective with pairwise source-domain MMD."""

from __future__ import annotations

from itertools import combinations

import torch
from torch import nn

from task2.methods.dan import multi_kernel_mmd


def dan_dg_objective(
    backbone: nn.Module,
    classifier: nn.Module,
    domain_images: dict[str, torch.Tensor],
    domain_labels: dict[str, torch.Tensor],
    alignment_weight: float = 1.0,
    bandwidth_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, torch.Tensor]:
    """Compute ERM loss plus average MMD over all source-domain pairs."""

    features = {name: backbone(images) for name, images in domain_images.items()}
    logits = {name: classifier(value) for name, value in features.items()}
    classification_loss = torch.stack(
        [nn.functional.cross_entropy(logits[name], domain_labels[name]) for name in logits]
    ).mean()

    pair_losses = []
    bandwidths = []
    for first, second in combinations(features, 2):
        loss, bandwidth = multi_kernel_mmd(
            features[first],
            features[second],
            bandwidth_multipliers=bandwidth_multipliers,
        )
        pair_losses.append(loss)
        bandwidths.append(bandwidth)
    alignment_loss = torch.stack(pair_losses).mean()
    mean_bandwidth = torch.stack(bandwidths).mean()
    total_loss = classification_loss + float(alignment_weight) * alignment_loss
    return {
        "total_loss": total_loss,
        "classification_loss": classification_loss,
        "alignment_loss": alignment_loss,
        "mmd_bandwidth": mean_bandwidth,
    }
