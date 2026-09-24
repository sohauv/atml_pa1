"""Frozen open-set unknownness scores used by Task 4."""

from __future__ import annotations

import numpy as np
import torch


def logit_unknownness_scores(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return scores where larger values consistently mean more unknown-like."""

    probabilities = logits.softmax(dim=1)
    return {
        "msp": 1.0 - probabilities.max(dim=1).values,
        "mls": -logits.max(dim=1).values,
        "energy": -torch.logsumexp(logits, dim=1),
    }


def fit_shared_diagonal_gaussian(
    features: torch.Tensor,
    labels: torch.Tensor,
    number_of_classes: int,
    regularization: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit class means and one pooled diagonal covariance on known features."""

    if features.ndim != 2:
        raise ValueError("Features must have shape [examples, feature_dimension].")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("Labels must be one-dimensional and match the features.")

    means = []
    residuals = []
    for class_id in range(number_of_classes):
        class_features = features[labels == class_id]
        if class_features.shape[0] == 0:
            raise ValueError(f"Class {class_id} has no fitting examples.")
        class_mean = class_features.mean(dim=0)
        means.append(class_mean)
        residuals.append(class_features - class_mean)
    means = torch.stack(means)
    pooled_residuals = torch.cat(residuals, dim=0)
    variance = pooled_residuals.square().mean(dim=0).clamp_min(regularization)
    return means, variance


def mahalanobis_unknownness(
    features: torch.Tensor,
    class_means: torch.Tensor,
    shared_diagonal_variance: torch.Tensor,
) -> torch.Tensor:
    """Minimum class-conditional squared Mahalanobis distance."""

    differences = features[:, None, :] - class_means[None, :, :]
    distances = (
        differences.square() / shared_diagonal_variance[None, None, :]
    ).sum(dim=2)
    return distances.min(dim=1).values


def percentile_threshold(scores: np.ndarray, percentile: float = 95.0) -> float:
    """Fix an unknownness threshold using known validation scores only."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Threshold scores must be a non-empty vector.")
    if not np.isfinite(scores).all():
        raise ValueError("Threshold scores contain a non-finite value.")
    return float(np.percentile(scores, percentile))
