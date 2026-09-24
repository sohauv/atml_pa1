"""Held-out source-versus-target domain separability diagnostic."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    data_loader,
    device: torch.device,
) -> np.ndarray:
    """Extract frozen features without reading class labels."""

    backbone.eval()
    features = []
    for batch in data_loader:
        images = batch["image"] if isinstance(batch, dict) else batch[0]
        images = images.to(device, non_blocking=True)
        features.append(backbone(images).cpu().numpy())
    return np.concatenate(features, axis=0)


def domain_separability_score(
    source_features: np.ndarray,
    target_features: np.ndarray,
    seed: int = 6304,
    test_fraction: float = 0.3,
    c: float = 1.0,
) -> dict:
    """Train a balanced logistic-regression domain classifier on equal samples."""

    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("Domain features must be two-dimensional matrices.")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Source and target feature dimensions must match.")

    rng = np.random.default_rng(seed)
    number_per_domain = min(len(source_features), len(target_features))
    if number_per_domain < 2:
        raise ValueError("At least two examples per domain are required.")
    source_indices = rng.choice(len(source_features), number_per_domain, replace=False)
    target_indices = rng.choice(len(target_features), number_per_domain, replace=False)
    features = np.concatenate(
        [source_features[source_indices], target_features[target_indices]],
        axis=0,
    )
    domain_labels = np.concatenate(
        [
            np.zeros(number_per_domain, dtype=np.int64),
            np.ones(number_per_domain, dtype=np.int64),
        ]
    )

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        domain_labels,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=domain_labels,
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "number_per_domain": int(number_per_domain),
        "train_examples": int(len(train_labels)),
        "test_examples": int(len(test_labels)),
        "test_fraction": float(test_fraction),
        "logistic_regression_c": float(c),
        "seed": int(seed),
        "domain_label_order": ["source", "target"],
        "confusion_matrix": confusion_matrix(
            test_labels,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }
