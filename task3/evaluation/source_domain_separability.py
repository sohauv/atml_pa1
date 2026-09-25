"""Linear source-domain separability diagnostic."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def extract_features(backbone, loader, device) -> np.ndarray:
    backbone.eval()
    parts = []
    for batch in loader:
        images = batch["image"] if isinstance(batch, dict) else batch[0]
        parts.append(backbone(images.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(parts)


def source_domain_separability(feature_sets: dict[str, np.ndarray], seed: int = 6304) -> float:
    """Fit a balanced multinomial linear probe on equally sized domains."""

    rng = np.random.default_rng(seed)
    count = min(len(values) for values in feature_sets.values())
    features, labels = [], []
    for domain_id, (_, values) in enumerate(feature_sets.items()):
        chosen = rng.choice(len(values), size=count, replace=False)
        features.append(values[chosen])
        labels.append(np.full(count, domain_id))
    x = np.concatenate(features)
    y = np.concatenate(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=5000,
        ),
    )
    probe.fit(x_train, y_train)
    return float(accuracy_score(y_test, probe.predict(x_test)))
