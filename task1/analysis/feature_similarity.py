from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional


def validate_feature_pair(
    clean_features: torch.Tensor,
    transformed_features: torch.Tensor,
) -> None:
    if clean_features.ndim != 2:
        raise ValueError(
            "Clean features must have shape "
            "[number_of_images, feature_dimension]."
        )

    if transformed_features.ndim != 2:
        raise ValueError(
            "Transformed features must have shape "
            "[number_of_images, feature_dimension]."
        )

    if clean_features.shape != transformed_features.shape:
        raise ValueError(
            "Clean and transformed features must have "
            "identical shapes, but received "
            f"{tuple(clean_features.shape)} and "
            f"{tuple(transformed_features.shape)}."
        )

    if clean_features.shape[0] == 0:
        raise ValueError(
            "Cannot compute stability for empty features."
        )


def compute_pairwise_cosine_similarity(
    clean_features: torch.Tensor,
    transformed_features: torch.Tensor,
) -> torch.Tensor:
    validate_feature_pair(
        clean_features=clean_features,
        transformed_features=transformed_features,
    )

    clean_features = clean_features.float()
    transformed_features = transformed_features.float()

    normalized_clean = functional.normalize(
        clean_features,
        p=2,
        dim=1,
    )
    normalized_transformed = functional.normalize(
        transformed_features,
        p=2,
        dim=1,
    )

    return (
        normalized_clean
        * normalized_transformed
    ).sum(dim=1)


def summarize_cosine_stability(
    clean_features: torch.Tensor,
    transformed_features: torch.Tensor,
) -> dict[str, Any]:
    similarities = compute_pairwise_cosine_similarity(
        clean_features=clean_features,
        transformed_features=transformed_features,
    )

    return {
        "number_of_pairs": int(similarities.numel()),
        "mean_cosine_stability": float(
            similarities.mean().item()
        ),
        "standard_deviation": float(
            similarities.std(
                unbiased=False
            ).item()
        ),
        "median": float(
            similarities.median().item()
        ),
        "minimum": float(
            similarities.min().item()
        ),
        "maximum": float(
            similarities.max().item()
        ),
    }


def average_stability_records(
    records: list[dict[str, Any]],
) -> dict[str, float]:
    if not records:
        raise ValueError(
            "At least one stability record is required."
        )

    return {
        "mean_cosine_stability": float(
            sum(
                record["mean_cosine_stability"]
                for record in records
            )
            / len(records)
        ),
        "standard_deviation_across_directions": float(
            torch.tensor(
                [
                    record["mean_cosine_stability"]
                    for record in records
                ],
                dtype=torch.float32,
            )
            .std(unbiased=False)
            .item()
        ),
        "number_of_directions": int(len(records)),
        "number_of_pairs_per_direction": int(
            records[0]["number_of_pairs"]
        ),
    }