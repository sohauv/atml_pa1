from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.lines import Line2D

from task1.analysis.feature_similarity import (
    validate_feature_pair,
)


def fit_joint_umap(
    clean_features: torch.Tensor,
    transformed_features: torch.Tensor,
    seed: int = 6304,
    number_of_neighbors: int = 15,
    minimum_distance: float = 0.1,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    validate_feature_pair(
        clean_features=clean_features,
        transformed_features=transformed_features,
    )

    combined_features = torch.cat(
        [
            clean_features.float(),
            transformed_features.float(),
        ],
        dim=0,
    ).cpu().numpy()

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=number_of_neighbors,
        min_dist=minimum_distance,
        metric=metric,
        random_state=seed,
        transform_seed=seed,
    )

    combined_projection = reducer.fit_transform(
        combined_features
    )

    number_of_clean_examples = clean_features.shape[0]

    clean_projection = combined_projection[
        :number_of_clean_examples
    ]
    transformed_projection = combined_projection[
        number_of_clean_examples:
    ]

    return clean_projection, transformed_projection


def create_joint_umap_plot(
    clean_features: torch.Tensor,
    transformed_features: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    model_name: str,
    intervention_name: str,
    output_path: str | Path,
    seed: int = 6304,
    number_of_neighbors: int = 15,
    minimum_distance: float = 0.1,
    metric: str = "cosine",
) -> None:
    if len(labels) != clean_features.shape[0]:
        raise ValueError(
            "Labels and feature pairs have different "
            f"lengths: {len(labels)} and "
            f"{clean_features.shape[0]}."
        )

    clean_projection, transformed_projection = (
        fit_joint_umap(
            clean_features=clean_features,
            transformed_features=transformed_features,
            seed=seed,
            number_of_neighbors=number_of_neighbors,
            minimum_distance=minimum_distance,
            metric=metric,
        )
    )

    labels_array = labels.cpu().numpy()
    color_map = plt.get_cmap(
        "tab10",
        len(class_names),
    )

    figure, axis = plt.subplots(
        figsize=(9, 7),
    )

    for class_id, class_name in enumerate(class_names):
        class_mask = labels_array == class_id

        if not np.any(class_mask):
            continue

        axis.scatter(
            clean_projection[class_mask, 0],
            clean_projection[class_mask, 1],
            color=color_map(class_id),
            marker="o",
            s=24,
            alpha=0.72,
            edgecolors="none",
        )

        axis.scatter(
            transformed_projection[class_mask, 0],
            transformed_projection[class_mask, 1],
            color=color_map(class_id),
            marker="^",
            s=28,
            alpha=0.72,
            edgecolors="none",
        )

    class_legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map(class_id),
            markeredgecolor="none",
            label=class_name,
            markersize=7,
        )
        for class_id, class_name in enumerate(class_names)
    ]

    condition_legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="black",
            label="Clean",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            color="black",
            label="Transformed",
            markersize=7,
        ),
    ]

    class_legend = axis.legend(
        handles=class_legend_handles,
        title="Ground-truth class",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
    )
    axis.add_artist(class_legend)

    axis.legend(
        handles=condition_legend_handles,
        title="Condition",
        bbox_to_anchor=(1.02, 0.42),
        loc="upper left",
        borderaxespad=0.0,
    )

    axis.set_title(
        f"{model_name}: clean vs. {intervention_name}"
    )
    axis.set_xlabel("UMAP dimension 1")
    axis.set_ylabel("UMAP dimension 2")
    axis.grid(alpha=0.15)

    settings_text = (
        f"n_neighbors={number_of_neighbors}, "
        f"min_dist={minimum_distance}, "
        f"metric={metric}, seed={seed}"
    )

    figure.text(
        0.5,
        0.01,
        settings_text,
        ha="center",
        fontsize=9,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout(rect=(0.0, 0.04, 0.82, 1.0))
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)