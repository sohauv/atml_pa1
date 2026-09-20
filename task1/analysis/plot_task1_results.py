"""Create aggregate Task 1 figures from saved experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("task1/results")
FIGURES_DIR = RESULTS_DIR / "figures" / "summary"
TABLES_DIR = RESULTS_DIR / "summary_tables"

MODEL_ORDER = [
    "resnet50_linear",
    "vit_b_16_linear",
    "clip_vit_b_32_linear",
    "clip_zero_shot",
]
REPRESENTATION_MODEL_ORDER = MODEL_ORDER[:3]
MODEL_LABELS = {
    "resnet50_linear": "ResNet-50\nlinear",
    "vit_b_16_linear": "ViT-B/16\nlinear",
    "clip_vit_b_32_linear": "CLIP ViT-B/32\nlinear",
    "clip_zero_shot": "CLIP ViT-B/32\nzero-shot",
}
MODEL_COLORS = {
    "resnet50_linear": "#4C78A8",
    "vit_b_16_linear": "#F58518",
    "clip_vit_b_32_linear": "#54A24B",
    "clip_zero_shot": "#B279A2",
}


def load_json(filename: str) -> dict:
    with (RESULTS_DIR / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_figure(figure: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {FIGURES_DIR / filename}")


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.set_axisbelow(True)


def plot_clean_and_interventions(clean: dict, interventions: dict) -> list[dict]:
    conditions = ["clean", "grayscale", "hue_rotation_0.5", "patch_shuffle_4x4"]
    condition_labels = ["Clean", "Grayscale", "Hue rotation", "Patch shuffle"]
    x = np.arange(len(MODEL_ORDER))
    width = 0.19
    rows = []

    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    for condition_index, (condition, label) in enumerate(zip(conditions, condition_labels)):
        values = []
        for model in MODEL_ORDER:
            metrics = clean[model] if condition == "clean" else interventions[model][condition]
            accuracy = metrics["accuracy"]
            values.append(accuracy)
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "accuracy": accuracy,
                    "macro_f1": metrics["macro_f1"],
                    "mean_maximum_confidence": metrics["mean_maximum_confidence"],
                }
            )

        offset = (condition_index - (len(conditions) - 1) / 2) * width
        axis.bar(x + offset, values, width=width, label=label)

    axis.set_title("Classification performance under image interventions")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.78, 1.0)
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axis.legend(ncols=2, frameon=False, loc="lower left")
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, "performance_interventions.png")
    return rows


def plot_translation_robustness(interventions: dict) -> list[dict]:
    displacements = [0, 8, 16, 32]
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for model in MODEL_ORDER:
        averaged = interventions[model]["translation_averaged"]
        accuracies = [averaged[str(value)]["accuracy"] for value in displacements]
        consistencies = [
            averaged[str(value)]["prediction_consistency"] for value in displacements
        ]

        axes[0].plot(
            displacements,
            accuracies,
            marker="o",
            linewidth=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model].replace("\n", " "),
        )
        axes[1].plot(
            displacements,
            consistencies,
            marker="o",
            linewidth=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model].replace("\n", " "),
        )

        for displacement, accuracy, consistency in zip(
            displacements, accuracies, consistencies
        ):
            rows.append(
                {
                    "model": model,
                    "displacement_pixels": displacement,
                    "accuracy": accuracy,
                    "prediction_consistency": consistency,
                }
            )

    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Score")
    axes[1].set_title("Prediction consistency with clean input")
    for axis in axes:
        axis.set_xlabel("Translation displacement (pixels)")
        axis.set_xticks(displacements)
        axis.set_ylim(0.92, 1.005)
        style_axis(axis)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    figure.suptitle("Translation robustness averaged across four directions")
    figure.tight_layout()
    save_figure(figure, "translation_robustness.png")
    return rows


def plot_cue_conflicts(cue_results: dict) -> list[dict]:
    x = np.arange(len(MODEL_ORDER))
    shape_counts = []
    texture_counts = []
    other_counts = []
    shape_biases = []
    coverages = []
    rows = []

    for model in MODEL_ORDER:
        metrics = cue_results["models"][model]["overall"]
        coverage = metrics["shape_texture_decisions"] / metrics["number_of_images"]
        shape_counts.append(metrics["shape_choices"])
        texture_counts.append(metrics["texture_choices"])
        other_counts.append(metrics["other_choices"])
        shape_biases.append(metrics["shape_bias"])
        coverages.append(coverage)
        rows.append(
            {
                "model": model,
                "number_of_images": metrics["number_of_images"],
                "shape_choices": metrics["shape_choices"],
                "texture_choices": metrics["texture_choices"],
                "other_choices": metrics["other_choices"],
                "shape_bias": metrics["shape_bias"],
                "coverage": coverage,
            }
        )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].bar(x, shape_counts, label="Shape", color="#4C78A8")
    axes[0].bar(x, texture_counts, bottom=shape_counts, label="Texture", color="#F58518")
    stacked_bottom = np.asarray(shape_counts) + np.asarray(texture_counts)
    axes[0].bar(x, other_counts, bottom=stacked_bottom, label="Other", color="#BAB0AC")
    axes[0].set_title("Prediction counts")
    axes[0].set_ylabel("Number of cue-conflict images")
    axes[0].legend(frameon=False, ncols=3, fontsize=8)

    width = 0.36
    axes[1].bar(x - width / 2, shape_biases, width, label="Shape bias", color="#54A24B")
    axes[1].bar(x + width / 2, coverages, width, label="Coverage", color="#E45756")
    axes[1].set_title("Shape bias and decision coverage")
    axes[1].set_ylabel("Proportion")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
        style_axis(axis)
    figure.suptitle("Cue-conflict evaluation")
    figure.tight_layout()
    save_figure(figure, "cue_conflict_summary.png")
    return rows


def plot_representation_stability(representation_results: dict) -> list[dict]:
    stability_conditions = ["grayscale", "patch_shuffle_4x4", "cue_conflict"]
    condition_labels = ["Grayscale", "Patch shuffle", "Cue conflict"]
    displacements = [8, 16, 32]
    x = np.arange(len(REPRESENTATION_MODEL_ORDER))
    width = 0.24
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    for condition_index, (condition, label) in enumerate(
        zip(stability_conditions, condition_labels)
    ):
        values = []
        for model in REPRESENTATION_MODEL_ORDER:
            value = representation_results["models"][model]["stability"][condition][
                "mean_cosine_stability"
            ]
            values.append(value)
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "mean_cosine_stability": value,
                }
            )
        offset = (condition_index - 1) * width
        axes[0].bar(x + offset, values, width=width, label=label)

    for model in REPRESENTATION_MODEL_ORDER:
        translation = representation_results["models"][model]["stability"]["translation"]
        values = [
            translation[f"displacement_{displacement}_averaged"][
                "mean_cosine_stability"
            ]
            for displacement in displacements
        ]
        axes[1].plot(
            displacements,
            values,
            marker="o",
            linewidth=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model].replace("\n", " "),
        )
        for displacement, value in zip(displacements, values):
            rows.append(
                {
                    "model": model,
                    "condition": f"translation_{displacement}_averaged",
                    "mean_cosine_stability": value,
                }
            )

    axes[0].set_title("Non-translation interventions")
    axes[0].set_ylabel("Mean cosine stability")
    axes[0].set_xticks(x, [MODEL_LABELS[model] for model in REPRESENTATION_MODEL_ORDER])
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_title("Translations")
    axes[1].set_xlabel("Translation displacement (pixels)")
    axes[1].set_ylabel("Mean cosine stability")
    axes[1].set_xticks(displacements)
    axes[1].set_ylim(0.9, 1.005)
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        style_axis(axis)
    figure.suptitle("Frozen-backbone representation stability")
    figure.tight_layout()
    save_figure(figure, "representation_stability.png")
    return rows


def write_csv(filename: str, rows: list[dict]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / filename
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")


def main() -> None:
    clean = load_json("clean_baselines.json")
    interventions = load_json("intervention_results.json")
    cue_results = load_json("cue_conflict_results.json")
    representation_results = load_json("representation_results.json")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 120,
        }
    )

    write_csv(
        "performance_interventions.csv",
        plot_clean_and_interventions(clean, interventions),
    )
    write_csv(
        "translation_robustness.csv",
        plot_translation_robustness(interventions),
    )
    write_csv("cue_conflict_summary.csv", plot_cue_conflicts(cue_results))
    write_csv(
        "representation_stability.csv",
        plot_representation_stability(representation_results),
    )
    print("Finished creating Task 1 summary figures and tables.")


if __name__ == "__main__":
    main()
