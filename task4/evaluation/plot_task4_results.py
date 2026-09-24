"""Create Task 4 report figures and compact machine-readable tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR100


DISPLAY_NAMES = {
    "vanilla_mls": "Vanilla MLS",
    "gcsc_mls": "GCSC MLS",
    "proser_mls": "PROSER MLS",
    "proser_placeholder": "PROSER placeholder",
}
DOMAIN_NAMES = {"known": "CIFAR-10 known", "near": "Near", "far": "Far"}
DOMAIN_COLORS = {"known": "#4c78a8", "near": "#f58518", "far": "#54a24b"}


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")


def accepted_mask(frame, score_name):
    values = frame[f"accepted_{score_name}"]
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq("true")


def make_compact_tables(summary, predictions, table_directory):
    table_directory.mkdir(parents=True, exist_ok=True)
    posthoc = summary[summary["method"] == "vanilla"].copy()
    posthoc.to_csv(table_directory / "vanilla_posthoc_scores.csv", index=False)

    comparison_names = list(DISPLAY_NAMES)
    trained = summary.set_index("run_name").loc[comparison_names].reset_index()
    trained.insert(1, "display_name", trained["run_name"].map(DISPLAY_NAMES))
    trained.to_csv(table_directory / "trained_method_comparison.csv", index=False)

    rows = []
    score_by_run = {
        "vanilla_mls": ("vanilla", "mls"),
        "gcsc_mls": ("gcsc", "mls"),
        "proser_mls": ("proser", "mls"),
        "proser_placeholder": ("proser", "placeholder"),
    }
    for run_name, (prediction_name, score_name) in score_by_run.items():
        frame = predictions[prediction_name]
        for domain in ("near", "far"):
            domain_frame = frame[frame["domain"] == domain].copy()
            domain_frame["accepted"] = accepted_mask(domain_frame, score_name)
            for class_name, class_frame in domain_frame.groupby("true_class_name"):
                most_common_prediction = class_frame[
                    "predicted_known_class_name"
                ].value_counts().index[0]
                rows.append(
                    {
                        "run_name": run_name,
                        "display_name": DISPLAY_NAMES[run_name],
                        "score": score_name,
                        "unknown_group": domain,
                        "unknown_class": class_name,
                        "count": len(class_frame),
                        "mean_unknownness": class_frame[score_name].mean(),
                        "rejection_rate": 1.0 - class_frame["accepted"].mean(),
                        "most_common_known_prediction": most_common_prediction,
                    }
                )
    per_class = pd.DataFrame(rows)
    per_class.to_csv(table_directory / "per_unknown_class_metrics.csv", index=False)
    return posthoc, trained, per_class


def plot_score_distributions(vanilla, thresholds, output_path):
    score_labels = {
        "msp": "MSP unknownness",
        "mls": "MLS unknownness",
        "mahalanobis": "Mahalanobis distance",
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for axis, (score_name, label) in zip(axes, score_labels.items()):
        pooled = vanilla[score_name].to_numpy()
        low, high = np.quantile(pooled, [0.005, 0.995])
        bins = np.linspace(low, high, 55)
        for domain in ("known", "near", "far"):
            values = vanilla.loc[vanilla["domain"] == domain, score_name]
            axis.hist(
                values.clip(low, high), bins=bins, density=True, alpha=0.42,
                label=DOMAIN_NAMES[domain], color=DOMAIN_COLORS[domain],
            )
        axis.axvline(
            thresholds[score_name], color="black", linestyle="--", linewidth=1.5,
            label="Validation threshold" if score_name == "msp" else None,
        )
        axis.set_title(label)
        axis.set_xlabel("Unknownness score")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.suptitle("Frozen Vanilla score distributions", y=1.05)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_method_comparison(trained, output_path):
    trained = trained.set_index("run_name").loc[list(DISPLAY_NAMES)].reset_index()
    labels = [DISPLAY_NAMES[name] for name in trained["run_name"]]
    positions = np.arange(len(labels))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(positions, trained["known_test_accuracy"], color=colors)
    axes[0].set_title("Known classification")
    axes[0].set_ylabel("CIFAR-10 test accuracy")
    axes[0].set_ylim(0.90, 0.97)

    width = 0.36
    axes[1].bar(
        positions - width / 2, trained["known_vs_near_auroc"], width,
        label="Near", color="#f58518",
    )
    axes[1].bar(
        positions + width / 2, trained["known_vs_far_auroc"], width,
        label="Far", color="#54a24b",
    )
    axes[1].set_title("Unknown ranking")
    axes[1].set_ylabel("AUROC")
    axes[1].set_ylim(0.70, 0.95)
    axes[1].legend(frameon=False)

    axes[2].bar(
        positions - width / 2, trained["near_rejection"], width,
        label="Near", color="#f58518",
    )
    axes[2].bar(
        positions + width / 2, trained["far_rejection"], width,
        label="Far", color="#54a24b",
    )
    axes[2].set_title("Validation-calibrated rejection")
    axes[2].set_ylabel("Rejection rate")
    axes[2].set_ylim(0.0, 0.7)
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Task 4 trained-model comparison")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_per_class_rejection(per_class, output_path):
    methods = ["vanilla_mls", "gcsc_mls", "proser_mls"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharey=True)
    for axis, domain in zip(axes, ("near", "far")):
        subset = per_class[
            (per_class["unknown_group"] == domain)
            & (per_class["run_name"].isin(methods))
        ]
        class_names = sorted(subset["unknown_class"].unique())
        positions = np.arange(len(class_names))
        width = 0.25
        for method_index, (method, color) in enumerate(zip(methods, colors)):
            values = (
                subset[subset["run_name"] == method]
                .set_index("unknown_class")
                .loc[class_names, "rejection_rate"]
            )
            axis.bar(
                positions + (method_index - 1) * width, values, width,
                label=DISPLAY_NAMES[method], color=color,
            )
        axis.set_title(f"{domain.title()} unknown classes")
        axis.set_xticks(
            positions, [name.replace("_", " ") for name in class_names],
            rotation=25, ha="right",
        )
        axis.set_ylabel("Rejection rate")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3, frameon=False)
    figure.suptitle("Class-level rejection using the common MLS score")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_training_curves(results_directory, output_path):
    run_names = ("vanilla", "gcsc", "proser")
    labels = ("Vanilla", "GCSC", "PROSER")
    colors = ("#4c78a8", "#f58518", "#54a24b")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run_name, label, color in zip(run_names, labels, colors):
        history = pd.read_csv(results_directory / "training" / f"{run_name}_history.csv")
        loss_column = "train_total_loss" if run_name == "proser" else "train_loss"
        axes[0].plot(history["epoch"], history[loss_column], label=label, color=color)
        axes[1].plot(
            history["epoch"], history["validation_accuracy"],
            label=label, color=color,
        )
    axes[0].set_title("Training objective")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[1].set_title("Checkpoint-selection metric")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CIFAR-10 validation accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, output_path)


def select_diverse_failures(frame, domain, count=3):
    candidates = frame[
        (frame["domain"] == domain) & accepted_mask(frame, "mls")
    ].sort_values(["maximum_known_probability", "index"], ascending=[False, True])
    selected_indices = []
    selected_classes = set()
    for row_index, row in candidates.iterrows():
        if row["true_class_name"] not in selected_classes:
            selected_indices.append(row_index)
            selected_classes.add(row["true_class_name"])
        if len(selected_indices) == count:
            break
    if len(selected_indices) < count:
        for row_index in candidates.index:
            if row_index not in selected_indices:
                selected_indices.append(row_index)
            if len(selected_indices) == count:
                break
    return candidates.loc[selected_indices].copy()


def plot_failure_examples(vanilla, cifar100_root, threshold, figure_path, table_path):
    near = select_diverse_failures(vanilla, "near")
    far = select_diverse_failures(vanilla, "far")
    failures = pd.concat([near, far], ignore_index=True)
    failures.insert(0, "selection_reason", "accepted_by_frozen_vanilla_mls")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    failures.to_csv(table_path, index=False)

    dataset = CIFAR100(root=cifar100_root, train=False, download=True)
    figure, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for axis, (_, row) in zip(axes.flat, failures.iterrows()):
        image, _ = dataset[int(row["index"])]
        axis.imshow(image)
        axis.set_title(
            f"{row['domain'].title()}: {row['true_class_name'].replace('_', ' ')}\n"
            f"predicted {row['predicted_known_class_name']} | "
            f"confidence {row['maximum_known_probability']:.3f}\n"
            f"MLS {row['mls']:.3f} ≤ threshold {threshold:.3f}",
            fontsize=9,
        )
        axis.axis("off")
    figure.suptitle("Unknowns incorrectly accepted by frozen Vanilla MLS")
    figure.tight_layout()
    save_figure(figure, figure_path)
    print(f"Saved {table_path}")


def run(args):
    results_directory = Path(args.results_directory)
    final_directory = results_directory / "final"
    output_directory = results_directory / "reporting"
    table_directory = output_directory / "tables"
    output_directory.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(final_directory / "final_summary.csv")
    predictions = {
        name: pd.read_csv(final_directory / "predictions" / f"{name}_predictions.csv")
        for name in ("vanilla", "gcsc", "proser")
    }
    thresholds = json.loads(
        (results_directory / "calibration" / "open_set_thresholds.json").read_text(
            encoding="utf-8"
        )
    )
    vanilla_thresholds = {
        name: details["threshold"]
        for name, details in thresholds["runs"]["vanilla"]["scores"].items()
    }

    _, trained, per_class = make_compact_tables(
        summary, predictions, table_directory
    )
    plot_score_distributions(
        predictions["vanilla"], vanilla_thresholds,
        output_directory / "score_distributions.png",
    )
    plot_method_comparison(
        trained, output_directory / "method_comparison.png"
    )
    plot_per_class_rejection(
        per_class, output_directory / "per_class_rejection.png"
    )
    plot_training_curves(
        results_directory, output_directory / "training_curves.png"
    )
    plot_failure_examples(
        predictions["vanilla"], args.cifar100_root, vanilla_thresholds["mls"],
        output_directory / "vanilla_mls_failure_examples.png",
        table_directory / "vanilla_mls_failure_examples.csv",
    )
    print("Finished Task 4 reporting artifacts.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-directory", default="task4/results"
    )
    parser.add_argument("--cifar100-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
