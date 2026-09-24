"""Create deterministic Task 2 figures and tables from locked final results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


DISPLAY_NAMES = {
    "source_only": "Source-only",
    "dan": "DAN (lambda=1)",
    "dann": "DANN",
    "cdan": "CDAN",
    "dan_lambda0_1": "DAN (lambda=0.1)",
    "dan_lambda10": "DAN (lambda=10)",
}

AXIS_NAMES = {
    "source_only": "Source-\nonly",
    "dan": "DAN\n(lambda=1)",
    "dann": "DANN",
    "cdan": "CDAN",
}

MAIN_RUNS = ("source_only", "dan", "dann", "cdan")
DAN_RUNS = ("dan_lambda0_1", "dan", "dan_lambda10")
DAN_LAMBDAS = (0.1, 1.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="task2/results/final")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="task2/results/reporting")
    return parser.parse_args()


def load_predictions(results_dir: Path, run_name: str) -> pd.DataFrame:
    path = results_dir / "predictions" / f"{run_name}_target_predictions.csv"
    frame = pd.read_csv(path)
    required = {
        "path", "true_class_id", "true_class_name", "predicted_class_id",
        "predicted_class_name", "maximum_confidence",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def resolve_image_root(data_root: Path) -> Path:
    candidates = (
        data_root,
        data_root / "images",
        data_root / "pacs" / "images",
        data_root / "PACS" / "images",
    )
    for candidate in candidates:
        if (candidate / "sketch").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate PACS images below {data_root}")


def save_main_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    main = summary.set_index("run_name").loc[list(MAIN_RUNS)].reset_index()
    x = np.arange(len(main))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].bar(x - width / 2, main["target_accuracy"], width, label="Accuracy")
    axes[0].bar(x + width / 2, main["target_macro_f1"], width, label="Macro-F1")
    axes[0].set_xticks(x, [AXIS_NAMES[name] for name in main["run_name"]])
    axes[0].set_ylim(0, 0.8)
    axes[0].set_ylabel("Sketch score")
    axes[0].set_title("Target classification")
    axes[0].legend()

    axes[1].bar(x, main["domain_separability"], color="#d95f02")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    axes[1].set_xticks(x, [AXIS_NAMES[name] for name in main["run_name"]])
    axes[1].set_ylim(0.45, 1.02)
    axes[1].set_ylabel("Domain-classifier accuracy")
    axes[1].set_title("Source-target separability")
    axes[1].legend()

    fig.suptitle("PACS Sketch adaptation: fixed-checkpoint evaluation")
    fig.tight_layout()
    path = output_dir / "main_method_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def save_dan_study(summary: pd.DataFrame, output_dir: Path) -> None:
    indexed = summary.set_index("run_name")
    rows = indexed.loc[list(DAN_RUNS)]
    table = pd.DataFrame({
        "lambda": DAN_LAMBDAS,
        "source_validation_macro_f1": rows["mean_source_validation_macro_f1"].to_numpy(),
        "target_macro_f1": rows["target_macro_f1"].to_numpy(),
        "domain_separability": rows["domain_separability"].to_numpy(),
    })
    table.to_csv(output_dir / "dan_lambda_study.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.semilogx(table["lambda"], table["target_macro_f1"], marker="o", label="Sketch macro-F1")
    ax.semilogx(
        table["lambda"], table["source_validation_macro_f1"], marker="o",
        label="Source-val macro-F1",
    )
    ax.semilogx(
        table["lambda"], table["domain_separability"], marker="o",
        label="Domain separability",
    )
    ax.set_xticks(DAN_LAMBDAS, ["0.1", "1", "10"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("MMD weight (lambda, log scale)")
    ax.set_ylabel("Score")
    ax.set_title("DAN alignment-strength controlled study")
    ax.grid(alpha=0.25)
    ax.legend()
    path = output_dir / "dan_lambda_study.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_class_tables(
    predictions: dict[str, pd.DataFrame], class_names: list[str], output_dir: Path
) -> pd.DataFrame:
    rows = []
    labels = list(range(len(class_names)))
    for run_name, frame in predictions.items():
        precision, recall, f1, support = precision_recall_fscore_support(
            frame["true_class_id"], frame["predicted_class_id"], labels=labels,
            zero_division=0,
        )
        for index, class_name in enumerate(class_names):
            rows.append({
                "run_name": run_name,
                "method": DISPLAY_NAMES[run_name],
                "class_name": class_name,
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
            })
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "per_class_metrics.csv", index=False)
    return table


def save_per_class_plot(table: pd.DataFrame, output_dir: Path) -> None:
    main = table[table["run_name"].isin(MAIN_RUNS)].copy()
    main["method"] = pd.Categorical(
        main["method"], [DISPLAY_NAMES[name] for name in MAIN_RUNS], ordered=True
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=main, x="class_name", y="f1", hue="method", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("PACS class")
    ax.set_ylabel("Sketch F1")
    ax.set_title("Per-class target performance")
    ax.legend(title=None, ncol=2)
    path = output_dir / "per_class_f1.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def save_confusion_matrices(
    predictions: dict[str, pd.DataFrame], class_names: list[str], output_dir: Path
) -> None:
    labels = list(range(len(class_names)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, run_name in zip(axes.flat, MAIN_RUNS):
        frame = predictions[run_name]
        matrix = confusion_matrix(
            frame["true_class_id"], frame["predicted_class_id"], labels=labels,
            normalize="true",
        )
        sns.heatmap(
            matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False,
        )
        ax.set_title(DISPLAY_NAMES[run_name])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
    fig.suptitle("Row-normalized Sketch confusion matrices")
    fig.tight_layout()
    path = output_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def select_failures(predictions: dict[str, pd.DataFrame], count: int = 8) -> pd.DataFrame:
    base = predictions["source_only"].copy()
    base = base[base["true_class_id"] != base["predicted_class_id"]].copy()
    for run_name in MAIN_RUNS[1:]:
        other = predictions[run_name].set_index("path")
        base[f"{run_name}_prediction"] = base["path"].map(other["predicted_class_name"])
        base[f"{run_name}_confidence"] = base["path"].map(other["maximum_confidence"])
    base["agreement_count"] = sum(
        base[f"{run_name}_prediction"].eq(base["predicted_class_name"])
        for run_name in MAIN_RUNS[1:]
    )
    selected = (
        base.sort_values(
            ["agreement_count", "maximum_confidence", "path"],
            ascending=[False, False, True],
        )
        .groupby("true_class_name", sort=True, group_keys=False)
        .head(2)
        .head(count)
        .reset_index(drop=True)
    )
    return selected


def save_failure_examples(
    selected: pd.DataFrame, image_root: Path, output_dir: Path
) -> None:
    columns = 2
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4.6 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (_, row) in zip(axes, selected.iterrows()):
        with Image.open(image_root / row["path"]) as image_file:
            ax.imshow(image_file.convert("RGB"))
        lines = [
            f"True: {row['true_class_name']}",
            f"Source-only: {row['predicted_class_name']} ({row['maximum_confidence']:.2f})",
        ]
        for run_name in MAIN_RUNS[1:]:
            lines.append(
                f"{DISPLAY_NAMES[run_name]}: {row[f'{run_name}_prediction']} "
                f"({row[f'{run_name}_confidence']:.2f})"
            )
        ax.set_title("\n".join(lines), fontsize=9)
        ax.axis("off")
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle("High-confidence source-only Sketch failures and adaptation responses")
    fig.tight_layout()
    path = output_dir / "target_failure_examples.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = resolve_image_root(Path(args.data_root))

    summary = pd.read_csv(results_dir / "final_summary.csv")
    summary["display_name"] = summary["run_name"].map(DISPLAY_NAMES)
    summary.to_csv(output_dir / "method_summary.csv", index=False)

    with (results_dir / "final_results.json").open(encoding="utf-8") as file:
        results = json.load(file)
    if not results["protocol"].get("target_labels_used_only_in_final_evaluation", False):
        raise ValueError("Protocol does not confirm final-only target-label use.")

    predictions = {
        run_name: load_predictions(results_dir, run_name)
        for run_name in DISPLAY_NAMES
    }
    class_frame = predictions["source_only"][["true_class_id", "true_class_name"]]
    class_names = (
        class_frame.drop_duplicates().sort_values("true_class_id")["true_class_name"].tolist()
    )

    sns.set_theme(style="whitegrid", context="talk")
    save_main_comparison(summary, output_dir)
    save_dan_study(summary, output_dir)
    class_table = make_class_tables(predictions, class_names, output_dir)
    save_per_class_plot(class_table, output_dir)
    save_confusion_matrices(predictions, class_names, output_dir)
    failures = select_failures(predictions)
    failures.to_csv(output_dir / "target_failure_examples.csv", index=False)
    save_failure_examples(failures, image_root, output_dir)
    print("Finished Task 2 reporting artifacts.")


if __name__ == "__main__":
    main()
