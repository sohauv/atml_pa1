"""Create Task 3 report figures and compact CSV tables from fixed results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from shared.pacs import resolve_pacs_root


LABELS = {
    "erm": "ERM",
    "dan_dg_lambda0_1": "DAN-DG (λ=0.1)",
    "dan_dg": "DAN-DG (λ=1)",
    "dan_dg_lambda10": "DAN-DG (λ=10)",
    "sam": "SAM (ρ=0.05)",
}
COLORS = {"erm": "#4c78a8", "dan_dg_lambda0_1": "#59a14f", "dan_dg": "#f28e2b", "dan_dg_lambda10": "#e15759", "sam": "#b279a2"}


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure); print(f"Saved {path}")


def load_results(root):
    return json.loads((root / "final/final_results.json").read_text(encoding="utf-8"))


def make_main_comparison(results, output, tables):
    runs = ["erm", "dan_dg", "sam"]
    rows = []
    for run in runs:
        item = results["runs"][run]
        source = item["source_diagnostics"]["source_summary"]
        target = item["target"]
        rows.append({"run_name": run, "method": LABELS[run], "mean_source_macro_f1": source["mean_macro_f1"],
                     "worst_source_macro_f1": source["worst_macro_f1"], "sketch_accuracy": target["accuracy"],
                     "sketch_macro_f1": target["macro_f1"], "sketch_macro_f1_change_vs_erm": target["macro_f1_change_vs_erm"]})
    frame = pd.DataFrame(rows); frame.to_csv(tables / "main_method_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(frame)); width = .34
    axes[0].bar(x-width/2, frame.mean_source_macro_f1, width, label="Mean source", color="#4c78a8")
    axes[0].bar(x+width/2, frame.worst_source_macro_f1, width, label="Worst source", color="#9ecae9")
    axes[1].bar(x-width/2, frame.sketch_accuracy, width, label="Accuracy", color="#f28e2b")
    axes[1].bar(x+width/2, frame.sketch_macro_f1, width, label="Macro-F1", color="#59a14f")
    for axis, title in zip(axes, ["Source validation", "Held-out Sketch"]):
        axis.set_title(title); axis.set_ylim(0, 1.02); axis.set_xticks(x, frame.method); axis.tick_params(axis="x", rotation=12); axis.legend(); axis.grid(axis="y", alpha=.25)
    fig.suptitle("Task 3 fixed-checkpoint evaluation"); fig.tight_layout()
    save_figure(fig, output / "main_method_comparison.png")


def make_lambda_study(results, output, tables):
    mapping = [(0.1, "dan_dg_lambda0_1"), (1.0, "dan_dg"), (10.0, "dan_dg_lambda10")]
    rows = []
    for weight, run in mapping:
        item = results["runs"][run]; source = item["source_diagnostics"]
        rows.append({"alignment_weight": weight, "source_macro_f1": source["source_summary"]["mean_macro_f1"],
                     "sketch_macro_f1": item["target"]["macro_f1"], "source_domain_separability": source["source_domain_separability"]})
    frame = pd.DataFrame(rows); frame.to_csv(tables / "dan_dg_lambda_study.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for column, label, color in [("source_macro_f1", "Source macro-F1", "#4c78a8"), ("sketch_macro_f1", "Sketch macro-F1", "#f28e2b"), ("source_domain_separability", "Domain separability", "#59a14f")]:
        axis.plot(frame.alignment_weight, frame[column], marker="o", linewidth=2, label=label, color=color)
    axis.axhline(1/3, linestyle="--", color="gray", linewidth=1, label="Separability chance")
    axis.set_xscale("log"); axis.set_xticks([.1, 1, 10], ["0.1", "1", "10"]); axis.set_ylim(0, 1.02)
    axis.set_xlabel("MMD alignment weight λ"); axis.set_ylabel("Score"); axis.set_title("DAN-DG controlled alignment-strength study")
    axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); save_figure(fig, output / "dan_dg_lambda_study.png")


def make_diagnostics(results, output, tables):
    runs = ["erm", "dan_dg_lambda0_1", "dan_dg", "dan_dg_lambda10", "sam"]
    rows = []
    for run in runs:
        source = results["runs"][run]["source_diagnostics"]
        rows.append({"run_name": run, "method": LABELS[run], "source_domain_separability": source["source_domain_separability"],
                     "sharpness_loss_increase": source["sharpness_proxy"]["loss_increase"]})
    frame = pd.DataFrame(rows); frame.to_csv(tables / "diagnostics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2)); x = np.arange(len(frame)); colors = [COLORS[r] for r in runs]
    axes[0].bar(x, frame.source_domain_separability, color=colors); axes[0].axhline(1/3, color="black", linestyle="--", label="Chance")
    axes[0].set_title("Source-domain separability"); axes[0].set_ylim(0, 1); axes[0].legend()
    axes[1].bar(x, frame.sharpness_loss_increase, color=colors); axes[1].set_title("Sharpness proxy (loss increase)")
    for axis in axes:
        axis.set_xticks(x, frame.method, rotation=18, ha="right"); axis.grid(axis="y", alpha=.25)
    fig.suptitle("Representation and optimization diagnostics"); fig.tight_layout(); save_figure(fig, output / "diagnostics.png")


def make_training_curves(repo_root, output):
    files = {"ERM": repo_root/"task2/results/training/source_only_history.csv", "DAN-DG (λ=0.1)": repo_root/"task3/results/training/dan_dg_lambda0_1_history.csv",
             "DAN-DG (λ=1)": repo_root/"task3/results/training/dan_dg_history.csv", "DAN-DG (λ=10)": repo_root/"task3/results/training/dan_dg_lambda10_history.csv", "SAM": repo_root/"task3/results/training/sam_history.csv"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for label, path in files.items():
        frame = pd.read_csv(path)
        axes[0, 0].plot(frame.epoch, frame.train_classification_loss, marker="o", markersize=3, label=label)
        axes[1, 0].plot(frame.epoch, frame.train_total_loss, marker="o", markersize=3, label=label)
        axes[1, 1].plot(frame.epoch, frame.mean_source_validation_macro_f1, marker="o", markersize=3, label=label)
        if "train_alignment_loss" in frame:
            axes[0, 1].plot(frame.epoch, frame.train_alignment_loss, marker="o", markersize=3, label=label)
    axes[0, 0].set_title("Classification loss"); axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_title("DAN-DG MMD alignment loss"); axes[0, 1].set_ylabel("MMD")
    axes[1, 0].set_title("Total training objective"); axes[1, 0].set_ylabel("Loss"); axes[1, 0].set_yscale("log")
    axes[1, 1].set_title("Source validation selection metric"); axes[1, 1].set_ylabel("Mean macro-F1"); axes[1, 1].set_ylim(0, 1)
    for axis in axes.flat:
        axis.set_xlabel("Epoch"); axis.grid(alpha=.25); axis.legend(fontsize=7)
    fig.tight_layout(); save_figure(fig, output / "training_curves.png")


def make_per_class_and_confusions(results, output, tables):
    runs = ["erm", "dan_dg_lambda0_1", "dan_dg", "sam"]
    class_names = list(results["runs"]["erm"]["target"]["per_class"])
    rows = []
    for run in runs:
        for name in class_names:
            rows.append({"run_name": run, "method": LABELS[run], "class_name": name, **results["runs"][run]["target"]["per_class"][name]})
    frame = pd.DataFrame(rows); frame.to_csv(tables / "per_class_metrics.csv", index=False)
    method_order = [LABELS[run] for run in runs]
    pivot = frame.pivot(index="class_name", columns="method", values="f1").loc[class_names, method_order]
    fig, axis = plt.subplots(figsize=(10, 4.8)); pivot.plot.bar(ax=axis, width=.8)
    axis.set_ylim(0, 1); axis.set_ylabel("Sketch F1"); axis.set_xlabel("PACS class"); axis.set_title("Per-class Sketch performance"); axis.grid(axis="y", alpha=.25); axis.legend(fontsize=8); fig.tight_layout()
    save_figure(fig, output / "per_class_f1.png")
    display_runs = ["erm", "dan_dg_lambda0_1", "sam"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, run in zip(axes, display_runs):
        matrix = np.asarray(results["runs"][run]["target"]["confusion_matrix"], dtype=float)
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=axis, xticklabels=class_names, yticklabels=class_names)
        axis.set_title(LABELS[run]); axis.set_xlabel("Predicted"); axis.set_ylabel("True")
    fig.suptitle("Row-normalized Sketch confusion matrices"); fig.tight_layout(); save_figure(fig, output / "confusion_matrices.png")


def make_task2_comparison(repo_root, results, output, tables):
    task2 = pd.read_csv(repo_root / "task2/results/final/final_summary.csv")
    task2 = task2[task2.run_name.isin(["source_only", "dan", "dann", "cdan"])][["run_name", "target_accuracy", "target_macro_f1"]]
    task2["task"] = "Task 2 adaptation"
    task2["method"] = task2.run_name.map({"source_only": "ERM", "dan": "DAN", "dann": "DANN", "cdan": "CDAN"})
    task3_rows = []
    for run in ["erm", "dan_dg_lambda0_1", "dan_dg", "sam"]:
        target = results["runs"][run]["target"]
        task3_rows.append({"run_name": run, "target_accuracy": target["accuracy"], "target_macro_f1": target["macro_f1"], "task": "Task 3 generalization", "method": LABELS[run]})
    frame = pd.concat([task2, pd.DataFrame(task3_rows)], ignore_index=True); frame.to_csv(tables / "task2_task3_comparison.csv", index=False)
    fig, axis = plt.subplots(figsize=(9, 6)); y=np.arange(len(frame)); height=.36
    axis.barh(y+height/2, frame.target_accuracy, height, label="Accuracy", color="#4c78a8")
    axis.barh(y-height/2, frame.target_macro_f1, height, label="Macro-F1", color="#f28e2b")
    task_labels = frame.task.map({"Task 2 adaptation": "T2 adaptation", "Task 3 generalization": "T3 generalization"})
    axis.set_yticks(y, [f"{task}: {method}" for task, method in zip(task_labels, frame.method)])
    axis.invert_yaxis(); axis.set_xlim(0,1); axis.set_xlabel("Sketch score")
    axis.set_title("Task 2 adaptation and Task 3 generalization on Sketch")
    axis.grid(axis="x", alpha=.25); axis.legend(); fig.tight_layout()
    save_figure(fig, output / "task2_task3_comparison.png")


def make_failures(repo_root, data_root, output, tables):
    runs = ["erm", "dan_dg_lambda0_1", "sam"]; selected=[]
    for run in runs:
        frame = pd.read_csv(repo_root / f"task3/results/final/predictions/{run}_sketch_predictions.csv")
        failures = frame[frame.true_class_id != frame.predicted_class_id].sort_values("maximum_confidence", ascending=False)
        for _, row in failures.drop_duplicates("true_class_name").head(2).iterrows(): selected.append({"run_name":run, **row.to_dict()})
    frame=pd.DataFrame(selected); frame.to_csv(tables / "target_failure_examples.csv", index=False)
    pacs_root=resolve_pacs_root(data_root); fig, axes=plt.subplots(2,3,figsize=(10,7));
    for axis, row in zip(axes.flat, selected):
        with Image.open(pacs_root / row["path"]) as image: axis.imshow(image.convert("RGB"))
        axis.set_title(f"{LABELS[row['run_name']]}\ntrue: {row['true_class_name']} → pred: {row['predicted_class_name']}\nconfidence: {row['maximum_confidence']:.2f}", fontsize=9); axis.axis("off")
    fig.suptitle("High-confidence Sketch failure examples"); fig.tight_layout(); save_figure(fig, output / "target_failure_examples.png")


def run(args):
    repo_root=Path(".").resolve(); result_root=repo_root/"task3/results"; output=result_root/"reporting"; tables=output/"tables"; tables.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook"); results=load_results(result_root)
    make_main_comparison(results,output,tables); make_lambda_study(results,output,tables); make_diagnostics(results,output,tables); make_training_curves(repo_root,output)
    make_per_class_and_confusions(results,output,tables); make_task2_comparison(repo_root,results,output,tables); make_failures(repo_root,args.data_root,output,tables)
    print("Finished Task 3 reporting artifacts.")


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-root",required=True); return parser.parse_args()


if __name__=="__main__": run(parse_args())
