"""One-time final Task 3 evaluation on the held-out PACS Sketch domain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from shared.pacs_protocol import build_source_datasets, build_task2_target_dataset
from task3.evaluate_sources import RUNS, load_model, native


@torch.no_grad()
def predict(backbone, classifier, loader, class_names, device):
    backbone.eval(); classifier.eval()
    rows, true_labels, predicted_labels = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = classifier(backbone(images))
        probabilities = logits.softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        labels = batch["label"]
        for index in range(len(labels)):
            truth = int(labels[index])
            prediction = int(predictions[index].cpu())
            true_labels.append(truth); predicted_labels.append(prediction)
            rows.append({
                "path": batch["path"][index],
                "domain": batch["domain"][index],
                "true_class_id": truth,
                "true_class_name": class_names[truth],
                "predicted_class_id": prediction,
                "predicted_class_name": class_names[prediction],
                "maximum_confidence": float(confidence[index].cpu()),
            })
    return rows, np.asarray(true_labels), np.asarray(predicted_labels)


def metrics(true_labels, predicted_labels, class_names):
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=np.arange(len(class_names)), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro")),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            } for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=np.arange(len(class_names))
        ).tolist(),
    }


def write_predictions(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, manifest = build_source_datasets(args.data_root, args.split_manifest)
    target_dataset = build_task2_target_dataset(
        args.data_root, manifest["class_to_index"], training=False, return_metadata=True
    )
    target_loader = DataLoader(
        target_dataset, batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    print(f"FINAL-ONLY target evaluation size: {len(target_dataset)}")
    source_results = json.loads(Path(args.source_diagnostics).read_text(encoding="utf-8"))
    output = Path(args.output_directory)
    prediction_directory = output / "predictions"
    results = {
        "protocol": {
            "source_domains": ["photo", "art_painting", "cartoon"],
            "target_domain": "sketch",
            "target_labels_used_only_in_final_evaluation": True,
            "all_training_selection_and_diagnostics_fixed_before_target_evaluation": True,
            "seed": 6304,
        },
        "runs": {},
    }
    summary_rows = []
    for run_name, checkpoint_path in RUNS.items():
        print("=" * 70); print(f"Final Sketch evaluation: {run_name}"); print("=" * 70)
        checkpoint, backbone, classifier = load_model(checkpoint_path, device)
        prediction_rows, truth, predictions = predict(
            backbone, classifier, target_loader, manifest["class_names"], device
        )
        target_metrics = metrics(truth, predictions, manifest["class_names"])
        source = source_results["runs"][run_name]
        write_predictions(prediction_rows, prediction_directory / f"{run_name}_sketch_predictions.csv")
        results["runs"][run_name] = native({
            "checkpoint": checkpoint_path,
            "selected_epoch": int(checkpoint["epoch"]),
            "source_diagnostics": source,
            "target": target_metrics,
        })
        summary_rows.append({
            "run_name": run_name,
            "selected_epoch": int(checkpoint["epoch"]),
            "mean_source_validation_accuracy": source["source_summary"]["mean_accuracy"],
            "worst_source_validation_accuracy": source["source_summary"]["worst_accuracy"],
            "mean_source_validation_macro_f1": source["source_summary"]["mean_macro_f1"],
            "worst_source_validation_macro_f1": source["source_summary"]["worst_macro_f1"],
            "source_domain_separability": source["source_domain_separability"],
            "sharpness_loss_increase": source["sharpness_proxy"]["loss_increase"],
            "target_accuracy": target_metrics["accuracy"],
            "target_macro_f1": target_metrics["macro_f1"],
        })
        print(f"Sketch accuracy {target_metrics['accuracy']:.4f} | macro-F1 {target_metrics['macro_f1']:.4f}")

    erm_accuracy = next(row["target_accuracy"] for row in summary_rows if row["run_name"] == "erm")
    erm_f1 = next(row["target_macro_f1"] for row in summary_rows if row["run_name"] == "erm")
    for row in summary_rows:
        row["target_accuracy_change_vs_erm"] = row["target_accuracy"] - erm_accuracy
        row["target_macro_f1_change_vs_erm"] = row["target_macro_f1"] - erm_f1
        results["runs"][row["run_name"]]["target"]["accuracy_change_vs_erm"] = row["target_accuracy_change_vs_erm"]
        results["runs"][row["run_name"]]["target"]["macro_f1_change_vs_erm"] = row["target_macro_f1_change_vs_erm"]

    output.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output / "final_results.json", output / "final_summary.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    print(f"Saved {json_path}"); print(f"Saved {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", default="shared/splits/pacs_sketch_seed6304.json")
    parser.add_argument("--source-diagnostics", default="task3/results/source_diagnostics/source_diagnostics.json")
    parser.add_argument("--output-directory", default="task3/results/final")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
