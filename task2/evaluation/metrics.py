"""Classification metrics and model evaluation for PACS."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn


def classification_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
    class_names: list[str],
) -> dict:
    """Compute aggregate, per-class, and confusion metrics."""

    labels = np.asarray(list(labels), dtype=np.int64)
    predictions = np.asarray(list(predictions), dtype=np.int64)
    class_indices = np.arange(len(class_names))
    matrix = confusion_matrix(labels, predictions, labels=class_indices)
    totals = matrix.sum(axis=1)
    correct = np.diag(matrix)
    per_class_accuracy = np.divide(
        correct,
        totals,
        out=np.zeros_like(correct, dtype=np.float64),
        where=totals > 0,
    )
    return {
        "number_of_examples": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_accuracy": {
            class_name: float(per_class_accuracy[index])
            for index, class_name in enumerate(class_names)
        },
        "class_counts": {
            class_name: int(totals[index])
            for index, class_name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "class_order": class_names,
    }


@torch.no_grad()
def evaluate_classifier(
    backbone: nn.Module,
    classifier: nn.Module,
    data_loader,
    class_names: list[str],
    device: torch.device,
) -> tuple[dict, list[dict]]:
    """Evaluate one domain and return metrics plus per-example predictions."""

    backbone.eval()
    classifier.eval()
    labels: list[int] = []
    predictions: list[int] = []
    prediction_rows: list[dict] = []

    for batch in data_loader:
        if isinstance(batch, dict):
            images = batch["image"].to(device, non_blocking=True)
            batch_labels = batch["label"].to(device, non_blocking=True)
            paths = list(batch["path"])
            domains = list(batch["domain"])
            class_name_rows = list(batch["class_name"])
        else:
            images, batch_labels = batch
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            paths = [""] * len(batch_labels)
            domains = [""] * len(batch_labels)
            class_name_rows = [class_names[int(value)] for value in batch_labels.cpu()]

        logits = classifier(backbone(images))
        probabilities = logits.softmax(dim=1)
        confidence, batch_predictions = probabilities.max(dim=1)

        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(batch_predictions.cpu().tolist())
        for index in range(len(batch_labels)):
            prediction_rows.append(
                {
                    "path": paths[index],
                    "domain": domains[index],
                    "true_class_id": int(batch_labels[index].item()),
                    "true_class_name": class_name_rows[index],
                    "predicted_class_id": int(batch_predictions[index].item()),
                    "predicted_class_name": class_names[
                        int(batch_predictions[index].item())
                    ],
                    "maximum_confidence": float(confidence[index].item()),
                }
            )

    return classification_metrics(labels, predictions, class_names), prediction_rows


def summarize_source_validation(domain_metrics: dict[str, dict]) -> dict:
    """Mean and worst source validation metrics across the three domains."""

    accuracies = [metrics["accuracy"] for metrics in domain_metrics.values()]
    macro_f1_values = [metrics["macro_f1"] for metrics in domain_metrics.values()]
    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_macro_f1": float(np.mean(macro_f1_values)),
        "worst_accuracy": float(np.min(accuracies)),
        "worst_macro_f1": float(np.min(macro_f1_values)),
    }
