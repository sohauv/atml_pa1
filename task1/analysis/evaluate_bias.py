from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()

    probabilities = torch.softmax(logits, dim=1)
    confidences, predictions = probabilities.max(dim=1)

    labels_numpy = labels.numpy()
    predictions_numpy = predictions.numpy()

    return {
        "accuracy": accuracy_score(
            labels_numpy,
            predictions_numpy,
        ),
        "macro_f1": f1_score(
            labels_numpy,
            predictions_numpy,
            average="macro",
        ),
        "mean_maximum_confidence": confidences.mean().item(),
    }


def compute_prediction_consistency(
    clean_predictions: torch.Tensor,
    transformed_predictions: torch.Tensor,
) -> float:
    clean_predictions = clean_predictions.detach().cpu()
    transformed_predictions = transformed_predictions.detach().cpu()

    if clean_predictions.shape != transformed_predictions.shape:
        raise ValueError(
            "Clean and transformed predictions must have the same shape."
        )

    return (
        clean_predictions == transformed_predictions
    ).float().mean().item()


def compute_accuracy_change(
    clean_accuracy: float,
    transformed_accuracy: float,
) -> float:
    return transformed_accuracy - clean_accuracy


def save_prediction_table(
    output_path: str | Path,
    identifiers: list[str],
    indices: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    class_names: list[str],
    condition: str,
    model_name: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logits = logits.detach().cpu()
    labels = labels.detach().cpu()
    indices = indices.detach().cpu()

    probabilities = torch.softmax(logits, dim=1)
    confidences, predictions = probabilities.max(dim=1)

    number_of_examples = labels.shape[0]

    if not (
        len(identifiers)
        == number_of_examples
        == indices.shape[0]
        == logits.shape[0]
    ):
        raise ValueError(
            "Identifiers, indices, labels, and logits are misaligned."
        )

    rows: list[dict[str, Any]] = []

    for row_index in range(number_of_examples):
        true_class_id = labels[row_index].item()
        predicted_class_id = predictions[row_index].item()

        row = {
            "model": model_name,
            "condition": condition,
            "identifier": identifiers[row_index],
            "dataset_index": indices[row_index].item(),
            "true_class_id": true_class_id,
            "true_class_name": class_names[true_class_id],
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": class_names[predicted_class_id],
            "maximum_confidence": confidences[row_index].item(),
            "correct": predicted_class_id == true_class_id,
        }

        for class_id, class_name in enumerate(class_names):
            row[f"probability_{class_name}"] = (
                probabilities[row_index, class_id].item()
            )

        rows.append(row)

    prediction_table = pd.DataFrame(rows)
    prediction_table.to_csv(output_path, index=False)