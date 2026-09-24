"""Class-level positive/negative transfer and dominant-confusion analysis."""

from __future__ import annotations

import numpy as np


def per_class_changes(
    baseline_metrics: dict,
    adapted_metrics: dict,
) -> dict[str, float]:
    """Adapted minus source-only target accuracy for each class."""

    baseline = baseline_metrics["per_class_accuracy"]
    adapted = adapted_metrics["per_class_accuracy"]
    if set(baseline) != set(adapted):
        raise ValueError("Baseline and adapted class sets do not match.")
    return {
        class_name: float(adapted[class_name] - baseline[class_name])
        for class_name in baseline
    }


def dominant_confusions(metrics: dict, top_k: int = 2) -> dict[str, list[dict]]:
    """Return each class's most frequent incorrect predictions."""

    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    class_names = metrics["class_order"]
    output: dict[str, list[dict]] = {}
    for true_index, true_class in enumerate(class_names):
        row = matrix[true_index].copy()
        row[true_index] = 0
        ranked_indices = sorted(
            range(len(class_names)),
            key=lambda predicted_index: (-row[predicted_index], predicted_index),
        )
        output[true_class] = [
            {
                "predicted_class": class_names[predicted_index],
                "count": int(row[predicted_index]),
            }
            for predicted_index in ranked_indices[:top_k]
            if row[predicted_index] > 0
        ]
    return output


def transfer_summary(
    baseline_metrics: dict,
    adapted_metrics: dict,
    number_to_report: int = 3,
) -> dict:
    """Summarize the largest class improvements and degradations."""

    changes = per_class_changes(baseline_metrics, adapted_metrics)
    ordered = sorted(changes.items(), key=lambda item: (-item[1], item[0]))
    return {
        "per_class_accuracy_change": changes,
        "largest_improvements": [
            {"class_name": name, "accuracy_change": change}
            for name, change in ordered[:number_to_report]
        ],
        "largest_degradations": [
            {"class_name": name, "accuracy_change": change}
            for name, change in sorted(changes.items(), key=lambda item: (item[1], item[0]))[
                :number_to_report
            ]
        ],
        "adapted_dominant_confusions": dominant_confusions(adapted_metrics),
    }
