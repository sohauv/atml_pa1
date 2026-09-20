import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10


SEED = 6304
CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def make_train_validation_split(labels: np.ndarray) -> tuple[list[int], list[int]]:
    all_indices = np.arange(len(labels))

    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=SEED,
        shuffle=True,
        stratify=labels,
    )

    return train_indices.tolist(), validation_indices.tolist()


def make_balanced_evaluation_subset(
    labels: np.ndarray,
    examples_per_class: int = 50,
) -> list[int]:
    rng = np.random.default_rng(SEED)
    selected_indices = []

    for class_id in range(len(CLASS_NAMES)):
        class_indices = np.flatnonzero(labels == class_id)

        if len(class_indices) < examples_per_class:
            chosen_indices = class_indices
        else:
            chosen_indices = rng.choice(
                class_indices,
                size=examples_per_class,
                replace=False,
            )

        selected_indices.extend(chosen_indices.tolist())

    # Avoid evaluating classes in ten consecutive blocks.
    selected_indices = rng.permutation(selected_indices)

    return selected_indices.tolist()


def verify_splits(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_indices: list[int],
    validation_indices: list[int],
    evaluation_indices: list[int],
) -> None:
    assert len(train_indices) == 4000
    assert len(validation_indices) == 1000
    assert len(set(train_indices) & set(validation_indices)) == 0

    combined_indices = set(train_indices) | set(validation_indices)
    assert len(combined_indices) == 5000

    evaluation_labels = test_labels[evaluation_indices]
    evaluation_counts = np.bincount(
        evaluation_labels,
        minlength=len(CLASS_NAMES),
    )

    assert len(evaluation_indices) == 500
    assert np.all(evaluation_counts == 50)


def save_splits(
    output_path: Path,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_indices: list[int],
    validation_indices: list[int],
    evaluation_indices: list[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_data = {
        "dataset": "STL-10",
        "seed": SEED,
        "class_names": CLASS_NAMES,
        "train_partition": {
            "split": "train",
            "indices": train_indices,
            "identifiers": [
                f"train_{index:05d}" for index in train_indices
            ],
            "labels": train_labels[train_indices].tolist(),
        },
        "validation_partition": {
            "split": "train",
            "indices": validation_indices,
            "identifiers": [
                f"train_{index:05d}" for index in validation_indices
            ],
            "labels": train_labels[validation_indices].tolist(),
        },
        "evaluation_partition": {
            "split": "test",
            "indices": evaluation_indices,
            "identifiers": [
                f"test_{index:05d}" for index in evaluation_indices
            ],
            "labels": test_labels[evaluation_indices].tolist(),
        },
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(split_data, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reproducible STL-10 splits for Task 1."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("task1/results/stl10_splits.json"),
    )
    args = parser.parse_args()

    train_dataset = STL10(
        root=args.data_root,
        split="train",
        download=True,
    )
    test_dataset = STL10(
        root=args.data_root,
        split="test",
        download=True,
    )

    train_labels = np.asarray(train_dataset.labels)
    test_labels = np.asarray(test_dataset.labels)

    train_indices, validation_indices = make_train_validation_split(
        train_labels
    )
    evaluation_indices = make_balanced_evaluation_subset(test_labels)

    verify_splits(
        train_labels=train_labels,
        test_labels=test_labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        evaluation_indices=evaluation_indices,
    )

    save_splits(
        output_path=args.output,
        train_labels=train_labels,
        test_labels=test_labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        evaluation_indices=evaluation_indices,
    )

    print(f"Training examples:   {len(train_indices)}")
    print(f"Validation examples: {len(validation_indices)}")
    print(f"Evaluation examples: {len(evaluation_indices)}")
    print(f"Saved splits to:     {args.output}")


if __name__ == "__main__":
    main()