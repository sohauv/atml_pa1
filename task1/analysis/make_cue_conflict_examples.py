"""Select reproducible cue-conflict examples and create evidence sheets."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


RESULTS_DIR = Path("task1/results")
CUE_DIR = RESULTS_DIR / "cue_conflicts"
PREDICTIONS_DIR = CUE_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures" / "cue_conflict_examples"
TABLES_DIR = RESULTS_DIR / "summary_tables"

MODEL_ORDER = [
    "resnet50_linear",
    "vit_b_16_linear",
    "clip_vit_b_32_linear",
    "clip_zero_shot",
]
MODEL_LABELS = {
    "resnet50_linear": "ResNet",
    "vit_b_16_linear": "ViT",
    "clip_vit_b_32_linear": "CLIP-L",
    "clip_zero_shot": "CLIP-ZS",
}
PREDICTION_FILES = {
    "resnet50_linear": "resnet50_linear_cue_conflicts.csv",
    "vit_b_16_linear": "vit_b_16_linear_cue_conflicts.csv",
    "clip_vit_b_32_linear": "clip_vit_b_32_linear_cue_conflicts.csv",
    "clip_zero_shot": "clip_zero_shot_cue_conflicts.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def load_merged_predictions() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for model in MODEL_ORDER:
        rows = read_csv(PREDICTIONS_DIR / PREDICTION_FILES[model])
        for row in rows:
            candidate_id = row["candidate_id"]
            if candidate_id not in merged:
                merged[candidate_id] = {
                    "candidate_id": candidate_id,
                    "pair_id": row["pair_id"],
                    "direction": row["direction"],
                    "shape_class_name": row["shape_class_name"],
                    "texture_class_name": row["texture_class_name"],
                    "image_path": row["image_path"],
                    "predictions": {},
                }
            merged[candidate_id]["predictions"][model] = {
                "predicted_class_name": row["predicted_class_name"],
                "choice": row["choice"].lower(),
                "maximum_confidence": float(row["maximum_confidence"]),
                "shape_probability": float(row["shape_probability"]),
                "texture_probability": float(row["texture_probability"]),
            }

    incomplete = [
        candidate_id
        for candidate_id, row in merged.items()
        if set(row["predictions"]) != set(MODEL_ORDER)
    ]
    if incomplete:
        raise ValueError(f"Missing model predictions for: {incomplete[:5]}")
    return merged


def mean_confidence(row: dict) -> float:
    return sum(
        row["predictions"][model]["maximum_confidence"] for model in MODEL_ORDER
    ) / len(MODEL_ORDER)


def choice_counts(row: dict) -> dict[str, int]:
    counts = {"shape": 0, "texture": 0, "other": 0}
    for model in MODEL_ORDER:
        choice = row["predictions"][model]["choice"]
        counts[choice] = counts.get(choice, 0) + 1
    return counts


def take_unique(
    candidates: list[dict],
    number: int,
    used: set[str],
) -> list[dict]:
    selected = []
    for row in candidates:
        if row["candidate_id"] in used:
            continue
        selected.append(row)
        used.add(row["candidate_id"])
        if len(selected) == number:
            break
    return selected


def select_model_examples(merged: dict[str, dict]) -> list[tuple[str, dict]]:
    rows = list(merged.values())
    used: set[str] = set()
    selected: list[tuple[str, dict]] = []

    unanimous_shape = [row for row in rows if choice_counts(row)["shape"] == 4]
    unanimous_shape.sort(key=lambda row: (-mean_confidence(row), row["candidate_id"]))

    unanimous_texture = [row for row in rows if choice_counts(row)["texture"] == 4]
    unanimous_texture.sort(key=lambda row: (-mean_confidence(row), row["candidate_id"]))

    disagreements = [
        row
        for row in rows
        if choice_counts(row)["shape"] > 0 and choice_counts(row)["texture"] > 0
    ]
    disagreements.sort(
        key=lambda row: (
            -min(choice_counts(row)["shape"], choice_counts(row)["texture"]),
            -mean_confidence(row),
            row["candidate_id"],
        )
    )

    other_predictions = [row for row in rows if choice_counts(row)["other"] > 0]
    other_predictions.sort(
        key=lambda row: (
            -choice_counts(row)["other"],
            -mean_confidence(row),
            row["candidate_id"],
        )
    )

    categories = [
        ("Unanimous shape", unanimous_shape),
        ("Unanimous texture", unanimous_texture),
        ("Shape–texture disagreement", disagreements),
        ("Other-class response", other_predictions),
    ]
    for category, candidates in categories:
        chosen = take_unique(candidates, 2, used)
        if not chosen:
            print(f"Warning: no examples found for {category}")
        selected.extend((category, row) for row in chosen)
    return selected


def prediction_text(row: dict) -> str:
    lines = []
    for model in MODEL_ORDER:
        prediction = row["predictions"][model]
        lines.append(
            f"{MODEL_LABELS[model]}: {prediction['predicted_class_name']} "
            f"[{prediction['choice']}, {prediction['maximum_confidence']:.2f}]"
        )
    return "\n".join(lines)


def plot_model_examples(selected: list[tuple[str, dict]]) -> None:
    if not selected:
        raise ValueError("No model examples were selected.")
    columns = 2
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 5.2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, (category, row) in zip(axes, selected):
        image = Image.open(row["image_path"]).convert("RGB")
        axis.imshow(image)
        axis.set_title(
            f"{category}: {row['candidate_id']}\n"
            f"shape={row['shape_class_name']}, texture={row['texture_class_name']}",
            fontsize=10,
        )
        axis.set_xlabel(prediction_text(row), fontsize=8, labelpad=8)
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in axes[len(selected) :]:
        axis.axis("off")
    figure.suptitle("Deterministically selected cue-conflict model responses", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output_path = FIGURES_DIR / "model_response_examples.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


def write_model_example_table(selected: list[tuple[str, dict]]) -> None:
    rows = []
    for category, row in selected:
        output = {
            "selection_category": category,
            "candidate_id": row["candidate_id"],
            "pair_id": row["pair_id"],
            "direction": row["direction"],
            "shape_class_name": row["shape_class_name"],
            "texture_class_name": row["texture_class_name"],
            "image_path": row["image_path"],
            "mean_model_confidence": mean_confidence(row),
        }
        for model in MODEL_ORDER:
            prediction = row["predictions"][model]
            output[f"{model}_prediction"] = prediction["predicted_class_name"]
            output[f"{model}_choice"] = prediction["choice"]
            output[f"{model}_confidence"] = prediction["maximum_confidence"]
        rows.append(output)

    output_path = TABLES_DIR / "cue_conflict_model_examples.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")


def select_generation_failures(candidate_rows: list[dict], limit: int = 6) -> list[dict]:
    rejected = [row for row in candidate_rows if parse_bool(row["accepted"]) is False]
    rejected.sort(key=lambda row: row["candidate_id"])

    by_reason: dict[str, list[dict]] = defaultdict(list)
    for row in rejected:
        reason = row["rejection_reason"].strip() or "unspecified"
        by_reason[reason].append(row)

    selected = []
    used = set()
    for reason in sorted(by_reason):
        row = by_reason[reason][0]
        selected.append(row)
        used.add(row["candidate_id"])
        if len(selected) == limit:
            return selected

    for row in rejected:
        if row["candidate_id"] not in used:
            selected.append(row)
            if len(selected) == limit:
                break
    return selected


def plot_generation_failures(selected: list[dict]) -> None:
    if not selected:
        print("Warning: no explicitly rejected generation failures were found.")
        return
    columns = 3
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4.6 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, row in zip(axes, selected):
        image = Image.open(row["image_path"]).convert("RGB")
        axis.imshow(image)
        reason = row["rejection_reason"].strip() or "unspecified"
        axis.set_title(
            f"{row['candidate_id']}: shape={row['shape_class_name']}, "
            f"texture={row['texture_class_name']}",
            fontsize=9,
        )
        axis.set_xlabel(f"Rejection: {reason}", fontsize=8, labelpad=6, wrap=True)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[len(selected) :]:
        axis.axis("off")

    figure.suptitle("Examples rejected during cue-conflict quality control", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = FIGURES_DIR / "generation_failure_examples.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


def write_failure_table(selected: list[dict]) -> None:
    if not selected:
        return
    fields = [
        "candidate_id",
        "pair_id",
        "direction",
        "shape_class_name",
        "texture_class_name",
        "image_path",
        "rejection_reason",
    ]
    output_path = TABLES_DIR / "cue_conflict_generation_failures.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    print(f"Saved {output_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    merged = load_merged_predictions()
    model_examples = select_model_examples(merged)
    plot_model_examples(model_examples)
    write_model_example_table(model_examples)

    candidate_rows = read_csv(CUE_DIR / "candidates.csv")
    failures = select_generation_failures(candidate_rows)
    plot_generation_failures(failures)
    write_failure_table(failures)

    print(f"Selected {len(model_examples)} accepted model-response examples.")
    print(f"Selected {len(failures)} rejected generation-failure examples.")


if __name__ == "__main__":
    main()
