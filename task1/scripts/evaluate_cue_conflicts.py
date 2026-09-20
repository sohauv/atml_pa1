from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common.seed import set_seed
from task1.analysis.extract_features import (
    extract_features,
    load_feature_cache,
    save_feature_cache,
)
from task1.data.cue_conflict_dataset import (
    build_cue_conflict_dataloader,
    parse_accepted_column,
)
from task1.data.make_subset import CLASS_NAMES
from task1.models.backbones import (
    CLIPViTB32Backbone,
    build_backbone,
)
from task1.scripts.evaluate_interventions import (
    MODEL_CONFIGURATIONS,
    evaluate_linear_features,
    load_classifier,
    load_config,
    make_zero_shot_logits,
    save_json,
)


def load_accepted_metadata(
    metadata_path: str | Path,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)

    accepted_mask = parse_accepted_column(
        metadata["accepted"]
    )

    accepted_metadata = metadata.loc[
        accepted_mask
    ].copy()

    accepted_metadata = accepted_metadata.reset_index(drop=True)

    if accepted_metadata.empty:
        raise ValueError(
            "No accepted cue-conflict candidates were found."
        )

    return accepted_metadata


def get_or_extract_features(
    backbone: torch.nn.Module,
    model_cache_name: str,
    metadata_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
    overwrite_cache: bool,
) -> dict[str, Any]:
    cache_path = (
        Path("task1/cache")
        / model_cache_name
        / "cue_conflicts.pt"
    )

    if cache_path.exists() and not overwrite_cache:
        print(f"Using existing cache: {cache_path}")
        return load_feature_cache(cache_path)

    dataloader = build_cue_conflict_dataloader(
        metadata_path=metadata_path,
        image_size=config["dataset"]["image_size"],
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=(
            config["dataloader"]["pin_memory"]
            and device.type == "cuda"
        ),
        accepted_only=True,
    )

    feature_data = extract_features(
        backbone=backbone,
        dataloader=dataloader,
        device=device,
        split_name="cue_conflicts",
    )

    save_feature_cache(
        feature_data=feature_data,
        output_path=cache_path,
        model_name=model_cache_name,
        split_name="cue_conflicts",
    )

    print(
        f"Saved {feature_data['features'].shape} "
        f"features to {cache_path}"
    )

    return feature_data


def verify_metadata_alignment(
    metadata: pd.DataFrame,
    feature_data: dict[str, Any],
) -> None:
    metadata_identifiers = (
        metadata["candidate_id"].astype(str).tolist()
    )

    feature_identifiers = [
        str(identifier)
        for identifier in feature_data["identifiers"]
    ]

    if metadata_identifiers != feature_identifiers:
        raise ValueError(
            "Cue-conflict metadata and extracted features "
            "are not in the same order. Delete the existing "
            "cue-conflict cache or use --overwrite-cache."
        )


def build_prediction_table(
    metadata: pd.DataFrame,
    logits: torch.Tensor,
    model_name: str,
) -> pd.DataFrame:
    if len(metadata) != logits.shape[0]:
        raise ValueError(
            "Metadata and logits have different lengths: "
            f"{len(metadata)} and {logits.shape[0]}."
        )

    probabilities = torch.softmax(
        logits.float(),
        dim=1,
    )

    predictions = logits.argmax(dim=1).cpu().numpy()
    confidence = (
        probabilities.max(dim=1).values.cpu().numpy()
    )

    shape_labels = (
        metadata["shape_class_id"]
        .astype(int)
        .to_numpy()
    )
    texture_labels = (
        metadata["texture_class_id"]
        .astype(int)
        .to_numpy()
    )

    row_indices = torch.arange(logits.shape[0])

    shape_probabilities = probabilities[
        row_indices,
        torch.tensor(shape_labels, dtype=torch.long),
    ].cpu().numpy()

    texture_probabilities = probabilities[
        row_indices,
        torch.tensor(texture_labels, dtype=torch.long),
    ].cpu().numpy()

    choice = np.full(
        shape=len(metadata),
        fill_value="other",
        dtype=object,
    )
    choice[predictions == shape_labels] = "shape"
    choice[predictions == texture_labels] = "texture"

    table = metadata[
        [
            "candidate_id",
            "pair_id",
            "direction",
            "content_identifier",
            "shape_class_id",
            "shape_class_name",
            "style_identifier",
            "texture_class_id",
            "texture_class_name",
            "style_strength",
            "image_path",
        ]
    ].copy()

    table.insert(0, "model_name", model_name)

    table["predicted_class_id"] = predictions
    table["predicted_class_name"] = [
        CLASS_NAMES[prediction]
        for prediction in predictions
    ]
    table["choice"] = choice
    table["maximum_confidence"] = confidence
    table["shape_probability"] = shape_probabilities
    table["texture_probability"] = texture_probabilities

    return table


def summarize_predictions(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    total_count = len(predictions)

    shape_count = int(
        (predictions["choice"] == "shape").sum()
    )
    texture_count = int(
        (predictions["choice"] == "texture").sum()
    )
    other_count = int(
        (predictions["choice"] == "other").sum()
    )

    shape_texture_count = shape_count + texture_count

    if shape_texture_count > 0:
        shape_bias = shape_count / shape_texture_count
        texture_bias = (
            texture_count / shape_texture_count
        )
    else:
        shape_bias = None
        texture_bias = None

    return {
        "number_of_images": int(total_count),
        "shape_choices": shape_count,
        "texture_choices": texture_count,
        "other_choices": other_count,
        "shape_choice_rate": float(
            shape_count / total_count
        ),
        "texture_choice_rate": float(
            texture_count / total_count
        ),
        "other_choice_rate": float(
            other_count / total_count
        ),
        "shape_texture_decisions": int(
            shape_texture_count
        ),
        "shape_bias": (
            None
            if shape_bias is None
            else float(shape_bias)
        ),
        "texture_bias": (
            None
            if texture_bias is None
            else float(texture_bias)
        ),
        "mean_maximum_confidence": float(
            predictions[
                "maximum_confidence"
            ].mean()
        ),
        "mean_shape_probability": float(
            predictions["shape_probability"].mean()
        ),
        "mean_texture_probability": float(
            predictions["texture_probability"].mean()
        ),
    }


def summarize_by_column(
    predictions: pd.DataFrame,
    column_name: str,
) -> dict[str, Any]:
    return {
        str(group_name): summarize_predictions(group)
        for group_name, group in predictions.groupby(
            column_name,
            sort=True,
        )
    }


def evaluate_logits(
    metadata: pd.DataFrame,
    logits: torch.Tensor,
    model_name: str,
    predictions_directory: Path,
) -> dict[str, Any]:
    prediction_table = build_prediction_table(
        metadata=metadata,
        logits=logits,
        model_name=model_name,
    )

    output_path = (
        predictions_directory
        / f"{model_name}_cue_conflicts.csv"
    )
    prediction_table.to_csv(output_path, index=False)

    return {
        "overall": summarize_predictions(
            prediction_table
        ),
        "by_direction": summarize_by_column(
            prediction_table,
            "direction",
        ),
        "by_pair": summarize_by_column(
            prediction_table,
            "pair_id",
        ),
        "prediction_table": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate shape and texture choices on "
            "Task 1 cue-conflict images."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("task1/configs/task1.yaml"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "task1/results/cue_conflicts/candidates.csv"
        ),
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    metadata = load_accepted_metadata(args.metadata)

    total_metadata = pd.read_csv(args.metadata)
    accepted_mask = parse_accepted_column(
        total_metadata["accepted"]
    )

    print(f"Total candidates: {len(total_metadata)}")
    print(f"Accepted candidates: {int(accepted_mask.sum())}")
    print(
        "Rejected candidates: "
        f"{int((~accepted_mask).sum())}"
    )

    minimum_valid_images = config[
        "cue_conflicts"
    ]["minimum_valid_images"]

    if len(metadata) < minimum_valid_images:
        raise ValueError(
            f"Only {len(metadata)} accepted images were found, "
            f"but at least {minimum_valid_images} are required."
        )

    predictions_directory = Path(
        "task1/results/cue_conflicts/predictions"
    )
    predictions_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results: dict[str, Any] = {
        "dataset": {
            "total_candidates": int(
                len(total_metadata)
            ),
            "accepted_candidates": int(
                accepted_mask.sum()
            ),
            "rejected_candidates": int(
                (~accepted_mask).sum()
            ),
            "style_strength": float(
                metadata["style_strength"].iloc[0]
            ),
        },
        "models": {},
    }

    for model_configuration in MODEL_CONFIGURATIONS:
        result_name = model_configuration["result_name"]
        backbone_name = model_configuration["backbone_name"]
        cache_name = model_configuration["cache_name"]

        print("\n" + "=" * 70)
        print(f"Evaluating cue conflicts: {result_name}")
        print("=" * 70)

        backbone = build_backbone(
            backbone_name
        ).to(device)
        backbone.eval()

        feature_data = get_or_extract_features(
            backbone=backbone,
            model_cache_name=cache_name,
            metadata_path=args.metadata,
            config=config,
            device=device,
            overwrite_cache=args.overwrite_cache,
        )

        verify_metadata_alignment(
            metadata=metadata,
            feature_data=feature_data,
        )

        classifier = load_classifier(
            checkpoint_path=(
                Path("task1/checkpoints")
                / f"{result_name}.pth"
            ),
            device=device,
        )

        linear_evaluation = evaluate_linear_features(
            classifier=classifier,
            feature_data=feature_data,
            device=device,
            batch_size=config["dataloader"]["batch_size"],
            seed=seed,
        )

        linear_results = evaluate_logits(
            metadata=metadata,
            logits=linear_evaluation["logits"],
            model_name=result_name,
            predictions_directory=predictions_directory,
        )

        all_results["models"][
            result_name
        ] = linear_results

        print(
            "Overall linear results: "
            f"{linear_results['overall']}"
        )

        if isinstance(backbone, CLIPViTB32Backbone):
            text_features = backbone.encode_text_prompts(
                class_names=CLASS_NAMES,
                prompt_template=config[
                    "models"
                ]["clip"]["prompt"],
            )

            zero_shot_logits = make_zero_shot_logits(
                backbone=backbone,
                image_features=feature_data["features"],
                text_features=text_features,
                device=device,
            )

            zero_shot_results = evaluate_logits(
                metadata=metadata,
                logits=zero_shot_logits,
                model_name="clip_zero_shot",
                predictions_directory=(
                    predictions_directory
                ),
            )

            all_results["models"][
                "clip_zero_shot"
            ] = zero_shot_results

            print(
                "Overall zero-shot results: "
                f"{zero_shot_results['overall']}"
            )

        del backbone
        del classifier
        del feature_data

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(
        "task1/results/cue_conflict_results.json"
    )

    save_json(
        data=all_results,
        output_path=output_path,
    )

    print("\nFinished cue-conflict evaluation.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()