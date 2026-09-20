import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from common.seed import set_seed
from task1.analysis.evaluate_bias import (
    compute_classification_metrics,
    compute_prediction_consistency,
    save_prediction_table,
)
from task1.analysis.extract_features import (
    extract_features,
    load_feature_cache,
    save_feature_cache,
)
from task1.data.datasets import build_task1_datasets
from task1.data.intervention_dataset import (
    build_intervention_dataloader,
)
from task1.data.make_subset import CLASS_NAMES
from task1.models.backbones import (
    CLIPViTB32Backbone,
    build_backbone,
)
from task1.models.linear_head import (
    LinearClassifier,
    evaluate_classifier,
    make_feature_dataloader,
)


MODEL_CONFIGURATIONS = [
    {
        "result_name": "resnet50_linear",
        "backbone_name": "resnet50",
        "cache_name": "resnet50",
    },
    {
        "result_name": "vit_b_16_linear",
        "backbone_name": "vit_b_16",
        "cache_name": "vit_b_16",
    },
    {
        "result_name": "clip_vit_b_32_linear",
        "backbone_name": "clip",
        "cache_name": "clip_vit_b_32",
    },
]


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(
    data: dict[str, Any],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_classifier(
    checkpoint_path: str | Path,
    device: torch.device,
) -> LinearClassifier:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    classifier = LinearClassifier(
        feature_dim=checkpoint["feature_dim"],
        number_of_classes=checkpoint["number_of_classes"],
    )

    classifier.load_state_dict(checkpoint["state_dict"])
    classifier = classifier.to(device)
    classifier.eval()

    return classifier


def build_intervention_specifications(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    specifications = [
        {
            "condition": "grayscale",
        },
        {
            "condition": "hue_rotation",
            "hue_factor": config[
                "interventions"
            ]["additional_color"]["hue_factor"],
        },
        {
            "condition": "patch_shuffle",
            "grid_size": config[
                "interventions"
            ]["patch_shuffle"]["grid_rows"],
        },
    ]

    displacements = config[
        "interventions"
    ]["translation"]["displacements"]

    directions = config[
        "interventions"
    ]["translation"]["directions"]

    for displacement in displacements:
        if displacement == 0:
            continue

        for direction in directions:
            specifications.append(
                {
                    "condition": "translation",
                    "displacement": displacement,
                    "direction": direction,
                }
            )

    return specifications


def get_or_extract_intervention_features(
    backbone: torch.nn.Module,
    model_cache_name: str,
    clean_evaluation_dataset: torch.utils.data.Dataset,
    specification: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    overwrite_cache: bool,
) -> tuple[str, dict[str, Any]]:
    dataloader = build_intervention_dataloader(
        clean_dataset=clean_evaluation_dataset,
        condition=specification["condition"],
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=(
            config["dataloader"]["pin_memory"]
            and device.type == "cuda"
        ),
        seed=config["seed"],
        hue_factor=specification.get(
            "hue_factor",
            config[
                "interventions"
            ]["additional_color"]["hue_factor"],
        ),
        grid_size=specification.get(
            "grid_size",
            config[
                "interventions"
            ]["patch_shuffle"]["grid_rows"],
        ),
        displacement=specification.get("displacement", 0),
        direction=specification.get("direction", "right"),
    )

    condition_name = dataloader.dataset.get_condition_name()

    cache_path = (
        Path("task1/cache")
        / model_cache_name
        / "interventions"
        / f"{condition_name}.pt"
    )

    if cache_path.exists() and not overwrite_cache:
        print(f"Using existing cache: {cache_path}")
        feature_data = load_feature_cache(cache_path)
        return condition_name, feature_data

    feature_data = extract_features(
        backbone=backbone,
        dataloader=dataloader,
        device=device,
        split_name=condition_name,
    )

    save_feature_cache(
        feature_data=feature_data,
        output_path=cache_path,
        model_name=model_cache_name,
        split_name=condition_name,
    )

    print(
        f"Saved {feature_data['features'].shape} "
        f"features to {cache_path}"
    )

    return condition_name, feature_data


def evaluate_linear_features(
    classifier: LinearClassifier,
    feature_data: dict[str, Any],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    dataloader = make_feature_dataloader(
        feature_data=feature_data,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    return evaluate_classifier(
        classifier=classifier,
        dataloader=dataloader,
        device=device,
    )


@torch.inference_mode()
def make_zero_shot_logits(
    backbone: CLIPViTB32Backbone,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    image_features = image_features.to(device)
    text_features = text_features.to(device)

    logits = (
        backbone.get_logit_scale()
        * image_features
        @ text_features.T
    )

    return logits.cpu()


def create_result_record(
    logits: torch.Tensor,
    labels: torch.Tensor,
    clean_predictions: torch.Tensor,
    clean_accuracy: float,
) -> dict[str, float]:
    metrics = compute_classification_metrics(
        logits=logits,
        labels=labels,
    )

    transformed_predictions = logits.argmax(dim=1)

    consistency = compute_prediction_consistency(
        clean_predictions=clean_predictions,
        transformed_predictions=transformed_predictions,
    )

    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "mean_maximum_confidence": float(
            metrics["mean_maximum_confidence"]
        ),
        "accuracy_change": float(
            metrics["accuracy"] - clean_accuracy
        ),
        "prediction_consistency": float(consistency),
    }


def aggregate_translation_results(
    model_results: dict[str, Any],
    clean_metrics: dict[str, float],
) -> dict[str, Any]:
    aggregated_results = {
        "0": {
            "accuracy": float(clean_metrics["accuracy"]),
            "macro_f1": float(clean_metrics["macro_f1"]),
            "mean_maximum_confidence": float(
                clean_metrics["mean_maximum_confidence"]
            ),
            "accuracy_change": 0.0,
            "prediction_consistency": 1.0,
        }
    }

    for displacement in [8, 16, 32]:
        direction_records = []

        for direction in ["left", "right", "up", "down"]:
            condition_name = (
                f"translation_{displacement}_{direction}"
            )
            direction_records.append(
                model_results[condition_name]
            )

        aggregated_results[str(displacement)] = {
            metric_name: float(
                np.mean(
                    [
                        record[metric_name]
                        for record in direction_records
                    ]
                )
            )
            for metric_name in [
                "accuracy",
                "macro_f1",
                "mean_maximum_confidence",
                "accuracy_change",
                "prediction_consistency",
            ]
        }

    return aggregated_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Task 1 image interventions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("task1/configs/task1.yaml"),
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

    datasets = build_task1_datasets(
        data_root=config["dataset"]["root"],
        split_path="task1/results/stl10_splits.json",
        image_size=config["dataset"]["image_size"],
        download=False,
    )

    clean_evaluation_dataset = datasets["evaluation"]

    predictions_directory = Path(
        "task1/results/predictions"
    )
    predictions_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    specifications = build_intervention_specifications(config)
    all_results = {}

    for model_configuration in MODEL_CONFIGURATIONS:
        result_name = model_configuration["result_name"]
        backbone_name = model_configuration["backbone_name"]
        cache_name = model_configuration["cache_name"]

        print("\n" + "=" * 70)
        print(f"Evaluating interventions: {result_name}")
        print("=" * 70)

        backbone = build_backbone(backbone_name).to(device)
        backbone.eval()

        classifier = load_classifier(
            checkpoint_path=(
                Path("task1/checkpoints")
                / f"{result_name}.pth"
            ),
            device=device,
        )

        clean_feature_data = load_feature_cache(
            Path("task1/cache")
            / cache_name
            / "evaluation.pt"
        )

        clean_evaluation = evaluate_linear_features(
            classifier=classifier,
            feature_data=clean_feature_data,
            device=device,
            batch_size=config["dataloader"]["batch_size"],
            seed=seed,
        )

        clean_predictions = clean_evaluation[
            "predictions"
        ]
        clean_accuracy = float(clean_evaluation["accuracy"])

        model_results = {}

        clip_text_features = None
        zero_shot_clean_predictions = None
        zero_shot_clean_accuracy = None
        zero_shot_results = {}

        if isinstance(backbone, CLIPViTB32Backbone):
            clip_text_features = backbone.encode_text_prompts(
                class_names=CLASS_NAMES,
                prompt_template=config[
                    "models"
                ]["clip"]["prompt"],
            )

            zero_shot_clean_logits = make_zero_shot_logits(
                backbone=backbone,
                image_features=clean_feature_data["features"],
                text_features=clip_text_features,
                device=device,
            )

            zero_shot_clean_metrics = (
                compute_classification_metrics(
                    logits=zero_shot_clean_logits,
                    labels=clean_feature_data["labels"],
                )
            )

            zero_shot_clean_predictions = (
                zero_shot_clean_logits.argmax(dim=1)
            )
            zero_shot_clean_accuracy = float(
                zero_shot_clean_metrics["accuracy"]
            )

        for specification in specifications:
            condition_name, feature_data = (
                get_or_extract_intervention_features(
                    backbone=backbone,
                    model_cache_name=cache_name,
                    clean_evaluation_dataset=(
                        clean_evaluation_dataset
                    ),
                    specification=specification,
                    config=config,
                    device=device,
                    overwrite_cache=args.overwrite_cache,
                )
            )

            transformed_evaluation = evaluate_linear_features(
                classifier=classifier,
                feature_data=feature_data,
                device=device,
                batch_size=config["dataloader"]["batch_size"],
                seed=seed,
            )

            record = create_result_record(
                logits=transformed_evaluation["logits"],
                labels=feature_data["labels"],
                clean_predictions=clean_predictions,
                clean_accuracy=clean_accuracy,
            )

            model_results[condition_name] = record

            print(f"{condition_name}: {record}")

            save_prediction_table(
                output_path=(
                    predictions_directory
                    / f"{result_name}_{condition_name}.csv"
                ),
                identifiers=feature_data["identifiers"],
                indices=feature_data["indices"],
                labels=feature_data["labels"],
                logits=transformed_evaluation["logits"],
                class_names=CLASS_NAMES,
                condition=condition_name,
                model_name=result_name,
            )

            if isinstance(backbone, CLIPViTB32Backbone):
                zero_shot_logits = make_zero_shot_logits(
                    backbone=backbone,
                    image_features=feature_data["features"],
                    text_features=clip_text_features,
                    device=device,
                )

                zero_shot_record = create_result_record(
                    logits=zero_shot_logits,
                    labels=feature_data["labels"],
                    clean_predictions=(
                        zero_shot_clean_predictions
                    ),
                    clean_accuracy=zero_shot_clean_accuracy,
                )

                zero_shot_results[
                    condition_name
                ] = zero_shot_record

                save_prediction_table(
                    output_path=(
                        predictions_directory
                        / (
                            "clip_zero_shot_"
                            f"{condition_name}.csv"
                        )
                    ),
                    identifiers=feature_data["identifiers"],
                    indices=feature_data["indices"],
                    labels=feature_data["labels"],
                    logits=zero_shot_logits,
                    class_names=CLASS_NAMES,
                    condition=condition_name,
                    model_name="clip_zero_shot",
                )

        model_results["translation_averaged"] = (
            aggregate_translation_results(
                model_results=model_results,
                clean_metrics={
                    "accuracy": clean_evaluation["accuracy"],
                    "macro_f1": clean_evaluation["macro_f1"],
                    "mean_maximum_confidence": (
                        clean_evaluation[
                            "mean_maximum_confidence"
                        ]
                    ),
                },
            )
        )

        all_results[result_name] = model_results

        if isinstance(backbone, CLIPViTB32Backbone):
            zero_shot_clean_metrics_for_aggregation = {
                key: float(value)
                for key, value in (
                    compute_classification_metrics(
                        logits=zero_shot_clean_logits,
                        labels=clean_feature_data["labels"],
                    )
                ).items()
            }

            zero_shot_results["translation_averaged"] = (
                aggregate_translation_results(
                    model_results=zero_shot_results,
                    clean_metrics=(
                        zero_shot_clean_metrics_for_aggregation
                    ),
                )
            )

            all_results["clip_zero_shot"] = (
                zero_shot_results
            )

        del backbone
        del classifier
        del clean_feature_data

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(
        data=all_results,
        output_path="task1/results/intervention_results.json",
    )

    print("\nFinished intervention evaluation.")
    print(
        "Results saved to "
        "task1/results/intervention_results.json"
    )


if __name__ == "__main__":
    main()