import argparse
import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from common.seed import set_seed
from task1.analysis.evaluate_bias import (
    compute_classification_metrics,
    save_prediction_table,
)
from task1.analysis.extract_features import (
    extract_and_cache_splits,
    load_feature_cache,
)
from task1.data.datasets import (
    build_task1_dataloaders,
    build_task1_datasets,
)
from task1.data.make_subset import CLASS_NAMES
from task1.models.backbones import (
    CLIPViTB32Backbone,
    build_backbone,
)
from task1.models.linear_head import (
    evaluate_classifier,
    make_feature_dataloader,
    train_linear_classifier,
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


def save_classifier_checkpoint(
    classifier: torch.nn.Module,
    feature_dim: int,
    best_epoch: int,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in classifier.state_dict().items()
    }

    torch.save(
        {
            "feature_dim": feature_dim,
            "number_of_classes": len(CLASS_NAMES),
            "best_epoch": best_epoch,
            "state_dict": state_dict,
        },
        output_path,
    )


@torch.inference_mode()
def evaluate_zero_shot_clip(
    backbone: CLIPViTB32Backbone,
    evaluation_features: dict[str, Any],
    device: torch.device,
    prompt_template: str,
) -> torch.Tensor:
    backbone = backbone.to(device)
    backbone.eval()

    image_features = evaluation_features["features"].to(device)

    text_features = backbone.encode_text_prompts(
        class_names=CLASS_NAMES,
        prompt_template=prompt_template,
    )

    logit_scale = backbone.get_logit_scale()

    logits = logit_scale * image_features @ text_features.T

    return logits.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Task 1 clean-baseline experiment."
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    split_path = Path("task1/results/stl10_splits.json")

    if not split_path.exists():
        raise FileNotFoundError(
            "The STL-10 split file does not exist. Run:\n"
            "python -m task1.data.make_subset"
        )

    datasets = build_task1_datasets(
        data_root=config["dataset"]["root"],
        split_path=split_path,
        image_size=config["dataset"]["image_size"],
        download=True,
    )

    dataloaders = build_task1_dataloaders(
        datasets=datasets,
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=(
            config["dataloader"]["pin_memory"]
            and device.type == "cuda"
        ),
    )

    results_directory = Path("task1/results")
    predictions_directory = results_directory / "predictions"
    training_directory = results_directory / "training"
    checkpoints_directory = Path("task1/checkpoints")

    predictions_directory.mkdir(parents=True, exist_ok=True)
    training_directory.mkdir(parents=True, exist_ok=True)
    checkpoints_directory.mkdir(parents=True, exist_ok=True)

    all_clean_metrics = {}

    for model_configuration in MODEL_CONFIGURATIONS:
        result_name = model_configuration["result_name"]
        backbone_name = model_configuration["backbone_name"]
        cache_name = model_configuration["cache_name"]

        print("\n" + "=" * 70)
        print(f"Running clean baseline: {result_name}")
        print("=" * 70)

        # Reset randomness before constructing each model and classifier.
        set_seed(seed)

        backbone = build_backbone(backbone_name)

        cache_paths = extract_and_cache_splits(
            model_name=cache_name,
            backbone=backbone,
            dataloaders=dataloaders,
            device=device,
            cache_directory="task1/cache",
            overwrite=args.overwrite_cache,
        )

        train_features = load_feature_cache(
            cache_paths["train"]
        )
        validation_features = load_feature_cache(
            cache_paths["validation"]
        )
        evaluation_features = load_feature_cache(
            cache_paths["evaluation"]
        )

        # Reset the seed before initializing and training the linear head.
        set_seed(seed)

        classifier, history, best_epoch = train_linear_classifier(
            train_features=train_features,
            validation_features=validation_features,
            device=device,
            batch_size=config["dataloader"]["batch_size"],
            max_epochs=config["training"]["max_epochs"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            patience=config["training"]["patience"],
            seed=seed,
        )

        evaluation_dataloader = make_feature_dataloader(
            feature_data=evaluation_features,
            batch_size=config["dataloader"]["batch_size"],
            shuffle=False,
            seed=seed,
        )

        evaluation_results = evaluate_classifier(
            classifier=classifier,
            dataloader=evaluation_dataloader,
            device=device,
        )

        clean_metrics = {
            "accuracy": float(evaluation_results["accuracy"]),
            "macro_f1": float(evaluation_results["macro_f1"]),
            "mean_maximum_confidence": float(
                evaluation_results[
                    "mean_maximum_confidence"
                ]
            ),
            "best_epoch": int(best_epoch),
            "feature_dimension": int(
                evaluation_features["features"].shape[1]
            ),
        }

        all_clean_metrics[result_name] = clean_metrics

        print(f"Clean metrics: {clean_metrics}")

        save_prediction_table(
            output_path=(
                predictions_directory
                / f"{result_name}_clean.csv"
            ),
            identifiers=evaluation_features["identifiers"],
            indices=evaluation_features["indices"],
            labels=evaluation_features["labels"],
            logits=evaluation_results["logits"],
            class_names=CLASS_NAMES,
            condition="clean",
            model_name=result_name,
        )

        pd.DataFrame(history).to_csv(
            training_directory
            / f"{result_name}_history.csv",
            index=False,
        )

        save_classifier_checkpoint(
            classifier=classifier,
            feature_dim=clean_metrics["feature_dimension"],
            best_epoch=best_epoch,
            output_path=(
                checkpoints_directory
                / f"{result_name}.pth"
            ),
        )

        if isinstance(backbone, CLIPViTB32Backbone):
            prompt_template = config["models"]["clip"]["prompt"]

            zero_shot_logits = evaluate_zero_shot_clip(
                backbone=backbone,
                evaluation_features=evaluation_features,
                device=device,
                prompt_template=prompt_template,
            )

            zero_shot_metrics = compute_classification_metrics(
                logits=zero_shot_logits,
                labels=evaluation_features["labels"],
            )

            zero_shot_metrics = {
                key: float(value)
                for key, value in zero_shot_metrics.items()
            }

            all_clean_metrics["clip_zero_shot"] = (
                zero_shot_metrics
            )

            print(
                f"CLIP zero-shot metrics: {zero_shot_metrics}"
            )

            save_prediction_table(
                output_path=(
                    predictions_directory
                    / "clip_zero_shot_clean.csv"
                ),
                identifiers=evaluation_features["identifiers"],
                indices=evaluation_features["indices"],
                labels=evaluation_features["labels"],
                logits=zero_shot_logits,
                class_names=CLASS_NAMES,
                condition="clean",
                model_name="clip_zero_shot",
            )

        del backbone
        del classifier
        del train_features
        del validation_features
        del evaluation_features

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(
        data=all_clean_metrics,
        output_path=results_directory / "clean_baselines.json",
    )

    print("\nFinished all clean-baseline experiments.")
    print(
        "Results saved to "
        "task1/results/clean_baselines.json"
    )


if __name__ == "__main__":
    main()