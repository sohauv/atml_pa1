from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from common.seed import set_seed
from task1.analysis.extract_features import (
    extract_features,
    load_feature_cache,
    save_feature_cache,
)
from task1.analysis.feature_similarity import (
    average_stability_records,
    compute_pairwise_cosine_similarity,
    summarize_cosine_stability,
)
from task1.analysis.representation import (
    create_joint_umap_plot,
)
from task1.data.cue_conflict_dataset import (
    build_cue_conflict_content_dataloader,
)
from task1.data.make_subset import CLASS_NAMES
from task1.models.backbones import build_backbone
from task1.scripts.evaluate_interventions import (
    MODEL_CONFIGURATIONS,
    load_config,
    save_json,
)


DIRECTIONS = ["left", "right", "up", "down"]
DISPLACEMENTS = [8, 16, 32]


def identifiers_as_strings(
    feature_data: dict[str, Any],
) -> list[str]:
    return [
        str(identifier)
        for identifier in feature_data["identifiers"]
    ]


def validate_aligned_feature_data(
    clean_data: dict[str, Any],
    transformed_data: dict[str, Any],
    comparison_name: str,
) -> None:
    clean_identifiers = identifiers_as_strings(clean_data)
    transformed_identifiers = identifiers_as_strings(
        transformed_data
    )

    if clean_identifiers != transformed_identifiers:
        raise ValueError(
            f"Identifier mismatch for {comparison_name}."
        )

    if not torch.equal(
        clean_data["labels"].long(),
        transformed_data["labels"].long(),
    ):
        raise ValueError(
            f"Label mismatch for {comparison_name}."
        )


def save_similarity_table(
    clean_data: dict[str, Any],
    transformed_data: dict[str, Any],
    model_name: str,
    condition_name: str,
    output_path: str | Path,
) -> None:
    similarities = compute_pairwise_cosine_similarity(
        clean_features=clean_data["features"],
        transformed_features=(
            transformed_data["features"]
        ),
    )

    labels = clean_data["labels"].long().cpu().tolist()

    table = pd.DataFrame(
        {
            "model_name": model_name,
            "condition": condition_name,
            "identifier": identifiers_as_strings(
                clean_data
            ),
            "class_id": labels,
            "class_name": [
                CLASS_NAMES[class_id]
                for class_id in labels
            ],
            "cosine_similarity": (
                similarities.cpu().numpy()
            ),
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    table.to_csv(output_path, index=False)


def analyze_feature_pair(
    clean_data: dict[str, Any],
    transformed_data: dict[str, Any],
    model_name: str,
    condition_name: str,
    tables_directory: Path,
) -> dict[str, Any]:
    validate_aligned_feature_data(
        clean_data=clean_data,
        transformed_data=transformed_data,
        comparison_name=condition_name,
    )

    summary = summarize_cosine_stability(
        clean_features=clean_data["features"],
        transformed_features=(
            transformed_data["features"]
        ),
    )

    table_path = (
        tables_directory
        / f"{model_name}_{condition_name}.csv"
    )

    save_similarity_table(
        clean_data=clean_data,
        transformed_data=transformed_data,
        model_name=model_name,
        condition_name=condition_name,
        output_path=table_path,
    )

    summary["per_example_table"] = str(table_path)

    return summary


def get_or_extract_clean_cue_features(
    backbone: torch.nn.Module,
    cache_name: str,
    config: dict[str, Any],
    metadata_path: str | Path,
    device: torch.device,
    overwrite_cache: bool,
) -> dict[str, Any]:
    cache_path = (
        Path("task1/cache")
        / cache_name
        / "cue_conflict_clean_content.pt"
    )

    if cache_path.exists() and not overwrite_cache:
        print(f"Using existing cache: {cache_path}")
        return load_feature_cache(cache_path)

    dataloader = build_cue_conflict_content_dataloader(
        metadata_path=metadata_path,
        data_root=config["dataset"]["root"],
        image_size=config["dataset"]["image_size"],
        batch_size=config["dataloader"]["batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=(
            config["dataloader"]["pin_memory"]
            and device.type == "cuda"
        ),
    )

    feature_data = extract_features(
        backbone=backbone,
        dataloader=dataloader,
        device=device,
        split_name="cue_conflict_clean_content",
    )

    save_feature_cache(
        feature_data=feature_data,
        output_path=cache_path,
        model_name=cache_name,
        split_name="cue_conflict_clean_content",
    )

    print(
        f"Saved {feature_data['features'].shape} "
        f"features to {cache_path}"
    )

    return feature_data


def load_intervention_features(
    cache_name: str,
    condition_name: str,
) -> dict[str, Any]:
    cache_path = (
        Path("task1/cache")
        / cache_name
        / "interventions"
        / f"{condition_name}.pt"
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing intervention cache: {cache_path}. "
            "Run evaluate_interventions.py first."
        )

    return load_feature_cache(cache_path)


def create_umap_for_pair(
    clean_data: dict[str, Any],
    transformed_data: dict[str, Any],
    model_name: str,
    condition_name: str,
    figures_directory: Path,
    seed: int,
) -> str:
    validate_aligned_feature_data(
        clean_data=clean_data,
        transformed_data=transformed_data,
        comparison_name=condition_name,
    )

    output_path = (
        figures_directory
        / f"{model_name}_{condition_name}_umap.png"
    )

    create_joint_umap_plot(
        clean_features=clean_data["features"],
        transformed_features=(
            transformed_data["features"]
        ),
        labels=clean_data["labels"],
        class_names=CLASS_NAMES,
        model_name=model_name,
        intervention_name=condition_name,
        output_path=output_path,
        seed=seed,
        number_of_neighbors=15,
        minimum_distance=0.1,
        metric="cosine",
    )

    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Task 1 representation stability "
            "and joint UMAP visualizations."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("task1/configs/task1.yaml"),
    )
    parser.add_argument(
        "--cue-metadata",
        type=Path,
        default=Path(
            "task1/results/cue_conflicts/candidates.csv"
        ),
    )
    parser.add_argument(
        "--overwrite-cue-cache",
        action="store_true",
    )
    parser.add_argument(
        "--skip-umap",
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

    figures_directory = Path(
        "task1/results/figures/representations"
    )
    tables_directory = Path(
        "task1/results/representation_tables"
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    tables_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results: dict[str, Any] = {
        "umap_settings": {
            "number_of_neighbors": 15,
            "minimum_distance": 0.1,
            "metric": "cosine",
            "seed": seed,
            "joint_fit": True,
        },
        "models": {},
    }

    for model_configuration in MODEL_CONFIGURATIONS:
        model_name = model_configuration["result_name"]
        backbone_name = model_configuration["backbone_name"]
        cache_name = model_configuration["cache_name"]

        print("\n" + "=" * 70)
        print(f"Representation analysis: {model_name}")
        print("=" * 70)

        clean_data = load_feature_cache(
            Path("task1/cache")
            / cache_name
            / "evaluation.pt"
        )

        grayscale_data = load_intervention_features(
            cache_name=cache_name,
            condition_name="grayscale",
        )

        patch_data = load_intervention_features(
            cache_name=cache_name,
            condition_name="patch_shuffle_4x4",
        )

        backbone = build_backbone(
            backbone_name
        ).to(device)
        backbone.eval()

        clean_cue_data = (
            get_or_extract_clean_cue_features(
                backbone=backbone,
                cache_name=cache_name,
                config=config,
                metadata_path=args.cue_metadata,
                device=device,
                overwrite_cache=(
                    args.overwrite_cue_cache
                ),
            )
        )

        generated_cue_data = load_feature_cache(
            Path("task1/cache")
            / cache_name
            / "cue_conflicts.pt"
        )

        model_results: dict[str, Any] = {
            "stability": {},
            "umap_figures": {},
        }

        model_results["stability"]["grayscale"] = (
            analyze_feature_pair(
                clean_data=clean_data,
                transformed_data=grayscale_data,
                model_name=model_name,
                condition_name="grayscale",
                tables_directory=tables_directory,
            )
        )

        model_results["stability"]["patch_shuffle_4x4"] = (
            analyze_feature_pair(
                clean_data=clean_data,
                transformed_data=patch_data,
                model_name=model_name,
                condition_name="patch_shuffle_4x4",
                tables_directory=tables_directory,
            )
        )

        model_results["stability"]["cue_conflict"] = (
            analyze_feature_pair(
                clean_data=clean_cue_data,
                transformed_data=generated_cue_data,
                model_name=model_name,
                condition_name="cue_conflict",
                tables_directory=tables_directory,
            )
        )

        translation_results: dict[str, Any] = {}

        for displacement in DISPLACEMENTS:
            direction_records = []

            for direction in DIRECTIONS:
                condition_name = (
                    f"translation_{displacement}_{direction}"
                )

                transformed_data = (
                    load_intervention_features(
                        cache_name=cache_name,
                        condition_name=condition_name,
                    )
                )

                record = analyze_feature_pair(
                    clean_data=clean_data,
                    transformed_data=transformed_data,
                    model_name=model_name,
                    condition_name=condition_name,
                    tables_directory=tables_directory,
                )

                translation_results[
                    condition_name
                ] = record
                direction_records.append(record)

            translation_results[
                f"displacement_{displacement}_averaged"
            ] = average_stability_records(
                direction_records
            )

        model_results["stability"][
            "translation"
        ] = translation_results

        if not args.skip_umap:
            model_results["umap_figures"]["grayscale"] = (
                create_umap_for_pair(
                    clean_data=clean_data,
                    transformed_data=grayscale_data,
                    model_name=model_name,
                    condition_name="grayscale",
                    figures_directory=figures_directory,
                    seed=seed,
                )
            )

            model_results["umap_figures"][
                "patch_shuffle_4x4"
            ] = create_umap_for_pair(
                clean_data=clean_data,
                transformed_data=patch_data,
                model_name=model_name,
                condition_name="patch_shuffle_4x4",
                figures_directory=figures_directory,
                seed=seed,
            )

            translation_32_right = (
                load_intervention_features(
                    cache_name=cache_name,
                    condition_name=(
                        "translation_32_right"
                    ),
                )
            )

            model_results["umap_figures"][
                "translation_32_right"
            ] = create_umap_for_pair(
                clean_data=clean_data,
                transformed_data=translation_32_right,
                model_name=model_name,
                condition_name="translation_32_right",
                figures_directory=figures_directory,
                seed=seed,
            )

            model_results["umap_figures"][
                "cue_conflict"
            ] = create_umap_for_pair(
                clean_data=clean_cue_data,
                transformed_data=generated_cue_data,
                model_name=model_name,
                condition_name="cue_conflict",
                figures_directory=figures_directory,
                seed=seed,
            )

        all_results["models"][model_name] = model_results

        print(
            "Grayscale stability:",
            model_results["stability"]["grayscale"][
                "mean_cosine_stability"
            ],
        )
        print(
            "Patch stability:",
            model_results["stability"][
                "patch_shuffle_4x4"
            ]["mean_cosine_stability"],
        )
        print(
            "Cue-conflict stability:",
            model_results["stability"]["cue_conflict"][
                "mean_cosine_stability"
            ],
        )

        del backbone
        del clean_data
        del grayscale_data
        del patch_data
        del clean_cue_data
        del generated_cue_data

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(
        "task1/results/representation_results.json"
    )

    save_json(
        data=all_results,
        output_path=output_path,
    )

    print("\nFinished representation analysis.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()