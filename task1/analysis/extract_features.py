from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from task1.models.backbones import FrozenBackbone


@torch.inference_mode()
def extract_features(
    backbone: FrozenBackbone,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
) -> dict[str, Any]:
    backbone = backbone.to(device)
    backbone.eval()

    all_features = []
    all_labels = []
    all_indices = []
    all_identifiers = []

    progress_bar = tqdm(
        dataloader,
        desc=f"Extracting {split_name} features",
    )

    for batch in progress_bar:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        features = backbone(images)

        all_features.append(features.float().cpu())
        all_labels.append(batch["label"].cpu())
        all_indices.append(batch["index"].cpu())
        all_identifiers.extend(batch["identifier"])

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    indices = torch.cat(all_indices, dim=0)

    assert features.ndim == 2
    assert features.shape[0] == labels.shape[0]
    assert features.shape[0] == indices.shape[0]
    assert features.shape[0] == len(all_identifiers)
    assert features.shape[1] == backbone.feature_dim

    return {
        "features": features,
        "labels": labels,
        "indices": indices,
        "identifiers": all_identifiers,
    }


def save_feature_cache(
    feature_data: dict[str, Any],
    output_path: str | Path,
    model_name: str,
    split_name: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_data = {
        "model_name": model_name,
        "split_name": split_name,
        "feature_dim": feature_data["features"].shape[1],
        "features": feature_data["features"],
        "labels": feature_data["labels"],
        "indices": feature_data["indices"],
        "identifiers": feature_data["identifiers"],
    }

    torch.save(cache_data, output_path)


def load_feature_cache(
    cache_path: str | Path,
) -> dict[str, Any]:
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature cache does not exist: {cache_path}"
        )

    return torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
    )


def extract_and_cache_splits(
    model_name: str,
    backbone: FrozenBackbone,
    dataloaders: dict[str, DataLoader],
    device: torch.device,
    cache_directory: str | Path = "task1/cache",
    overwrite: bool = False,
) -> dict[str, Path]:
    cache_directory = Path(cache_directory)
    cache_paths = {}

    backbone = backbone.to(device)
    backbone.eval()

    for split_name, dataloader in dataloaders.items():
        cache_path = (
            cache_directory
            / model_name
            / f"{split_name}.pt"
        )
        cache_paths[split_name] = cache_path

        if cache_path.exists() and not overwrite:
            print(f"Using existing cache: {cache_path}")
            continue

        feature_data = extract_features(
            backbone=backbone,
            dataloader=dataloader,
            device=device,
            split_name=split_name,
        )

        save_feature_cache(
            feature_data=feature_data,
            output_path=cache_path,
            model_name=model_name,
            split_name=split_name,
        )

        print(
            f"Saved {feature_data['features'].shape} "
            f"features to {cache_path}"
        )

    return cache_paths