import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
from torchvision.transforms import Compose, Resize, ToTensor


class STL10IndexedSubset(Dataset):
    def __init__(
        self,
        dataset: STL10,
        indices: list[int],
        split_name: str,
        image_size: int = 224,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.split_name = split_name

        self.transform = Compose(
            [
                Resize(
                    size=(image_size, image_size),
                    antialias=True,
                ),
                ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, subset_index: int) -> dict[str, Any]:
        original_index = self.indices[subset_index]
        image, label = self.dataset[original_index]

        image = image.convert("RGB")
        image = self.transform(image)

        identifier = f"{self.split_name}_{original_index:05d}"

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(original_index, dtype=torch.long),
            "identifier": identifier,
        }


def load_split_indices(
    split_path: str | Path,
) -> dict[str, list[int]]:
    split_path = Path(split_path)

    with split_path.open("r", encoding="utf-8") as file:
        split_data = json.load(file)

    return {
        "train": split_data["train_partition"]["indices"],
        "validation": split_data["validation_partition"]["indices"],
        "evaluation": split_data["evaluation_partition"]["indices"],
    }


def build_task1_datasets(
    data_root: str | Path = "datasets",
    split_path: str | Path = "task1/results/stl10_splits.json",
    image_size: int = 224,
    download: bool = True,
) -> dict[str, STL10IndexedSubset]:
    split_indices = load_split_indices(split_path)

    official_train_dataset = STL10(
        root=data_root,
        split="train",
        download=download,
    )

    official_test_dataset = STL10(
        root=data_root,
        split="test",
        download=download,
    )

    datasets = {
        "train": STL10IndexedSubset(
            dataset=official_train_dataset,
            indices=split_indices["train"],
            split_name="train",
            image_size=image_size,
        ),
        "validation": STL10IndexedSubset(
            dataset=official_train_dataset,
            indices=split_indices["validation"],
            split_name="train",
            image_size=image_size,
        ),
        "evaluation": STL10IndexedSubset(
            dataset=official_test_dataset,
            indices=split_indices["evaluation"],
            split_name="test",
            image_size=image_size,
        ),
    }

    return datasets


def build_task1_dataloaders(
    datasets: dict[str, STL10IndexedSubset],
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    dataloaders = {}

    for split_name, dataset in datasets.items():
        dataloaders[split_name] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    return dataloaders