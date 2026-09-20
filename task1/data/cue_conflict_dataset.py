from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.datasets import STL10


def parse_accepted_column(
    values: pd.Series,
) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()

    valid_values = {"true", "false"}

    invalid_values = set(normalized.unique()) - valid_values

    if invalid_values:
        raise ValueError(
            "The accepted column contains invalid values: "
            f"{sorted(invalid_values)}"
        )

    return normalized == "true"


class CueConflictDataset(Dataset):
    def __init__(
        self,
        metadata_path: str | Path,
        image_size: int = 224,
        accepted_only: bool = True,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.metadata = pd.read_csv(self.metadata_path)

        required_columns = {
            "candidate_id",
            "pair_id",
            "direction",
            "shape_class_id",
            "shape_class_name",
            "texture_class_id",
            "texture_class_name",
            "image_path",
            "accepted",
        }

        missing_columns = (
            required_columns - set(self.metadata.columns)
        )

        if missing_columns:
            raise ValueError(
                "Cue-conflict metadata is missing columns: "
                f"{sorted(missing_columns)}"
            )

        accepted_mask = parse_accepted_column(
            self.metadata["accepted"]
        )

        if accepted_only:
            self.metadata = self.metadata.loc[
                accepted_mask
            ].copy()

        self.metadata = self.metadata.reset_index(drop=True)

        if self.metadata.empty:
            raise ValueError(
                "No cue-conflict images were selected."
            )

        self.transform = v2.Compose(
            [
                v2.Resize(
                    size=(image_size, image_size),
                    antialias=True,
                ),
                v2.ToImage(),
                v2.ToDtype(
                    dtype=torch.float32,
                    scale=True,
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, dataset_index: int) -> dict[str, Any]:
        row = self.metadata.iloc[dataset_index]
        image_path = Path(row["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Cue-conflict image not found: {image_path}"
            )

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        candidate_number = int(
            str(row["candidate_id"]).split("_")[-1]
        )

        return {
            "image": self.transform(image),
            "label": torch.tensor(
                int(row["shape_class_id"]),
                dtype=torch.long,
            ),
            "index": torch.tensor(
                candidate_number,
                dtype=torch.long,
            ),
            "identifier": str(row["candidate_id"]),
            "condition": "cue_conflict",
        }


def build_cue_conflict_dataloader(
    metadata_path: str | Path,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    accepted_only: bool = True,
) -> DataLoader:
    dataset = CueConflictDataset(
        metadata_path=metadata_path,
        image_size=image_size,
        accepted_only=accepted_only,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

class CueConflictContentDataset(Dataset):
    def __init__(
        self,
        metadata_path: str | Path,
        data_root: str | Path,
        image_size: int = 224,
    ) -> None:
        generated_dataset = CueConflictDataset(
            metadata_path=metadata_path,
            image_size=image_size,
            accepted_only=True,
        )

        self.metadata = generated_dataset.metadata
        self.transform = generated_dataset.transform

        self.test_dataset = STL10(
            root=data_root,
            split="test",
            download=False,
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self,
        dataset_index: int,
    ) -> dict[str, Any]:
        row = self.metadata.iloc[dataset_index]

        content_index = int(row["content_index"])
        image, dataset_label = self.test_dataset[
            content_index
        ]

        shape_label = int(row["shape_class_id"])

        if int(dataset_label) != shape_label:
            raise ValueError(
                "Content-image label does not match metadata "
                f"for {row['candidate_id']}: "
                f"{dataset_label} != {shape_label}."
            )

        candidate_number = int(
            str(row["candidate_id"]).split("_")[-1]
        )

        return {
            "image": self.transform(image.convert("RGB")),
            "label": torch.tensor(
                shape_label,
                dtype=torch.long,
            ),
            "index": torch.tensor(
                candidate_number,
                dtype=torch.long,
            ),
            "identifier": str(row["candidate_id"]),
            "condition": "cue_conflict_clean_content",
        }


def build_cue_conflict_content_dataloader(
    metadata_path: str | Path,
    data_root: str | Path,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = CueConflictContentDataset(
        metadata_path=metadata_path,
        data_root=data_root,
        image_size=image_size,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )