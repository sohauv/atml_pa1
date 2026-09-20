from typing import Any

from torch.utils.data import DataLoader, Dataset

from task1.data.transforms import (
    convert_to_grayscale,
    rotate_hue,
    shuffle_patches,
    translate_image,
)


class InterventionDataset(Dataset):
    def __init__(
        self,
        clean_dataset: Dataset,
        condition: str,
        seed: int = 6304,
        hue_factor: float = 0.5,
        grid_size: int = 4,
        displacement: int = 0,
        direction: str = "right",
    ) -> None:
        self.clean_dataset = clean_dataset
        self.condition = condition
        self.seed = seed
        self.hue_factor = hue_factor
        self.grid_size = grid_size
        self.displacement = displacement
        self.direction = direction

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, subset_index: int) -> dict[str, Any]:
        example = self.clean_dataset[subset_index]

        clean_image = example["image"]
        original_index = example["index"].item()

        if self.condition == "grayscale":
            transformed_image = convert_to_grayscale(clean_image)

        elif self.condition == "hue_rotation":
            transformed_image = rotate_hue(
                clean_image,
                hue_factor=self.hue_factor,
            )

        elif self.condition == "patch_shuffle":
            transformed_image = shuffle_patches(
                clean_image,
                grid_size=self.grid_size,
                seed=self.seed,
                image_index=original_index,
            )

        elif self.condition == "translation":
            transformed_image = translate_image(
                clean_image,
                displacement=self.displacement,
                direction=self.direction,
            )

        else:
            raise ValueError(
                f"Unsupported intervention: {self.condition}"
            )

        return {
            "image": transformed_image,
            "label": example["label"],
            "index": example["index"],
            "identifier": example["identifier"],
            "condition": self.get_condition_name(),
        }

    def get_condition_name(self) -> str:
        if self.condition == "translation":
            return (
                f"translation_{self.displacement}_"
                f"{self.direction}"
            )

        if self.condition == "hue_rotation":
            return f"hue_rotation_{self.hue_factor}"

        if self.condition == "patch_shuffle":
            return f"patch_shuffle_{self.grid_size}x{self.grid_size}"

        return self.condition


def build_intervention_dataloader(
    clean_dataset: Dataset,
    condition: str,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 6304,
    hue_factor: float = 0.5,
    grid_size: int = 4,
    displacement: int = 0,
    direction: str = "right",
) -> DataLoader:
    intervention_dataset = InterventionDataset(
        clean_dataset=clean_dataset,
        condition=condition,
        seed=seed,
        hue_factor=hue_factor,
        grid_size=grid_size,
        displacement=displacement,
        direction=direction,
    )

    return DataLoader(
        dataset=intervention_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )