from typing import Literal

import torch
import torchvision.transforms.functional as vision_functional
from torch.nn import functional as torch_functional


TranslationDirection = Literal["left", "right", "up", "down"]


def convert_to_grayscale(
    image: torch.Tensor,
) -> torch.Tensor:
    return vision_functional.rgb_to_grayscale(
        image,
        num_output_channels=3,
    )


def rotate_hue(
    image: torch.Tensor,
    hue_factor: float = 0.5,
) -> torch.Tensor:
    if not -0.5 <= hue_factor <= 0.5:
        raise ValueError(
            "hue_factor must be between -0.5 and 0.5."
        )

    return vision_functional.adjust_hue(
        image,
        hue_factor=hue_factor,
    )


def translate_image(
    image: torch.Tensor,
    displacement: int,
    direction: TranslationDirection,
) -> torch.Tensor:
    if displacement < 0:
        raise ValueError("Displacement cannot be negative.")

    if direction not in {"left", "right", "up", "down"}:
        raise ValueError(
            f"Unsupported translation direction: {direction}"
        )

    if displacement == 0:
        return image.clone()

    _, height, width = image.shape

    padded_image = torch_functional.pad(
        image,
        pad=(
            displacement,
            displacement,
            displacement,
            displacement,
        ),
        mode="reflect",
    )

    crop_top = displacement
    crop_left = displacement

    if direction == "right":
        crop_left = 0
    elif direction == "left":
        crop_left = 2 * displacement
    elif direction == "down":
        crop_top = 0
    elif direction == "up":
        crop_top = 2 * displacement

    translated_image = padded_image[
        :,
        crop_top : crop_top + height,
        crop_left : crop_left + width,
    ]

    return translated_image


def make_patch_permutation(
    number_of_patches: int,
    seed: int,
    image_index: int,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed + image_index)

    permutation = torch.randperm(
        number_of_patches,
        generator=generator,
    )

    identity = torch.arange(number_of_patches)

    # The assignment requires a non-identity permutation.
    if torch.equal(permutation, identity):
        permutation = torch.roll(permutation, shifts=1)

    return permutation


def shuffle_patches(
    image: torch.Tensor,
    grid_size: int = 4,
    seed: int = 6304,
    image_index: int = 0,
) -> torch.Tensor:
    channels, height, width = image.shape

    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError(
            "Image height and width must be divisible by grid_size."
        )

    patch_height = height // grid_size
    patch_width = width // grid_size
    number_of_patches = grid_size * grid_size

    patches = (
        image.unfold(
            dimension=1,
            size=patch_height,
            step=patch_height,
        )
        .unfold(
            dimension=2,
            size=patch_width,
            step=patch_width,
        )
        .permute(1, 2, 0, 3, 4)
        .reshape(
            number_of_patches,
            channels,
            patch_height,
            patch_width,
        )
    )

    permutation = make_patch_permutation(
        number_of_patches=number_of_patches,
        seed=seed,
        image_index=image_index,
    )

    shuffled_patches = patches[permutation]

    shuffled_image = (
        shuffled_patches.reshape(
            grid_size,
            grid_size,
            channels,
            patch_height,
            patch_width,
        )
        .permute(2, 0, 3, 1, 4)
        .reshape(channels, height, width)
    )

    return shuffled_image