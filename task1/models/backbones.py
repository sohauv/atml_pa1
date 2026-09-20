from abc import ABC, abstractmethod

import open_clip
import torch
from torch import nn
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class FrozenBackbone(nn.Module, ABC):
    def __init__(
        self,
        feature_dim: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim

        self.register_buffer(
            "normalization_mean",
            torch.tensor(mean).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "normalization_std",
            torch.tensor(std).view(1, 3, 1, 1),
        )

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (
            images - self.normalization_mean
        ) / self.normalization_std

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

        self.eval()

    def train(self, mode: bool = True) -> "FrozenBackbone":
        # The backbone must remain in evaluation mode even while a separate
        # classifier head is being trained.
        super().train(False)
        return self

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ResNet50Backbone(FrozenBackbone):
    def __init__(self) -> None:
        super().__init__(
            feature_dim=2048,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        self.model = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        # Replace the original 1000-class ImageNet classifier with identity.
        # The model will now return its 2048-dimensional representation.
        self.model.fc = nn.Identity()

        self.freeze()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized_images = self.normalize(images)
        return self.model(normalized_images)


class ViTB16Backbone(FrozenBackbone):
    def __init__(self) -> None:
        super().__init__(
            feature_dim=768,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        self.model = vit_b_16(
            weights=ViT_B_16_Weights.IMAGENET1K_V1
        )

        # Torchvision applies the classifier head to the final class token.
        # Replacing it with identity makes forward() return that token.
        self.model.heads = nn.Identity()

        self.freeze()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized_images = self.normalize(images)
        return self.model(normalized_images)


class CLIPViTB32Backbone(FrozenBackbone):
    def __init__(self) -> None:
        super().__init__(
            feature_dim=512,
            mean=CLIP_MEAN,
            std=CLIP_STD,
        )

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="openai",
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.freeze()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized_images = self.normalize(images)

        # normalize=True gives unit-length CLIP image embeddings, as required.
        return self.model.encode_image(
            normalized_images,
            normalize=True,
        )

    @torch.no_grad()
    def encode_text_prompts(
        self,
        class_names: list[str],
        prompt_template: str = "a photo of a {}.",
    ) -> torch.Tensor:
        prompts = [
            prompt_template.format(class_name)
            for class_name in class_names
        ]

        tokens = self.tokenizer(prompts).to(
            self.normalization_mean.device
        )

        return self.model.encode_text(
            tokens,
            normalize=True,
        )

    def get_logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale.exp()


def build_backbone(name: str) -> FrozenBackbone:
    normalized_name = name.lower()

    if normalized_name == "resnet50":
        return ResNet50Backbone()

    if normalized_name == "vit_b_16":
        return ViTB16Backbone()

    if normalized_name in {"clip", "clip_vit_b_32", "vit-b-32"}:
        return CLIPViTB32Backbone()

    raise ValueError(f"Unsupported backbone: {name}")