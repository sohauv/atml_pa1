import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import save_image
from tqdm.auto import tqdm



CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

DECODER_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/"
    "releases/download/v0.0.0/decoder.pth"
)

VGG_URLS = [
    (
        "https://github.com/naoto0804/pytorch-AdaIN/"
        "releases/download/v0.0.0/vgg_normalised.pth"
    ),
    (
        "https://github.com/naoto0804/pytorch-AdaIN/"
        "releases/download/v0.0.0/vgg_normalized.pth"
    ),
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_file(
    urls: list[str],
    output_path: Path,
) -> None:
    if output_path.exists():
        print(f"Using existing weights: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = None

    for url in urls:
        try:
            print(f"Downloading {url}")
            torch.hub.download_url_to_file(
                url=url,
                dst=str(output_path),
                progress=True,
            )
            return
        except Exception as error:
            last_error = error

            if output_path.exists():
                output_path.unlink()

            print(f"Download failed: {error}")

    raise RuntimeError(
        f"Could not download weights to {output_path}"
    ) from last_error


def load_adain_models(
    adain_root: Path,
    weights_directory: Path,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, Any]:
    if not (adain_root / "net.py").exists():
        raise FileNotFoundError(
            "AdaIN submodule is missing. Run:\n"
            "git submodule update --init --recursive"
        )

    sys.path.insert(0, str(adain_root.resolve()))

    import net as adain_networks
    from function import adaptive_instance_normalization

    decoder_path = weights_directory / "decoder.pth"
    vgg_path = weights_directory / "vgg_normalised.pth"

    download_file(
        urls=[DECODER_URL],
        output_path=decoder_path,
    )
    download_file(
        urls=VGG_URLS,
        output_path=vgg_path,
    )

    decoder = adain_networks.decoder
    vgg = adain_networks.vgg

    decoder.load_state_dict(
        torch.load(
            decoder_path,
            map_location="cpu",
            weights_only=True,
        )
    )

    vgg.load_state_dict(
        torch.load(
            vgg_path,
            map_location="cpu",
            weights_only=True,
        )
    )

    # AdaIN uses VGG only through relu4_1.
    vgg = nn.Sequential(
        *list(vgg.children())[:31]
    )

    decoder = decoder.to(device).eval()
    vgg = vgg.to(device).eval()

    for model in [decoder, vgg]:
        for parameter in model.parameters():
            parameter.requires_grad = False

        def style_transfer(
            vgg: nn.Module,
            decoder: nn.Module,
            content: torch.Tensor,
            style: torch.Tensor,
            alpha: float = 1.0,
        ) -> torch.Tensor:
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    "Style strength alpha must be between 0 and 1."
                )

            content_features = vgg(content)
            style_features = vgg(style)

            stylized_features = adaptive_instance_normalization(
                content_features,
                style_features,
            )

            blended_features = (
                alpha * stylized_features
                + (1.0 - alpha) * content_features
            )

            return decoder(blended_features)

    return vgg, decoder, style_transfer


def load_evaluation_indices(
    split_path: Path,
) -> list[int]:
    with split_path.open("r", encoding="utf-8") as file:
        split_data = json.load(file)

    return split_data["evaluation_partition"]["indices"]


def create_candidate_records(
    test_labels: torch.Tensor,
    evaluation_indices: list[int],
    class_pairs: list[list[str]],
    candidates_per_direction: int,
    seed: int,
    style_strength: float,
    output_directory: Path,
) -> list[dict[str, Any]]:
    rng = torch.Generator()
    rng.manual_seed(seed)

    class_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(CLASS_NAMES)
    }

    records = []
    candidate_number = 0

    labels = torch.as_tensor(test_labels)
    evaluation_tensor = torch.tensor(evaluation_indices)

    for pair_number, pair in enumerate(class_pairs):
        class_a, class_b = pair

        directions = [
            (class_a, class_b),
            (class_b, class_a),
        ]

        for shape_class, texture_class in directions:
            shape_class_id = class_to_id[shape_class]
            texture_class_id = class_to_id[texture_class]

            content_pool = evaluation_tensor[
                labels[evaluation_tensor] == shape_class_id
            ]
            style_pool = evaluation_tensor[
                labels[evaluation_tensor] == texture_class_id
            ]

            if len(content_pool) < candidates_per_direction:
                raise ValueError(
                    f"Not enough {shape_class} content images."
                )

            if len(style_pool) < candidates_per_direction:
                raise ValueError(
                    f"Not enough {texture_class} style images."
                )

            content_order = torch.randperm(
                len(content_pool),
                generator=rng,
            )[:candidates_per_direction]

            style_order = torch.randperm(
                len(style_pool),
                generator=rng,
            )[:candidates_per_direction]

            selected_content = content_pool[content_order]
            selected_styles = style_pool[style_order]

            for content_index, style_index in zip(
                selected_content.tolist(),
                selected_styles.tolist(),
            ):
                candidate_id = f"cc_{candidate_number:03d}"

                filename = (
                    f"{candidate_id}_"
                    f"shape-{shape_class}_"
                    f"texture-{texture_class}.png"
                )

                records.append(
                    {
                        "candidate_id": candidate_id,
                        "pair_id": pair_number,
                        "direction": (
                            f"{shape_class}_to_{texture_class}"
                        ),
                        "content_index": content_index,
                        "content_identifier": (
                            f"test_{content_index:05d}"
                        ),
                        "shape_class_id": shape_class_id,
                        "shape_class_name": shape_class,
                        "style_index": style_index,
                        "style_identifier": (
                            f"test_{style_index:05d}"
                        ),
                        "texture_class_id": texture_class_id,
                        "texture_class_name": texture_class,
                        "style_strength": style_strength,
                        "image_path": str(
                            output_directory / filename
                        ),
                        "accepted": "",
                        "rejection_reason": "",
                    }
                )

                candidate_number += 1

    return records


class CueConflictPairDataset(Dataset):
    def __init__(
        self,
        stl10_test: STL10,
        records: list[dict[str, Any]],
        image_size: int,
    ) -> None:
        self.stl10_test = stl10_test
        self.records = records

        self.transform = Compose(
            [
                Resize(
                    (image_size, image_size),
                    antialias=True,
                ),
                ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        record = self.records[index]

        content_image, _ = self.stl10_test[
            record["content_index"]
        ]
        style_image, _ = self.stl10_test[
            record["style_index"]
        ]

        return {
            "candidate_id": record["candidate_id"],
            "content": self.transform(
                content_image.convert("RGB")
            ),
            "style": self.transform(
                style_image.convert("RGB")
            ),
            "output_path": record["image_path"],
        }


@torch.inference_mode()
def generate_conflicts(
    dataloader: DataLoader,
    vgg: nn.Module,
    decoder: nn.Module,
    style_transfer: Any,
    style_strength: float,
    device: torch.device,
) -> None:
    for batch in tqdm(
        dataloader,
        desc="Generating cue conflicts",
    ):
        content_images = batch["content"].to(device)
        style_images = batch["style"].to(device)

        generated_images = style_transfer(
            vgg=vgg,
            decoder=decoder,
            content=content_images,
            style=style_images,
            alpha=style_strength,
        )

        for generated_image, output_path in zip(
            generated_images.cpu(),
            batch["output_path"],
        ):
            output_path = Path(output_path)
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_image(
                generated_image.clamp(0, 1),
                output_path,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AdaIN cue-conflict candidates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("task1/configs/task1.yaml"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets"),
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path(
            "task1/results/stl10_splits.json"
        ),
    )
    parser.add_argument(
        "--adain-root",
        type=Path,
        default=Path(
            "task1/external/pytorch-AdaIN"
        ),
    )
    parser.add_argument(
        "--weights-directory",
        type=Path,
        default=Path(
            "task1/external_weights/adain"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "task1/generated_images/cue_conflicts"
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path(
            "task1/results/cue_conflicts/candidates.csv"
        ),
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    cue_config = config["cue_conflicts"]
    seed = config["seed"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    torch.manual_seed(seed)

    stl10_test = STL10(
        root=args.data_root,
        split="test",
        download=False,
    )

    evaluation_indices = load_evaluation_indices(
        args.split_path
    )

    records = create_candidate_records(
        test_labels=torch.as_tensor(stl10_test.labels),
        evaluation_indices=evaluation_indices,
        class_pairs=cue_config["class_pairs"],
        candidates_per_direction=(
            cue_config["candidates_per_direction"]
        ),
        seed=seed,
        style_strength=cue_config["style_strength"],
        output_directory=args.output_directory,
    )

    expected_candidates = (
        len(cue_config["class_pairs"])
        * 2
        * cue_config["candidates_per_direction"]
    )

    assert len(records) == expected_candidates

    pair_dataset = CueConflictPairDataset(
        stl10_test=stl10_test,
        records=records,
        image_size=cue_config["image_size"],
    )

    dataloader = DataLoader(
        pair_dataset,
        batch_size=cue_config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    vgg, decoder, style_transfer = load_adain_models(
        adain_root=args.adain_root,
        weights_directory=args.weights_directory,
        device=device,
    )

    generate_conflicts(
        dataloader=dataloader,
        vgg=vgg,
        decoder=decoder,
        style_transfer=style_transfer,
        style_strength=cue_config["style_strength"],
        device=device,
    )

    args.metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(records).to_csv(
        args.metadata_path,
        index=False,
    )

    design_path = (
        args.metadata_path.parent
        / "generation_design.json"
    )

    with design_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "method": cue_config["method"],
                "seed": seed,
                "style_strength": (
                    cue_config["style_strength"]
                ),
                "candidates_per_direction": (
                    cue_config[
                        "candidates_per_direction"
                    ]
                ),
                "total_candidates": len(records),
                "minimum_valid_images": (
                    cue_config["minimum_valid_images"]
                ),
                "class_pairs": cue_config["class_pairs"],
                "rejection_rule": (
                    cue_config["rejection_rule"]
                ),
                "implementation": (
                    "naoto0804/pytorch-AdaIN"
                ),
            },
            file,
            indent=2,
        )

    print(f"Generated candidates: {len(records)}")
    print(f"Metadata saved to: {args.metadata_path}")
    print(f"Design saved to: {design_path}")


if __name__ == "__main__":
    main()