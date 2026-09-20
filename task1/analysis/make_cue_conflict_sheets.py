from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torchvision.datasets import STL10
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create contact sheets for manual cue-conflict inspection."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("task1/results/cue_conflicts/candidates.csv"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("task1/data/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "task1/results/cue_conflicts/contact_sheets"
        ),
    )
    parser.add_argument(
        "--rows-per-sheet",
        type=int,
        default=10,
    )
    return parser.parse_args()


def load_generated_image(image_path: str) -> Image.Image:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Generated image does not exist: {path}"
        )

    return Image.open(path).convert("RGB")


def create_sheet(
    rows: pd.DataFrame,
    test_dataset: STL10,
    output_path: Path,
) -> None:
    number_of_rows = len(rows)

    figure, axes = plt.subplots(
        number_of_rows,
        3,
        figsize=(9, 2.8 * number_of_rows),
        squeeze=False,
    )

    column_titles = [
        "Shape / content",
        "Texture / style",
        "Generated cue conflict",
    ]

    for column_index, title in enumerate(column_titles):
        axes[0, column_index].set_title(
            title,
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

    for row_position, (_, candidate) in enumerate(rows.iterrows()):
        content_image, _ = test_dataset[
            int(candidate["content_index"])
        ]
        style_image, _ = test_dataset[
            int(candidate["style_index"])
        ]
        generated_image = load_generated_image(
            candidate["image_path"]
        )

        images = [
            content_image,
            style_image,
            generated_image,
        ]

        labels = [
            (
                f'{candidate["content_identifier"]}\n'
                f'Shape: {candidate["shape_class_name"]}'
            ),
            (
                f'{candidate["style_identifier"]}\n'
                f'Texture: {candidate["texture_class_name"]}'
            ),
            (
                f'{candidate["candidate_id"]}\n'
                f'{candidate["direction"]}'
            ),
        ]

        for column_position, (image, label) in enumerate(
            zip(images, labels)
        ):
            axis = axes[row_position, column_position]
            axis.imshow(image)
            axis.set_xlabel(label, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])

            for spine in axis.spines.values():
                spine.set_visible(False)

    first_candidate = rows.iloc[0]["candidate_id"]
    last_candidate = rows.iloc[-1]["candidate_id"]

    figure.suptitle(
        (
            "Cue-conflict candidate inspection\n"
            f"{first_candidate} to {last_candidate}"
        ),
        fontsize=14,
        fontweight="bold",
        y=1.002,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()

    if args.rows_per_sheet <= 0:
        raise ValueError("--rows-per-sheet must be positive.")

    metadata = pd.read_csv(args.metadata)

    required_columns = {
        "candidate_id",
        "direction",
        "content_index",
        "content_identifier",
        "shape_class_name",
        "style_index",
        "style_identifier",
        "texture_class_name",
        "image_path",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = STL10(
        root=args.data_root,
        split="test",
        download=True,
    )

    number_of_sheets = math.ceil(
        len(metadata) / args.rows_per_sheet
    )

    for sheet_index in tqdm(
        range(number_of_sheets),
        desc="Creating contact sheets",
    ):
        start = sheet_index * args.rows_per_sheet
        stop = start + args.rows_per_sheet
        rows = metadata.iloc[start:stop]

        output_path = (
            args.output_dir
            / f"cue_conflicts_{sheet_index + 1:02d}.png"
        )

        create_sheet(
            rows=rows,
            test_dataset=test_dataset,
            output_path=output_path,
        )

    print(f"Candidates reviewed: {len(metadata)}")
    print(f"Contact sheets created: {number_of_sheets}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()