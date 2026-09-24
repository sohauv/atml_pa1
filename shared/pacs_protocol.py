"""Fixed PACS source protocol shared by Tasks 2 and 3."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from shared.pacs import (
    SOURCE_DOMAINS,
    TARGET_DOMAIN,
    PACSDataset,
    PACSRecord,
    discover_class_names,
    discover_domain_records,
    records_from_dicts,
    resolve_pacs_root,
)


DEFAULT_SEED = 6304
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_SPLIT_PATH = Path("shared/splits/pacs_sketch_seed6304.json")


def training_transform():
    """Assignment-specified PACS training preprocessing."""

    normalization = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization.mean, std=normalization.std),
        ]
    )


def evaluation_transform():
    """Assignment-specified PACS validation and evaluation preprocessing."""

    normalization = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization.mean, std=normalization.std),
        ]
    )


def _record_to_dict(record: PACSRecord) -> dict:
    return {
        "path": record.path,
        "label": record.label,
        "class_name": record.class_name,
        "domain": record.domain,
    }


def _class_counts(records: list[PACSRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.class_name for record in records).items()))


def create_source_split_manifest(
    pacs_root: str | Path,
    output_path: str | Path = DEFAULT_SPLIT_PATH,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    overwrite: bool = False,
) -> dict:
    """Create source-only stratified splits without loading the Sketch directory."""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Split manifest already exists at {output_path}. Load it instead, or "
            "pass overwrite=True intentionally."
        )

    pacs_root = resolve_pacs_root(pacs_root, required_domains=SOURCE_DOMAINS)
    class_names = discover_class_names(pacs_root, domains=SOURCE_DOMAINS)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    domains: dict[str, dict] = {}

    for domain in SOURCE_DOMAINS:
        records = discover_domain_records(pacs_root, domain, class_to_index)
        indices = list(range(len(records)))
        labels = [record.label for record in records]
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        train_records = [records[index] for index in sorted(train_indices)]
        validation_records = [records[index] for index in sorted(validation_indices)]
        domains[domain] = {
            "train": [_record_to_dict(record) for record in train_records],
            "validation": [
                _record_to_dict(record) for record in validation_records
            ],
            "counts": {
                "total": len(records),
                "train": len(train_records),
                "validation": len(validation_records),
                "train_by_class": _class_counts(train_records),
                "validation_by_class": _class_counts(validation_records),
            },
        }

    manifest = {
        "dataset": "PACS",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "source_domains": list(SOURCE_DOMAINS),
        "held_out_target_domain": TARGET_DOMAIN,
        "class_names": class_names,
        "class_to_index": class_to_index,
        "domains": domains,
        "target_records_included": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def load_source_split_manifest(
    split_path: str | Path = DEFAULT_SPLIT_PATH,
) -> dict:
    """Load and validate the fixed source split manifest."""

    split_path = Path(split_path)
    with split_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest["source_domains"] != list(SOURCE_DOMAINS):
        raise ValueError("Unexpected source domains in PACS split manifest.")
    if manifest["held_out_target_domain"] != TARGET_DOMAIN:
        raise ValueError("PACS split manifest does not hold out Sketch.")
    if manifest.get("target_records_included", True):
        raise ValueError("Source split manifest must not contain target records.")
    return manifest


def build_source_datasets(
    pacs_root: str | Path,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
) -> tuple[dict[str, PACSDataset], dict[str, PACSDataset], dict]:
    """Build source training and validation datasets from the saved manifest."""

    manifest = load_source_split_manifest(split_path)
    train_datasets = {}
    validation_datasets = {}
    for domain in SOURCE_DOMAINS:
        domain_manifest = manifest["domains"][domain]
        train_datasets[domain] = PACSDataset(
            pacs_root,
            records_from_dicts(domain_manifest["train"]),
            transform=training_transform(),
        )
        validation_datasets[domain] = PACSDataset(
            pacs_root,
            records_from_dicts(domain_manifest["validation"]),
            transform=evaluation_transform(),
            return_metadata=True,
        )
    return train_datasets, validation_datasets, manifest


def build_task2_target_dataset(
    pacs_root: str | Path,
    class_to_index: dict[str, int],
    training: bool,
    return_metadata: bool = True,
) -> PACSDataset:
    """Build Sketch for Task 2 without exposing labels during adaptation."""

    target_records = discover_domain_records(
        pacs_root,
        TARGET_DOMAIN,
        class_to_index=class_to_index,
    )
    transform = training_transform() if training else evaluation_transform()
    return PACSDataset(
        pacs_root,
        target_records,
        transform=transform,
        return_metadata=return_metadata,
    )
