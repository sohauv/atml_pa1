"""PACS dataset discovery and loading shared by Tasks 2 and 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset


SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
TARGET_DOMAIN = "sketch"
ALL_DOMAINS = SOURCE_DOMAINS + (TARGET_DOMAIN,)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class PACSRecord:
    """One image and its immutable metadata."""

    path: str
    label: int
    class_name: str
    domain: str


def resolve_pacs_root(
    root: str | Path,
    required_domains: Sequence[str] = ALL_DOMAINS,
) -> Path:
    """Resolve common PACS archive layouts to the directory containing domains."""

    root = Path(root).expanduser().resolve()
    candidates = (
        root,
        root / "kfold",
        root / "PACS",
        root / "PACS" / "kfold",
        root / "pacs",
        root / "pacs" / "kfold",
    )
    for candidate in candidates:
        if all((candidate / domain).is_dir() for domain in required_domains):
            return candidate
    expected = ", ".join(required_domains)
    raise FileNotFoundError(
        f"Could not find PACS domain folders ({expected}) below {root}."
    )


def discover_class_names(
    pacs_root: str | Path,
    domains: Sequence[str] = ALL_DOMAINS,
) -> list[str]:
    """Return the shared class names after validating every PACS domain."""

    pacs_root = resolve_pacs_root(pacs_root, required_domains=domains)
    domain_classes = {
        domain: sorted(path.name for path in (pacs_root / domain).iterdir() if path.is_dir())
        for domain in domains
    }
    reference_domain = domains[0]
    reference = domain_classes[reference_domain]
    if not reference:
        raise ValueError(f"No class directories found in {pacs_root / reference_domain}.")
    for domain, class_names in domain_classes.items():
        if class_names != reference:
            raise ValueError(
                f"Class folders for {domain} do not match {reference_domain}: "
                f"{class_names} != {reference}"
            )
    return reference


def discover_domain_records(
    pacs_root: str | Path,
    domain: str,
    class_to_index: dict[str, int] | None = None,
) -> list[PACSRecord]:
    """Discover images for one domain in deterministic path order."""

    if domain not in ALL_DOMAINS:
        raise ValueError(f"Unknown PACS domain {domain!r}; expected one of {ALL_DOMAINS}.")
    pacs_root = resolve_pacs_root(pacs_root, required_domains=(domain,))
    class_names = discover_class_names(pacs_root, domains=(domain,))
    if class_to_index is None:
        class_to_index = {name: index for index, name in enumerate(class_names)}
    elif set(class_to_index) != set(class_names):
        raise ValueError("Provided class mapping does not match the PACS class folders.")

    records: list[PACSRecord] = []
    for class_name in class_names:
        class_directory = pacs_root / domain / class_name
        for image_path in sorted(class_directory.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    PACSRecord(
                        path=image_path.relative_to(pacs_root).as_posix(),
                        label=class_to_index[class_name],
                        class_name=class_name,
                        domain=domain,
                    )
                )
    if not records:
        raise ValueError(f"No images found for PACS domain {domain!r}.")
    return records


class PACSDataset(Dataset):
    """Dataset backed by an explicit list of PACS records."""

    def __init__(
        self,
        pacs_root: str | Path,
        records: Sequence[PACSRecord | dict],
        transform: Callable | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.records = [
            record if isinstance(record, PACSRecord) else PACSRecord(**record)
            for record in records
        ]
        record_domains = tuple(sorted({record.domain for record in self.records}))
        if not record_domains:
            raise ValueError("PACSDataset requires at least one record.")
        self.pacs_root = resolve_pacs_root(
            pacs_root,
            required_domains=record_domains,
        )
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.pacs_root / record.path
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if not self.return_metadata:
            return image, record.label
        return {
            "image": image,
            "label": record.label,
            "class_name": record.class_name,
            "domain": record.domain,
            "path": record.path,
            "index": index,
        }


def records_from_dicts(rows: Sequence[dict]) -> list[PACSRecord]:
    """Convert JSON-compatible rows back into immutable records."""

    return [PACSRecord(**row) for row in rows]
