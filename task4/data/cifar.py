"""Leakage-safe CIFAR-10 splits and fixed CIFAR-100 unknown groups."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms


NEAR_UNKNOWN_CLASSES = ("bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel")
FAR_UNKNOWN_CLASSES = ("bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class ArrayDataset(Dataset):
    def __init__(self, data, targets, transform, class_names, indices=None, domain="known"):
        self.data = data
        self.targets = np.asarray(targets)
        self.indices = np.arange(len(self.targets)) if indices is None else np.asarray(indices)
        self.transform = transform
        self.class_names = class_names
        self.domain = domain

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = int(self.indices[index])
        image = Image.fromarray(self.data[original_index])
        label = int(self.targets[original_index])
        return {
            "image": self.transform(image), "label": label,
            "class_name": self.class_names[label], "index": original_index,
            "domain": self.domain,
        }


def training_transform(randaugment=False, num_ops=2, magnitude=9):
    operations = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if randaugment:
        operations.append(transforms.RandAugment(num_ops=num_ops, magnitude=magnitude))
    operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(operations)


def evaluation_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def create_split(root, output_path, seed=6304, validation_fraction=0.1):
    dataset = datasets.CIFAR10(root=root, train=True, download=True)
    all_indices = np.arange(len(dataset.targets))
    train_indices, validation_indices = train_test_split(
        all_indices, test_size=validation_fraction, random_state=seed,
        stratify=np.asarray(dataset.targets),
    )
    manifest = {
        "seed": seed, "validation_fraction": validation_fraction,
        "class_names": dataset.classes,
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
    }
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_cifar10(root, split_path, randaugment=False, create=False, seed=6304):
    path = Path(split_path)
    manifest = create_split(root, path, seed=seed) if create and not path.exists() else json.loads(path.read_text(encoding="utf-8"))
    train_base = datasets.CIFAR10(root=root, train=True, download=True)
    test_base = datasets.CIFAR10(root=root, train=False, download=True)
    train = ArrayDataset(train_base.data, train_base.targets, training_transform(randaugment), train_base.classes, manifest["train_indices"], "cifar10_train")
    validation = ArrayDataset(train_base.data, train_base.targets, evaluation_transform(), train_base.classes, manifest["validation_indices"], "cifar10_validation")
    test = ArrayDataset(test_base.data, test_base.targets, evaluation_transform(), test_base.classes, domain="cifar10_test")
    return train, validation, test, manifest


def build_cifar10_evaluation_sets(root, split_path):
    """Build unaugmented train/validation/test views from a fixed split."""

    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split manifest {path}. Run Task 4 training first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    train_base = datasets.CIFAR10(root=root, train=True, download=True)
    test_base = datasets.CIFAR10(root=root, train=False, download=True)
    transform = evaluation_transform()
    train = ArrayDataset(
        train_base.data,
        train_base.targets,
        transform,
        train_base.classes,
        manifest["train_indices"],
        "cifar10_train_unaugmented",
    )
    validation = ArrayDataset(
        train_base.data,
        train_base.targets,
        transform,
        train_base.classes,
        manifest["validation_indices"],
        "cifar10_validation",
    )
    test = ArrayDataset(
        test_base.data,
        test_base.targets,
        transform,
        test_base.classes,
        domain="cifar10_test",
    )
    return train, validation, test, manifest


def build_unknowns(root):
    base = datasets.CIFAR100(root=root, train=False, download=True)
    mapping = {name: index for index, name in enumerate(base.classes)}
    near_ids = {mapping[name] for name in NEAR_UNKNOWN_CLASSES}
    far_ids = {mapping[name] for name in FAR_UNKNOWN_CLASSES}
    targets = np.asarray(base.targets)
    near_indices = np.flatnonzero(np.isin(targets, list(near_ids)))
    far_indices = np.flatnonzero(np.isin(targets, list(far_ids)))
    near = ArrayDataset(base.data, base.targets, evaluation_transform(), base.classes, near_indices, "near_unknown")
    far = ArrayDataset(base.data, base.targets, evaluation_transform(), base.classes, far_indices, "far_unknown")
    return near, far
