"""Common Task 2 training pipeline for Source-only, DAN, DANN, and CDAN."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from shared.pacs import SOURCE_DOMAINS
from shared.pacs_protocol import (
    build_source_datasets,
    build_task2_target_dataset,
    create_source_split_manifest,
)
from task2.evaluation.metrics import evaluate_classifier, summarize_source_validation
from task2.methods.cdan import cdan_objective
from task2.methods.dan import dan_objective
from task2.methods.dann import dann_objective
from task2.methods.source_only import source_only_objective
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import (
    DomainDiscriminator,
    gradient_reversal_schedule,
)


METHOD_CONFIGS = {
    "source_only": Path("task2/configs/source_only.yaml"),
    "dan": Path("task2/configs/dan.yaml"),
    "dann": Path("task2/configs/dann.yaml"),
    "cdan": Path("task2/configs/cdan.yaml"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_config(method: str) -> dict:
    base = load_yaml("task2/configs/base.yaml")
    method_config = load_yaml(METHOD_CONFIGS[method])
    return deep_merge(base, method_config)


def next_cycled(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def build_loaders(config: dict, data_root: str | Path, create_split: bool):
    split_path = Path(config["data"]["split_manifest"])
    if not split_path.exists():
        if not create_split:
            raise FileNotFoundError(
                f"Missing {split_path}. Re-run once with --create-split."
            )
        create_source_split_manifest(
            data_root,
            output_path=split_path,
            seed=config["experiment"]["seed"],
            validation_fraction=config["data"]["validation_fraction"],
        )

    train_datasets, validation_datasets, manifest = build_source_datasets(
        data_root,
        split_path,
    )
    source_loaders = {
        domain: DataLoader(
            train_datasets[domain],
            batch_size=config["data"]["source_batch_size_per_domain"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
        for domain in SOURCE_DOMAINS
    }
    validation_loaders = {
        domain: DataLoader(
            validation_datasets[domain],
            batch_size=64,
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )
        for domain in SOURCE_DOMAINS
    }

    target_loader = None
    if config["method"]["uses_unlabeled_target"]:
        target_dataset = build_task2_target_dataset(
            data_root,
            manifest["class_to_index"],
            training=True,
            return_metadata=False,
        )
        target_loader = DataLoader(
            target_dataset,
            batch_size=config["data"]["target_batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
    return source_loaders, validation_loaders, target_loader, manifest


def build_models(config: dict, device: torch.device):
    backbone = ResNet18Backbone(
        pretrained=True,
        freeze_batchnorm_statistics=config["model"][
            "freeze_batchnorm_running_statistics"
        ],
    ).to(device)
    classifier = ClassifierHead(
        config["model"]["feature_dimension"],
        config["model"]["number_of_classes"],
    ).to(device)
    discriminator = None
    method = config["method"]["name"]
    if method in {"dann", "cdan"}:
        input_dimension = config["model"]["feature_dimension"]
        if method == "cdan":
            input_dimension *= config["model"]["number_of_classes"]
        discriminator = DomainDiscriminator(
            input_dimension=input_dimension,
            hidden_dimension=config["method"]["discriminator_hidden_dimension"],
            dropout=config["method"]["discriminator_dropout"],
        ).to(device)
    return backbone, classifier, discriminator


def build_optimizer(config: dict, backbone, classifier, discriminator):
    parameters = list(backbone.parameters()) + list(classifier.parameters())
    if discriminator is not None:
        parameters += list(discriminator.parameters())
    return torch.optim.AdamW(
        parameters,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )


def compute_objective(
    config: dict,
    backbone,
    classifier,
    discriminator,
    source_images,
    source_labels,
    target_images,
    progress: float,
):
    method = config["method"]["name"]
    if method == "source_only":
        return source_only_objective(
            backbone,
            classifier,
            source_images,
            source_labels,
        )
    if method == "dan":
        return dan_objective(
            backbone,
            classifier,
            source_images,
            source_labels,
            target_images,
            mmd_weight=config["method"]["mmd_loss_weight"],
            bandwidth_multipliers=config["method"][
                "kernel_bandwidth_multipliers"
            ],
        )

    reversal_strength = gradient_reversal_schedule(
        progress,
        maximum_strength=config["method"]["gradient_reversal_maximum"],
        gamma=config["method"]["gradient_reversal_gamma"],
    )
    if method == "dann":
        output = dann_objective(
            backbone,
            classifier,
            discriminator,
            source_images,
            source_labels,
            target_images,
            reversal_strength,
            domain_loss_weight=config["method"]["domain_loss_weight"],
        )
    elif method == "cdan":
        output = cdan_objective(
            backbone,
            classifier,
            discriminator,
            source_images,
            source_labels,
            target_images,
            reversal_strength,
            domain_loss_weight=config["method"]["domain_loss_weight"],
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    output["gradient_reversal_strength"] = source_images.new_tensor(
        reversal_strength
    )
    return output


def average_running_metrics(sums: dict[str, float], steps: int) -> dict[str, float]:
    return {key: value / steps for key, value in sums.items()}


def train_one_epoch(
    config: dict,
    backbone,
    classifier,
    discriminator,
    optimizer,
    scaler,
    source_loaders,
    target_loader,
    epoch: int,
    device: torch.device,
):
    backbone.train()
    classifier.train()
    if discriminator is not None:
        discriminator.train()

    source_iterators = {domain: iter(loader) for domain, loader in source_loaders.items()}
    target_iterator = iter(target_loader) if target_loader is not None else None
    steps_per_epoch = max(len(loader) for loader in source_loaders.values())
    maximum_epochs = config["training"]["maximum_epochs"]
    total_planned_steps = maximum_epochs * steps_per_epoch
    running: dict[str, float] = {}

    for step in range(steps_per_epoch):
        source_images_parts = []
        source_label_parts = []
        for domain in SOURCE_DOMAINS:
            batch, source_iterators[domain] = next_cycled(
                source_loaders[domain],
                source_iterators[domain],
            )
            images, labels = batch
            source_images_parts.append(images)
            source_label_parts.append(labels)
        source_images = torch.cat(source_images_parts, dim=0).to(
            device,
            non_blocking=True,
        )
        source_labels = torch.cat(source_label_parts, dim=0).to(
            device,
            non_blocking=True,
        )

        target_images = None
        if target_loader is not None:
            target_batch, target_iterator = next_cycled(target_loader, target_iterator)
            target_images = target_batch[0].to(device, non_blocking=True)

        global_step = epoch * steps_per_epoch + step
        progress = global_step / max(total_planned_steps - 1, 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=scaler.is_enabled(),
        ):
            output = compute_objective(
                config,
                backbone,
                classifier,
                discriminator,
                source_images,
                source_labels,
                target_images,
                progress,
            )
        scaler.scale(output["total_loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        tracked_keys = (
            "total_loss",
            "classification_loss",
            "alignment_loss",
            "domain_loss",
            "domain_accuracy",
            "mmd_bandwidth",
            "gradient_reversal_strength",
        )
        for key in tracked_keys:
            if key in output:
                running[key] = running.get(key, 0.0) + float(
                    output[key].detach().item()
                )
    return average_running_metrics(running, steps_per_epoch)


def validate_sources(
    backbone,
    classifier,
    validation_loaders,
    class_names,
    device,
):
    domain_metrics = {}
    for domain in SOURCE_DOMAINS:
        metrics, _ = evaluate_classifier(
            backbone,
            classifier,
            validation_loaders[domain],
            class_names,
            device,
        )
        domain_metrics[domain] = metrics
    return domain_metrics, summarize_source_validation(domain_metrics)


def save_checkpoint(
    path: Path,
    epoch: int,
    config: dict,
    backbone,
    classifier,
    discriminator,
    optimizer,
    validation_metrics,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "config": config,
        "backbone_state_dict": backbone.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "source_validation": validation_metrics,
    }
    if discriminator is not None:
        checkpoint["discriminator_state_dict"] = discriminator.state_dict()
    torch.save(checkpoint, path)


def write_history(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def run_training(args) -> None:
    config = load_config(args.method)
    config["data"]["root"] = args.data_root
    if args.mmd_weight is not None:
        if args.method != "dan":
            raise ValueError("--mmd-weight is only valid for DAN.")
        config["method"]["mmd_loss_weight"] = args.mmd_weight

    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    source_loaders, validation_loaders, target_loader, manifest = build_loaders(
        config,
        args.data_root,
        args.create_split,
    )
    print(
        "Source train sizes:",
        {domain: len(loader.dataset) for domain, loader in source_loaders.items()},
    )
    print(
        "Source validation sizes:",
        {domain: len(loader.dataset) for domain, loader in validation_loaders.items()},
    )
    if target_loader is not None:
        print(f"Unlabeled target adaptation size: {len(target_loader.dataset)}")

    backbone, classifier, discriminator = build_models(config, device)
    optimizer = build_optimizer(config, backbone, classifier, discriminator)
    amp_enabled = bool(
        config["training"]["use_automatic_mixed_precision"]
        and device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_name = args.run_name or args.method
    checkpoint_path = Path(config["experiment"]["checkpoint_directory"]) / (
        f"{run_name}_best.pt"
    )
    history_path = Path(config["experiment"]["output_directory"]) / "training" / (
        f"{run_name}_history.csv"
    )
    config_path = Path(config["experiment"]["output_directory"]) / "configs" / (
        f"{run_name}.json"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    best_macro_f1 = -float("inf")
    epochs_without_improvement = 0
    history = []
    maximum_epochs = config["training"]["maximum_epochs"]
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(maximum_epochs):
        train_metrics = train_one_epoch(
            config,
            backbone,
            classifier,
            discriminator,
            optimizer,
            scaler,
            source_loaders,
            target_loader,
            epoch,
            device,
        )
        domain_validation, source_summary = validate_sources(
            backbone,
            classifier,
            validation_loaders,
            manifest["class_names"],
            device,
        )
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "mean_source_validation_accuracy": source_summary["mean_accuracy"],
            "mean_source_validation_macro_f1": source_summary["mean_macro_f1"],
        }
        for domain, metrics in domain_validation.items():
            row[f"{domain}_validation_accuracy"] = metrics["accuracy"]
            row[f"{domain}_validation_macro_f1"] = metrics["macro_f1"]
        history.append(row)
        write_history(history, history_path)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"loss {train_metrics['total_loss']:.4f} | "
            f"source val macro-F1 {source_summary['mean_macro_f1']:.4f}"
        )
        checkpoint_score = source_summary["mean_macro_f1"]
        if checkpoint_score > best_macro_f1:
            best_macro_f1 = checkpoint_score
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                epoch + 1,
                config,
                backbone,
                classifier,
                discriminator,
                optimizer,
                {
                    "by_domain": domain_validation,
                    "summary": source_summary,
                },
            )
            print(f"Saved new best checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping after epoch {epoch + 1}; "
                    f"best source validation macro-F1: {best_macro_f1:.4f}"
                )
                break

    print(f"Training complete: {run_name}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_CONFIGS), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--create-split", action="store_true")
    parser.add_argument("--mmd-weight", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
