"""Train source-only DAN-DG and SAM models for Task 3."""

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
from shared.pacs_protocol import build_source_datasets
from task2.evaluation.metrics import evaluate_classifier, summarize_source_validation
from task3.methods.dan_dg import dan_dg_objective
from task3.methods.sam import ascent_step, restore_parameters
from task3.models import ClassifierHead, ResNet18Backbone


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = deep_merge(result[key], value) if key in result and isinstance(result[key], dict) and isinstance(value, dict) else value
    return result


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_config(method: str) -> dict:
    return deep_merge(load_yaml("task3/configs/base.yaml"), load_yaml(f"task3/configs/{method}.yaml"))


def next_cycled(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def build_loaders(config: dict, data_root: str | Path):
    train_sets, validation_sets, manifest = build_source_datasets(
        data_root, config["data"]["split_manifest"]
    )
    train_loaders = {
        domain: DataLoader(
            train_sets[domain], batch_size=config["data"]["batch_size_per_domain"],
            shuffle=True, num_workers=config["data"]["num_workers"], pin_memory=True,
            drop_last=True,
        ) for domain in SOURCE_DOMAINS
    }
    validation_loaders = {
        domain: DataLoader(
            validation_sets[domain], batch_size=config["data"]["validation_batch_size"],
            shuffle=False, num_workers=config["data"]["num_workers"], pin_memory=True,
        ) for domain in SOURCE_DOMAINS
    }
    return train_loaders, validation_loaders, manifest


def source_batch(loaders, iterators, device):
    images, labels = {}, {}
    for domain in SOURCE_DOMAINS:
        batch, iterators[domain] = next_cycled(loaders[domain], iterators[domain])
        images[domain] = batch[0].to(device, non_blocking=True)
        labels[domain] = batch[1].to(device, non_blocking=True)
    return images, labels


def erm_loss(backbone, classifier, images, labels):
    all_images = torch.cat([images[domain] for domain in SOURCE_DOMAINS])
    all_labels = torch.cat([labels[domain] for domain in SOURCE_DOMAINS])
    return nn.functional.cross_entropy(classifier(backbone(all_images)), all_labels)


def train_epoch(config, backbone, classifier, optimizer, loaders, device):
    backbone.train()
    classifier.train()
    iterators = {domain: iter(loaders[domain]) for domain in SOURCE_DOMAINS}
    steps = max(len(loader) for loader in loaders.values())
    running: dict[str, float] = {}
    parameters = list(backbone.parameters()) + list(classifier.parameters())
    for _ in range(steps):
        images, labels = source_batch(loaders, iterators, device)
        optimizer.zero_grad(set_to_none=True)
        if config["method"]["name"] == "dan_dg":
            output = dan_dg_objective(
                backbone, classifier, images, labels,
                alignment_weight=config["method"]["alignment_weight"],
                bandwidth_multipliers=tuple(config["method"]["bandwidth_multipliers"]),
                normalize_features_for_mmd=config["method"]["normalize_features_for_mmd"],
            )
            output["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip_norm"])
            optimizer.step()
        else:
            first_loss = erm_loss(backbone, classifier, images, labels)
            first_loss.backward()
            perturbations = ascent_step(parameters, config["method"]["rho"])
            optimizer.zero_grad(set_to_none=True)
            second_loss = erm_loss(backbone, classifier, images, labels)
            second_loss.backward()
            restore_parameters(perturbations)
            torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip_norm"])
            optimizer.step()
            output = {
                "total_loss": second_loss,
                "classification_loss": first_loss,
                "sam_perturbed_loss": second_loss,
                "sam_loss_increase": second_loss - first_loss,
            }
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                running[key] = running.get(key, 0.0) + float(value.detach().item())
    return {key: value / steps for key, value in running.items()}


def validate(backbone, classifier, loaders, class_names, device):
    results = {}
    for domain in SOURCE_DOMAINS:
        results[domain], _ = evaluate_classifier(backbone, classifier, loaders[domain], class_names, device)
    return results, summarize_source_validation(results)


def save_checkpoint(path, epoch, config, backbone, classifier, optimizer, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "config": config,
        "backbone_state_dict": backbone.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "source_validation": metrics,
    }, path)


def write_history(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def run(args) -> None:
    config = load_config(args.method)
    if args.alignment_weight is not None:
        if args.method != "dan_dg": raise ValueError("--alignment-weight requires DAN-DG")
        config["method"]["alignment_weight"] = args.alignment_weight
    if args.rho is not None:
        if args.method != "sam": raise ValueError("--rho requires SAM")
        config["method"]["rho"] = args.rho
    config["data"]["root"] = args.data_root
    run_name = args.run_name or args.method
    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loaders, validation_loaders, manifest = build_loaders(config, args.data_root)
    print("Source train sizes:", {d: len(x.dataset) for d, x in train_loaders.items()})
    print("Source validation sizes:", {d: len(x.dataset) for d, x in validation_loaders.items()})
    print("Sketch is not loaded by this program.")

    backbone = ResNet18Backbone(pretrained=True, freeze_batchnorm_statistics=True).to(device)
    classifier = ClassifierHead(config["model"]["feature_dimension"], config["model"]["number_of_classes"]).to(device)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"],
    )
    checkpoint_path = Path("task3/checkpoints") / f"{run_name}_best.pt"
    history_path = Path("task3/results/training") / f"{run_name}_history.csv"
    config_path = Path("task3/results/configs") / f"{run_name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    history, best, stale = [], -float("inf"), 0
    for epoch in range(config["training"]["maximum_epochs"]):
        train_metrics = train_epoch(config, backbone, classifier, optimizer, train_loaders, device)
        domain_metrics, summary = validate(backbone, classifier, validation_loaders, manifest["class_names"], device)
        score = summary["mean_macro_f1"]
        row = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()},
               "mean_source_validation_accuracy": summary["mean_accuracy"],
               "mean_source_validation_macro_f1": score,
               "worst_source_validation_accuracy": summary["worst_accuracy"],
               "worst_source_validation_macro_f1": summary["worst_macro_f1"]}
        for domain, metrics in domain_metrics.items():
            row[f"{domain}_validation_accuracy"] = metrics["accuracy"]
            row[f"{domain}_validation_macro_f1"] = metrics["macro_f1"]
        history.append(row); write_history(history, history_path)
        print(f"Epoch {epoch + 1:02d} | loss {train_metrics['total_loss']:.4f} | source val macro-F1 {score:.4f}")
        if score > best:
            best, stale = score, 0
            save_checkpoint(checkpoint_path, epoch + 1, config, backbone, classifier, optimizer, domain_metrics)
            print(f"Saved new best checkpoint to {checkpoint_path}")
        else:
            stale += 1
            if stale >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping after epoch {epoch + 1}; best source validation macro-F1: {best:.4f}")
                break
    print(f"Training complete: {run_name}\nBest checkpoint: {checkpoint_path}\nHistory: {history_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dan_dg", "sam"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--alignment-weight", type=float)
    parser.add_argument("--rho", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
