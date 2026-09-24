"""Fine-tune PROSER from the selected frozen Vanilla checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from task4.data.cifar import build_cifar10
from task4.methods.proser import proser_objective
from task4.models import PROSERResNet18
from task4.train import evaluate, load_config, seed_everything


def initialize_from_vanilla(config, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PROSERResNet18(
        number_of_classes=config["model"]["number_of_classes"],
        dummy_classifier_count=config["method"]["dummy_classifier_count"],
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    expected_missing = {
        "dummy_classifier.weight", "dummy_classifier.bias"
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Vanilla initialization mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device), checkpoint


def write_history(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_epoch(model, loader, optimizer, scaler, scheduler, config, device):
    model.train()
    totals = {
        "total_loss": 0.0,
        "classifier_placeholder_loss": 0.0,
        "classification_loss": 0.0,
        "runner_up_dummy_loss": 0.0,
        "data_placeholder_loss": 0.0,
        "mean_mixup_lambda": 0.0,
    }
    example_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            output = proser_objective(
                model,
                images,
                labels,
                beta=config["method"]["classifier_placeholder_weight"],
                gamma=config["method"]["data_placeholder_weight"],
                mixup_alpha=config["method"]["mixup_alpha"],
            )
        if not torch.isfinite(output["total_loss"]):
            raise FloatingPointError("PROSER produced a non-finite training loss.")
        scaler.scale(output["total_loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        example_count += batch_size
        for key in totals:
            totals[key] += float(output[key].detach()) * batch_size
    scheduler.step()
    return {key: value / example_count for key, value in totals.items()}


def run(args):
    config = load_config("proser")
    config["data"]["root"] = args.data_root
    config["initialization"] = {
        "checkpoint": str(args.vanilla_checkpoint),
        "rule": "selected_vanilla_checkpoint",
    }
    seed_everything(config["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train, validation, _, manifest = build_cifar10(
        args.data_root,
        config["data"]["split_manifest"],
        randaugment=False,
        create=False,
        seed=config["experiment"]["seed"],
    )
    loader_options = {
        "batch_size": config["data"]["batch_size"],
        "num_workers": config["data"]["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)
    print(
        f"Train: {len(train)} | validation: {len(validation)} | "
        "CIFAR-100 is not loaded"
    )

    model, vanilla_checkpoint = initialize_from_vanilla(
        config, args.vanilla_checkpoint, device
    )
    if vanilla_checkpoint["class_names"] != manifest["class_names"]:
        raise ValueError("Vanilla checkpoint classes do not match the fixed split.")
    print(f"Initialized from Vanilla epoch {vanilla_checkpoint['epoch']}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            config["training"]["use_automatic_mixed_precision"]
            and device.type == "cuda"
        ),
    )
    run_name = args.run_name
    checkpoint_path = Path("task4/checkpoints") / f"{run_name}_best.pt"
    history_path = Path("task4/results/training") / f"{run_name}_history.csv"
    config_path = Path("task4/results/configs") / f"{run_name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_accuracy = -1.0
    for epoch in range(config["training"]["epochs"]):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, scheduler, config, device
        )
        validation_metrics = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(row)
        write_history(history, history_path)
        print(
            f"Epoch {epoch + 1:03d} | loss {row['train_total_loss']:.4f} | "
            f"val acc {row['validation_accuracy']:.4f}"
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "config": config,
                    "class_names": manifest["class_names"],
                    "model_state_dict": model.state_dict(),
                    "validation": validation_metrics,
                    "vanilla_selected_epoch": vanilla_checkpoint["epoch"],
                },
                checkpoint_path,
            )
            print(f"Saved {checkpoint_path}")
    print(
        f"Training complete: {run_name} | "
        f"best validation accuracy {best_accuracy:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--vanilla-checkpoint", default="task4/checkpoints/vanilla_best.pt"
    )
    parser.add_argument("--run-name", default="proser")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
