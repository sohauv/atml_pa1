"""Final Task 2 evaluation after all checkpoints and settings are fixed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from shared.pacs import SOURCE_DOMAINS
from shared.pacs_protocol import build_source_datasets, build_task2_target_dataset
from task2.evaluation.class_analysis import transfer_summary
from task2.evaluation.domain_separability import (
    domain_separability_score,
    extract_features,
)
from task2.evaluation.metrics import evaluate_classifier, summarize_source_validation
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead


MAIN_METHODS = ("source_only", "dan", "dann", "cdan")


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Runs must have the form name=path/to/checkpoint.pt"
        )
    name, checkpoint = value.split("=", 1)
    if not name.strip() or not checkpoint.strip():
        raise argparse.ArgumentTypeError("Run name and checkpoint path cannot be empty.")
    return name.strip(), Path(checkpoint.strip())


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_evaluation_loaders(config: dict, data_root: str | Path):
    _, validation_datasets, manifest = build_source_datasets(
        data_root,
        config["data"]["split_manifest"],
    )
    workers = config["data"]["num_workers"]
    source_validation_loaders = {
        domain: DataLoader(
            validation_datasets[domain],
            batch_size=64,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        for domain in SOURCE_DOMAINS
    }
    pooled_source_loader = DataLoader(
        ConcatDataset([validation_datasets[domain] for domain in SOURCE_DOMAINS]),
        batch_size=64,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    target_dataset = build_task2_target_dataset(
        data_root,
        manifest["class_to_index"],
        training=False,
        return_metadata=True,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return (
        source_validation_loaders,
        pooled_source_loader,
        target_loader,
        manifest,
    )


def load_classifier(checkpoint: dict, device: torch.device):
    config = checkpoint["config"]
    backbone = ResNet18Backbone(
        pretrained=False,
        freeze_batchnorm_statistics=config["model"][
            "freeze_batchnorm_running_statistics"
        ],
    ).to(device)
    classifier = ClassifierHead(
        feature_dimension=config["model"]["feature_dimension"],
        number_of_classes=config["model"]["number_of_classes"],
    ).to(device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
    classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
    backbone.eval()
    classifier.eval()
    return backbone, classifier


def evaluate_run(
    run_name: str,
    checkpoint_path: Path,
    data_root: str | Path,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {run_name}: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    (
        source_validation_loaders,
        pooled_source_loader,
        target_loader,
        manifest,
    ) = build_evaluation_loaders(config, data_root)
    class_names = manifest["class_names"]
    backbone, classifier = load_classifier(checkpoint, device)

    source_by_domain = {}
    for domain in SOURCE_DOMAINS:
        metrics, _ = evaluate_classifier(
            backbone,
            classifier,
            source_validation_loaders[domain],
            class_names,
            device,
        )
        source_by_domain[domain] = metrics
    source_summary = summarize_source_validation(source_by_domain)

    target_metrics, target_predictions = evaluate_classifier(
        backbone,
        classifier,
        target_loader,
        class_names,
        device,
    )
    source_features = extract_features(backbone, pooled_source_loader, device)
    target_features = extract_features(backbone, target_loader, device)
    separability = domain_separability_score(
        source_features,
        target_features,
        seed=config["evaluation"]["domain_separability"]["seed"],
        test_fraction=config["evaluation"]["domain_separability"][
            "test_fraction"
        ],
        c=config["evaluation"]["domain_separability"]["c"],
    )
    result = {
        "run_name": run_name,
        "method": config["method"]["name"],
        "checkpoint": checkpoint_path.as_posix(),
        "selected_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_rule": "mean_source_validation_macro_f1",
        "source_validation": {
            "by_domain": source_by_domain,
            "summary": source_summary,
        },
        "target": target_metrics,
        "domain_separability": separability,
        "config": config,
    }
    return result, target_predictions


def summary_row(result: dict) -> dict:
    by_domain = result["source_validation"]["by_domain"]
    source_summary = result["source_validation"]["summary"]
    target = result["target"]
    return {
        "run_name": result["run_name"],
        "method": result["method"],
        "selected_epoch": result["selected_epoch"],
        "photo_validation_accuracy": by_domain["photo"]["accuracy"],
        "photo_validation_macro_f1": by_domain["photo"]["macro_f1"],
        "art_painting_validation_accuracy": by_domain["art_painting"]["accuracy"],
        "art_painting_validation_macro_f1": by_domain["art_painting"]["macro_f1"],
        "cartoon_validation_accuracy": by_domain["cartoon"]["accuracy"],
        "cartoon_validation_macro_f1": by_domain["cartoon"]["macro_f1"],
        "mean_source_validation_accuracy": source_summary["mean_accuracy"],
        "mean_source_validation_macro_f1": source_summary["mean_macro_f1"],
        "target_accuracy": target["accuracy"],
        "target_macro_f1": target["macro_f1"],
        "target_accuracy_change": result["target_accuracy_change"],
        "domain_separability": result["domain_separability"]["accuracy"],
    }


def main(args) -> None:
    runs = dict(args.run)
    if "source_only" not in runs:
        raise ValueError("A source_only checkpoint is required as the comparison baseline.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_directory = Path(args.output_directory)
    predictions_directory = output_directory / "predictions"
    predictions_directory.mkdir(parents=True, exist_ok=True)

    results = {}
    for run_name, checkpoint_path in runs.items():
        print("=" * 70)
        print(f"Final evaluation: {run_name}")
        print("=" * 70)
        result, predictions = evaluate_run(
            run_name,
            checkpoint_path,
            args.data_root,
            device,
        )
        results[run_name] = result
        write_rows(
            predictions_directory / f"{run_name}_target_predictions.csv",
            predictions,
        )
        print(
            f"Target accuracy: {result['target']['accuracy']:.4f} | "
            f"macro-F1: {result['target']['macro_f1']:.4f} | "
            f"domain separability: {result['domain_separability']['accuracy']:.4f}"
        )

    baseline_target = results["source_only"]["target"]
    baseline_accuracy = baseline_target["accuracy"]
    for run_name, result in results.items():
        result["target_accuracy_change"] = float(
            result["target"]["accuracy"] - baseline_accuracy
        )
        result["class_transfer_analysis"] = transfer_summary(
            baseline_target,
            result["target"],
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    final_json = output_directory / "final_results.json"
    with final_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "protocol": {
                    "source_domains": list(SOURCE_DOMAINS),
                    "target_domain": "sketch",
                    "target_labels_used_only_in_final_evaluation": True,
                    "domain_separability_split": "70/30",
                },
                "runs": results,
            },
            file,
            indent=2,
        )
    write_rows(
        output_directory / "final_summary.csv",
        [summary_row(result) for result in results.values()],
    )
    print(f"Saved {final_json}")
    print(f"Saved {output_directory / 'final_summary.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Repeat name=checkpoint for every fixed run.",
    )
    parser.add_argument("--output-directory", default="task2/results/final")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
