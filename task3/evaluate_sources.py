"""Evaluate fixed Task 3 checkpoints using source validation data only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from shared.pacs import SOURCE_DOMAINS
from shared.pacs_protocol import build_source_datasets
from task2.evaluation.metrics import evaluate_classifier
from task3.evaluation.sharpness import fixed_domain_batch, sharpness_proxy
from task3.evaluation.source_domain_separability import (
    extract_features,
    source_domain_separability,
)
from task3.models import ClassifierHead, ResNet18Backbone


RUNS = {
    "erm": "task2/checkpoints/source_only_best.pt",
    "dan_dg_lambda0_1": "task3/checkpoints/dan_dg_lambda0_1_best.pt",
    "dan_dg": "task3/checkpoints/dan_dg_best.pt",
    "dan_dg_lambda10": "task3/checkpoints/dan_dg_lambda10_best.pt",
    "sam": "task3/checkpoints/sam_best.pt",
}


def native(value):
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist() if value.numel() > 1 else value.item()
    return value


def load_model(checkpoint_path: str | Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone = ResNet18Backbone(pretrained=False, freeze_batchnorm_statistics=True).to(device)
    classifier = ClassifierHead().to(device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    backbone.eval(); classifier.eval()
    return checkpoint, backbone, classifier


def summary(domain_metrics: dict) -> dict:
    accuracies = [domain_metrics[domain]["accuracy"] for domain in SOURCE_DOMAINS]
    macro_f1s = [domain_metrics[domain]["macro_f1"] for domain in SOURCE_DOMAINS]
    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "worst_accuracy": float(np.min(accuracies)),
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "worst_macro_f1": float(np.min(macro_f1s)),
    }


def run(args) -> None:
    seed = 6304
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, validation_sets, manifest = build_source_datasets(args.data_root, args.split_manifest)
    loaders = {
        domain: DataLoader(
            validation_sets[domain], batch_size=64, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        ) for domain in SOURCE_DOMAINS
    }
    sharpness_images, sharpness_labels = fixed_domain_batch(
        validation_sets, count_per_domain=32, seed=seed
    )
    missing = [path for path in RUNS.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fixed checkpoints: {missing}")

    results = {
        "protocol": {
            "domains": list(SOURCE_DOMAINS),
            "target_domain_used": False,
            "selection_metric": "mean_source_validation_macro_f1",
            "separability_probe": "balanced multinomial logistic regression, C=1, seeded 70/30 split",
            "separability_chance": 1.0 / len(SOURCE_DOMAINS),
            "sharpness_radius": 0.05,
            "sharpness_batch_per_domain": 32,
            "seed": seed,
        },
        "runs": {},
    }
    rows = []
    for run_name, checkpoint_path in RUNS.items():
        print("=" * 70); print(f"Source diagnostics: {run_name}"); print("=" * 70)
        checkpoint, backbone, classifier = load_model(checkpoint_path, device)
        domain_metrics = {}
        feature_sets = {}
        for domain in SOURCE_DOMAINS:
            domain_metrics[domain], _ = evaluate_classifier(
                backbone, classifier, loaders[domain], manifest["class_names"], device
            )
            feature_sets[domain] = extract_features(backbone, loaders[domain], device)
        aggregate = summary(domain_metrics)
        separability = source_domain_separability(feature_sets, seed=seed)
        sharpness = sharpness_proxy(
            backbone, classifier, sharpness_images, sharpness_labels,
            device=device, radius=0.05,
        )
        run_result = {
            "checkpoint": checkpoint_path,
            "selected_epoch": int(checkpoint["epoch"]),
            "source_domains": domain_metrics,
            "source_summary": aggregate,
            "source_domain_separability": separability,
            "sharpness_proxy": sharpness,
        }
        results["runs"][run_name] = native(run_result)
        row = {
            "run_name": run_name,
            "selected_epoch": int(checkpoint["epoch"]),
            **aggregate,
            "source_domain_separability": separability,
            "sharpness_base_loss": sharpness["base_loss"],
            "sharpness_perturbed_loss": sharpness["perturbed_loss"],
            "sharpness_loss_increase": sharpness["loss_increase"],
        }
        for domain in SOURCE_DOMAINS:
            row[f"{domain}_accuracy"] = domain_metrics[domain]["accuracy"]
            row[f"{domain}_macro_f1"] = domain_metrics[domain]["macro_f1"]
        rows.append(row)
        print(
            f"mean F1 {aggregate['mean_macro_f1']:.4f} | "
            f"worst F1 {aggregate['worst_macro_f1']:.4f} | "
            f"separability {separability:.4f} | "
            f"sharpness increase {sharpness['loss_increase']:.4f}"
        )

    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "source_diagnostics.json"
    csv_path = output / "source_diagnostics.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved {json_path}"); print(f"Saved {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", default="shared/splits/pacs_sketch_seed6304.json")
    parser.add_argument("--output-directory", default="task3/results/source_diagnostics")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
