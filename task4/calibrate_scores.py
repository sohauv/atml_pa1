"""Fit frozen Vanilla score statistics and CIFAR-10 validation thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from task4.data.cifar import build_cifar10_evaluation_sets
from task4.evaluation.open_set_scores import (
    fit_shared_diagonal_gaussian,
    logit_unknownness_scores,
    mahalanobis_unknownness,
    percentile_threshold,
)
from task4.models import CIFARResNet18


def load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    model = CIFARResNet18(number_of_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, class_names


@torch.no_grad()
def extract_outputs(model, loader, device):
    outputs = {
        "features": [], "logits": [], "labels": [], "indices": [],
        "class_names": [],
    }
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features = model.forward_features(images)
        logits = model.classifier(features)
        outputs["features"].append(features.cpu())
        outputs["logits"].append(logits.cpu())
        outputs["labels"].append(batch["label"].cpu())
        outputs["indices"].append(batch["index"].cpu())
        outputs["class_names"].extend(batch["class_name"])
    for key in ("features", "logits", "labels", "indices"):
        outputs[key] = torch.cat(outputs[key], dim=0)
    return outputs


def classification_metrics(outputs):
    truth = outputs["labels"].numpy()
    predictions = outputs["logits"].argmax(dim=1).numpy()
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
    }


def score_validation(outputs, score_names, class_means=None, variance=None):
    available = logit_unknownness_scores(outputs["logits"])
    if class_means is not None and variance is not None:
        available["mahalanobis"] = mahalanobis_unknownness(
            outputs["features"], class_means, variance
        )
    return {name: available[name].numpy() for name in score_names}


def threshold_summary(scores, percentile):
    summary = {}
    for name, values in scores.items():
        threshold = percentile_threshold(values, percentile)
        summary[name] = {
            "threshold": threshold,
            "validation_known_acceptance": float(np.mean(values <= threshold)),
            "validation_score_mean": float(np.mean(values)),
            "validation_score_std": float(np.std(values)),
        }
    return summary


def write_validation_rows(path, outputs, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = outputs["logits"].argmax(dim=1).numpy()
    confidence = outputs["logits"].softmax(dim=1).max(dim=1).values.numpy()
    fieldnames = [
        "index", "true_class_id", "true_class_name", "predicted_class_id",
        "maximum_confidence", *scores,
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(len(predictions)):
            row = {
                "index": int(outputs["indices"][row_index]),
                "true_class_id": int(outputs["labels"][row_index]),
                "true_class_name": outputs["class_names"][row_index],
                "predicted_class_id": int(predictions[row_index]),
                "maximum_confidence": float(confidence[row_index]),
            }
            row.update(
                {name: float(values[row_index]) for name, values in scores.items()}
            )
            writer.writerow(row)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train, validation, _, manifest = build_cifar10_evaluation_sets(
        args.data_root, args.split_manifest
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train, shuffle=False, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)
    print(
        f"Unaugmented fit set: {len(train)} | validation: {len(validation)} | "
        "CIFAR-10 test and CIFAR-100 are not evaluated"
    )

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": {
            "seed": manifest["seed"],
            "threshold_percentile": args.percentile,
            "threshold_fit_split": "cifar10_validation",
            "mahalanobis_fit_split": "unaugmented_cifar10_train",
            "larger_score_means": "more_unknown_like",
            "cifar100_used": False,
        },
        "runs": {},
    }

    vanilla, vanilla_checkpoint, vanilla_classes = load_model(
        args.vanilla_checkpoint, device
    )
    if vanilla_classes != manifest["class_names"]:
        raise ValueError("Vanilla checkpoint classes do not match the split manifest.")
    print("Extracting unaugmented Vanilla training features...")
    vanilla_train = extract_outputs(vanilla, train_loader, device)
    class_means, variance = fit_shared_diagonal_gaussian(
        vanilla_train["features"], vanilla_train["labels"],
        number_of_classes=len(vanilla_classes),
        regularization=args.covariance_regularization,
    )
    statistics_path = output_directory / "vanilla_mahalanobis_stats.npz"
    np.savez_compressed(
        statistics_path,
        class_means=class_means.numpy(),
        shared_diagonal_variance=variance.numpy(),
        class_names=np.asarray(vanilla_classes),
        regularization=np.asarray(args.covariance_regularization),
    )
    print("Scoring the Vanilla validation split...")
    vanilla_validation = extract_outputs(vanilla, validation_loader, device)
    vanilla_scores = score_validation(
        vanilla_validation, ("msp", "mls", "energy", "mahalanobis"),
        class_means, variance,
    )
    result["runs"]["vanilla"] = {
        "checkpoint": str(args.vanilla_checkpoint),
        "selected_epoch": vanilla_checkpoint["epoch"],
        "classification": classification_metrics(vanilla_validation),
        "scores": threshold_summary(vanilla_scores, args.percentile),
        "mahalanobis_statistics": str(statistics_path),
    }
    write_validation_rows(
        output_directory / "vanilla_validation_scores.csv",
        vanilla_validation, vanilla_scores,
    )
    del vanilla, vanilla_train, vanilla_validation

    gcsc, gcsc_checkpoint, gcsc_classes = load_model(args.gcsc_checkpoint, device)
    if gcsc_classes != manifest["class_names"]:
        raise ValueError("GCSC checkpoint classes do not match the split manifest.")
    print("Scoring the GCSC validation split...")
    gcsc_validation = extract_outputs(gcsc, validation_loader, device)
    gcsc_scores = score_validation(gcsc_validation, ("mls",))
    result["runs"]["gcsc"] = {
        "checkpoint": str(args.gcsc_checkpoint),
        "selected_epoch": gcsc_checkpoint["epoch"],
        "classification": classification_metrics(gcsc_validation),
        "scores": threshold_summary(gcsc_scores, args.percentile),
    }
    write_validation_rows(
        output_directory / "gcsc_validation_scores.csv",
        gcsc_validation, gcsc_scores,
    )

    result_path = output_directory / "open_set_thresholds.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {result_path}")
    print(f"Saved {statistics_path}")
    for run_name, run_result in result["runs"].items():
        print(f"{run_name}: validation accuracy {run_result['classification']['accuracy']:.4f}")
        for score_name, score_result in run_result["scores"].items():
            print(
                f"  {score_name}: threshold {score_result['threshold']:.6f} | "
                f"known acceptance {score_result['validation_known_acceptance']:.4f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--split-manifest", default="task4/results/cifar10_seed6304_split.json"
    )
    parser.add_argument(
        "--vanilla-checkpoint", default="task4/checkpoints/vanilla_best.pt"
    )
    parser.add_argument(
        "--gcsc-checkpoint", default="task4/checkpoints/gcsc_best.pt"
    )
    parser.add_argument("--output-directory", default="task4/results/calibration")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--covariance-regularization", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
