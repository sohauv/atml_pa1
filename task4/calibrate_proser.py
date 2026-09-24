"""Freeze PROSER and calibrate its MLS and placeholder scores on validation."""

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
from task4.evaluation.open_set_scores import percentile_threshold
from task4.methods.proser import placeholder_unknownness
from task4.models import PROSERResNet18


@torch.no_grad()
def extract_validation(model, loader, device, temperature):
    result = {
        "indices": [], "labels": [], "class_names": [], "known_logits": [],
        "dummy_logits": [], "mls": [], "placeholder": [],
    }
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        known_logits, dummy_logits, _ = model.forward_all(images)
        result["indices"].append(batch["index"].cpu())
        result["labels"].append(batch["label"].cpu())
        result["class_names"].extend(batch["class_name"])
        result["known_logits"].append(known_logits.cpu())
        result["dummy_logits"].append(dummy_logits.cpu())
        result["mls"].append((-known_logits.max(dim=1).values).cpu())
        result["placeholder"].append(
            placeholder_unknownness(
                known_logits, dummy_logits, temperature=temperature
            ).cpu()
        )
    for key in (
        "indices", "labels", "known_logits", "dummy_logits", "mls", "placeholder"
    ):
        result[key] = torch.cat(result[key], dim=0)
    return result


def score_summary(values, percentile):
    values = values.numpy()
    threshold = percentile_threshold(values, percentile)
    return {
        "threshold": threshold,
        "validation_known_acceptance": float(np.mean(values <= threshold)),
        "validation_score_mean": float(np.mean(values)),
        "validation_score_std": float(np.std(values)),
    }


def write_rows(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = result["known_logits"].argmax(dim=1)
    strongest_dummy = result["dummy_logits"].max(dim=1).values
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "index", "true_class_id", "true_class_name", "predicted_class_id",
            "strongest_dummy_logit", "mls", "placeholder",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(predictions)):
            writer.writerow(
                {
                    "index": int(result["indices"][index]),
                    "true_class_id": int(result["labels"][index]),
                    "true_class_name": result["class_names"][index],
                    "predicted_class_id": int(predictions[index]),
                    "strongest_dummy_logit": float(strongest_dummy[index]),
                    "mls": float(result["mls"][index]),
                    "placeholder": float(result["placeholder"][index]),
                }
            )


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.proser_checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    model = PROSERResNet18(
        number_of_classes=config["model"]["number_of_classes"],
        dummy_classifier_count=config["method"]["dummy_classifier_count"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, validation, _, manifest = build_cifar10_evaluation_sets(
        args.data_root, args.split_manifest
    )
    if checkpoint["class_names"] != manifest["class_names"]:
        raise ValueError("PROSER checkpoint classes do not match the fixed split.")
    loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    temperature = config["method"]["placeholder_temperature"]
    result = extract_validation(model, loader, device, temperature)
    predictions = result["known_logits"].argmax(dim=1).numpy()
    truth = result["labels"].numpy()

    threshold_path = Path(args.thresholds)
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    thresholds["runs"]["proser"] = {
        "checkpoint": str(args.proser_checkpoint),
        "selected_epoch": checkpoint["epoch"],
        "classification": {
            "accuracy": float(accuracy_score(truth, predictions)),
            "macro_f1": float(f1_score(truth, predictions, average="macro")),
        },
        "scores": {
            "mls": score_summary(result["mls"], args.percentile),
            "placeholder": score_summary(result["placeholder"], args.percentile),
        },
        "placeholder_definition": (
            "max dummy probability minus max known probability after reducing "
            f"five dummies to their strongest logit; temperature={temperature}"
        ),
    }
    threshold_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    output_csv = threshold_path.parent / "proser_validation_scores.csv"
    write_rows(output_csv, result)
    print(f"Saved {threshold_path}")
    print(f"Saved {output_csv}")
    print(
        f"PROSER epoch {checkpoint['epoch']} | validation accuracy "
        f"{thresholds['runs']['proser']['classification']['accuracy']:.4f}"
    )
    for score_name, summary in thresholds["runs"]["proser"]["scores"].items():
        print(
            f"  {score_name}: threshold {summary['threshold']:.6f} | "
            f"known acceptance {summary['validation_known_acceptance']:.4f}"
        )
    print("CIFAR-100 was not loaded.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--proser-checkpoint", default="task4/checkpoints/proser_best.pt"
    )
    parser.add_argument(
        "--split-manifest", default="task4/results/cifar10_seed6304_split.json"
    )
    parser.add_argument(
        "--thresholds", default="task4/results/calibration/open_set_thresholds.json"
    )
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
