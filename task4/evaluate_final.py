"""One-time final Task 4 evaluation on fixed CIFAR-10/100 test sets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from task4.data.cifar import (
    FAR_UNKNOWN_CLASSES,
    NEAR_UNKNOWN_CLASSES,
    build_cifar10_evaluation_sets,
    build_unknowns,
)
from task4.evaluation.open_set_scores import (
    logit_unknownness_scores,
    mahalanobis_unknownness,
)
from task4.methods.proser import placeholder_unknownness
from task4.models import CIFARResNet18, PROSERResNet18


def load_closed_set_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CIFARResNet18(len(checkpoint["class_names"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_proser_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = PROSERResNet18(
        number_of_classes=config["model"]["number_of_classes"],
        dummy_classifier_count=config["method"]["dummy_classifier_count"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def extract_closed_set(model, loader, device):
    output = {
        "indices": [], "labels": [], "class_names": [], "domains": [],
        "features": [], "logits": [],
    }
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features = model.forward_features(images)
        logits = model.classifier(features)
        output["indices"].append(batch["index"].cpu())
        output["labels"].append(batch["label"].cpu())
        output["class_names"].extend(batch["class_name"])
        output["domains"].extend(batch["domain"])
        output["features"].append(features.cpu())
        output["logits"].append(logits.cpu())
    for key in ("indices", "labels", "features", "logits"):
        output[key] = torch.cat(output[key], dim=0)
    return output


@torch.no_grad()
def extract_proser(model, loader, device, temperature):
    output = {
        "indices": [], "labels": [], "class_names": [], "domains": [],
        "features": [], "logits": [], "dummy_logits": [],
        "placeholder": [],
    }
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        known_logits, dummy_logits, features = model.forward_all(images)
        output["indices"].append(batch["index"].cpu())
        output["labels"].append(batch["label"].cpu())
        output["class_names"].extend(batch["class_name"])
        output["domains"].extend(batch["domain"])
        output["features"].append(features.cpu())
        output["logits"].append(known_logits.cpu())
        output["dummy_logits"].append(dummy_logits.cpu())
        output["placeholder"].append(
            placeholder_unknownness(
                known_logits, dummy_logits, temperature=temperature
            ).cpu()
        )
    for key in (
        "indices", "labels", "features", "logits", "dummy_logits", "placeholder"
    ):
        output[key] = torch.cat(output[key], dim=0)
    return output


def closed_set_metrics(output):
    truth = output["labels"].numpy()
    predictions = output["logits"].argmax(dim=1).numpy()
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
    }


def auroc(known_scores, unknown_scores):
    truth = np.concatenate(
        [np.zeros(len(known_scores)), np.ones(len(unknown_scores))]
    )
    scores = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(truth, scores))


def evaluate_score(known, near, far, threshold):
    all_unknown = np.concatenate([near, far])
    known_acceptance = float(np.mean(known <= threshold))
    near_rejection = float(np.mean(near > threshold))
    far_rejection = float(np.mean(far > threshold))
    all_rejection = float(np.mean(all_unknown > threshold))
    return {
        "threshold": float(threshold),
        "known_test_acceptance": known_acceptance,
        "near_rejection": near_rejection,
        "far_rejection": far_rejection,
        "all_unknown_rejection": all_rejection,
        "near_fpr_at_95_tpr": 1.0 - near_rejection,
        "far_fpr_at_95_tpr": 1.0 - far_rejection,
        "all_unknown_fpr_at_95_tpr": 1.0 - all_rejection,
        "known_vs_near_auroc": auroc(known, near),
        "known_vs_far_auroc": auroc(known, far),
        "known_vs_all_auroc": auroc(known, all_unknown),
    }


def write_prediction_rows(path, outputs_by_domain, scores_by_domain, threshold_by_score):
    path.parent.mkdir(parents=True, exist_ok=True)
    score_names = list(scores_by_domain["known"])
    fieldnames = [
        "domain", "index", "true_class_id", "true_class_name",
        "predicted_known_class_id", "predicted_known_class_name",
        "maximum_known_probability", *score_names,
        *[f"accepted_{name}" for name in score_names],
    ]
    known_class_names = outputs_by_domain["known_class_names"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for domain_name in ("known", "near", "far"):
            output = outputs_by_domain[domain_name]
            predictions = output["logits"].argmax(dim=1).numpy()
            confidence = output["logits"].softmax(dim=1).max(dim=1).values.numpy()
            for row_index in range(len(predictions)):
                row = {
                    "domain": domain_name,
                    "index": int(output["indices"][row_index]),
                    "true_class_id": int(output["labels"][row_index]),
                    "true_class_name": output["class_names"][row_index],
                    "predicted_known_class_id": int(predictions[row_index]),
                    "predicted_known_class_name": known_class_names[predictions[row_index]],
                    "maximum_known_probability": float(confidence[row_index]),
                }
                for score_name in score_names:
                    value = float(scores_by_domain[domain_name][score_name][row_index])
                    row[score_name] = value
                    row[f"accepted_{score_name}"] = (
                        value <= threshold_by_score[score_name]
                    )
                writer.writerow(row)


def add_result_row(rows, run_name, method, score_name, checkpoint, classification, metrics):
    rows.append(
        {
            "run_name": run_name,
            "method": method,
            "score": score_name,
            "selected_epoch": checkpoint["epoch"],
            "known_test_accuracy": classification["accuracy"],
            "known_test_macro_f1": classification["macro_f1"],
            **metrics,
        }
    )


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    _, _, known_test, manifest = build_cifar10_evaluation_sets(
        args.cifar10_root, args.split_manifest
    )
    near, far = build_unknowns(args.cifar100_root)
    if len(near) != 800 or len(far) != 800:
        raise ValueError(
            f"Expected 800 Near and 800 Far images, received {len(near)} and {len(far)}."
        )
    print(f"Known test: {len(known_test)} | Near: {len(near)} | Far: {len(far)}")
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    loaders = {
        "known": DataLoader(known_test, **loader_options),
        "near": DataLoader(near, **loader_options),
        "far": DataLoader(far, **loader_options),
    }

    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    statistics = np.load(args.mahalanobis_statistics)
    class_means = torch.from_numpy(statistics["class_means"])
    variance = torch.from_numpy(statistics["shared_diagonal_variance"])
    output_directory = Path(args.output_directory)
    predictions_directory = output_directory / "predictions"
    rows = []

    vanilla, vanilla_checkpoint = load_closed_set_model(
        args.vanilla_checkpoint, device
    )
    vanilla_outputs = {
        name: extract_closed_set(vanilla, loader, device)
        for name, loader in loaders.items()
    }
    vanilla_outputs["known_class_names"] = manifest["class_names"]
    vanilla_scores = {}
    for domain_name in ("known", "near", "far"):
        output = vanilla_outputs[domain_name]
        scores = {
            name: values.numpy()
            for name, values in logit_unknownness_scores(output["logits"]).items()
        }
        scores["mahalanobis"] = mahalanobis_unknownness(
            output["features"], class_means, variance
        ).numpy()
        vanilla_scores[domain_name] = scores
    vanilla_classification = closed_set_metrics(vanilla_outputs["known"])
    vanilla_thresholds = {
        name: value["threshold"]
        for name, value in thresholds["runs"]["vanilla"]["scores"].items()
    }
    for score_name in ("msp", "mls", "energy", "mahalanobis"):
        metrics = evaluate_score(
            vanilla_scores["known"][score_name],
            vanilla_scores["near"][score_name],
            vanilla_scores["far"][score_name],
            vanilla_thresholds[score_name],
        )
        add_result_row(
            rows, f"vanilla_{score_name}", "vanilla", score_name,
            vanilla_checkpoint, vanilla_classification, metrics,
        )
    write_prediction_rows(
        predictions_directory / "vanilla_predictions.csv",
        vanilla_outputs, vanilla_scores, vanilla_thresholds,
    )
    del vanilla, vanilla_outputs

    gcsc, gcsc_checkpoint = load_closed_set_model(args.gcsc_checkpoint, device)
    gcsc_outputs = {
        name: extract_closed_set(gcsc, loader, device)
        for name, loader in loaders.items()
    }
    gcsc_outputs["known_class_names"] = manifest["class_names"]
    gcsc_scores = {
        name: {"mls": -output["logits"].max(dim=1).values.numpy()}
        for name, output in gcsc_outputs.items()
        if name != "known_class_names"
    }
    gcsc_threshold = thresholds["runs"]["gcsc"]["scores"]["mls"]["threshold"]
    metrics = evaluate_score(
        gcsc_scores["known"]["mls"], gcsc_scores["near"]["mls"],
        gcsc_scores["far"]["mls"], gcsc_threshold,
    )
    add_result_row(
        rows, "gcsc_mls", "gcsc", "mls", gcsc_checkpoint,
        closed_set_metrics(gcsc_outputs["known"]), metrics,
    )
    write_prediction_rows(
        predictions_directory / "gcsc_predictions.csv", gcsc_outputs,
        gcsc_scores, {"mls": gcsc_threshold},
    )
    del gcsc, gcsc_outputs

    proser, proser_checkpoint = load_proser_model(args.proser_checkpoint, device)
    temperature = proser_checkpoint["config"]["method"]["placeholder_temperature"]
    proser_outputs = {
        name: extract_proser(proser, loader, device, temperature)
        for name, loader in loaders.items()
    }
    proser_outputs["known_class_names"] = manifest["class_names"]
    proser_scores = {
        name: {
            "mls": -output["logits"].max(dim=1).values.numpy(),
            "placeholder": output["placeholder"].numpy(),
        }
        for name, output in proser_outputs.items()
        if name != "known_class_names"
    }
    proser_thresholds = {
        name: value["threshold"]
        for name, value in thresholds["runs"]["proser"]["scores"].items()
    }
    proser_classification = closed_set_metrics(proser_outputs["known"])
    for score_name in ("mls", "placeholder"):
        metrics = evaluate_score(
            proser_scores["known"][score_name],
            proser_scores["near"][score_name],
            proser_scores["far"][score_name],
            proser_thresholds[score_name],
        )
        add_result_row(
            rows, f"proser_{score_name}", "proser", score_name,
            proser_checkpoint, proser_classification, metrics,
        )
    write_prediction_rows(
        predictions_directory / "proser_predictions.csv", proser_outputs,
        proser_scores, proser_thresholds,
    )

    final_result = {
        "protocol": {
            "known_dataset": "CIFAR-10 official test",
            "near_unknown_classes": list(NEAR_UNKNOWN_CLASSES),
            "far_unknown_classes": list(FAR_UNKNOWN_CLASSES),
            "near_count": len(near),
            "far_count": len(far),
            "threshold_source": "95th percentile of CIFAR-10 validation unknownness",
            "checkpoints_and_thresholds_frozen_before_cifar100": True,
        },
        "runs": {row["run_name"]: row for row in rows},
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "final_results.json"
    csv_path = output_directory / "final_summary.csv"
    json_path.write_text(json.dumps(final_result, indent=2), encoding="utf-8")
    write_summary(csv_path, rows)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    for row in rows:
        print(
            f"{row['run_name']}: CSA {row['known_test_accuracy']:.4f} | "
            f"AUROC near/far/all {row['known_vs_near_auroc']:.4f}/"
            f"{row['known_vs_far_auroc']:.4f}/{row['known_vs_all_auroc']:.4f} | "
            f"reject near/far {row['near_rejection']:.4f}/{row['far_rejection']:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar10-root", required=True)
    parser.add_argument("--cifar100-root", required=True)
    parser.add_argument(
        "--split-manifest", default="task4/results/cifar10_seed6304_split.json"
    )
    parser.add_argument(
        "--thresholds", default="task4/results/calibration/open_set_thresholds.json"
    )
    parser.add_argument(
        "--mahalanobis-statistics",
        default="task4/results/calibration/vanilla_mahalanobis_stats.npz",
    )
    parser.add_argument(
        "--vanilla-checkpoint", default="task4/checkpoints/vanilla_best.pt"
    )
    parser.add_argument(
        "--gcsc-checkpoint", default="task4/checkpoints/gcsc_best.pt"
    )
    parser.add_argument(
        "--proser-checkpoint", default="task4/checkpoints/proser_best.pt"
    )
    parser.add_argument("--output-directory", default="task4/results/final")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
