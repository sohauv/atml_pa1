from copy import deepcopy
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LinearClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        number_of_classes: int = 10,
    ) -> None:
        super().__init__()

        self.classifier = nn.Linear(
            in_features=feature_dim,
            out_features=number_of_classes,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def make_feature_dataloader(
    feature_data: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    seed: int = 6304,
) -> DataLoader:
    dataset = TensorDataset(
        feature_data["features"].float(),
        feature_data["labels"].long(),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


@torch.inference_mode()
def evaluate_classifier(
    classifier: LinearClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    classifier.eval()

    loss_function = nn.CrossEntropyLoss()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_examples = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        logits = classifier(features)
        loss = loss_function(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probabilities = torch.softmax(logits, dim=1)
    confidences, predictions = probabilities.max(dim=1)

    labels_numpy = labels.numpy()
    predictions_numpy = predictions.numpy()

    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(
            labels_numpy,
            predictions_numpy,
        ),
        "macro_f1": f1_score(
            labels_numpy,
            predictions_numpy,
            average="macro",
        ),
        "mean_maximum_confidence": confidences.mean().item(),
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "labels": labels,
    }


def train_linear_classifier(
    train_features: dict[str, Any],
    validation_features: dict[str, Any],
    device: torch.device,
    batch_size: int = 64,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 6304,
) -> tuple[LinearClassifier, list[dict[str, float]], int]:
    feature_dim = train_features["features"].shape[1]

    classifier = LinearClassifier(
        feature_dim=feature_dim,
        number_of_classes=10,
    ).to(device)

    train_dataloader = make_feature_dataloader(
        feature_data=train_features,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    validation_dataloader = make_feature_dataloader(
        feature_data=validation_features,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_validation_accuracy = -np.inf
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        classifier.train()

        training_loss = 0.0
        training_correct = 0
        training_examples = 0

        for features, labels in train_dataloader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = classifier(features)
            loss = loss_function(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size_current = labels.shape[0]
            training_loss += loss.item() * batch_size_current
            training_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            training_examples += batch_size_current

        validation_results = evaluate_classifier(
            classifier=classifier,
            dataloader=validation_dataloader,
            device=device,
        )

        epoch_results = {
            "epoch": epoch,
            "training_loss": training_loss / training_examples,
            "training_accuracy": (
                training_correct / training_examples
            ),
            "validation_loss": validation_results["loss"],
            "validation_accuracy": validation_results["accuracy"],
            "validation_macro_f1": validation_results["macro_f1"],
        }

        history.append(epoch_results)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss: {epoch_results['training_loss']:.4f} | "
            f"train acc: {epoch_results['training_accuracy']:.4f} | "
            f"val loss: {epoch_results['validation_loss']:.4f} | "
            f"val acc: {epoch_results['validation_accuracy']:.4f}"
        )

        current_accuracy = validation_results["accuracy"]

        if current_accuracy > best_validation_accuracy:
            best_validation_accuracy = current_accuracy
            best_epoch = epoch
            best_state = deepcopy(classifier.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping after epoch {epoch}. "
                f"Best epoch: {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("No classifier checkpoint was selected.")

    classifier.load_state_dict(best_state)

    return classifier, history, best_epoch