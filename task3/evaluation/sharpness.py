"""Fixed-batch normalized-ascent sharpness proxy."""

from __future__ import annotations

import random

import torch
from torch import nn

from task3.methods.sam import ascent_step, restore_parameters


def fixed_domain_batch(datasets: dict, count_per_domain: int = 32, seed: int = 6304):
    rng = random.Random(seed)
    images, labels = [], []
    for domain in datasets:
        indices = rng.sample(range(len(datasets[domain])), count_per_domain)
        for index in indices:
            sample = datasets[domain][index]
            images.append(sample["image"] if isinstance(sample, dict) else sample[0])
            labels.append(sample["label"] if isinstance(sample, dict) else sample[1])
    return torch.stack(images), torch.as_tensor(labels, dtype=torch.long)


def sharpness_proxy(backbone, classifier, images, labels, device, radius: float = 0.05):
    """Return base loss, perturbed loss, and their increase without changing weights."""

    backbone.eval()
    classifier.eval()
    images, labels = images.to(device), labels.to(device)
    parameters = list(backbone.parameters()) + list(classifier.parameters())
    for parameter in parameters:
        parameter.grad = None
    base_loss = nn.functional.cross_entropy(classifier(backbone(images)), labels)
    base_loss.backward()
    perturbations = ascent_step(parameters, radius)
    try:
        with torch.no_grad():
            perturbed_loss = nn.functional.cross_entropy(classifier(backbone(images)), labels)
    finally:
        restore_parameters(perturbations)
        for parameter in parameters:
            parameter.grad = None
    return {
        "base_loss": float(base_loss.item()),
        "perturbed_loss": float(perturbed_loss.item()),
        "loss_increase": float((perturbed_loss - base_loss).item()),
        "radius": float(radius),
    }
