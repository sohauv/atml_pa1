"""CIFAR ResNet-18 augmented with PROSER dummy classifiers."""

from __future__ import annotations

from torch import nn

from task4.models.resnet_cifar import CIFARResNet18


class PROSERResNet18(CIFARResNet18):
    """Ten known classifiers plus independently initialized dummy classifiers."""

    def __init__(self, number_of_classes: int = 10, dummy_classifier_count: int = 5):
        super().__init__(number_of_classes=number_of_classes)
        self.dummy_classifier = nn.Linear(512, dummy_classifier_count)

    def classify_features(self, features):
        return self.classifier(features), self.dummy_classifier(features)

    def forward_all(self, inputs):
        features = self.forward_features(inputs)
        known_logits, dummy_logits = self.classify_features(features)
        return known_logits, dummy_logits, features

    def forward(self, inputs):
        """Return known logits so closed-set evaluation remains unambiguous."""

        features = self.forward_features(inputs)
        return self.classifier(features)
