"""CIFAR-appropriate ResNet-18 with accessible intermediate features."""

from torch import nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    def __init__(self, number_of_classes=10):
        super().__init__()
        network = resnet18(weights=None)
        network.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        network.maxpool = nn.Identity()
        self.conv1, self.bn1, self.relu = network.conv1, network.bn1, network.relu
        self.layer1, self.layer2, self.layer3, self.layer4 = network.layer1, network.layer2, network.layer3, network.layer4
        self.avgpool = network.avgpool
        self.classifier = nn.Linear(512, number_of_classes)

    def forward_to_layer2(self, x):
        x = self.relu(self.bn1(self.conv1(x))); x = self.layer1(x); return self.layer2(x)

    def forward_from_layer2(self, x):
        x = self.layer3(x); x = self.layer4(x); return self.avgpool(x).flatten(1)

    def forward_features(self, x):
        return self.forward_from_layer2(self.forward_to_layer2(x))

    def forward(self, x):
        return self.classifier(self.forward_features(x))
