"""DenseNet-121 model for binary TB classification."""

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


def build_densenet121(
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build a binary TB/non-TB DenseNet-121 classifier."""

    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)

    # Keep the new classification head trainable.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model