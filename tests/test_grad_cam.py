"""Tests for the Grad-CAM implementation."""

import pytest
import torch

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.explainability.grad_cam import GradCAM


@pytest.fixture(scope="module")
def model():
    network = build_densenet121(
        pretrained=False,
        freeze_backbone=True,
    )
    network.eval()
    return network


def test_grad_cam_output_shape_and_range(model):
    target_layer = model.features.denseblock4
    input_tensor = torch.randn(1, 3, 224, 224)

    with GradCAM(model, target_layer) as grad_cam:
        heatmap = grad_cam.generate(input_tensor, target_class=1)

    assert heatmap.shape == (224, 224)
    assert torch.isfinite(heatmap).all()
    assert heatmap.min().item() >= 0.0
    assert heatmap.max().item() <= 1.0


def test_grad_cam_rejects_batch_of_multiple_images(model):
    target_layer = model.features.denseblock4
    input_tensor = torch.randn(2, 3, 224, 224)

    with GradCAM(model, target_layer) as grad_cam:
        with pytest.raises(ValueError):
            grad_cam.generate(input_tensor, target_class=1)


def test_grad_cam_rejects_invalid_class(model):
    target_layer = model.features.denseblock4
    input_tensor = torch.randn(1, 3, 224, 224)

    with GradCAM(model, target_layer) as grad_cam:
        with pytest.raises(ValueError):
            grad_cam.generate(input_tensor, target_class=2)