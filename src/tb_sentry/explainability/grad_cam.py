"""Grad-CAM implementation for DenseNet-121."""

import torch
from torch import nn


class GradCAM:
    """Generate Grad-CAM heatmaps for a target class."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )
        self._backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """Return a normalized Grad-CAM heatmap for one image.

        Args:
            input_tensor: Preprocessed image with shape (1, 3, H, W).
            target_class: Class index to explain (0=non-TB, 1=TB).

        Returns:
            Heatmap with shape (H, W), normalized to [0, 1].
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                "Expected one image with shape (1, 3, H, W)."
            )

        if target_class not in (0, 1):
            raise ValueError("target_class must be 0 or 1.")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Gradients are needed with respect to the feature maps,
        # even though the pretrained backbone parameters are frozen.
        input_tensor = input_tensor.detach().requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(input_tensor)

            if logits.ndim != 2 or logits.shape != (1, 2):
                raise ValueError(
                    f"Expected logits of shape (1, 2); got {tuple(logits.shape)}."
                )

            score = logits[0, target_class]
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        # Global average pooling of gradients gives one importance
        # weight per feature channel.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of the activation maps.
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # Grad-CAM keeps positive evidence for the target class.
        cam = torch.relu(cam)

        # Resize the heatmap to the input resolution.
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        # Normalize safely, including the all-zero case.
        cam_min = cam.min()
        cam_max = cam.max()
        denominator = cam_max - cam_min

        if denominator.item() > 1e-8:
            cam = (cam - cam_min) / denominator
        else:
            cam = torch.zeros_like(cam)

        return cam.detach()

    def close(self):
        """Remove registered hooks."""
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()