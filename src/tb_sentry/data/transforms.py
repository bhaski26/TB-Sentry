"""Image transformation utilities for TB-Sentry."""

from torchvision import transforms


def build_evaluation_transform(image_size: int = 224):
    """Build deterministic preprocessing for evaluation."""

    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )