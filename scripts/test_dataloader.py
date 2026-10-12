"""Smoke test for the TB-Sentry DataLoader and DenseNet-121."""

import torch
from torch.utils.data import DataLoader
from torchvision.models import densenet121, DenseNet121_Weights

from tb_sentry.data.dataset import TBX11KDataset


MANIFEST = "data/interim/tbx11k_manifest.csv"
BATCH_SIZE = 4


def collate_samples(samples):
    """Batch tensors while preserving variable-length metadata."""
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "label": torch.stack([sample["label"] for sample in samples]),
        "image_path": [sample["image_path"] for sample in samples],
        "source_category": [
            sample["source_category"] for sample in samples
        ],
        "boxes": [sample["boxes"] for sample in samples],
        "localization_categories": [
            sample["localization_categories"] for sample in samples
        ],
        "annotation_status": [
            sample["annotation_status"] for sample in samples
        ],
    }


def main():
    dataset = TBX11KDataset(
        manifest_path=MANIFEST,
        split="train",
        image_size=224,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )

    batch = next(iter(loader))
    images = batch["image"]
    labels = batch["label"]

    print("Batch image shape:", tuple(images.shape))
    print("Batch label shape:", tuple(labels.shape))
    print("Labels:", labels.tolist())
    print("Image dtype:", images.dtype)
    print("Number of image paths:", len(batch["image_path"]))
    print("Localization boxes per image:", [
        len(boxes) for boxes in batch["boxes"]
    ])

    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)
    model.classifier = torch.nn.Linear(
        model.classifier.in_features,
        2,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(images)

    print("Model output shape:", tuple(outputs.shape))
    print("DataLoader and model smoke test PASSED")


if __name__ == "__main__":
    main()