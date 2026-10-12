"""Export baseline validation predictions for threshold analysis."""

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.data.dataset import TBX11KDataset


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data" / "interim" / "tbx11k_manifest.csv"
CHECKPOINT = ROOT / "models" / "checkpoints" / "densenet121_baseline.pt"

OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_FILE = OUTPUT_DIR / "baseline_validation_predictions.csv"

BATCH_SIZE = 4


def collate_samples(samples):
    """Batch images and labels while preserving sample metadata."""
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "label": torch.stack([sample["label"] for sample in samples]),
        "image_path": [sample["image_path"] for sample in samples],
    }


def main():
    device = torch.device("cpu")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    dataset = TBX11KDataset(
        manifest_path=MANIFEST,
        split="val",
        image_size=224,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )

    model = build_densenet121(
        pretrained=False,
        freeze_backbone=True,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exported = 0

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "image_path",
            "true_label",
            "tb_probability",
            "predicted_label_at_0.5",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"]

                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)[:, 1]

                for image_path, label, probability in zip(
                    batch["image_path"],
                    labels.tolist(),
                    probabilities.tolist(),
                ):
                    writer.writerow({
                        "image_path": image_path,
                        "true_label": label,
                        "tb_probability": f"{probability:.10f}",
                        "predicted_label_at_0.5": int(probability >= 0.5),
                    })

                    exported += 1

    print("Validation predictions exported successfully.")
    print("Checkpoint:", CHECKPOINT)
    print("Validation examples:", exported)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()