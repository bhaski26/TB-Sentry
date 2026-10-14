"""Train and evaluate DenseNet-121 after targeted near-duplicate exclusion."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.data.dataset import TBX11KDataset


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest_near_duplicate_excluded.csv"
)

RESULTS = ROOT / "results"
TABLES_DIR = RESULTS / "tables"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"

METRICS_JSON = RESULTS / "near_duplicate_excluded_metrics.json"
METRICS_CSV = RESULTS / "near_duplicate_excluded_metrics.csv"

PREDICTIONS_CSV = (
    TABLES_DIR
    / "near_duplicate_excluded_validation_predictions.csv"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "densenet121_near_duplicate_excluded.pt"
)

BATCH_SIZE = 4
EPOCHS = 1
SEED = 42
LEARNING_RATE = 1e-3

THRESHOLD = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_samples(samples):
    """Batch tensors and preserve image paths and localization metadata."""
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


def make_loader(dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_samples,
    )


def evaluate(model, loader, criterion, device):
    """Evaluate the model and retain per-image predictions."""
    model.eval()

    total_loss = 0.0
    total_examples = 0

    labels_all = []
    probabilities_all = []
    paths_all = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            probabilities = torch.softmax(logits, dim=1)[:, 1]

            labels_all.extend(labels.cpu().tolist())
            probabilities_all.extend(probabilities.cpu().tolist())
            paths_all.extend(batch["image_path"])

    y_true = np.asarray(labels_all, dtype=np.int64)
    y_prob = np.asarray(probabilities_all, dtype=np.float64)
    y_pred = (y_prob >= THRESHOLD).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    metrics = {
        "validation_loss": total_loss / total_examples,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true, y_prob),
        # This is average precision, not trapezoidal PR-AUC.
        "average_precision": average_precision_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": THRESHOLD,
        "validation_examples": int(len(y_true)),
    }

    predictions = [
        {
            "image_path": path,
            "true_label": int(label),
            "tb_probability": float(probability),
            "predicted_label": int(prediction),
            "threshold": THRESHOLD,
        }
        for path, label, probability, prediction in zip(
            paths_all,
            y_true,
            y_prob,
            y_pred,
        )
    ]

    return metrics, predictions


def save_metrics(metrics: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    with METRICS_JSON.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with METRICS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(metrics.keys()),
        )
        writer.writeheader()
        writer.writerow(metrics)


def save_predictions(predictions: list[dict]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if not predictions:
        raise ValueError("No validation predictions were generated.")

    with PREDICTIONS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(predictions[0].keys()),
        )
        writer.writeheader()
        writer.writerows(predictions)


def main():
    set_seed(SEED)

    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Training manifest not found: {MANIFEST}"
        )

    device = torch.device("cpu")

    print("Device:", device)
    print("Training manifest:", MANIFEST)
    print("Checkpoint:", CHECKPOINT_PATH)

    train_dataset = TBX11KDataset(
        MANIFEST,
        split="train",
        image_size=224,
    )

    val_dataset = TBX11KDataset(
        MANIFEST,
        split="val",
        image_size=224,
    )

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)

    train_labels = [
        int(row["binary_label"])
        for row in train_dataset.records
    ]

    counts = np.bincount(train_labels, minlength=2)

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))
    print("Training class counts [non-TB, TB]:", counts.tolist())

    model = build_densenet121(
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    # Preserve the inverse-frequency class-weighting strategy.
    class_weights = len(train_labels) / (
        2.0 * np.maximum(counts, 1)
    )

    weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    )

    weights = weights / weights.mean()
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    running_loss = 0.0
    processed = 0

    for epoch in range(EPOCHS):
        model.train()

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            processed += batch_size

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Training loss: {running_loss / processed:.4f}"
                )

    if processed == 0:
        raise RuntimeError("No training examples were processed.")

    training_loss = running_loss / processed

    metrics, predictions = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    metrics.update(
        {
            "experiment": "near_duplicate_excluded",
            "epoch": EPOCHS,
            "training_loss": training_loss,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "class_weights": weights.cpu().tolist(),
            "training_examples": len(train_dataset),
            "training_non_tb": int(counts[0]),
            "training_tb": int(counts[1]),
            "validation_examples": len(val_dataset),
            "threshold": THRESHOLD,
            "excluded_training_images": [
                "data/raw/tbx11k/TBX11K/imgs/sick/s1174.png",
                "data/raw/tbx11k/TBX11K/imgs/sick/s2023.png",
            ],
        }
    )

    # Save the checkpoint with the experiment's metrics.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": EPOCHS,
            "metrics": metrics,
            "seed": SEED,
        },
        CHECKPOINT_PATH,
    )

    save_metrics(metrics)
    save_predictions(predictions)

    print("\n" + "=" * 60)
    print("NEAR-DUPLICATE-EXCLUDED EXPERIMENT COMPLETED")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nCheckpoint:", CHECKPOINT_PATH)
    print("Metrics JSON:", METRICS_JSON)
    print("Metrics CSV:", METRICS_CSV)
    print("Validation predictions:", PREDICTIONS_CSV)


if __name__ == "__main__":
    main()