"""Train and evaluate a frozen-backbone DenseNet-121 baseline."""

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
from torchvision.models import DenseNet121_Weights

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.data.dataset import TBX11KDataset


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest_deduplicated.csv"
)
RESULTS = ROOT / "results"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"

BATCH_SIZE = 4
EPOCHS = 1
SEED = 42
LEARNING_RATE = 1e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_samples(samples):
    """Batch tensors and preserve variable-length localization metadata."""
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
    model.eval()

    total_loss = 0.0
    total_examples = 0
    labels_all = []
    probabilities_all = []

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

    y_true = np.asarray(labels_all, dtype=np.int64)
    y_prob = np.asarray(probabilities_all, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    metrics = {
        "validation_loss": total_loss / total_examples,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": 0.5,
        "validation_examples": int(len(y_true)),
    }

    return metrics


def save_metrics(metrics: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS / "duplicate_aware_metrics.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    csv_path = RESULTS / "duplicate_aware_metrics.csv"
    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main():
    set_seed(SEED)

    device = torch.device("cpu")
    print("Device:", device)
    print("Training manifest:", MANIFEST)

    train_dataset = TBX11KDataset(
        MANIFEST, split="train", image_size=224
    )
    val_dataset = TBX11KDataset(
        MANIFEST, split="val", image_size=224
    )

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)

    train_labels = [
        int(row["binary_label"]) for row in train_dataset.records
    ]
    counts = np.bincount(train_labels, minlength=2)

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))
    print("Training class counts [non-TB, TB]:", counts.tolist())

    model = build_densenet121(
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    # Inverse-frequency weights; the mean is normalized to 1.
    class_weights = len(train_labels) / (
        2.0 * np.maximum(counts, 1)
    )
    weights = torch.tensor(class_weights, dtype=torch.float32)
    weights = weights / weights.mean()
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "densenet121_duplicate_aware.pt"

    best_pr_auc = -1.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        processed = 0

        for batch_index, batch in enumerate(train_loader, start=1):
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

        train_loss = running_loss / processed
        metrics = evaluate(model, val_loader, criterion, device)
        metrics["epoch"] = epoch + 1
        metrics["training_loss"] = train_loss
        metrics["seed"] = SEED
        metrics["batch_size"] = BATCH_SIZE
        metrics["learning_rate"] = LEARNING_RATE
        metrics["class_weights"] = weights.cpu().tolist()

        print(f"\nEpoch {epoch + 1} results")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # PR-AUC is used for checkpoint selection because TB is the
        # minority class. With one epoch, this is just the first checkpoint.
        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "metrics": metrics,
                    "seed": SEED,
                },
                checkpoint_path,
            )
            print("Checkpoint saved:", checkpoint_path)

        save_metrics(metrics)

    print("\nDUPLICATE-AWARE EXPERIMENT COMPLETED")
    print("Metrics:", RESULTS / "duplicate_aware_metrics.json")


if __name__ == "__main__":
    main()