"""Run a reproducible, preliminary multi-seed TB classification experiment."""

import csv
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
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

RESULTS_DIR = ROOT / "results" / "multiseed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"

SEEDS = [42, 123, 2026]
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224
THRESHOLD = 0.5


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_samples(samples):
    """Batch tensors while preserving image metadata."""
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


def make_loader(dataset, shuffle: bool, seed: int) -> DataLoader:
    """Create a data loader with an explicitly seeded generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_samples,
        generator=generator,
    )


def evaluate(model, loader, criterion, device):
    """Evaluate classification metrics at the prespecified threshold."""
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
    y_pred = (y_prob >= THRESHOLD).astype(np.int64)

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
        "average_precision": average_precision_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": THRESHOLD,
        "validation_examples": int(len(y_true)),
    }

    return metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries to a CSV file."""
    if not rows:
        raise ValueError(f"No rows to write to {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(seed, train_dataset, val_dataset, device):
    """Train one model and return its validation metrics."""
    print("\n" + "=" * 65)
    print(f"STARTING EXPERIMENT WITH SEED {seed}")
    print("=" * 65)

    set_seed(seed)

    # Construct loaders after setting the seed.
    train_loader = make_loader(
        train_dataset, shuffle=True, seed=seed
    )
    val_loader = make_loader(
        val_dataset, shuffle=False, seed=seed
    )

    train_labels = [
        int(row["binary_label"]) for row in train_dataset.records
    ]
    counts = np.bincount(train_labels, minlength=2)

    class_weights = len(train_labels) / (
        2.0 * np.maximum(counts, 1)
    )
    class_weights = class_weights / class_weights.mean()

    weights = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )

    model = build_densenet121(
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
    )

    running_loss = 0.0
    processed = 0

    model.train()

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
                f"Seed {seed} | "
                f"Batch {batch_index}/{len(train_loader)} | "
                f"Training loss: {running_loss / processed:.4f}"
            )

    train_loss = running_loss / processed
    metrics = evaluate(model, val_loader, criterion, device)

    metrics.update({
        "seed": seed,
        "epoch": EPOCHS,
        "training_loss": train_loss,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE,
        "class_weights": weights.cpu().tolist(),
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
    })

    checkpoint_path = (
        CHECKPOINT_DIR / f"densenet121_multiseed_{seed}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "metrics": metrics,
            "configuration": {
                "manifest": str(MANIFEST),
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "image_size": IMAGE_SIZE,
                "threshold": THRESHOLD,
                "pretrained": True,
                "freeze_backbone": True,
            },
        },
        checkpoint_path,
    )

    metrics["checkpoint"] = str(
        checkpoint_path.relative_to(ROOT)
    )

    seed_dir = RESULTS_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    with (seed_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print(f"\nRESULTS FOR SEED {seed}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def calculate_summary(results):
    """Calculate mean and sample standard deviation across seeds."""
    metric_names = [
        "training_loss",
        "validation_loss",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    summary = {
        "number_of_seeds": len(results),
        "seeds": SEEDS,
        "metrics": {},
    }

    for metric in metric_names:
        values = np.asarray(
            [result[metric] for result in results],
            dtype=np.float64,
        )

        summary["metrics"][metric] = {
            "mean": float(np.mean(values)),
            "sample_standard_deviation": (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
            ),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    return summary


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {MANIFEST}"
        )

    device = torch.device("cpu")

    print("TB-Sentry multi-seed experiment")
    print("Device:", device)
    print("Manifest:", MANIFEST)
    print("Seeds:", SEEDS)
    print("Training epochs per seed:", EPOCHS)

    # Load datasets once. All runs use the same manifest and image data.
    train_dataset = TBX11KDataset(
        MANIFEST, split="train", image_size=IMAGE_SIZE
    )
    val_dataset = TBX11KDataset(
        MANIFEST, split="val", image_size=IMAGE_SIZE
    )

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))

    # Enforce deterministic algorithms where supported.
    # This can raise an error if an operation has no deterministic CPU
    # implementation; we prefer a visible error over silent assumptions.
    torch.use_deterministic_algorithms(True)

    all_results = []

    for seed in SEEDS:
        result = run_experiment(
            seed, train_dataset, val_dataset, device
        )
        all_results.append(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        RESULTS_DIR / "per_seed_metrics.csv",
        all_results,
    )

    summary = calculate_summary(all_results)

    summary["configuration"] = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "architecture": "DenseNet-121",
        "pretrained_weights": "DenseNet121_Weights.DEFAULT",
        "backbone_frozen": True,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "threshold": THRESHOLD,
        "device": str(device),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary["limitations"] = [
        "Three seeds provide only a preliminary estimate of training variability.",
        "The runs do not establish patient-level generalisation.",
        "The validation set has been examined repeatedly and is not an untouched final test set.",
        "One training epoch is insufficient to establish a converged model.",
        "Mean and sample standard deviation across three seeds are descriptive, not definitive statistical evidence.",
    ]

    with (RESULTS_DIR / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    summary_rows = []
    for metric, values in summary["metrics"].items():
        summary_rows.append({
            "metric": metric,
            "mean": values["mean"],
            "sample_standard_deviation": values[
                "sample_standard_deviation"
            ],
            "minimum": values["minimum"],
            "maximum": values["maximum"],
        })

    write_csv(
        RESULTS_DIR / "summary_metrics.csv",
        summary_rows,
    )

    print("\n" + "=" * 65)
    print("MULTI-SEED EXPERIMENT COMPLETE")
    print("=" * 65)
    print("Per-seed metrics:", RESULTS_DIR / "per_seed_metrics.csv")
    print("Summary metrics:", RESULTS_DIR / "summary_metrics.csv")
    print("Full summary:", RESULTS_DIR / "summary.json")


if __name__ == "__main__":
    main()