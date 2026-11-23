"""
TB-Sentry: DenseNet-121 Fine-Tuning Experiment

Experiment:
    Full-backbone fine-tuning using the TBX11K dataset.

Configuration:
    - DenseNet-121 (ImageNet pretrained)
    - Full backbone trainable
    - Class-weighted CrossEntropyLoss
    - Adam optimizer
    - 3 epochs
    - Batch size 4
    - Learning rate 1e-4
    - Seed 42

Outputs:
    experiments/finetuned_densenet121_seed42/
        epoch_1.pt
        epoch_2.pt
        epoch_3.pt
        best_model.pt
        training_history.csv
"""

# ============================================================
# IMPORTS
# ============================================================

import sys
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tb_sentry.classification.densenet import build_densenet121
from src.tb_sentry.data.dataset import TBX11KDataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest_near_duplicate_excluded.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "finetuned_densenet121_seed42"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# CUSTOM COLLATE FUNCTION
# ============================================================

def classification_collate(batch):
    """
    Batch only classification inputs.

    Localization boxes and other metadata have variable lengths,
    so they are intentionally excluded from classification batches.
    """

    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[:, 1]

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities,
    )

    metrics["loss"] = (
        total_loss / len(loader.dataset)
    )

    return metrics


# ============================================================
# TRAINING — ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    epoch,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):

        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        # Progress update every 100 batches
        if (batch_idx + 1) % 100 == 0:

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Elapsed: {elapsed / 60:.2f} min",
                flush=True,
            )

    return {
        "train_loss": total_loss / total,
        "train_accuracy": correct / total,
    }


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 60)
    print("TB-SENTRY — DENSENET121 FINE-TUNING EXPERIMENT")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Manifest: {MANIFEST_PATH}")

    # ========================================================
    # DATASETS
    # ========================================================

    print("\nLoading datasets...")

    train_dataset = TBX11KDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=IMAGE_SIZE,
    )

    val_dataset = TBX11KDataset(
        manifest_path=MANIFEST_PATH,
        split="val",
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=classification_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=classification_collate,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # ========================================================
    # MODEL
    # ========================================================

    print("\nInitializing DenseNet-121...")

    model = build_densenet121(
        pretrained=True,
        freeze_backbone=False,
    )

    model.to(DEVICE)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    print("\nCalculating class weights...")

    labels = np.array([
        train_dataset[i]["label"].item()
        for i in range(len(train_dataset))
    ])

    class_counts = np.bincount(
        labels,
        minlength=2,
    )

    class_weights = (
        len(labels) / (2.0 * class_counts)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    history = []

    best_f1 = -1.0
    best_epoch = None

    overall_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        print("\n" + "=" * 60)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 60)

        epoch_start = time.time()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print("\nRunning validation...")

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
        )

        epoch_time = time.time() - epoch_start

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        print("\nTRAIN RESULTS")
        print("-" * 40)

        print(
            f"Train Loss: "
            f"{train_metrics['train_loss']:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_metrics['train_accuracy']:.4f}"
        )

        print("\nVALIDATION RESULTS")
        print("-" * 40)

        for key, value in val_metrics.items():

            print(
                f"{key}: {value:.4f}"
            )

        print(
            f"\nEpoch duration: "
            f"{epoch_time / 60:.2f} minutes"
        )

        # ----------------------------------------------------
        # RECORD HISTORY
        # ----------------------------------------------------

        record = {
            "epoch": epoch,
            **train_metrics,
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            },
            "epoch_time_seconds": epoch_time,
        }

        history.append(record)

        history_df = pd.DataFrame(history)

        history_df.to_csv(
            OUTPUT_DIR / "training_history.csv",
            index=False,
        )

        # ----------------------------------------------------
        # SAVE EPOCH CHECKPOINT
        # ----------------------------------------------------

        checkpoint_path = (
            OUTPUT_DIR / f"epoch_{epoch}.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "seed": SEED,
                "metrics": val_metrics,
                "config": {
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "image_size": IMAGE_SIZE,
                    "model": "DenseNet-121",
                    "backbone_trainable": True,
                    "class_weighted_loss": True,
                    "manifest": str(MANIFEST_PATH),
                },
            },
            checkpoint_path,
        )

        print(
            f"\nCheckpoint saved: {checkpoint_path}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]
            best_epoch = epoch

            best_model_path = (
                OUTPUT_DIR / "best_model.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "seed": SEED,
                    "metrics": val_metrics,
                    "config": {
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "image_size": IMAGE_SIZE,
                        "model": "DenseNet-121",
                        "backbone_trainable": True,
                        "class_weighted_loss": True,
                        "manifest": str(MANIFEST_PATH),
                    },
                },
                best_model_path,
            )

            print(
                f"New best model saved "
                f"(Validation F1: {best_f1:.4f})"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_training_time = (
        time.time() - overall_start
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best Validation F1: {best_f1:.4f}"
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Total Training Time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    print(
        f"Output Directory: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print("\nTRAINING HISTORY")

    print(
        pd.DataFrame(history).to_string(
            index=False
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()