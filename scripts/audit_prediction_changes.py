"""Compare individual predictions from both TB-Sentry baselines."""

import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.data.dataset import TBX11KDataset


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest_deduplicated.csv"
)

ORIGINAL_PREDICTIONS = (
    ROOT / "results" / "tables"
    / "baseline_validation_predictions.csv"
)

CHECKPOINT = (
    ROOT / "models" / "checkpoints"
    / "densenet121_duplicate_aware.pt"
)

OUTPUT_DIR = ROOT / "results" / "data_audit"
NEW_PREDICTIONS = (
    ROOT / "results" / "tables"
    / "duplicate_aware_validation_predictions.csv"
)
COMPARISON_CSV = OUTPUT_DIR / "prediction_changes.csv"
SUMMARY_JSON = OUTPUT_DIR / "prediction_change_summary.json"

BATCH_SIZE = 4
THRESHOLD = 0.5


def collate_samples(samples):
    """Batch image tensors while preserving image paths."""
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "label": torch.stack([s["label"] for s in samples]),
        "image_path": [s["image_path"] for s in samples],
    }


def load_original_predictions():
    with ORIGINAL_PREDICTIONS.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    predictions = {}

    for row in rows:
        path = row["image_path"]

        if path in predictions:
            raise ValueError(
                f"Duplicate image path in original predictions: {path}"
            )

        predictions[path] = {
            "true_label": int(row["true_label"]),
            "tb_probability": float(row["tb_probability"]),
            "predicted_label": int(row["predicted_label_at_0.5"]),
        }

    return predictions


def calculate_metrics(labels, probabilities):
    labels = torch.tensor(labels).numpy()
    probabilities = torch.tensor(probabilities).numpy()
    predicted = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predicted, labels=[0, 1]
    ).ravel()

    return {
        "examples": int(len(labels)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NEW_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    print("Loading duplicate-aware validation dataset...")

    dataset = TBX11KDataset(
        MANIFEST,
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

    print("Loading duplicate-aware checkpoint...")

    model = build_densenet121(
        pretrained=False,
        freeze_backbone=True,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    new_predictions = {}

    print("Generating predictions...")

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].tolist()
            paths = batch["image_path"]

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            for path, label, probability in zip(
                paths, labels, probabilities.tolist()
            ):
                if path in new_predictions:
                    raise ValueError(
                        f"Duplicate validation path: {path}"
                    )

                new_predictions[path] = {
                    "true_label": int(label),
                    "tb_probability": float(probability),
                    "predicted_label": int(
                        probability >= THRESHOLD
                    ),
                }

            if batch_index % 100 == 0:
                print(
                    f"Processed {len(new_predictions)}"
                    f"/{len(dataset)} images"
                )

    original = load_original_predictions()

    if set(original) != set(new_predictions):
        missing_from_new = sorted(set(original) - set(new_predictions))
        missing_from_original = sorted(
            set(new_predictions) - set(original)
        )

        raise ValueError(
            "The prediction sets do not match.\n"
            f"Missing from new predictions: {missing_from_new[:5]}\n"
            f"Missing from original predictions: "
            f"{missing_from_original[:5]}"
        )

    labels = []
    old_probabilities = []
    new_probabilities = []
    transitions = Counter()
    comparison_rows = []
    new_prediction_rows = []

    for path in sorted(original):
        old = original[path]
        new = new_predictions[path]

        if old["true_label"] != new["true_label"]:
            raise ValueError(f"Label mismatch for {path}")

        label = old["true_label"]
        old_pred = old["predicted_label"]
        new_pred = new["predicted_label"]

        transition = (
            f"true_{label}_old_{old_pred}_new_{new_pred}"
        )
        transitions[transition] += 1

        labels.append(label)
        old_probabilities.append(old["tb_probability"])
        new_probabilities.append(new["tb_probability"])

        comparison_rows.append({
            "image_path": path,
            "true_label": label,
            "original_tb_probability": old["tb_probability"],
            "duplicate_aware_tb_probability": new["tb_probability"],
            "original_prediction": old_pred,
            "duplicate_aware_prediction": new_pred,
            "prediction_changed": int(old_pred != new_pred),
            "transition": transition,
        })

        new_prediction_rows.append({
            "image_path": path,
            "true_label": label,
            "tb_probability": new["tb_probability"],
            "predicted_label_at_0.5": new_pred,
        })

    with NEW_PREDICTIONS.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "true_label",
                "tb_probability",
                "predicted_label_at_0.5",
            ],
        )
        writer.writeheader()
        writer.writerows(new_prediction_rows)

    with COMPARISON_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(comparison_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    changed_count = sum(
        row["prediction_changed"] for row in comparison_rows
    )

    summary = {
        "validation_examples": len(comparison_rows),
        "classification_threshold": THRESHOLD,
        "changed_predictions": int(changed_count),
        "unchanged_predictions": int(
            len(comparison_rows) - changed_count
        ),
        "transitions": dict(transitions),
        "original_metrics_recomputed": calculate_metrics(
            labels, old_probabilities
        ),
        "duplicate_aware_metrics_recomputed": calculate_metrics(
            labels, new_probabilities
        ),
        "notes": [
            "Both models evaluated on the same image paths.",
            "Predictions matched by image_path.",
            "This comparison does not establish clinical utility.",
            "Validation contains six within-split exact-duplicate groups.",
        ],
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nPREDICTION CHANGE SUMMARY")
    print("=" * 60)
    print("Validation images:", len(comparison_rows))
    print("Changed predictions:", changed_count)
    print("Unchanged predictions:", len(comparison_rows) - changed_count)

    print("\nTransitions (actual class, old prediction -> new prediction):")
    for transition, count in sorted(transitions.items()):
        print(f"{transition}: {count}")

    print("\nRecomputed metrics:")
    print(json.dumps({
        "original": summary["original_metrics_recomputed"],
        "duplicate_aware": summary["duplicate_aware_metrics_recomputed"],
    }, indent=2))

    print("\nFiles saved:")
    print(NEW_PREDICTIONS)
    print(COMPARISON_CSV)
    print(SUMMARY_JSON)


if __name__ == "__main__":
    main()