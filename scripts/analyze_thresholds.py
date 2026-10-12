"""Analyze the baseline classifier's validation thresholds."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "tables" / "baseline_validation_predictions.csv"
OUTPUT_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95,
]


def load_predictions():
    with INPUT_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    y_true = np.array(
        [int(row["true_label"]) for row in rows],
        dtype=np.int64,
    )
    y_prob = np.array(
        [float(row["tb_probability"]) for row in rows],
        dtype=np.float64,
    )

    if len(rows) != 1800:
        raise ValueError(f"Expected 1800 predictions, got {len(rows)}")

    if not np.isfinite(y_prob).all():
        raise ValueError("Probabilities contain non-finite values.")

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must lie between 0 and 1.")

    return rows, y_true, y_prob


def calculate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.int64)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    _, y_true, y_prob = load_predictions()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        calculate_metrics(y_true, y_prob, threshold)
        for threshold in THRESHOLDS
    ]

    output_csv = OUTPUT_DIR / "baseline_threshold_analysis.csv"

    with output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nTHRESHOLD ANALYSIS")
    print(
        "Threshold | Sensitivity | Specificity | Precision | "
        "F1 | Balanced Acc. | FN | FP"
    )

    for result in results:
        print(
            f"{result['threshold']:9.2f} | "
            f"{result['sensitivity']:11.3f} | "
            f"{result['specificity']:11.3f} | "
            f"{result['precision']:9.3f} | "
            f"{result['f1']:5.3f} | "
            f"{result['balanced_accuracy']:13.3f} | "
            f"{result['fn']:3d} | {result['fp']:3d}"
        )

    # Plot sensitivity and specificity against the threshold.
    thresholds = [row["threshold"] for row in results]
    sensitivity = [row["sensitivity"] for row in results]
    specificity = [row["specificity"] for row in results]

    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, sensitivity, marker="o", label="Sensitivity")
    plt.plot(thresholds, specificity, marker="s", label="Specificity")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("TB-Sentry Baseline: Validation Threshold Analysis")
    plt.ylim(0, 1.02)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figure_path = FIGURE_DIR / "baseline_threshold_analysis.png"
    plt.savefig(figure_path, dpi=160)
    plt.close()

    print("\nSaved threshold table:", output_csv)
    print("Saved plot:", figure_path)


if __name__ == "__main__":
    main()