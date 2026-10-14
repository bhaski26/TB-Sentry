"""Compare original and duplicate-aware TB-Sentry baselines."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ORIGINAL = RESULTS / "baseline_metrics.json"
CORRECTED = RESULTS / "duplicate_aware_metrics.json"

METRICS = [
    "training_loss",
    "validation_loss",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "roc_auc",
    "pr_auc",
]


def load_metrics(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    original = load_metrics(ORIGINAL)
    corrected = load_metrics(CORRECTED)

    rows = []

    print("\nTB-SENTRY BASELINE COMPARISON")
    print("=" * 78)
    print(f"{'Metric':<20}{'Original':>15}{'Corrected':>15}{'Difference':>15}")
    print("-" * 78)

    for metric in METRICS:
        old_value = float(original[metric])
        new_value = float(corrected[metric])
        difference = new_value - old_value

        rows.append({
            "metric": metric,
            "original": old_value,
            "duplicate_aware": new_value,
            "difference": difference,
        })

        print(
            f"{metric:<20}"
            f"{old_value:>15.6f}"
            f"{new_value:>15.6f}"
            f"{difference:>15.6f}"
        )

    print("\nCONFUSION MATRIX COMPARISON")
    print("-" * 78)

    for metric in ("tn", "fp", "fn", "tp"):
        old_value = int(original[metric])
        new_value = int(corrected[metric])

        print(
            f"{metric.upper():<20}"
            f"{old_value:>15}"
            f"{new_value:>15}"
            f"{new_value - old_value:>15}"
        )

    output_path = RESULTS / "baseline_experiment_comparison.csv"

    with output_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "original",
                "duplicate_aware",
                "difference",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nComparison saved to:", output_path)


if __name__ == "__main__":
    main()