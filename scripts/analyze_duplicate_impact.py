"""Measure the sensitivity of baseline metrics to cross-split duplicates.

This analysis evaluates the existing predictions before and after
excluding validation images identified in the exact-duplicate audit.

It does not retrain the model or establish patient-level independence.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_CSV = (
    ROOT
    / "results"
    / "tables"
    / "baseline_validation_predictions.csv"
)

DUPLICATE_REPORT_CSV = (
    ROOT
    / "results"
    / "data_audit"
    / "exact_duplicate_report.csv"
)

OUTPUT_DIR = ROOT / "results" / "data_audit"


def load_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def calculate_metrics(rows):
    """Calculate classification metrics at the existing 0.50 threshold."""
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset.")

    y_true = np.array(
        [int(row["true_label"]) for row in rows],
        dtype=int,
    )

    probabilities = np.array(
        [float(row["tb_probability"]) for row in rows],
        dtype=float,
    )

    y_pred = (probabilities >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": len(rows),
        "threshold": 0.50,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": float(
            tn / (tn + fp) if tn + fp else float("nan")
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "positive_images": int(y_true.sum()),
        "negative_images": int((y_true == 0).sum()),
    }


def main():
    for path in (PREDICTIONS_CSV, DUPLICATE_REPORT_CSV):
        if not path.is_file():
            raise FileNotFoundError(path)

    predictions = load_csv(PREDICTIONS_CSV)
    duplicate_rows = load_csv(DUPLICATE_REPORT_CSV)

    # Identify validation image paths from the duplicate audit.
    affected_validation_paths = {
        row["image_path"]
        for row in duplicate_rows
        if row["split"] == "val"
    }

    prediction_paths = {
        row["image_path"]
        for row in predictions
    }

    matched_paths = affected_validation_paths & prediction_paths
    unmatched_paths = affected_validation_paths - prediction_paths

    # Preserve all original predictions for the first evaluation.
    full_metrics = calculate_metrics(predictions)

    # Exclude only the validation paths flagged by the audit.
    filtered_predictions = [
        row
        for row in predictions
        if row["image_path"] not in affected_validation_paths
    ]

    filtered_metrics = calculate_metrics(filtered_predictions)

    # Verify that every identified validation duplicate was matched.
    report = {
        "experiment": (
            "Sensitivity of baseline validation metrics "
            "to identified exact cross-split duplicates"
        ),
        "duplicate_validation_paths_identified": len(
            affected_validation_paths
        ),
        "duplicate_validation_paths_matched": len(matched_paths),
        "duplicate_validation_paths_not_in_predictions": sorted(
            unmatched_paths
        ),
        "full_validation": full_metrics,
        "excluding_identified_validation_duplicates": filtered_metrics,
        "metric_changes_filtered_minus_full": {},
        "limitations": [
            "The model was not retrained.",
            "This is a sensitivity analysis of the existing predictions.",
            "Excluding records does not correct any effects of duplicates on model training.",
            "No exact duplicates does not establish patient-level independence.",
            "The filtered subset is not an independently collected external test set.",
            "The audit checks byte-identical files, not near-duplicates.",
        ],
    }

    for metric in (
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "average_precision",
    ):
        report["metric_changes_filtered_minus_full"][metric] = (
            filtered_metrics[metric] - full_metrics[metric]
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "duplicate_impact_analysis.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    csv_path = OUTPUT_DIR / "duplicate_impact_metrics.csv"

    metric_names = [
        "n",
        "positive_images",
        "negative_images",
        "tn",
        "fp",
        "fn",
        "tp",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["evaluation", *metric_names],
        )
        writer.writeheader()

        for name, metrics in (
            ("full_validation", full_metrics),
            (
                "excluding_identified_validation_duplicates",
                filtered_metrics,
            ),
        ):
            writer.writerow(
                {
                    "evaluation": name,
                    **{
                        key: metrics[key]
                        for key in metric_names
                    },
                }
            )

    print("\nDUPLICATE-IMPACT ANALYSIS")
    print(
        "Validation duplicate paths identified:",
        len(affected_validation_paths),
    )
    print("Matched against predictions:", len(matched_paths))
    print("Unmatched paths:", len(unmatched_paths))

    print("\nMETRIC COMPARISON")
    print(
        f"{'Metric':<22}"
        f"{'Full validation':>19}"
        f"{'Filtered validation':>22}"
        f"{'Change':>12}"
    )

    for metric in (
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "average_precision",
    ):
        print(
            f"{metric:<22}"
            f"{full_metrics[metric]:>19.4f}"
            f"{filtered_metrics[metric]:>22.4f}"
            f"{report['metric_changes_filtered_minus_full'][metric]:>12.4f}"
        )

    print("\nConfusion matrices:")
    print(
        "Full validation:",
        {
            key: full_metrics[key]
            for key in ("tn", "fp", "fn", "tp")
        },
    )
    print(
        "Filtered validation:",
        {
            key: filtered_metrics[key]
            for key in ("tn", "fp", "fn", "tp")
        },
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()