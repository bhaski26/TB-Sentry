"""Analyze spatial consistency by classification outcome and confidence."""

import csv
import json
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "spatial_consistency"
    / "per_image_metrics.csv"
)

OUTPUT_DIR = ROOT / "results" / "spatial_consistency"

METRICS = [
    "box_coverage_union",
    "heatmap_precision_union",
    "mean_individual_box_coverage",
]

CONFIDENCE_BINS = [
    ("0.00–0.20", 0.00, 0.20),
    ("0.20–0.40", 0.20, 0.40),
    ("0.40–0.60", 0.40, 0.60),
    ("0.60–0.80", 0.60, 0.80),
    ("0.80–1.00", 0.80, 1.00),
]


def summarize(values):
    """Summarize finite metric values."""
    values = [value for value in values if value is not None]

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
        }

    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
    }


def metric_value(row, field):
    value = float(row[field])
    return value if value == value else None


def load_records():
    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        records = list(csv.DictReader(file))

    for row in records:
        row["tb_probability"] = float(row["tb_probability"])
        row["true_label"] = int(row["true_label"])
        row["predicted_label_at_0.5"] = int(
            row["predicted_label_at_0.5"]
        )

        for field in METRICS:
            row[field] = metric_value(row, field)

    return records


def classify_outcome(row):
    true_label = row["true_label"]
    predicted = row["predicted_label_at_0.5"]

    if true_label == 1 and predicted == 1:
        return "TP"
    if true_label == 1 and predicted == 0:
        return "FN"
    if true_label == 0 and predicted == 0:
        return "TN"
    return "FP"


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(INPUT_FILE)

    records = load_records()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for row in records:
        row["outcome"] = classify_outcome(row)

    # Overall and classification-outcome summaries.
    groups = [("ALL", records)]

    for outcome in ("TP", "FN", "TN", "FP"):
        groups.append(
            (
                outcome,
                [row for row in records if row["outcome"] == outcome],
            )
        )

    outcome_results = []

    print("\nSPATIAL CONSISTENCY BY CLASSIFICATION OUTCOME")
    print(
        f"{'Group':<8}{'N':>6}"
        f"{'Mean coverage':>18}{'Median coverage':>19}"
        f"{'Mean precision':>18}"
    )

    for name, group in groups:
        coverage = summarize(
            [row["box_coverage_union"] for row in group]
        )
        precision = summarize(
            [row["heatmap_precision_union"] for row in group]
        )

        result = {
            "group": name,
            "images": len(group),
            "box_coverage_union": coverage,
            "heatmap_precision_union": precision,
        }
        outcome_results.append(result)

        print(
            f"{name:<8}{len(group):>6}"
            f"{coverage['mean'] if coverage['mean'] is not None else float('nan'):>18.4f}"
            f"{coverage['median'] if coverage['median'] is not None else float('nan'):>19.4f}"
            f"{precision['mean'] if precision['mean'] is not None else float('nan'):>18.4f}"
        )

    # Confidence-bin summaries.
    confidence_results = []

    print("\nSPATIAL CONSISTENCY BY TB PROBABILITY")
    print(
        f"{'Bin':<14}{'N':>6}"
        f"{'Mean coverage':>18}{'Median coverage':>19}"
        f"{'Mean precision':>18}"
    )

    for label, lower, upper in CONFIDENCE_BINS:
        if upper == 1.00:
            group = [
                row for row in records
                if lower <= row["tb_probability"] <= upper
            ]
        else:
            group = [
                row for row in records
                if lower <= row["tb_probability"] < upper
            ]

        coverage = summarize(
            [row["box_coverage_union"] for row in group]
        )
        precision = summarize(
            [row["heatmap_precision_union"] for row in group]
        )

        result = {
            "confidence_bin": label,
            "lower_bound": lower,
            "upper_bound": upper,
            "images": len(group),
            "box_coverage_union": coverage,
            "heatmap_precision_union": precision,
        }
        confidence_results.append(result)

        print(
            f"{label:<14}{len(group):>6}"
            f"{coverage['mean'] if coverage['mean'] is not None else float('nan'):>18.4f}"
            f"{coverage['median'] if coverage['median'] is not None else float('nan'):>19.4f}"
            f"{precision['mean'] if precision['mean'] is not None else float('nan'):>18.4f}"
        )

    # Save machine-readable summaries.
    summary = {
        "images_analyzed": len(records),
        "classification_threshold": 0.5,
        "confidence_bins": [
            {
                "label": label,
                "lower": lower,
                "upper": upper,
            }
            for label, lower, upper in CONFIDENCE_BINS
        ],
        "outcome_summaries": outcome_results,
        "confidence_summaries": confidence_results,
        "limitations": [
            "Analysis is restricted to TB-labelled validation images with localization boxes.",
            "Classification outcomes use the existing threshold of 0.5.",
            "Confidence bins are descriptive and are not evidence of calibration.",
            "Group differences are exploratory and do not establish causality.",
            "Bounding-box overlap does not establish clinical correctness or explanation faithfulness.",
        ],
    }

    output_json = OUTPUT_DIR / "confidence_analysis.json"

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nSaved:", output_json)


if __name__ == "__main__":
    main()