import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


INPUT_CSV = Path(
    "results/data_audit/prediction_changes.csv"
)

OUTPUT_DIR = Path("results/data_audit")

SUMMARY_JSON = OUTPUT_DIR / "probability_change_analysis.json"
CHANGED_CSV = OUTPUT_DIR / "changed_prediction_details.csv"


def load_predictions(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_predictions(INPUT_CSV)

    if not rows:
        raise ValueError("The prediction-change CSV is empty.")

    original_probabilities = np.array(
        [float(row["original_tb_probability"]) for row in rows]
    )

    new_probabilities = np.array(
        [float(row["duplicate_aware_tb_probability"]) for row in rows]
    )

    true_labels = np.array(
        [int(row["true_label"]) for row in rows]
    )

    original_predictions = np.array(
        [int(row["original_prediction"]) for row in rows]
    )

    new_predictions = np.array(
        [int(row["duplicate_aware_prediction"]) for row in rows]
    )

    changed_mask = original_predictions != new_predictions

    probability_differences = (
        new_probabilities - original_probabilities
    )

    absolute_differences = np.abs(probability_differences)

    transition_counts = Counter(
        row["transition"] for row in rows
    )

    changed_rows = [
        row for row, changed in zip(rows, changed_mask)
        if changed
    ]

    with CHANGED_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(changed_rows)

    results = {
        "number_of_images": len(rows),
        "changed_predictions": int(changed_mask.sum()),
        "unchanged_predictions": int((~changed_mask).sum()),
        "probability_change": {
            "mean_signed_change": float(
                probability_differences.mean()
            ),
            "median_signed_change": float(
                np.median(probability_differences)
            ),
            "mean_absolute_change": float(
                absolute_differences.mean()
            ),
            "median_absolute_change": float(
                np.median(absolute_differences)
            ),
            "images_with_increased_tb_probability": int(
                (probability_differences > 0).sum()
            ),
            "images_with_decreased_tb_probability": int(
                (probability_differences < 0).sum()
            ),
            "images_with_unchanged_tb_probability": int(
                (probability_differences == 0).sum()
            ),
        },
        "probability_change_by_true_class": {},
        "probability_change_by_transition": {},
        "prediction_transitions": dict(transition_counts),
    }

    for label in (0, 1):
        mask = true_labels == label
        differences = probability_differences[mask]

        results["probability_change_by_true_class"][
            str(label)
        ] = {
            "images": int(mask.sum()),
            "mean_signed_change": float(differences.mean()),
            "median_signed_change": float(np.median(differences)),
            "mean_absolute_change": float(
                np.abs(differences).mean()
            ),
        }

    for transition in sorted(transition_counts):
        mask = np.array([
            row["transition"] == transition
            for row in rows
        ])

        differences = probability_differences[mask]

        results["probability_change_by_transition"][
            transition
        ] = {
            "images": int(mask.sum()),
            "mean_signed_change": float(differences.mean()),
            "median_signed_change": float(
                np.median(differences)
            ),
            "mean_absolute_change": float(
                np.abs(differences).mean()
            ),
        }

    correlation = spearmanr(
        original_probabilities,
        new_probabilities,
    )

    results["probability_rank_association"] = {
        "spearman_rho": float(correlation.statistic),
        "p_value": float(correlation.pvalue),
    }

    with SUMMARY_JSON.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print("\nPROBABILITY CHANGE ANALYSIS")
    print("=" * 60)
    print(f"Validation images: {len(rows)}")
    print(f"Changed predictions: {changed_mask.sum()}")
    print(
        "Mean absolute probability change: "
        f"{absolute_differences.mean():.4f}"
    )
    print(
        "Median absolute probability change: "
        f"{np.median(absolute_differences):.4f}"
    )
    print(
        "Images with increased TB probability: "
        f"{(probability_differences > 0).sum()}"
    )
    print(
        "Images with decreased TB probability: "
        f"{(probability_differences < 0).sum()}"
    )

    print("\nProbability changes by actual class:")
    for label, metrics in results[
        "probability_change_by_true_class"
    ].items():
        class_name = "Non-TB" if label == "0" else "TB"
        print(
            f"{class_name}: "
            f"n={metrics['images']}, "
            f"mean signed change="
            f"{metrics['mean_signed_change']:.4f}, "
            f"mean absolute change="
            f"{metrics['mean_absolute_change']:.4f}"
        )

    print("\nPrediction transitions:")
    for transition, count in sorted(transition_counts.items()):
        print(f"{transition}: {count}")

    print(
        "\nSpearman correlation between original and "
        "duplicate-aware probabilities: "
        f"{correlation.statistic:.4f}"
    )

    print(f"\nSaved summary: {SUMMARY_JSON}")
    print(f"Saved changed predictions: {CHANGED_CSV}")


if __name__ == "__main__":
    main()