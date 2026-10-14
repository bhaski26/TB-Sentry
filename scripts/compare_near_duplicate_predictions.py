import csv
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

EXACT_DUPLICATE_CSV = (
    ROOT / "results/tables/duplicate_aware_validation_predictions.csv"
)
NEAR_DUPLICATE_CSV = (
    ROOT / "results/tables/near_duplicate_excluded_validation_predictions.csv"
)

OUTPUT_CSV = ROOT / "results/data_audit/near_duplicate_prediction_comparison.csv"
OUTPUT_JSON = (
    ROOT / "results/data_audit/near_duplicate_prediction_comparison_summary.json"
)

TARGET_IMAGES = {
    "s1964.png",
    "s3669.png",
}


def load_predictions(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No prediction rows found in {path}")

    required = {"image_path", "true_label", "tb_probability"}
    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {sorted(missing)}"
        )

    prediction_column = (
        "predicted_label_at_0.5"
        if "predicted_label_at_0.5" in rows[0]
        else "predicted_label"
    )

    if prediction_column not in rows[0]:
        raise ValueError(
            f"No supported prediction column found in {path.name}"
        )

    predictions = {}

    for row in rows:
        image_path = row["image_path"].replace("\\", "/")
        image_name = image_path.rsplit("/", 1)[-1]

        predictions[image_path] = {
            "image_name": image_name,
            "true_label": int(row["true_label"]),
            "tb_probability": float(row["tb_probability"]),
            "predicted_label": int(row[prediction_column]),
        }

    return predictions


def main():
    exact = load_predictions(EXACT_DUPLICATE_CSV)
    near = load_predictions(NEAR_DUPLICATE_CSV)

    exact_paths = set(exact)
    near_paths = set(near)

    if exact_paths != near_paths:
        raise ValueError(
            "The prediction files contain different image paths. "
            f"Exact-only: {len(exact_paths - near_paths)}; "
            f"Near-only: {len(near_paths - exact_paths)}."
        )

    comparison_rows = []

    for path in sorted(exact_paths):
        old = exact[path]
        new = near[path]

        probability_change = (
            new["tb_probability"] - old["tb_probability"]
        )

        comparison_rows.append({
            "image_path": path,
            "image_name": old["image_name"],
            "true_label": old["true_label"],
            "exact_duplicate_tb_probability": old["tb_probability"],
            "near_duplicate_excluded_tb_probability": new["tb_probability"],
            "probability_change": probability_change,
            "exact_duplicate_prediction": old["predicted_label"],
            "near_duplicate_excluded_prediction": new["predicted_label"],
            "prediction_changed": (
                old["predicted_label"] != new["predicted_label"]
            ),
        })

    changed = [
        row for row in comparison_rows
        if row["prediction_changed"]
    ]

    target_rows = [
        row for row in comparison_rows
        if row["image_name"] in TARGET_IMAGES
    ]

    summary = {
        "experiment_1": "Exact-duplicate-aware training",
        "experiment_2": "Near-duplicate-excluded training",
        "validation_images_compared": len(comparison_rows),
        "predictions_changed": len(changed),
        "predictions_unchanged": len(comparison_rows) - len(changed),
        "mean_absolute_probability_change": (
            sum(abs(row["probability_change"]) for row in comparison_rows)
            / len(comparison_rows)
        ),
        "target_images": target_rows,
        "limitations": [
            "Both experiments used one seed and one training epoch.",
            "Differences cannot be attributed causally to the two exclusions.",
            "The audit does not establish patient-level independence.",
            "The two candidate pairs require interpretation as image-level "
            "near-duplicates, not proof of patient identity."
        ],
    }

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=comparison_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Near-duplicate prediction comparison complete.")
    print(f"Validation images compared: {len(comparison_rows)}")
    print(f"Predictions changed: {len(changed)}")
    print(f"Predictions unchanged: {len(comparison_rows) - len(changed)}")
    print(
        "Mean absolute probability change: "
        f"{summary['mean_absolute_probability_change']:.6f}"
    )

    print("\nTARGET IMAGE RESULTS")

    if not target_rows:
        print("Neither target image was found in the prediction files.")

    for row in target_rows:
        print(f"\nImage: {row['image_name']}")
        print(f"True label: {row['true_label']}")
        print(
            "Exact-duplicate-aware TB probability: "
            f"{row['exact_duplicate_tb_probability']:.6f}"
        )
        print(
            "Near-duplicate-excluded TB probability: "
            f"{row['near_duplicate_excluded_tb_probability']:.6f}"
        )
        print(f"Probability change: {row['probability_change']:+.6f}")
        print(
            "Exact-duplicate-aware prediction: "
            f"{row['exact_duplicate_prediction']}"
        )
        print(
            "Near-duplicate-excluded prediction: "
            f"{row['near_duplicate_excluded_prediction']}"
        )
        print(f"Prediction changed: {row['prediction_changed']}")

    print(f"\nCSV saved to: {OUTPUT_CSV}")
    print(f"Summary saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()