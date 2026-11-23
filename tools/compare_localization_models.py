from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_PATH = Path("results/spatial_consistency/per_image_metrics.csv")

FINETUNED_PATH = Path(
    "results/finetuned_localization/finetuned_localization_metrics.csv"
)

OUTPUT_DIR = Path("results/localization_comparison")

BOOTSTRAP_ITERATIONS = 10000
PERMUTATION_ITERATIONS = 10000
SEED = 42


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def bootstrap_ci(data, iterations=BOOTSTRAP_ITERATIONS, seed=SEED):
    """Calculate mean and 95% bootstrap confidence interval."""

    rng = np.random.default_rng(seed)

    data = np.asarray(data, dtype=float)
    n = len(data)

    bootstrap_means = np.empty(iterations)

    for i in range(iterations):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    mean_value = float(np.mean(data))
    ci_low = float(np.percentile(bootstrap_means, 2.5))
    ci_high = float(np.percentile(bootstrap_means, 97.5))

    return mean_value, ci_low, ci_high


def paired_permutation_test(
    differences,
    iterations=PERMUTATION_ITERATIONS,
    seed=SEED,
):
    """Two-sided paired permutation test using sign flipping."""

    rng = np.random.default_rng(seed)

    differences = np.asarray(differences, dtype=float)
    observed_mean = np.mean(differences)

    count_extreme = 0

    for _ in range(iterations):
        signs = rng.choice([-1, 1], size=len(differences))
        permuted_mean = np.mean(signs * differences)

        if abs(permuted_mean) >= abs(observed_mean):
            count_extreme += 1

    p_value = (count_extreme + 1) / (iterations + 1)

    return float(p_value)


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_match_data():
    """Load both localization datasets and match images."""

    baseline = pd.read_csv(BASELINE_PATH)
    finetuned = pd.read_csv(FINETUNED_PATH)

    baseline["image"] = baseline["image_path"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    )

    finetuned["image"] = finetuned["image"].apply(
        lambda x: Path(str(x).replace("\\", "/")).name
    )

    if baseline["image"].duplicated().any():
        raise ValueError("Duplicate image filenames in baseline dataset.")

    if finetuned["image"].duplicated().any():
        raise ValueError("Duplicate image filenames in fine-tuned dataset.")

    baseline_images = set(baseline["image"])
    finetuned_images = set(finetuned["image"])

    baseline_only = baseline_images - finetuned_images
    finetuned_only = finetuned_images - baseline_images

    if baseline_only or finetuned_only:
        raise ValueError(
            "The image sets do not match.\n"
            f"Baseline-only images: {baseline_only}\n"
            f"Fine-tuned-only images: {finetuned_only}"
        )

    merged = baseline.merge(
        finetuned,
        on="image",
        how="inner",
        validate="one_to_one",
    )

    print(f"Matched images: {len(merged)}")

    return merged


# ============================================================
# METRIC COMPARISON
# ============================================================

def compare_metric(
    merged,
    metric_name,
    baseline_column,
    finetuned_column,
):
    """Perform paired comparison for a metric."""

    baseline_values = merged[baseline_column].astype(float).to_numpy()
    finetuned_values = merged[finetuned_column].astype(float).to_numpy()

    differences = finetuned_values - baseline_values

    baseline_mean = float(np.mean(baseline_values))
    finetuned_mean = float(np.mean(finetuned_values))

    mean_difference, ci_low, ci_high = bootstrap_ci(differences)

    median_difference = float(np.median(differences))

    improved = int(np.sum(differences > 0))
    worsened = int(np.sum(differences < 0))
    unchanged = int(np.sum(differences == 0))

    p_value = paired_permutation_test(differences)

    return {
        "metric": metric_name,
        "baseline_mean": baseline_mean,
        "finetuned_mean": finetuned_mean,
        "mean_paired_difference": mean_difference,
        "median_paired_difference": median_difference,
        "difference_ci_95_low": ci_low,
        "difference_ci_95_high": ci_high,
        "images_improved": improved,
        "images_worsened": worsened,
        "images_unchanged": unchanged,
        "improved_percent": improved / len(differences) * 100,
        "permutation_p_value": p_value,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged = load_and_match_data()

    comparisons = []

    # --------------------------------------------------------
    # Threshold 0.50
    # Baseline and fine-tuned metrics are available per image.
    # --------------------------------------------------------

    comparisons.append(
        compare_metric(
            merged,
            "Heatmap Precision @ 0.50",
            "heatmap_precision_union",
            "precision_05",
        )
    )

    comparisons.append(
        compare_metric(
            merged,
            "Bounding Box Coverage @ 0.50",
            "box_coverage_union",
            "coverage_05",
        )
    )

    results_df = pd.DataFrame(comparisons)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    comparison_csv = OUTPUT_DIR / "paired_localization_comparison.csv"
    matched_csv = OUTPUT_DIR / "matched_image_metrics.csv"
    report_txt = OUTPUT_DIR / "comparison_report.txt"

    results_df.to_csv(comparison_csv, index=False)
    merged.to_csv(matched_csv, index=False)

    # --------------------------------------------------------
    # Generate text report
    # --------------------------------------------------------

    with open(report_txt, "w", encoding="utf-8") as report:

        report.write("TB-SENTRY LOCALIZATION COMPARISON\n")
        report.write("=" * 60 + "\n\n")

        report.write(f"Matched images: {len(merged)}\n")
        report.write(f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}\n")
        report.write(f"Permutation iterations: {PERMUTATION_ITERATIONS}\n")
        report.write(f"Random seed: {SEED}\n\n")

        for _, row in results_df.iterrows():

            report.write("\n")
            report.write("=" * 60 + "\n")
            report.write(f"METRIC: {row['metric']}\n")
            report.write("=" * 60 + "\n")

            report.write(
                f"Baseline mean: "
                f"{row['baseline_mean']:.6f}\n"
            )

            report.write(
                f"Fine-tuned mean: "
                f"{row['finetuned_mean']:.6f}\n"
            )

            report.write(
                f"Mean paired difference: "
                f"{row['mean_paired_difference']:.6f}\n"
            )

            report.write(
                f"95% Bootstrap CI: "
                f"[{row['difference_ci_95_low']:.6f}, "
                f"{row['difference_ci_95_high']:.6f}]\n"
            )

            report.write(
                f"Median paired difference: "
                f"{row['median_paired_difference']:.6f}\n"
            )

            report.write(
                f"Images improved: "
                f"{row['images_improved']}\n"
            )

            report.write(
                f"Images worsened: "
                f"{row['images_worsened']}\n"
            )

            report.write(
                f"Images unchanged: "
                f"{row['images_unchanged']}\n"
            )

            report.write(
                f"Percentage improved: "
                f"{row['improved_percent']:.2f}%\n"
            )

            report.write(
                f"Permutation test p-value: "
                f"{row['permutation_p_value']:.6f}\n"
            )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("LOCALIZATION COMPARISON RESULTS")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nFiles generated:")
    print(f"  {comparison_csv}")
    print(f"  {matched_csv}")
    print(f"  {report_txt}")

    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()