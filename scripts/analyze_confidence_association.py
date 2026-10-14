"""Explore associations between TB confidence and Grad-CAM metrics.

This is exploratory analysis on TB-labelled validation images with
localization boxes. It does not establish clinical utility or
independent predictive value.
"""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    ROOT
    / "results"
    / "spatial_consistency"
    / "per_image_metrics.csv"
)

OUTPUT_DIR = ROOT / "results" / "spatial_consistency"

METRICS = (
    "box_coverage_union",
    "heatmap_precision_union",
    "mean_individual_box_coverage",
)

BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 42


def rank_values(values):
    """Calculate average ranks, including for tied values."""
    values = np.asarray(values, dtype=float)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    ranks = np.empty(len(values), dtype=float)

    start = 0

    while start < len(values):
        end = start + 1

        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        # Ranks are one-based; tied values receive average rank.
        average_rank = ((start + 1) + end) / 2
        ranks[order[start:end]] = average_rank

        start = end

    return ranks


def pearson_correlation(x, y):
    """Calculate Pearson correlation, handling constant arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return float("nan")

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(x, y):
    """Calculate Spearman correlation using average ranks."""
    return pearson_correlation(
        rank_values(x),
        rank_values(y),
    )


def bootstrap_spearman(x, y, rng):
    """Estimate a percentile bootstrap confidence interval."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    observed = spearman_correlation(x, y)

    if not np.isfinite(observed):
        return {
            "estimate": None,
            "ci_95_lower": None,
            "ci_95_upper": None,
        }

    bootstrap_values = []
    n = len(x)

    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, n, size=n)

        value = spearman_correlation(
            x[indices],
            y[indices],
        )

        if np.isfinite(value):
            bootstrap_values.append(value)

    if not bootstrap_values:
        return {
            "estimate": observed,
            "ci_95_lower": None,
            "ci_95_upper": None,
        }

    lower, upper = np.percentile(
        bootstrap_values,
        [2.5, 97.5],
    )

    return {
        "estimate": observed,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
    }


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError("The input CSV contains no records.")

    confidence = np.array(
        [float(row["tb_probability"]) for row in rows],
        dtype=float,
    )

    if not np.all(np.isfinite(confidence)):
        raise ValueError("Non-finite TB probabilities were found.")

    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("TB probabilities must be between 0 and 1.")

    rng = np.random.default_rng(RANDOM_SEED)

    results = {
        "experiment": (
            "Exploratory association between TB confidence "
            "and Grad-CAM spatial-consistency metrics"
        ),
        "input_file": str(INPUT_CSV.relative_to(ROOT)),
        "images_evaluated": len(rows),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": RANDOM_SEED,
        "results": {},
        "limitations": [
            "The analysis uses only TB-labelled validation images with localization boxes.",
            "Spearman correlation measures association, not causation.",
            "Association with confidence does not establish incremental predictive value.",
            "The bootstrap interval assumes the sampled images are appropriate independent sampling units; patient-level independence has not been established.",
            "Repeated threshold exploration and other analyses should be considered when interpreting the findings.",
            "Bounding-box overlap does not establish explanation faithfulness or clinical correctness.",
        ],
    }

    print("EXPLORATORY CONFIDENCE ASSOCIATION")
    print("Images:", len(rows))
    print("Bootstrap replicates:", BOOTSTRAP_REPLICATES)
    print()

    print(
        f"{'Metric':<35}"
        f"{'Spearman rho':>15}"
        f"{'95% CI':>29}"
    )

    for metric in METRICS:
        values = np.array(
            [float(row[metric]) for row in rows],
            dtype=float,
        )

        valid = np.isfinite(confidence) & np.isfinite(values)

        x = confidence[valid]
        y = values[valid]

        correlation = bootstrap_spearman(x, y, rng)

        results["results"][metric] = {
            "n": int(len(x)),
            "spearman": correlation,
        }

        estimate = correlation["estimate"]
        lower = correlation["ci_95_lower"]
        upper = correlation["ci_95_upper"]

        if estimate is None:
            print(f"{metric:<35}Insufficient variation")
            continue

        print(
            f"{metric:<35}"
            f"{estimate:>15.4f}"
            f"{f'[{lower:.4f}, {upper:.4f}]':>29}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "confidence_association.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print()
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()