"""Evaluate Grad-CAM spatial consistency at multiple thresholds.

This is an exploratory localization-consistency experiment.
Bounding boxes are coarse localization annotations, not pixel-perfect
lesion masks. Metrics must not be interpreted as clinical correctness.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.data.transforms import build_evaluation_transform
from tb_sentry.explainability.grad_cam import GradCAM

from evaluate_spatial_consistency import (
    IMAGE_SIZE,
    ROOT,
    MANIFEST,
    CHECKPOINT,
    OUTPUT_DIR,
    load_annotated_validation_records,
    boxes_to_mask,
    calculate_metrics,
    summarize,
)


# Predefined thresholds; do not select a threshold based on the results.
THRESHOLDS = (0.25, 0.50, 0.75)


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    records = load_annotated_validation_records()

    print("Annotated TB validation images:", len(records))
    print("Relative heatmap thresholds:", THRESHOLDS)

    device = torch.device("cpu")

    model = build_densenet121(
        pretrained=False,
        freeze_backbone=True,
    )

    # Load only the locally generated, trusted checkpoint.
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_evaluation_transform(
        image_size=IMAGE_SIZE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    category_counts = Counter()

    target_layer = model.features.denseblock4

    with GradCAM(model, target_layer) as grad_cam:
        for index, row in enumerate(records):
            relative_path = Path(row["image_path"])

            image_path = (
                relative_path
                if relative_path.is_absolute()
                else ROOT / relative_path
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")

            original_width, original_height = image.size

            input_tensor = transform(image).unsqueeze(0).to(device)

            # Explain the TB output class for every image,
            # irrespective of the model's predicted class.
            heatmap_tensor = grad_cam.generate(
                input_tensor,
                target_class=1,
            )

            heatmap = heatmap_tensor.cpu().numpy()

            union_mask, transformed_boxes = boxes_to_mask(
                row["boxes"],
                original_width,
                original_height,
            )

            # Evaluate each generated heatmap at all thresholds.
            for threshold in THRESHOLDS:
                union_metrics = calculate_metrics(
                    heatmap,
                    union_mask,
                    threshold=threshold,
                )

                individual_coverages = []
                individual_precisions = []

                for scaled_box in transformed_boxes:
                    single_box_mask = np.zeros_like(union_mask)

                    single_box_mask[
                        scaled_box["y1"]:scaled_box["y2"],
                        scaled_box["x1"]:scaled_box["x2"],
                    ] = True

                    box_metrics = calculate_metrics(
                        heatmap,
                        single_box_mask,
                        threshold=threshold,
                    )

                    individual_coverages.append(
                        box_metrics["box_coverage"]
                    )

                    individual_precisions.append(
                        box_metrics["heatmap_precision"]
                    )

                results.append(
                    {
                        "image_path": row["image_path"],
                        "threshold": threshold,
                        "annotation_count": len(row["boxes"]),
                        "box_coverage_union":
                            union_metrics["box_coverage"],
                        "heatmap_precision_union":
                            union_metrics["heatmap_precision"],
                        "mean_individual_box_coverage": float(
                            np.nanmean(individual_coverages)
                        ),
                        "median_individual_box_coverage": float(
                            np.nanmedian(individual_coverages)
                        ),
                        "heatmap_pixels":
                            union_metrics["heatmap_pixels"],
                        "union_box_pixels":
                            union_metrics["box_pixels"],
                        "intersection_pixels":
                            union_metrics["intersection_pixels"],
                    }
                )

            category_counts.update(row["categories"])

            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(
                    f"Processed {index + 1}/{len(records)} images"
                )

    # Save detailed results: one row per image per threshold.
    csv_path = OUTPUT_DIR / "threshold_sensitivity.csv"

    fieldnames = list(results[0].keys())

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summarize the distribution of image-level metrics.
    threshold_summaries = {}

    for threshold in THRESHOLDS:
        threshold_rows = [
            row for row in results
            if row["threshold"] == threshold
        ]

        threshold_summaries[str(threshold)] = {
            "images_evaluated": len(threshold_rows),
            "images_with_empty_heatmap": sum(
                row["heatmap_pixels"] == 0
                for row in threshold_rows
            ),
            "box_coverage_union": summarize(
                row["box_coverage_union"]
                for row in threshold_rows
            ),
            "heatmap_precision_union": summarize(
                row["heatmap_precision_union"]
                for row in threshold_rows
            ),
            "mean_individual_box_coverage": summarize(
                row["mean_individual_box_coverage"]
                for row in threshold_rows
            ),
            "median_individual_box_coverage": summarize(
                row["median_individual_box_coverage"]
                for row in threshold_rows
            ),
        }

    summary = {
        "experiment": (
            "TBX11K validation Grad-CAM threshold-sensitivity analysis"
        ),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "split": "val",
        "heatmap_target_class": 1,
        "thresholds": list(THRESHOLDS),
        "images_evaluated": len(records),
        "boxes_evaluated": sum(
            len(row["boxes"]) for row in records
        ),
        "localization_categories": dict(category_counts),
        "results_by_threshold": threshold_summaries,
        "limitations": [
            "Evaluation is restricted to TB-labelled validation images with localization boxes.",
            "Bounding boxes are coarse localization annotations, not pixel-perfect lesion masks.",
            "Thresholds are predefined exploratory settings, not clinically validated values.",
            "Metrics measure spatial consistency, not explanation faithfulness or clinical correctness.",
            "Results do not establish patient-level independence or external generalization.",
            "Threshold sensitivity alone does not establish whether spatial consistency adds value beyond model confidence.",
        ],
    }

    json_path = OUTPUT_DIR / "threshold_sensitivity.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nTHRESHOLD-SENSITIVITY ANALYSIS COMPLETE")
    print("Images evaluated:", len(records))
    print("Boxes evaluated:", summary["boxes_evaluated"])

    print("\nRESULTS BY THRESHOLD")

    print(
        f"{'Threshold':<12}"
        f"{'Mean coverage':>16}"
        f"{'Median coverage':>18}"
        f"{'Mean precision':>17}"
        f"{'Empty maps':>13}"
    )

    for threshold in THRESHOLDS:
        metrics = threshold_summaries[str(threshold)]

        print(
            f"{threshold:<12.2f}"
            f"{metrics['box_coverage_union']['mean']:>16.4f}"
            f"{metrics['box_coverage_union']['median']:>18.4f}"
            f"{metrics['heatmap_precision_union']['mean']:>17.4f}"
            f"{metrics['images_with_empty_heatmap']:>13}"
        )

    print("\nDetailed CSV:", csv_path)
    print("Summary JSON:", json_path)


if __name__ == "__main__":
    main()