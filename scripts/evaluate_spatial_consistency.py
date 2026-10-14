"""Evaluate Grad-CAM spatial consistency against TBX11K bounding boxes.

This is an exploratory localization-consistency experiment.
Bounding boxes are not pixel-perfect lesion masks, and the metrics
must not be interpreted as clinical correctness.
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.data.transforms import build_evaluation_transform
from tb_sentry.explainability.grad_cam import GradCAM


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data" / "interim" / "tbx11k_manifest.csv"
CHECKPOINT = ROOT / "models" / "checkpoints" / "densenet121_baseline.pt"

OUTPUT_DIR = ROOT / "results" / "spatial_consistency"
FIGURE_DIR = OUTPUT_DIR / "figures"

IMAGE_SIZE = 224

# Predefined exploratory threshold: 50% of each heatmap's maximum.
RELATIVE_HEATMAP_THRESHOLD = 0.50

# Save a small, fixed number of visual examples.
MAX_VISUALIZATIONS = 8


def load_annotated_validation_records():
    """Load validation images that have TB localization boxes."""
    with MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        records = list(csv.DictReader(file))

    selected = []

    for row in records:
        if row["split"] != "val":
            continue

        if row["annotation_status"] != "boxes_present":
            continue

        if int(row["binary_label"]) != 1:
            continue

        boxes = json.loads(row["localization_boxes"])
        categories = json.loads(row["localization_categories"])

        if not boxes:
            continue

        if len(boxes) != len(categories):
            raise ValueError(
                f"Box/category mismatch for {row['image_path']}"
            )

        row["boxes"] = boxes
        row["categories"] = categories
        selected.append(row)

    if not selected:
        raise RuntimeError(
            "No annotated TB validation records were found."
        )

    return selected


def boxes_to_mask(boxes, original_width, original_height):
    """Convert original-image XYWH boxes into a 224x224 binary mask."""
    mask = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=bool,
    )

    scale_x = IMAGE_SIZE / original_width
    scale_y = IMAGE_SIZE / original_height

    transformed_boxes = []

    for box in boxes:
        x, y, width, height = map(float, box)

        # Scale original-image coordinates to model-input coordinates.
        x1 = x * scale_x
        y1 = y * scale_y
        x2 = (x + width) * scale_x
        y2 = (y + height) * scale_y

        # Clip to image boundaries.
        x1 = max(0.0, min(float(IMAGE_SIZE), x1))
        y1 = max(0.0, min(float(IMAGE_SIZE), y1))
        x2 = max(0.0, min(float(IMAGE_SIZE), x2))
        y2 = max(0.0, min(float(IMAGE_SIZE), y2))

        # Pixel indices covering the continuous box.
        left = max(0, min(IMAGE_SIZE, math.floor(x1)))
        top = max(0, min(IMAGE_SIZE, math.floor(y1)))
        right = max(0, min(IMAGE_SIZE, math.ceil(x2)))
        bottom = max(0, min(IMAGE_SIZE, math.ceil(y2)))

        if right <= left or bottom <= top:
            raise ValueError(f"Box became empty after scaling: {box}")

        mask[top:bottom, left:right] = True

        transformed_boxes.append(
            {
                "x1": left,
                "y1": top,
                "x2": right,
                "y2": bottom,
            }
        )

    return mask, transformed_boxes


def calculate_metrics(heatmap, box_mask, threshold=RELATIVE_HEATMAP_THRESHOLD):
    """Calculate exploratory pixel-based spatial-consistency metrics."""
    heatmap_mask = heatmap >= threshold

    intersection = int(np.logical_and(heatmap_mask, box_mask).sum())

    heatmap_pixels = int(heatmap_mask.sum())
    box_pixels = int(box_mask.sum())

    box_coverage = (
        intersection / box_pixels
        if box_pixels > 0
        else float("nan")
    )

    heatmap_precision = (
        intersection / heatmap_pixels
        if heatmap_pixels > 0
        else float("nan")
    )

    return {
        "box_coverage": box_coverage,
        "heatmap_precision": heatmap_precision,
        "intersection_pixels": intersection,
        "heatmap_pixels": heatmap_pixels,
        "box_pixels": box_pixels,
    }


def save_visualization(
    image,
    heatmap,
    box_mask,
    transformed_boxes,
    image_name,
    output_path,
):
    """Save original image, Grad-CAM, overlay, and annotated boxes."""
    display_image = np.asarray(
        image.resize((IMAGE_SIZE, IMAGE_SIZE))
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(display_image)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM for TB")
    axes[1].axis("off")

    axes[2].imshow(display_image)
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Grad-CAM overlay")
    axes[2].axis("off")

    axes[3].imshow(display_image)
    axes[3].imshow(
        box_mask,
        cmap="Greens",
        alpha=0.35,
        vmin=0,
        vmax=1,
    )

    for box in transformed_boxes:
        axes[3].plot(
            [box["x1"], box["x2"], box["x2"], box["x1"], box["x1"]],
            [box["y1"], box["y1"], box["y2"], box["y2"], box["y1"]],
            linewidth=1.5,
        )

    axes[3].set_title("Localization boxes")
    axes[3].axis("off")

    fig.suptitle(
        f"{image_name} — TB-class spatial consistency",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def summarize(values):
    """Return count, mean, and median for finite metric values."""
    finite_values = [
        float(value)
        for value in values
        if np.isfinite(value)
    ]

    if not finite_values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
        }

    return {
        "count": len(finite_values),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
    }


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    records = load_annotated_validation_records()

    print("Annotated TB validation images:", len(records))
    print(
        "Relative heatmap threshold:",
        RELATIVE_HEATMAP_THRESHOLD,
    )

    device = torch.device("cpu")

    model = build_densenet121(
        pretrained=False,
        freeze_backbone=True,
    )

    # Only load the locally generated, trusted checkpoint.
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_evaluation_transform(
        image_size=IMAGE_SIZE
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    image_results = []
    box_results = []
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

            # Use exactly the same preprocessing as validation.
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Explain the TB output class, regardless of the prediction.
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

            union_metrics = calculate_metrics(
                heatmap,
                union_mask,
            )

            # Measure each annotated box separately.
            per_box_coverages = []

            for box_index, (box, category, scaled_box) in enumerate(
                zip(
                    row["boxes"],
                    row["categories"],
                    transformed_boxes,
                )
            ):
                single_box_mask = np.zeros_like(union_mask)

                single_box_mask[
                    scaled_box["y1"]:scaled_box["y2"],
                    scaled_box["x1"]:scaled_box["x2"],
                ] = True

                metrics = calculate_metrics(
                    heatmap,
                    single_box_mask,
                )

                per_box_coverages.append(
                    metrics["box_coverage"]
                )

                category_counts[category] += 1

                box_results.append(
                    {
                        "image_path": row["image_path"],
                        "box_index": box_index,
                        "localization_category": category,
                        "box_coverage": metrics["box_coverage"],
                        "heatmap_precision": metrics["heatmap_precision"],
                        "intersection_pixels": metrics["intersection_pixels"],
                        "heatmap_pixels": metrics["heatmap_pixels"],
                        "box_pixels": metrics["box_pixels"],
                    }
                )

            with torch.no_grad():
                logits = model(input_tensor)
                tb_probability = torch.softmax(
                    logits,
                    dim=1,
                )[0, 1].item()

            image_results.append(
                {
                    "image_path": row["image_path"],
                    "source_category": row["source_category"],
                    "true_label": int(row["binary_label"]),
                    "tb_probability": tb_probability,
                    "predicted_label_at_0.5": int(
                        tb_probability >= 0.5
                    ),
                    "annotation_count": len(row["boxes"]),
                    "localization_categories": json.dumps(
                        row["categories"]
                    ),
                    "box_coverage_union": union_metrics["box_coverage"],
                    "heatmap_precision_union": union_metrics["heatmap_precision"],
                    "mean_individual_box_coverage": float(
                        np.nanmean(per_box_coverages)
                    ),
                    "median_individual_box_coverage": float(
                        np.nanmedian(per_box_coverages)
                    ),
                    "heatmap_pixels": union_metrics["heatmap_pixels"],
                    "union_box_pixels": union_metrics["box_pixels"],
                    "intersection_pixels": union_metrics["intersection_pixels"],
                }
            )

            if index < MAX_VISUALIZATIONS:
                output_path = (
                    FIGURE_DIR
                    / f"spatial_consistency_{index + 1:02d}.png"
                )

                save_visualization(
                    image=image,
                    heatmap=heatmap,
                    box_mask=union_mask,
                    transformed_boxes=transformed_boxes,
                    image_name=relative_path.name,
                    output_path=output_path,
                )

            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(
                    f"Processed {index + 1}/{len(records)} images"
                )

    image_csv = OUTPUT_DIR / "per_image_metrics.csv"
    box_csv = OUTPUT_DIR / "per_box_metrics.csv"
    summary_json = OUTPUT_DIR / "summary.json"

    image_fields = list(image_results[0].keys())
    box_fields = list(box_results[0].keys())

    with image_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=image_fields)
        writer.writeheader()
        writer.writerows(image_results)

    with box_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=box_fields)
        writer.writeheader()
        writer.writerows(box_results)

    summary = {
        "experiment": "TBX11K validation Grad-CAM spatial consistency",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "split": "val",
        "images_evaluated": len(image_results),
        "boxes_evaluated": len(box_results),
        "heatmap_target_class": 1,
        "relative_heatmap_threshold": RELATIVE_HEATMAP_THRESHOLD,
        "localization_categories": dict(category_counts),
        "box_coverage_union": summarize(
            [row["box_coverage_union"] for row in image_results]
        ),
        "heatmap_precision_union": summarize(
            [row["heatmap_precision_union"] for row in image_results]
        ),
        "mean_individual_box_coverage": summarize(
            [row["mean_individual_box_coverage"] for row in image_results]
        ),
        "median_individual_box_coverage": summarize(
            [row["median_individual_box_coverage"] for row in image_results]
        ),
        "limitations": [
            "Evaluation is restricted to validation images with TB localization boxes.",
            "Bounding boxes are coarse localization annotations, not pixel-perfect lesion masks.",
            "The relative heatmap threshold is exploratory and not clinically validated.",
            "Metrics measure spatial consistency, not explanation faithfulness or clinical correctness.",
            "The evaluation does not establish patient-level independence or external generalization.",
        ],
    }

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nSPATIAL-CONSISTENCY EVALUATION COMPLETE")
    print("Images evaluated:", len(image_results))
    print("Boxes evaluated:", len(box_results))
    print("Localization categories:", dict(category_counts))
    print(
        "Mean box coverage (union):",
        summary["box_coverage_union"]["mean"],
    )
    print(
        "Mean heatmap precision (union):",
        summary["heatmap_precision_union"]["mean"],
    )
    print("Per-image metrics:", image_csv)
    print("Per-box metrics:", box_csv)
    print("Summary:", summary_json)
    print("Visualizations:", FIGURE_DIR)


if __name__ == "__main__":
    main()