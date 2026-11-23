"""
TB-Sentry Fine-Tuned Model Localization Evaluation

Evaluates Grad-CAM localization against TB bounding-box annotations
for the fine-tuned DenseNet-121 checkpoint.

This is an exploratory localization analysis, not clinical validation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.data.transforms import build_evaluation_transform
from tb_sentry.explainability.grad_cam import GradCAM


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

DATA_DIR = ROOT / "data" / "raw" / "tbx11k" / "TBX11K"
IMAGE_DIR = DATA_DIR / "imgs"
ANNOTATION_FILE = DATA_DIR / "annotations" / "json" / "all_val.json"

CHECKPOINT_FILE = (
    ROOT
    / "experiments"
    / "finetuned_densenet121_seed42"
    / "best_model.pt"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "finetuned_localization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "finetuned_localization_metrics.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "finetuned_localization_summary.txt"

THRESHOLDS = [0.25, 0.50, 0.75]


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    print("Building DenseNet-121...")

    model = build_densenet121(
        pretrained=False,
        freeze_backbone=False,
    )

    print("Loading fine-tuned checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)

    model.to(DEVICE)
    model.eval()

    print("Fine-tuned checkpoint loaded successfully.")

    return model


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

def load_annotations():

    print("Loading validation annotations...")

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    image_id_to_filename = {
        image["id"]: Path(image["file_name"]).name
        for image in data["images"]
    }

    annotations = {}

    for annotation in data["annotations"]:

        image_id = annotation["image_id"]

        filename = image_id_to_filename.get(image_id)

        if filename is None:
            continue

        annotations.setdefault(filename, []).append(
            annotation["bbox"]
        )

    print(f"Annotated images: {len(annotations)}")
    print(f"Bounding boxes: {sum(len(v) for v in annotations.values())}")

    return annotations


# ============================================================
# BOUNDING-BOX MASK
# ============================================================

def create_bbox_mask(image_size, boxes):

    width, height = image_size

    mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    for bbox in boxes:

        x, y, box_width, box_height = bbox

        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))

        x2 = min(
            width,
            int(round(x + box_width)),
        )

        y2 = min(
            height,
            int(round(y + box_height)),
        )

        if x2 > x1 and y2 > y1:

            mask[y1:y2, x1:x2] = True

    return mask


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(heatmap, bbox_mask):

    heatmap = np.asarray(
        heatmap,
        dtype=np.float64,
    )

    bbox_mask = np.asarray(
        bbox_mask,
        dtype=bool,
    )

    inside = heatmap[bbox_mask]
    outside = heatmap[~bbox_mask]

    total_activation = heatmap.sum()

    if total_activation > 0:

        activation_fraction_inside = (
            inside.sum() / total_activation
        )

    else:

        activation_fraction_inside = 0.0

    mean_inside = inside.mean() if inside.size else 0.0
    mean_outside = outside.mean() if outside.size else 0.0

    max_inside = inside.max() if inside.size else 0.0
    max_outside = outside.max() if outside.size else 0.0

    bbox_area = bbox_mask.sum()

    metrics = {
        "mean_inside": float(mean_inside),
        "mean_outside": float(mean_outside),
        "max_inside": float(max_inside),
        "max_outside": float(max_outside),
        "activation_fraction_inside": float(
            activation_fraction_inside
        ),
        "bbox_area_pixels": int(bbox_area),
    }

    for threshold in THRESHOLDS:

        activated = heatmap >= threshold

        activated_area = activated.sum()

        intersection = np.logical_and(
            activated,
            bbox_mask,
        ).sum()

        precision = (
            intersection / activated_area
            if activated_area > 0
            else 0.0
        )

        coverage = (
            intersection / bbox_area
            if bbox_area > 0
            else 0.0
        )

        area_fraction = (
            activated_area / heatmap.size
        )

        key = str(threshold).replace(".", "")

        metrics[f"precision_{key}"] = float(precision)
        metrics[f"coverage_{key}"] = float(coverage)
        metrics[f"area_fraction_{key}"] = float(area_fraction)

    return metrics


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("TB-SENTRY FINE-TUNED LOCALIZATION EVALUATION")
    print("=" * 70)

    model = load_model()

    grad_cam = GradCAM(
        model=model,
        target_layer=model.features.denseblock4,
    )

    transform = build_evaluation_transform(
        image_size=224,
    )

    annotations = load_annotations()

    records = []

    total_images = len(annotations)

    print(f"\nStarting evaluation on {total_images} annotated images.")

    # ========================================================
    # PROCESS ALL ANNOTATED IMAGES
    # ========================================================

    for index, (filename, boxes) in enumerate(annotations.items(), start=1):

        image_path = IMAGE_DIR / "tb" / filename

        if not image_path.exists():

            print(f"[SKIP] Missing image: {filename}")
            continue

        try:

            original = Image.open(
                image_path
            ).convert("RGB")

            original_width, original_height = original.size

            bbox_mask = create_bbox_mask(
                original.size,
                boxes,
            )

            input_tensor = transform(
                original
            ).unsqueeze(0).to(DEVICE)

            # ------------------------------------------------
            # Generate Grad-CAM
            # ------------------------------------------------

            heatmap_tensor = grad_cam.generate(
                input_tensor,
                target_class=1,
            )

            heatmap_tensor = (
                heatmap_tensor
                .unsqueeze(0)
                .unsqueeze(0)
            )

            heatmap_tensor = F.interpolate(
                heatmap_tensor,
                size=(
                    original_height,
                    original_width,
                ),
                mode="bilinear",
                align_corners=False,
            )

            heatmap = (
                heatmap_tensor
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )

            heatmap = np.nan_to_num(
                heatmap,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # ------------------------------------------------
            # Calculate localization metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                heatmap,
                bbox_mask,
            )

            record = {
                "image": filename,
                "image_width": original_width,
                "image_height": original_height,
                "num_boxes": len(boxes),
                **metrics,
            }

            records.append(record)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if index % 10 == 0 or index == total_images:

                print(
                    f"Processed {index}/{total_images} "
                    f"| Successfully evaluated: {len(records)}"
                )

        except Exception as error:

            print(
                f"[ERROR] {filename}: {error}"
            )

    grad_cam.close()

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    if not records:

        print("\nNo images successfully evaluated.")
        return

    df = pd.DataFrame(records)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_lines = []

    summary_lines.append(
        "TB-SENTRY FINE-TUNED LOCALIZATION EVALUATION"
    )

    summary_lines.append("=" * 70)

    summary_lines.append(
        f"Images successfully analyzed: {len(df)}"
    )

    summary_lines.append(
        "Model: Fine-tuned DenseNet-121 (Epoch 1)"
    )

    summary_lines.append(
        "Interpolation: Floating-point bilinear"
    )

    summary_lines.append(
        "Thresholds: 0.25, 0.50, 0.75"
    )

    summary_lines.append("")

    summary_lines.append("AVERAGE METRICS")
    summary_lines.append("-" * 70)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        summary_lines.append(
            f"{column}: {df[column].mean():.6f}"
        )

    summary_lines.append("")
    summary_lines.append("MEDIAN METRICS")
    summary_lines.append("-" * 70)

    for column in numeric_columns:

        summary_lines.append(
            f"{column}: {df[column].median():.6f}"
        )

    OUTPUT_SUMMARY.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # ========================================================
    # DISPLAY FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\nImages analyzed: {len(df)}")

    print("\nMean metrics:")

    print(
        df[
            [
                "mean_inside",
                "mean_outside",
                "activation_fraction_inside",
                "precision_025",
                "coverage_025",
                "precision_050",
                "coverage_050",
                "precision_075",
                "coverage_075",
            ]
        ].mean().to_string()
    )

    print(f"\nCSV saved to:\n{OUTPUT_CSV}")

    print(f"\nSummary saved to:\n{OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()