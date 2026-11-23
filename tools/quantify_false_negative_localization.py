"""
Quantitative localization analysis of selected TB false negatives
using the TB-Sentry DenseNet-121 and Grad-CAM implementation.

Method:
1. Load the existing baseline checkpoint.
2. Select five previously identified false-negative TB images.
3. Generate TB-class Grad-CAM heatmaps.
4. Resize heatmaps using floating-point bilinear interpolation.
5. Compare heatmap activation with annotated TB bounding boxes.
6. Calculate activation, precision, and coverage metrics.

This analysis is exploratory and does not constitute clinical validation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.data.transforms import build_evaluation_transform
from tb_sentry.explainability.grad_cam import GradCAM


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw" / "tbx11k" / "TBX11K"

IMAGE_DIR = DATA_DIR / "imgs"

ANNOTATION_FILE = (
    DATA_DIR / "annotations" / "json" / "all_val.json"
)

PREDICTIONS_FILE = (
    ROOT / "results" / "tables"
    / "baseline_validation_predictions.csv"
)

CHECKPOINT_FILE = (
    ROOT / "models" / "checkpoints"
    / "densenet121_baseline.pt"
)

OUTPUT_DIR = (
    ROOT / "results" / "false_negative_inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR / "quantitative_localization.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR / "quantitative_localization_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_IMAGES = [
    "tb0326.png",
    "tb0575.png",
    "tb0015.png",
    "tb0231.png",
    "tb0456.png",
]

THRESHOLDS = [
    0.25,
    0.50,
    0.75,
]

DEVICE = torch.device("cpu")


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    print("Building DenseNet-121...")

    model = build_densenet121(
        pretrained=False,
        freeze_backbone=False,
    )

    print("Loading baseline checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    # Handle supported checkpoint structures.
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Handle DataParallel prefixes if present.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict
    )

    model.to(DEVICE)

    model.eval()

    print("Baseline checkpoint loaded successfully.")

    return model


# ============================================================
# ANNOTATION LOADING
# ============================================================

def load_annotations():

    print("Loading TBX11K annotations...")

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

        annotations.setdefault(
            filename,
            [],
        ).append(
            annotation["bbox"]
        )

    print(
        f"Annotations loaded for "
        f"{len(annotations)} images."
    )

    return annotations


# ============================================================
# BOUNDING-BOX MASK
# ============================================================

def create_bbox_mask(
    image_size,
    boxes,
):

    width, height = image_size

    mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    for bbox in boxes:

        x, y, box_width, box_height = bbox

        x1 = max(
            0,
            int(round(x)),
        )

        y1 = max(
            0,
            int(round(y)),
        )

        x2 = min(
            width,
            int(round(x + box_width)),
        )

        y2 = min(
            height,
            int(round(y + box_height)),
        )

        if x2 > x1 and y2 > y1:

            mask[
                y1:y2,
                x1:x2
            ] = True

    return mask


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    heatmap,
    bbox_mask,
):

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

    # --------------------------------------------------------
    # Activation distribution
    # --------------------------------------------------------

    if total_activation > 0:

        activation_fraction_inside = (
            inside.sum() / total_activation
        )

    else:

        activation_fraction_inside = 0.0

    mean_inside = (
        inside.mean()
        if inside.size > 0
        else 0.0
    )

    mean_outside = (
        outside.mean()
        if outside.size > 0
        else 0.0
    )

    max_inside = (
        inside.max()
        if inside.size > 0
        else 0.0
    )

    max_outside = (
        outside.max()
        if outside.size > 0
        else 0.0
    )

    bbox_area = bbox_mask.sum()

    metrics = {

        "mean_inside": float(
            mean_inside
        ),

        "mean_outside": float(
            mean_outside
        ),

        "max_inside": float(
            max_inside
        ),

        "max_outside": float(
            max_outside
        ),

        "activation_fraction_inside": float(
            activation_fraction_inside
        ),

        "bbox_area_pixels": int(
            bbox_area
        ),
    }

    # --------------------------------------------------------
    # Threshold-based localization metrics
    # --------------------------------------------------------

    for threshold in THRESHOLDS:

        activated = (
            heatmap >= threshold
        )

        activated_area = activated.sum()

        intersection = np.logical_and(
            activated,
            bbox_mask,
        ).sum()

        # Precision:
        # Fraction of activated heatmap pixels
        # falling inside the annotated region.
        if activated_area > 0:

            precision = (
                intersection / activated_area
            )

        else:

            precision = 0.0

        # Coverage:
        # Fraction of annotated region activated.
        if bbox_area > 0:

            coverage = (
                intersection / bbox_area
            )

        else:

            coverage = 0.0

        # Fraction of image activated.
        area_fraction = (
            activated_area / heatmap.size
        )

        key = str(threshold).replace(
            ".",
            "",
        )

        metrics[
            f"precision_{key}"
        ] = float(precision)

        metrics[
            f"coverage_{key}"
        ] = float(coverage)

        metrics[
            f"area_fraction_{key}"
        ] = float(area_fraction)

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "TB-SENTRY FALSE NEGATIVE LOCALIZATION ANALYSIS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Grad-CAM setup
    # --------------------------------------------------------

    grad_cam = GradCAM(
        model=model,
        target_layer=model.features.denseblock4,
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    transform = build_evaluation_transform(
        image_size=224,
    )

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    annotations = load_annotations()

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    print("Loading validation predictions...")

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    false_negatives = predictions[
        (
            predictions["true_label"] == 1
        )
        &
        (
            predictions[
                "predicted_label_at_0.5"
            ] == 0
        )
    ]

    false_negative_lookup = {

        Path(row["image_path"]).name: row

        for _, row in false_negatives.iterrows()
    }

    print(
        f"False negatives in validation set: "
        f"{len(false_negatives)}"
    )

    print(
        f"Selected images: "
        f"{len(TARGET_IMAGES)}"
    )

    records = []

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for filename in TARGET_IMAGES:

        print("\n" + "-" * 70)

        print(
            f"Processing: {filename}"
        )

        # ----------------------------------------------------
        # Check false-negative record
        # ----------------------------------------------------

        if filename not in false_negative_lookup:

            print(
                "[WARNING] Image not found "
                "in false-negative prediction list."
            )

            continue

        prediction_row = (
            false_negative_lookup[filename]
        )

        tb_probability = float(
            prediction_row["tb_probability"]
        )

        # ----------------------------------------------------
        # Image path
        # ----------------------------------------------------

        image_path = (
            IMAGE_DIR / "tb" / filename
        )

        if not image_path.exists():

            print(
                f"[WARNING] Missing image: "
                f"{image_path}"
            )

            continue

        # ----------------------------------------------------
        # Check annotations
        # ----------------------------------------------------

        if filename not in annotations:

            print(
                "[WARNING] No annotation found."
            )

            continue

        boxes = annotations[filename]

        # ----------------------------------------------------
        # Load original image
        # ----------------------------------------------------

        original = Image.open(
            image_path
        ).convert("RGB")

        original_width, original_height = (
            original.size
        )

        # ----------------------------------------------------
        # Bounding-box mask
        # ----------------------------------------------------

        bbox_mask = create_bbox_mask(
            original.size,
            boxes,
        )

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        input_tensor = transform(
            original
        ).unsqueeze(0).to(DEVICE)

        # ====================================================
        # GENERATE GRAD-CAM
        # ====================================================

        heatmap_tensor = grad_cam.generate(
            input_tensor,
            target_class=1,
        )

        # GradCAM.generate() already returns:
        # [H, W]
        #
        # Convert to:
        # [N, C, H, W]
        # for interpolation.

        heatmap_tensor = (
            heatmap_tensor
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # ----------------------------------------------------
        # Floating-point bilinear resizing
        # ----------------------------------------------------

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

        # ====================================================
        # CALCULATE METRICS
        # ====================================================

        metrics = calculate_metrics(
            heatmap,
            bbox_mask,
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        record = {

            "image": filename,

            "tb_probability": tb_probability,

            "image_width": original_width,

            "image_height": original_height,

            **metrics,
        }

        records.append(
            record
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"TB probability: "
            f"{tb_probability:.6f}"
        )

        print(
            f"Bounding box area: "
            f"{metrics['bbox_area_pixels']}"
        )

        print(
            f"Mean activation inside: "
            f"{metrics['mean_inside']:.6f}"
        )

        print(
            f"Mean activation outside: "
            f"{metrics['mean_outside']:.6f}"
        )

        print(
            f"Activation fraction inside: "
            f"{metrics['activation_fraction_inside']:.6f}"
        )

        for threshold in THRESHOLDS:

            key = str(threshold).replace(
                ".",
                "",
            )

            print(
                f"\nThreshold {threshold}"
            )

            print(
                f"  Precision: "
                f"{metrics[f'precision_{key}']:.6f}"
            )

            print(
                f"  Coverage: "
                f"{metrics[f'coverage_{key}']:.6f}"
            )

            print(
                f"  Activated area: "
                f"{metrics[f'area_fraction_{key}']:.6f}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    if not records:

        print(
            "\nNo images were successfully analyzed."
        )

        grad_cam.close()

        return

    df = pd.DataFrame(
        records
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ========================================================
    # SUMMARY FILE
    # ========================================================

    summary_lines = []

    summary_lines.append(
        "TB-SENTRY FALSE NEGATIVE LOCALIZATION ANALYSIS"
    )

    summary_lines.append(
        "=" * 70
    )

    summary_lines.append(
        f"Images analyzed: {len(df)}"
    )

    summary_lines.append(
        "Interpolation: Floating-point bilinear"
    )

    summary_lines.append(
        "Quantization before interpolation: None"
    )

    summary_lines.append("")

    summary_lines.append(
        "AVERAGE METRICS"
    )

    summary_lines.append(
        "-" * 70
    )

    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        ).columns
    )

    for column in numeric_columns:

        summary_lines.append(
            f"{column}: "
            f"{df[column].mean():.6f}"
        )

    summary_lines.append("")

    summary_lines.append(
        "PER-IMAGE RESULTS"
    )

    summary_lines.append(
        "-" * 70
    )

    for _, row in df.iterrows():

        summary_lines.append(
            f"\n{row['image']}"
        )

        summary_lines.append(
            f"TB probability: "
            f"{row['tb_probability']:.6f}"
        )

        summary_lines.append(
            f"Mean inside: "
            f"{row['mean_inside']:.6f}"
        )

        summary_lines.append(
            f"Mean outside: "
            f"{row['mean_outside']:.6f}"
        )

        summary_lines.append(
            f"Activation fraction inside: "
            f"{row['activation_fraction_inside']:.6f}"
        )

        for threshold in THRESHOLDS:

            key = str(threshold).replace(
                ".",
                "",
            )

            summary_lines.append(
                f"Threshold {threshold}: "
                f"Precision="
                f"{row[f'precision_{key}']:.6f}, "
                f"Coverage="
                f"{row[f'coverage_{key}']:.6f}, "
                f"ActivatedArea="
                f"{row[f'area_fraction_{key}']:.6f}"
            )

    # --------------------------------------------------------
    # Write summary
    # --------------------------------------------------------

    OUTPUT_SUMMARY.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    grad_cam.close()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "LOCALIZATION ANALYSIS COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nCSV:\n{OUTPUT_CSV}"
    )

    print(
        f"\nSummary:\n{OUTPUT_SUMMARY}"
    )

    print(
        "\nAverage localization metrics:"
    )

    print(
        df[
            [
                "mean_inside",
                "mean_outside",
                "activation_fraction_inside",
            ]
        ].mean().to_string()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()