"""
Visual verification of Grad-CAM localization for selected
TB false-negative validation images.

Each image produces a 5-panel diagnostic:
1. Original chest X-ray
2. Ground-truth TB bounding boxes
3. TB-class Grad-CAM
4. Grad-CAM overlay
5. Grad-CAM overlay with TB bounding boxes

This is an exploratory model diagnostic, not clinical validation.
"""

import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms.functional import to_pil_image

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.data.transforms import build_evaluation_transform
from tb_sentry.explainability.grad_cam import GradCAM


# ============================================================
# PATHS
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
    / "visualization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device("cpu")

TARGET_IMAGES = [
    "tb0326.png",
    "tb0575.png",
    "tb0015.png",
    "tb0231.png",
    "tb0456.png",
]


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    print("Building DenseNet-121...")

    model = build_densenet121(
        pretrained=False,
        freeze_backbone=False,
    )

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

    model.load_state_dict(
        cleaned_state_dict
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded.")

    return model


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

def load_annotations():

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

        filename = image_id_to_filename.get(
            annotation["image_id"]
        )

        if filename is None:
            continue

        annotations.setdefault(
            filename,
            [],
        ).append(
            annotation["bbox"]
        )

    return annotations


# ============================================================
# DRAW BOUNDING BOXES
# ============================================================

def draw_boxes(ax, boxes):

    for bbox in boxes:

        x, y, w, h = bbox

        rectangle = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor="lime",
            linewidth=2.5,
        )

        ax.add_patch(rectangle)


# ============================================================
# GENERATE VISUALIZATION
# ============================================================

def visualize_image(
    filename,
    model,
    grad_cam,
    transform,
    annotations,
    predictions,
):

    image_path = (
        IMAGE_DIR / "tb" / filename
    )

    original = Image.open(
        image_path
    ).convert("RGB")

    boxes = annotations.get(
        filename,
        [],
    )

    prediction_row = predictions[
        predictions["image_path"].apply(
            lambda p: Path(p).name == filename
        )
    ]

    if prediction_row.empty:

        print(
            f"[WARNING] Prediction not found for {filename}"
        )

        return

    tb_probability = float(
        prediction_row.iloc[0]["tb_probability"]
    )

    input_tensor = transform(
        original
    ).unsqueeze(0).to(DEVICE)

    # Generate TB-class Grad-CAM.
    heatmap_tensor = grad_cam.generate(
        input_tensor,
        target_class=1,
    )

    heatmap = (
        heatmap_tensor
        .detach()
        .cpu()
        .numpy()
    )

    # Ensure visualization heatmap matches original image size.
    heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)

    heatmap_tensor = torch.nn.functional.interpolate(
        heatmap_tensor,
        size=(original.height, original.width),
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

    # ========================================================
    # PLOT
    # ========================================================

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(25, 6),
    )

    fig.suptitle(
        f"{filename} | TB probability = {tb_probability:.4f}",
        fontsize=15,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # 1. Original
    # --------------------------------------------------------

    axes[0].imshow(original)

    axes[0].set_title(
        "Original X-ray"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # 2. Ground truth bounding boxes
    # --------------------------------------------------------

    axes[1].imshow(original)

    draw_boxes(
        axes[1],
        boxes,
    )

    axes[1].set_title(
        "Annotated TB Bounding Boxes"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # 3. Grad-CAM
    # --------------------------------------------------------

    axes[2].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "TB-Class Grad-CAM"
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # 4. Grad-CAM overlay
    # --------------------------------------------------------

    axes[3].imshow(original)

    axes[3].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )

    axes[3].set_title(
        "Grad-CAM Overlay"
    )

    axes[3].axis("off")

    # --------------------------------------------------------
    # 5. Grad-CAM + annotations
    # --------------------------------------------------------

    axes[4].imshow(original)

    axes[4].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )

    draw_boxes(
        axes[4],
        boxes,
    )

    axes[4].set_title(
        "Grad-CAM + TB Bounding Boxes"
    )

    axes[4].axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{Path(filename).stem}_localization.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FALSE NEGATIVE LOCALIZATION VISUALIZATION")
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

    import pandas as pd

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    for filename in TARGET_IMAGES:

        print(
            f"\nProcessing {filename}"
        )

        visualize_image(
            filename=filename,
            model=model,
            grad_cam=grad_cam,
            transform=transform,
            annotations=annotations,
            predictions=predictions,
        )

    grad_cam.close()

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()