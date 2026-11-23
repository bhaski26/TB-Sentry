import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.explainability.grad_cam import GradCAM


ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = ROOT / "data/raw/tbx11k/TBX11K/imgs"
ANNOTATION_FILE = (
    ROOT / "data/raw/tbx11k/TBX11K/annotations/json/all_val.json"
)
CHECKPOINT = ROOT / "models/checkpoints/densenet121_baseline.pt"

OUTPUT_DIR = ROOT / "results/false_negative_inspection"

TARGET_IMAGES = [
    "tb/tb0326.png",
    "tb/tb0575.png",
    "tb/tb0015.png",
    "tb/tb0231.png",
    "tb/tb0456.png",
]

DEVICE = torch.device("cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_checkpoint():
    # Explicitly allow the NumPy globals present in this trusted
    # project checkpoint while retaining weights_only=True.
    import numpy as np

    torch.serialization.add_safe_globals([
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    ])

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    state_dict = checkpoint["model_state_dict"]

    model = build_densenet121(
        pretrained=False,
        freeze_backbone=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


def load_annotations():
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    images = {
        image["id"]: image
        for image in data["images"]
    }

    annotations = {}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        annotations.setdefault(image_id, []).append(annotation)

    records = {}

    for image in images.values():
        records[image["file_name"]] = {
            "image": image,
            "annotations": annotations.get(image["id"], []),
        }

    return records


def draw_boxes(ax, annotations):
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]

        rectangle = plt.Rectangle(
            (x, y),
            width,
            height,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )

        ax.add_patch(rectangle)


def save_visualization(
    original,
    annotations,
    tb_heatmap,
    non_tb_heatmap,
    image_name,
    probability,
):
    original_array = np.asarray(original)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_array, cmap="gray")
    draw_boxes(axes[0], annotations)
    axes[0].set_title("Original X-ray + TB annotation")

    axes[1].imshow(tb_heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM: TB class")

    axes[2].imshow(original_array, cmap="gray")
    axes[2].imshow(
        tb_heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    draw_boxes(axes[2], annotations)
    axes[2].set_title("TB Grad-CAM + annotation")

    axes[3].imshow(non_tb_heatmap, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("Grad-CAM: non-TB class")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{image_name} | Predicted TB probability: {probability:.4f}",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{Path(image_name).stem}_inspection.png"

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    print("Loading baseline model...")
    model = load_checkpoint()

    print("Loading validation annotations...")
    records = load_annotations()

    results = []

    target_layer = model.features.denseblock4

    with GradCAM(model, target_layer) as grad_cam:

        for image_name in TARGET_IMAGES:
            if image_name not in records:
                raise ValueError(
                    f"Image not found in validation annotations: {image_name}"
                )

            record = records[image_name]
            image_path = IMAGE_ROOT / image_name

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            original = Image.open(image_path).convert("RGB")
            input_tensor = TRANSFORM(original).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

            tb_probability = float(probabilities[1].item())
            non_tb_probability = float(probabilities[0].item())
            predicted_label = int(torch.argmax(probabilities).item())

            # Generate independent class-specific explanations.
            tb_heatmap = (
                grad_cam.generate(input_tensor, target_class=1)
                .cpu()
                .numpy()
            )

            non_tb_heatmap = (
                grad_cam.generate(input_tensor, target_class=0)
                .cpu()
                .numpy()
            )

            # Convert the heatmaps to the original image dimensions.
            tb_heatmap = np.asarray(
                Image.fromarray(
                    (tb_heatmap * 255).astype(np.uint8)
                ).resize(original.size, Image.Resampling.BILINEAR)
            ) / 255.0

            non_tb_heatmap = np.asarray(
                Image.fromarray(
                    (non_tb_heatmap * 255).astype(np.uint8)
                ).resize(original.size, Image.Resampling.BILINEAR)
            ) / 255.0

            annotations = record["annotations"]

            output_path = save_visualization(
                original,
                annotations,
                tb_heatmap,
                non_tb_heatmap,
                Path(image_name).name,
                tb_probability,
            )

            results.append({
                "image": image_name,
                "true_label": 1,
                "predicted_label": predicted_label,
                "tb_probability": tb_probability,
                "non_tb_probability": non_tb_probability,
                "annotation_count": len(annotations),
                "annotation_categories": ";".join(
                    str(a["category_id"]) for a in annotations
                ),
                "bounding_boxes_xywh": json.dumps([
                    a["bbox"] for a in annotations
                ]),
                "visualization": str(output_path.relative_to(ROOT)),
            })

            print(
                f"{image_name}: "
                f"TB probability={tb_probability:.6f}, "
                f"predicted={predicted_label}, "
                f"annotations={len(annotations)}"
            )

    csv_path = OUTPUT_DIR / "false_negative_inspection.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nInspection completed.")
    print("Visualizations:", OUTPUT_DIR)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()