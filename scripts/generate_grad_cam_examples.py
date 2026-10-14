"""Generate Grad-CAM visualizations for validation examples."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tb_sentry.classification.densenet import build_densenet121
from tb_sentry.data.transforms import build_evaluation_transform
from tb_sentry.explainability.grad_cam import GradCAM


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = ROOT / "models" / "checkpoints" / "densenet121_baseline.pt"
PREDICTIONS = ROOT / "results" / "tables" / "baseline_validation_predictions.csv"
OUTPUT_DIR = ROOT / "results" / "figures" / "grad_cam"

IMAGE_SIZE = 224

CLASS_NAMES = {
    0: "Non-TB-labelled",
    1: "TB-labelled",
}

OUTCOME_NAMES = {
    "TP": "True Positive",
    "FN": "False Negative",
    "TN": "True Negative",
    "FP": "False Positive",
}


def load_predictions():
    """Read saved validation predictions."""
    with PREDICTIONS.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_examples(rows):
    """Select one representative example per confusion-matrix outcome."""
    selected = {}

    for row in rows:
        true_label = int(row["true_label"])
        probability = float(row["tb_probability"])
        predicted_label = int(probability >= 0.5)

        if true_label == 1 and predicted_label == 1:
            outcome = "TP"
        elif true_label == 1 and predicted_label == 0:
            outcome = "FN"
        elif true_label == 0 and predicted_label == 0:
            outcome = "TN"
        else:
            outcome = "FP"

        # Choose an example near the decision threshold for each outcome.
        # This makes the examples informative, not necessarily representative.
        distance = abs(probability - 0.5)

        if outcome not in selected or distance < selected[outcome][0]:
            selected[outcome] = (distance, row)

    missing = {"TP", "FN", "TN", "FP"} - set(selected)
    if missing:
        raise RuntimeError(f"No validation examples found for: {sorted(missing)}")

    return {key: value[1] for key, value in selected.items()}


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not PREDICTIONS.is_file():
        raise FileNotFoundError(f"Predictions not found: {PREDICTIONS}")

    device = torch.device("cpu")

    model = build_densenet121(pretrained=False, freeze_backbone=True)

    # This checkpoint was created by our own training script.
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_evaluation_transform(image_size=IMAGE_SIZE)

    examples = select_examples(load_predictions())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_layer = model.features.denseblock4

    with GradCAM(model, target_layer) as grad_cam:
        for outcome, row in examples.items():
            image_path = Path(row["image_path"])

            if not image_path.is_absolute():
                image_path = ROOT / image_path

            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")

            input_tensor = transform(image).unsqueeze(0).to(device)

            # Explain the class actually predicted at threshold 0.5.
            with torch.no_grad():
                logits = model(input_tensor)
                probability = torch.softmax(logits, dim=1)[0, 1].item()

            predicted_label = int(probability >= 0.5)

            # Generate Grad-CAM outside no_grad().
            heatmap = grad_cam.generate(
                input_tensor,
                target_class=predicted_label,
            )

            # Prepare the RGB image for visualization.
            display_image = np.asarray(
                image.resize((IMAGE_SIZE, IMAGE_SIZE))
            )

            heatmap_array = heatmap.cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))

            axes[0].imshow(display_image)
            axes[0].set_title("Original X-ray")
            axes[0].axis("off")

            axes[1].imshow(heatmap_array, cmap="jet", vmin=0, vmax=1)
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")

            axes[2].imshow(display_image)
            axes[2].imshow(
                heatmap_array,
                cmap="jet",
                alpha=0.4,
                vmin=0,
                vmax=1,
            )
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            true_label = int(row["true_label"])

            fig.suptitle(
                f"{outcome} — {OUTCOME_NAMES[outcome]}\n"
                f"True: {CLASS_NAMES[true_label]} | "
                f"Predicted: {CLASS_NAMES[predicted_label]} | "
                f"TB probability: {probability:.3f}",
                fontsize=11,
            )

            fig.tight_layout()

            output_path = OUTPUT_DIR / f"{outcome.lower()}_grad_cam.png"
            fig.savefig(output_path, dpi=160, bbox_inches="tight")
            plt.close(fig)

            print(f"{outcome}: {image_path}")
            print(f"  TB probability: {probability:.4f}")
            print(f"  Saved: {output_path}")

    print("\nGrad-CAM example generation completed.")


if __name__ == "__main__":
    main()