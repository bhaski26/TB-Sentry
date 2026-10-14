"""Compare class-specific Grad-CAM maps for the same chest X-ray."""

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

CHECKPOINT = ROOT / "models" / "checkpoints" / "densenet121_baseline.pt"
IMAGE_PATH = ROOT / "data" / "raw" / "tbx11k" / "TBX11K" / "imgs" / "tb" / "tb1132.png"
OUTPUT_DIR = ROOT / "results" / "figures" / "grad_cam"

IMAGE_SIZE = 224

CLASS_NAMES = {
    0: "Non-TB-labelled (class 0)",
    1: "TB-labelled (class 1)",
}


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    device = torch.device("cpu")

    # Load the same trained model used in the baseline experiment.
    model = build_densenet121(
        pretrained=False,
        freeze_backbone=True,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_evaluation_transform(image_size=IMAGE_SIZE)

    with Image.open(IMAGE_PATH) as image_file:
        image = image_file.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    target_layer = model.features.denseblock4
    heatmaps = {}

    with GradCAM(model, target_layer) as grad_cam:
        # Record the model's prediction independently of Grad-CAM.
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        print("Image:", IMAGE_PATH)
        print("Non-TB probability:", f"{probabilities[0].item():.6f}")
        print("TB probability:", f"{probabilities[1].item():.6f}")
        print()

        # Explain each output class using the exact same image.
        for class_index in (0, 1):
            heatmap = grad_cam.generate(
                input_tensor,
                target_class=class_index,
            )
            heatmaps[class_index] = heatmap.cpu().numpy()

            print(
                f"Generated Grad-CAM for {CLASS_NAMES[class_index]} "
                f"| min={heatmaps[class_index].min():.6f} "
                f"| max={heatmaps[class_index].max():.6f} "
                f"| mean={heatmaps[class_index].mean():.6f}"
            )

    # Compare the normalized heatmaps quantitatively.
    non_tb_map = heatmaps[0]
    tb_map = heatmaps[1]

    absolute_difference = np.abs(non_tb_map - tb_map)
    mean_absolute_difference = float(absolute_difference.mean())

    correlation = float(
        np.corrcoef(non_tb_map.ravel(), tb_map.ravel())[0, 1]
    )

    print()
    print("Mean absolute heatmap difference:", f"{mean_absolute_difference:.6f}")
    print("Pearson heatmap correlation:", f"{correlation:.6f}")

    display_image = np.asarray(
        image.resize((IMAGE_SIZE, IMAGE_SIZE))
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    axes[0].imshow(display_image)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(heatmaps[0], cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM: non-TB")
    axes[1].axis("off")

    axes[2].imshow(heatmaps[1], cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM: TB")
    axes[2].axis("off")

    axes[3].imshow(absolute_difference, cmap="viridis")
    axes[3].set_title("|Non-TB CAM − TB CAM|")
    axes[3].axis("off")

    fig.suptitle(
        f"Class-specific Grad-CAM comparison\n"
        f"TB probability: {probabilities[1].item():.4f}",
        fontsize=12,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "tb1132_class_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Comparison saved to:", output_path)


if __name__ == "__main__":
    main()