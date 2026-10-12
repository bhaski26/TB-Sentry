"""PyTorch dataset for the TB-Sentry project."""

import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TBX11KDataset(Dataset):
    """Load TBX11K images and their binary labels from a CSV manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: int = 224,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = PROJECT_ROOT
        self.split = split

        if split not in {"train", "val"}:
            raise ValueError(
                "split must be 'train' or 'val'. "
                "The test split is intentionally excluded."
            )

        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        with self.manifest_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.DictReader(file)

            required_columns = {
                "image_path",
                "split",
                "binary_label",
                "localization_boxes",
                "localization_categories",
                "annotation_status",
            }

            missing = required_columns - set(reader.fieldnames or [])

            if missing:
                raise ValueError(
                    f"Manifest is missing columns: {sorted(missing)}"
                )

            self.records = [
                row for row in reader if row["split"] == split
            ]

        if not self.records:
            raise ValueError(
                f"No records found for split '{split}' in "
                f"{self.manifest_path}"
            )

        for row in self.records:
            label = int(row["binary_label"])

            if label not in (0, 1):
                raise ValueError(
                    f"Invalid binary label {label} for "
                    f"{row['image_path']}"
                )

            boxes = json.loads(row["localization_boxes"])
            categories = json.loads(row["localization_categories"])

            if len(boxes) != len(categories):
                raise ValueError(
                    "Localization box/category count mismatch for "
                    f"{row['image_path']}"
                )

            for box in boxes:
                if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
                    raise ValueError(
                        f"Invalid localization box for "
                        f"{row['image_path']}: {box}"
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]

        image_path = (
            self.project_root / row["image_path"]
        ).resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image = self.transform(image)

        boxes = json.loads(row["localization_boxes"])
        categories = json.loads(row["localization_categories"])

        return {
            "image": image,
            "label": torch.tensor(
                int(row["binary_label"]), dtype=torch.long
            ),
            "image_path": row["image_path"],
            "source_category": row["source_category"],
            "boxes": boxes,
            "localization_categories": categories,
            "annotation_status": row["annotation_status"],
        }