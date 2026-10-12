"""Build a reproducible TBX11K image manifest.

This script reads the official TBX11K train/validation lists and
COCO-style annotation JSON files. It does not modify source data.

Run from the TB-Sentry repository root:
    python scripts/build_manifest.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = REPO_ROOT / "data" / "raw" / "tbx11k" / "TBX11K"
IMAGE_ROOT = DATASET_ROOT / "imgs"
LIST_ROOT = DATASET_ROOT / "lists"
JSON_ROOT = DATASET_ROOT / "annotations" / "json"

OUTPUT_PATH = (
    REPO_ROOT / "data" / "interim" / "tbx11k_manifest.csv"
)

# Initial binary classification policy.
LABEL_MAP = {
    "health": 0,
    "sick": 0,
    "tb": 1,
}

SPLIT_CONFIG = {
    "train": {
        "list_file": "TBX11K_train.txt",
        "annotation_file": "TBX11K_train.json",
        "expected_images": 6600,
    },
    "val": {
        "list_file": "TBX11K_val.txt",
        "annotation_file": "TBX11K_val.json",
        "expected_images": 1800,
    },
}

MANIFEST_FIELDS = [
    "image_path",
    "split",
    "source_category",
    "binary_label",
    "image_width",
    "image_height",
    "localization_boxes",
    "localization_categories",
    "annotation_status",
]


def load_json(path: Path) -> dict:
    """Load a JSON file and ensure it has the expected structure."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(
                f"{path.name} is missing required key: {key}"
            )

    return data


def load_image_list(path: Path) -> list[str]:
    """Read a dataset list while preserving its relative image paths."""
    with path.open("r", encoding="utf-8-sig") as file:
        paths = [
            line.strip().replace("\\", "/")
            for line in file
            if line.strip()
        ]

    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate paths found in {path.name}")

    return paths


def index_annotations(data: dict) -> dict[int, list[dict]]:
    """Group localization annotations by image ID."""
    image_ids = {
        int(image["id"])
        for image in data["images"]
    }

    categories = {
        int(category["id"]): category["name"]
        for category in data["categories"]
    }

    indexed: dict[int, list[dict]] = {
        image_id: []
        for image_id in image_ids
    }

    for annotation in data["annotations"]:
        image_id = int(annotation["image_id"])

        if image_id not in indexed:
            raise ValueError(
                f"Annotation references unknown image ID {image_id}"
            )

        category_id = int(annotation["category_id"])

        if category_id not in categories:
            raise ValueError(
                f"Unknown localization category ID {category_id}"
            )

        bbox = annotation["bbox"]

        if len(bbox) != 4:
            raise ValueError(
                f"Invalid bounding box for image ID {image_id}"
            )

        x, y, width, height = map(float, bbox)

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Non-positive bounding-box dimensions "
                f"for image ID {image_id}"
            )

        indexed[image_id].append(
            {
                "bbox_xywh": [x, y, width, height],
                "category_id": category_id,
                "category_name": categories[category_id],
            }
        )

    return indexed


def build_split_records(split: str, config: dict) -> list[dict]:
    """Build manifest records for one official split."""
    list_path = LIST_ROOT / config["list_file"]
    json_path = JSON_ROOT / config["annotation_file"]

    if not list_path.is_file():
        raise FileNotFoundError(f"Missing list file: {list_path}")

    if not json_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {json_path}")

    listed_paths = load_image_list(list_path)
    annotation_data = load_json(json_path)

    json_images = {
        image["file_name"].replace("\\", "/"): image
        for image in annotation_data["images"]
    }

    if len(json_images) != len(annotation_data["images"]):
        raise ValueError(
            f"Duplicate image file names in {json_path.name}"
        )

    indexed_boxes = index_annotations(annotation_data)

    listed_set = set(listed_paths)
    json_set = set(json_images)

    if listed_set != json_set:
        missing_json = sorted(listed_set - json_set)
        missing_list = sorted(json_set - listed_set)

        raise ValueError(
            f"List/JSON mismatch in {split} split.\n"
            f"Paths missing from JSON: {missing_json[:10]}\n"
            f"Paths missing from list: {missing_list[:10]}"
        )

    if len(listed_paths) != config["expected_images"]:
        raise ValueError(
            f"{split}: expected {config['expected_images']} images, "
            f"found {len(listed_paths)}"
        )

    records = []

    for relative_path in listed_paths:
        parts = relative_path.split("/")

        if len(parts) != 2:
            raise ValueError(
                f"Unexpected image path structure: {relative_path}"
            )

        source_category, filename = parts

        if source_category not in LABEL_MAP:
            raise ValueError(
                f"Unknown source category: {source_category}"
            )

        image_path = IMAGE_ROOT / source_category / filename

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Listed image does not exist: {image_path}"
            )

        image_info = json_images[relative_path]
        image_id = int(image_info["id"])

        boxes = indexed_boxes[image_id]

        box_coordinates = [
            box["bbox_xywh"]
            for box in boxes
        ]

        box_categories = [
            box["category_name"]
            for box in boxes
        ]

        records.append(
            {
                "image_path": image_path.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "split": split,
                "source_category": source_category,
                "binary_label": LABEL_MAP[source_category],
                "image_width": int(image_info["width"]),
                "image_height": int(image_info["height"]),
                "localization_boxes": json.dumps(
                    box_coordinates,
                    separators=(",", ":"),
                ),
                "localization_categories": json.dumps(
                    box_categories,
                    separators=(",", ":"),
                ),
                "annotation_status": (
                    "boxes_present" if boxes else "no_boxes"
                ),
            }
        )

    return records


def validate_records(records: list[dict]) -> None:
    """Check core manifest invariants before saving."""
    seen_paths = set()

    for record in records:
        path = record["image_path"]

        if path in seen_paths:
            raise ValueError(f"Duplicate manifest image path: {path}")

        seen_paths.add(path)

        category = record["source_category"]
        expected_label = LABEL_MAP[category]

        if record["binary_label"] != expected_label:
            raise ValueError(f"Incorrect label mapping for {path}")

        if record["split"] not in SPLIT_CONFIG:
            raise ValueError(f"Unexpected split for {path}")


def main() -> int:
    try:
        all_records = []

        for split, config in SPLIT_CONFIG.items():
            records = build_split_records(split, config)
            all_records.extend(records)

        validate_records(all_records)

        train_paths = {
            record["image_path"]
            for record in all_records
            if record["split"] == "train"
        }

        val_paths = {
            record["image_path"]
            for record in all_records
            if record["split"] == "val"
        }

        overlap = train_paths & val_paths

        if overlap:
            raise ValueError(
                f"Train/validation overlap detected: "
                f"{sorted(overlap)[:10]}"
            )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=MANIFEST_FIELDS,
            )
            writer.writeheader()
            writer.writerows(all_records)

        print("\nTB-Sentry dataset manifest created successfully.")
        print(f"Output: {OUTPUT_PATH.relative_to(REPO_ROOT)}")
        print(f"Total images: {len(all_records)}")

        for split in ("train", "val"):
            split_records = [
                record
                for record in all_records
                if record["split"] == split
            ]

            labels = Counter(
                record["binary_label"]
                for record in split_records
            )

            sources = Counter(
                record["source_category"]
                for record in split_records
            )

            annotation_status = Counter(
                record["annotation_status"]
                for record in split_records
            )

            print(f"\n{split.upper()} SUMMARY")
            print(f"Images: {len(split_records)}")
            print(f"Source categories: {dict(sources)}")
            print(f"Binary labels: {dict(labels)}")
            print(f"Localization status: {dict(annotation_status)}")

        print("\nChecks passed:")
        print("- Every listed image exists.")
        print("- List files match the corresponding JSON image records.")
        print("- No duplicate paths across the two splits.")
        print("- Every image has a recognized source category.")
        print("- Binary labels follow the configured mapping.")

        print(
            "\nNote: These checks do not establish patient-level "
            "independence or clinical validity."
        )

        return 0

    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"\nMANIFEST BUILD FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())