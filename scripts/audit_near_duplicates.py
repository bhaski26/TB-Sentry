import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import imagehash
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data" / "interim" / "tbx11k_manifest_deduplicated.csv"

OUTPUT_DIR = ROOT / "results" / "data_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "near_duplicate_candidates.csv"
OUTPUT_JSON = OUTPUT_DIR / "near_duplicate_audit_summary.json"

# Candidate threshold, not a universal definition of a duplicate.
# Lower Hamming distance means more similar perceptual hashes.
HASH_DISTANCE_THRESHOLD = 6

HASH_SIZE = 8


def resolve_image_path(value):
    """Resolve a manifest image path against the repository root."""
    path = Path(value)

    if path.is_absolute():
        return path

    return ROOT / path


def compute_hash(path):
    """Calculate a perceptual hash after converting the image to grayscale."""
    with Image.open(path) as image:
        image = image.convert("L")
        return imagehash.phash(image, hash_size=HASH_SIZE)


def main():
    records = []
    missing_files = []

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {"image_path", "split", "binary_label"}
        missing_columns = required - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            path = resolve_image_path(row["image_path"])

            if not path.is_file():
                missing_files.append(str(path))
                continue

            try:
                perceptual_hash = compute_hash(path)
            except Exception as exc:
                print(f"Could not hash {path}: {exc}")
                continue

            records.append(
                {
                    "image_path": row["image_path"],
                    "split": row["split"],
                    "binary_label": row["binary_label"],
                    "hash": perceptual_hash,
                }
            )

    print(f"Manifest: {MANIFEST}")
    print(f"Successfully hashed images: {len(records)}")
    print(f"Missing files: {len(missing_files)}")

    # Compare hashes. This is a straightforward all-pairs audit.
    # At 8,280 images, it requires about 34 million comparisons.
    candidates = []

    for first, second in combinations(records, 2):
        distance = first["hash"] - second["hash"]

        if distance > HASH_DISTANCE_THRESHOLD:
            continue

        same_split = first["split"] == second["split"]

        candidates.append(
            {
                "image_path_1": first["image_path"],
                "split_1": first["split"],
                "label_1": first["binary_label"],
                "image_path_2": second["image_path"],
                "split_2": second["split"],
                "label_2": second["binary_label"],
                "hamming_distance": distance,
                "comparison": (
                    "within_split" if same_split else "cross_split"
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["hamming_distance"],
            item["comparison"],
            item["image_path_1"],
            item["image_path_2"],
        )
    )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "image_path_1",
            "split_1",
            "label_1",
            "image_path_2",
            "split_2",
            "label_2",
            "hamming_distance",
            "comparison",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    cross_split = [
        item for item in candidates
        if item["comparison"] == "cross_split"
    ]

    within_split = [
        item for item in candidates
        if item["comparison"] == "within_split"
    ]

    summary = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "successfully_hashed_images": len(records),
        "missing_files": len(missing_files),
        "hash_method": "Pillow ImageHash pHash",
        "hash_size": HASH_SIZE,
        "hamming_distance_threshold": HASH_DISTANCE_THRESHOLD,
        "candidate_pairs_total": len(candidates),
        "candidate_pairs_cross_split": len(cross_split),
        "candidate_pairs_within_split": len(within_split),
        "warning": (
            "Perceptual-hash matches are candidates only. They do not prove "
            "that images belong to the same patient or represent the same "
            "original radiograph. Review candidate images manually."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nAudit summary")
    print(json.dumps(summary, indent=2))
    print(f"\nCandidate CSV: {OUTPUT_CSV}")
    print(f"Summary JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()