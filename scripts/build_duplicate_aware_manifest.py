"""Create a duplicate-aware manifest without modifying the original."""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_MANIFEST = ROOT / "data" / "interim" / "tbx11k_manifest.csv"
OUTPUT_MANIFEST = (
    ROOT / "data" / "interim" / "tbx11k_manifest_deduplicated.csv"
)

OUTPUT_REPORT = ROOT / "results" / "data_audit" / (
    "duplicate_aware_manifest_report.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    with SOURCE_MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise ValueError("The source manifest has no column headers.")

    # Hash every image and validate that all files exist.
    for row in rows:
        image_path = ROOT / row["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing image: {row['image_path']}"
            )

        row["_sha256"] = sha256_file(image_path)

    validation_hashes = {
        row["_sha256"] for row in rows if row["split"] == "val"
    }

    kept_rows = []
    seen_training_hashes = set()

    excluded_cross_split = 0
    excluded_within_training = 0

    for row in rows:
        if row["split"] == "val":
            # Preserve every official validation record.
            kept_rows.append(row)
            continue

        if row["split"] != "train":
            # Preserve other splits without changing their membership.
            kept_rows.append(row)
            continue

        image_hash = row["_sha256"]

        if image_hash in validation_hashes:
            excluded_cross_split += 1
            continue

        if image_hash in seen_training_hashes:
            excluded_within_training += 1
            continue

        seen_training_hashes.add(image_hash)
        kept_rows.append(row)

    # Write only the original manifest columns.
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_MANIFEST.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in kept_rows:
            writer.writerow({
                column: row[column] for column in fieldnames
            })

    # Independently summarize the output split counts.
    split_counts = defaultdict(int)

    for row in kept_rows:
        split_counts[row["split"]] += 1

    import json

    report = {
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "output_manifest": str(OUTPUT_MANIFEST.relative_to(ROOT)),
        "source_records": len(rows),
        "output_records": len(kept_rows),
        "excluded_train_validation_duplicate_records":
            excluded_cross_split,
        "excluded_additional_training_duplicate_records":
            excluded_within_training,
        "output_split_counts": dict(split_counts),
        "validation_records_preserved": (
            split_counts["val"]
            == sum(row["split"] == "val" for row in rows)
        ),
        "method": (
            "SHA-256 exact-content deduplication; validation retained "
            "unchanged; one unique training image per content hash; "
            "training copies of validation images excluded."
        ),
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_REPORT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report, indent=2))
    print("\nOriginal manifest unchanged.")
    print("New manifest:", OUTPUT_MANIFEST)
    print("Audit report:", OUTPUT_REPORT)


if __name__ == "__main__":
    main()