import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_MANIFEST = (
    ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest_deduplicated.csv"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest_near_duplicate_excluded.csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "data_audit"
    / "near_duplicate_excluded_manifest_report.json"
)

# Exclude only the training-side members of the two investigated pairs.
EXCLUDED_TRAINING_PATHS = {
    "data/raw/tbx11k/TBX11K/imgs/sick/s1174.png",
    "data/raw/tbx11k/TBX11K/imgs/sick/s2023.png",
}


def normalize_path(value):
    return Path(value).as_posix()


def main():
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Source manifest not found: {SOURCE_MANIFEST}"
        )

    with SOURCE_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames

        if not fieldnames or "image_path" not in fieldnames:
            raise ValueError(
                "Source manifest must contain an image_path column."
            )

        if "split" not in fieldnames:
            raise ValueError(
                "Source manifest must contain a split column."
            )

        source_rows = list(reader)

    retained_rows = []
    excluded_rows = []
    found_excluded_paths = set()

    for row in source_rows:
        image_path = normalize_path(row["image_path"])

        if image_path in EXCLUDED_TRAINING_PATHS:
            if row["split"] != "train":
                raise ValueError(
                    f"Refusing to exclude a non-training record: "
                    f"{image_path}, split={row['split']}"
                )

            excluded_rows.append(row)
            found_excluded_paths.add(image_path)
        else:
            retained_rows.append(row)

    missing_exclusions = (
        EXCLUDED_TRAINING_PATHS - found_excluded_paths
    )

    if missing_exclusions:
        raise ValueError(
            "The following expected training records were not found: "
            f"{sorted(missing_exclusions)}"
        )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(retained_rows)

    source_split_counts = Counter(
        row["split"] for row in source_rows
    )

    output_split_counts = Counter(
        row["split"] for row in retained_rows
    )

    excluded_details = [
        {
            "image_path": normalize_path(row["image_path"]),
            "split": row["split"],
            "binary_label": row.get("binary_label"),
        }
        for row in excluded_rows
    ]

    report = {
        "source_manifest": str(
            SOURCE_MANIFEST.relative_to(ROOT)
        ),
        "output_manifest": str(
            OUTPUT_MANIFEST.relative_to(ROOT)
        ),
        "source_records": len(source_rows),
        "output_records": len(retained_rows),
        "excluded_training_records": len(excluded_rows),
        "source_split_counts": dict(source_split_counts),
        "output_split_counts": dict(output_split_counts),
        "excluded_records": excluded_details,
        "validation_record_count_preserved": (
            source_split_counts.get("val", 0)
            == output_split_counts.get("val", 0)
        ),
        "exclusion_scope": (
            "Only the two specified training-side members of "
            "probable near-duplicate pairs were excluded. "
            "Validation records were retained."
        ),
        "limitations": (
            "This is a targeted exclusion experiment, not proof "
            "that all near-duplicates or patient-level leakage "
            "have been eliminated."
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("NEAR-DUPLICATE EXCLUSION MANIFEST")
    print("=" * 60)

    print(f"Source records: {len(source_rows)}")
    print(f"Output records: {len(retained_rows)}")
    print(f"Excluded training records: {len(excluded_rows)}")

    print("\nSource split counts:")
    for split, count in sorted(source_split_counts.items()):
        print(f"  {split}: {count}")

    print("\nOutput split counts:")
    for split, count in sorted(output_split_counts.items()):
        print(f"  {split}: {count}")

    print("\nExcluded records:")
    for row in excluded_details:
        print(
            f"  {row['image_path']} "
            f"| split={row['split']} "
            f"| label={row['binary_label']}"
        )

    print(
        "\nValidation record count preserved:",
        report["validation_record_count_preserved"],
    )

    print(f"\nManifest saved to:\n{OUTPUT_MANIFEST}")
    print(f"\nReport saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()