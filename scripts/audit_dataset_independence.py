"""Audit exact duplicate image content across dataset splits.

This script reports duplicate file contents and does not modify
the dataset, manifest, or model checkpoint.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest.csv"
)

OUTPUT_DIR = ROOT / "results" / "data_audit"


def sha256_file(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    with MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    entries_by_hash = defaultdict(list)
    missing_files = []

    for row in rows:
        relative_path = Path(row["image_path"])

        image_path = (
            relative_path
            if relative_path.is_absolute()
            else ROOT / relative_path
        )

        if not image_path.is_file():
            missing_files.append(
                {
                    "image_path": row["image_path"],
                    "split": row["split"],
                }
            )
            continue

        digest = sha256_file(image_path)

        entries_by_hash[digest].append(
            {
                "image_path": row["image_path"],
                "split": row["split"],
                "binary_label": int(row["binary_label"]),
                "source_category": row["source_category"],
            }
        )

    duplicate_groups = []

    for digest, entries in entries_by_hash.items():
        if len(entries) < 2:
            continue

        splits = sorted(
            {entry["split"] for entry in entries}
        )

        # Record duplicate groups only when content crosses splits.
        if len(splits) > 1:
            duplicate_groups.append(
                {
                    "sha256": digest,
                    "splits": splits,
                    "entries": entries,
                }
            )

    affected_validation_paths = {
        entry["image_path"]
        for group in duplicate_groups
        for entry in group["entries"]
        if entry["split"] == "val"
    }

    validation_duplicate_records = sum(
        1
        for group in duplicate_groups
        for entry in group["entries"]
        if entry["split"] == "val"
    )

    groups_with_label_conflicts = []

    for group in duplicate_groups:
        labels = {
            entry["binary_label"]
            for entry in group["entries"]
        }

        if len(labels) > 1:
            groups_with_label_conflicts.append(group)

    report = {
        "manifest_records": len(rows),
        "missing_image_count": len(missing_files),
        "exact_cross_split_duplicate_group_count": len(
            duplicate_groups
        ),
        "validation_duplicate_record_count": (
            validation_duplicate_records
        ),
        "unique_affected_validation_path_count": len(
            affected_validation_paths
        ),
        "cross_split_label_conflict_group_count": len(
            groups_with_label_conflicts
        ),
        "missing_files": missing_files,
        "duplicate_groups": duplicate_groups,
        "label_conflict_groups": groups_with_label_conflicts,
        "limitations": [
            "Exact file hashes do not detect visually identical files with different encodings.",
            "This audit does not establish patient identity.",
            "No patient-level independence claim can be made from these results alone.",
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        OUTPUT_DIR
        / "exact_duplicate_report.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    csv_path = (
        OUTPUT_DIR
        / "exact_duplicate_report.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "sha256",
            "image_path",
            "split",
            "binary_label",
            "source_category",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for group in duplicate_groups:
            for entry in group["entries"]:
                writer.writerow(
                    {
                        "sha256": group["sha256"],
                        **entry,
                    }
                )

    print("\nEXACT DUPLICATE AUDIT")
    print("Manifest records:", len(rows))
    print("Missing files:", len(missing_files))

    print(
        "Duplicate groups spanning splits:",
        len(duplicate_groups),
    )

    print(
        "Validation duplicate records:",
        validation_duplicate_records,
    )

    print(
        "Unique affected validation paths:",
        len(affected_validation_paths),
    )

    print(
        "Duplicate groups with binary-label conflicts:",
        len(groups_with_label_conflicts),
    )

    print("\nDetailed JSON report:", json_path)
    print("Detailed CSV report:", csv_path)


if __name__ == "__main__":
    main()