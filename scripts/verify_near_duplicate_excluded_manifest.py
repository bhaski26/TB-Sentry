import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest_near_duplicate_excluded.csv"
)

SOURCE_MANIFEST = (
    ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest_deduplicated.csv"
)

EXCLUDED_PATHS = {
    "data/raw/tbx11k/TBX11K/imgs/sick/s1174.png",
    "data/raw/tbx11k/TBX11K/imgs/sick/s2023.png",
}


def read_manifest(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def main():
    rows = read_manifest(MANIFEST)
    source_rows = read_manifest(SOURCE_MANIFEST)

    split_counts = Counter(row["split"] for row in rows)

    missing_files = [
        row["image_path"]
        for row in rows
        if not (ROOT / row["image_path"]).is_file()
    ]

    present_excluded_paths = {
        Path(row["image_path"]).as_posix()
        for row in rows
    } & EXCLUDED_PATHS

    source_validation = {
        row["image_path"]
        for row in source_rows
        if row["split"] == "val"
    }

    output_validation = {
        row["image_path"]
        for row in rows
        if row["split"] == "val"
    }

    source_training = {
        row["image_path"]
        for row in source_rows
        if row["split"] == "train"
    }

    output_training = {
        row["image_path"]
        for row in rows
        if row["split"] == "train"
    }

    expected_training = source_training - EXCLUDED_PATHS

    checks = {
        "total_records_8278": len(rows) == 8278,
        "training_records_6478": split_counts["train"] == 6478,
        "validation_records_1800": split_counts["val"] == 1800,
        "no_missing_files": len(missing_files) == 0,
        "excluded_images_absent": len(present_excluded_paths) == 0,
        "validation_unchanged": source_validation == output_validation,
        "training_changed_only_as_intended": (
            expected_training == output_training
        ),
    }

    report = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "total_records": len(rows),
        "split_counts": dict(split_counts),
        "missing_file_count": len(missing_files),
        "missing_files": missing_files[:20],
        "excluded_paths_still_present": sorted(present_excluded_paths),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }

    report_path = (
        ROOT
        / "results"
        / "data_audit"
        / "near_duplicate_excluded_manifest_verification.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))

    if not all(checks.values()):
        raise SystemExit("VERIFICATION FAILED")

    print("\nPASS: All manifest verification checks passed.")


if __name__ == "__main__":
    main()