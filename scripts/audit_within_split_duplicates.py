import csv
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "interim" / "tbx11k_manifest.csv"


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    groups = defaultdict(list)
    missing = []

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        image_path = ROOT / row["image_path"]

        if not image_path.is_file():
            missing.append(str(image_path))
            continue

        image_hash = sha256_file(image_path)
        key = (row["split"], image_hash)
        groups[key].append(row["image_path"])

    print(f"Manifest records: {len(rows)}")
    print(f"Missing files: {len(missing)}")

    for split in ("train", "val"):
        duplicate_groups = [
            paths
            for (group_split, _), paths in groups.items()
            if group_split == split and len(paths) > 1
        ]

        extra_records = sum(
            len(paths) - 1 for paths in duplicate_groups
        )

        print(f"\nSplit: {split}")
        print(f"Duplicate content groups: {len(duplicate_groups)}")
        print(f"Extra duplicate records: {extra_records}")

        for paths in duplicate_groups[:5]:
            print("  ", paths)

    train_hashes = {
        image_hash
        for (split, image_hash), paths in groups.items()
        if split == "train"
    }

    val_hashes = {
        image_hash
        for (split, image_hash), paths in groups.items()
        if split == "val"
    }

    print(
        "\nDistinct image contents shared across train/val:",
        len(train_hashes & val_hashes),
    )

    if missing:
        print("\nFirst missing files:")
        for path in missing[:10]:
            print("  ", path)


if __name__ == "__main__":
    main()