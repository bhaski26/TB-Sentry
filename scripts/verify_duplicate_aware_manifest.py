"""Verify exact-content independence of the corrected manifest."""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest_deduplicated.csv"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    with MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    hashes_by_split = defaultdict(list)
    missing = []

    for row in rows:
        image_path = ROOT / row["image_path"]

        if not image_path.is_file():
            missing.append(row["image_path"])
            continue

        image_hash = sha256_file(image_path)
        hashes_by_split[row["split"]].append(image_hash)

    train_hashes = hashes_by_split["train"]
    val_hashes = hashes_by_split["val"]

    train_set = set(train_hashes)
    val_set = set(val_hashes)

    print("Manifest records:", len(rows))
    print("Missing files:", len(missing))

    print("\nTraining records:", len(train_hashes))
    print("Unique training contents:", len(train_set))
    print("Duplicate training records:", len(train_hashes) - len(train_set))

    print("\nValidation records:", len(val_hashes))
    print("Unique validation contents:", len(val_set))
    print("Duplicate validation records:", len(val_hashes) - len(val_set))

    print(
        "\nExact contents shared between train and validation:",
        len(train_set & val_set),
    )

    if missing:
        print("\nFirst missing paths:")
        for path in missing[:10]:
            print("  ", path)

    assert not missing, "Some manifest images are missing."
    assert len(train_hashes) == len(train_set), (
        "Duplicate image contents remain within training."
    )
    assert not (train_set & val_set), (
        "Exact image-content overlap remains across splits."
    )

    print("\nPASS: Exact-content training/validation separation verified.")


if __name__ == "__main__":
    main()