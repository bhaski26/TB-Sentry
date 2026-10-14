import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_CSV = (
    ROOT / "results" / "data_audit" / "near_duplicate_candidates.csv"
)

OUTPUT_DIR = ROOT / "results" / "data_audit"
CONTACT_SHEET = OUTPUT_DIR / "strongest_cross_split_candidates.png"
REPORT_JSON = OUTPUT_DIR / "strongest_cross_split_candidates.json"

# Inspect the closest cross-split matches first.
MAX_PAIRS = 20


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_path(value):
    return ROOT / Path(value)


def main():
    with CANDIDATES_CSV.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    cross_split = [
        row for row in rows
        if row["comparison"] == "cross_split"
    ]

    cross_split.sort(
        key=lambda row: (
            int(row["hamming_distance"]),
            row["image_path_1"],
            row["image_path_2"],
        )
    )

    selected = cross_split[:MAX_PAIRS]

    if not selected:
        print("No cross-split candidates found.")
        return

    report = []
    image_pairs = []

    for row in selected:
        path1 = resolve_path(row["image_path_1"])
        path2 = resolve_path(row["image_path_2"])

        hash1 = sha256_file(path1)
        hash2 = sha256_file(path2)

        exact_file_match = hash1 == hash2

        report.append(
            {
                **row,
                "sha256_1": hash1,
                "sha256_2": hash2,
                "exact_file_match": exact_file_match,
            }
        )

        with Image.open(path1) as image:
            image1 = image.convert("RGB").copy()

        with Image.open(path2) as image:
            image2 = image.convert("RGB").copy()

        image_pairs.append((row, image1, image2))

    # One row per candidate pair; two images per row.
    fig, axes = plt.subplots(
        len(image_pairs),
        2,
        figsize=(10, max(3 * len(image_pairs), 6)),
        squeeze=False,
    )

    for index, (row, image1, image2) in enumerate(image_pairs):
        axes[index, 0].imshow(image1, cmap="gray")
        axes[index, 0].set_title(
            f"{row['image_path_1']}\n"
            f"label={row['label_1']}"
        )
        axes[index, 0].axis("off")

        axes[index, 1].imshow(image2, cmap="gray")
        axes[index, 1].set_title(
            f"{row['image_path_2']}\n"
            f"label={row['label_2']} | "
            f"pHash distance={row['hamming_distance']}"
        )
        axes[index, 1].axis("off")

    fig.tight_layout()
    fig.savefig(CONTACT_SHEET, dpi=120, bbox_inches="tight")
    plt.close(fig)

    REPORT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    exact_matches = sum(
        item["exact_file_match"] for item in report
    )

    print(f"Cross-split pairs inspected: {len(report)}")
    print(f"Exact SHA-256 matches: {exact_matches}")
    print(f"Visually inspect: {CONTACT_SHEET}")
    print(f"Detailed report: {REPORT_JSON}")

    print("\nPair-level results:")

    for item in report:
        print(
            f"Distance={item['hamming_distance']} | "
            f"Exact file match={item['exact_file_match']} | "
            f"{item['image_path_1']} <-> {item['image_path_2']}"
        )


if __name__ == "__main__":
    main()