import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "data"
    / "interim"
    / "tbx11k_manifest_deduplicated.csv"
)

CANDIDATES_CSV = (
    ROOT
    / "results"
    / "data_audit"
    / "near_duplicate_candidates.csv"
)

OUTPUT_DIR = ROOT / "results" / "data_audit"

OUTPUT_CSV = OUTPUT_DIR / "pixel_near_duplicate_candidates.csv"
OUTPUT_JSON = OUTPUT_DIR / "pixel_near_duplicate_audit_summary.json"

# Strict initial screening criteria.
# These are candidate-selection rules, not proof of duplication.
MAX_MEAN_ABSOLUTE_DIFFERENCE = 0.02
MAX_CHANGED_PIXEL_PERCENTAGE = 1.0

# Compare only the strongest perceptual-hash candidates.
# The original candidate report remains unchanged.
MAX_PHASH_DISTANCE = 6


def resolve_path(value):
    """Resolve a manifest or candidate path against the repository root."""
    path = Path(value)

    if path.is_absolute():
        return path

    return ROOT / path


def sha256_file(path):
    """Compute a file's SHA-256 hash."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_image(path):
    """Load an image as an independent grayscale image."""
    with Image.open(path) as image:
        return image.convert("L").copy()


def compare_images(path1, path2):
    """
    Compare decoded grayscale pixels.

    Returns None when the image dimensions differ.
    """
    image1 = load_image(path1)
    image2 = load_image(path2)

    if image1.size != image2.size:
        return None

    difference = ImageChops.difference(image1, image2)
    mean_difference = ImageStat.Stat(difference).mean[0]

    histogram = difference.histogram()
    total_pixels = image1.width * image1.height
    unchanged_pixels = histogram[0]
    changed_pixels = total_pixels - unchanged_pixels

    changed_percentage = 100.0 * changed_pixels / total_pixels

    return {
        "width": image1.width,
        "height": image1.height,
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "changed_pixel_percentage": changed_percentage,
        "mean_absolute_difference": mean_difference,
        "maximum_pixel_difference": difference.getextrema()[1],
        "exact_pixel_equality": difference.getbbox() is None,
        "sha256_equal": sha256_file(path1) == sha256_file(path2),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    if not CANDIDATES_CSV.is_file():
        raise FileNotFoundError(
            f"Candidate CSV not found: {CANDIDATES_CSV}"
        )

    # Read split membership from the duplicate-aware manifest.
    manifest_records = {}

    with MANIFEST.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        for row in csv.DictReader(file):
            normalized_path = Path(row["image_path"]).as_posix()

            manifest_records[normalized_path] = {
                "split": row["split"],
                "label": row["binary_label"],
            }

    # Load the existing perceptual-hash candidate pairs.
    with CANDIDATES_CSV.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        candidate_rows = list(csv.DictReader(file))

    selected = [
        row
        for row in candidate_rows
        if int(row["hamming_distance"]) <= MAX_PHASH_DISTANCE
    ]

    print("=" * 65)
    print("TB-SENTRY PIXEL-LEVEL NEAR-DUPLICATE AUDIT")
    print("=" * 65)

    print(f"Manifest records: {len(manifest_records)}")
    print(f"Original pHash candidate pairs: {len(candidate_rows)}")
    print(f"Pairs selected for pixel comparison: {len(selected)}")

    results = []
    missing_images = 0
    dimension_mismatches = 0
    comparison_errors = 0

    for index, row in enumerate(selected, start=1):
        relative1 = Path(row["image_path_1"]).as_posix()
        relative2 = Path(row["image_path_2"]).as_posix()

        path1 = resolve_path(relative1)
        path2 = resolve_path(relative2)

        if not path1.is_file() or not path2.is_file():
            missing_images += 1
            continue

        try:
            comparison = compare_images(path1, path2)
        except Exception as exc:
            comparison_errors += 1
            print(f"Comparison error: {relative1} <-> {relative2}: {exc}")
            continue

        if comparison is None:
            dimension_mismatches += 1
            continue

        # Skip pairs whose content is byte-for-byte identical.
        # These should already have been addressed by the exact-duplicate
        # audit and duplicate-aware manifest.
        if comparison["sha256_equal"]:
            continue

        same_split = (
            manifest_records.get(relative1, {}).get("split")
            == manifest_records.get(relative2, {}).get("split")
        )

        # Keep only candidates meeting BOTH strict screening criteria.
        if (
            comparison["mean_absolute_difference"]
            > MAX_MEAN_ABSOLUTE_DIFFERENCE
            or comparison["changed_pixel_percentage"]
            > MAX_CHANGED_PIXEL_PERCENTAGE
        ):
            continue

        results.append(
            {
                "image_path_1": relative1,
                "image_path_2": relative2,
                "split_1": manifest_records.get(
                    relative1, {}
                ).get("split", row["split_1"]),
                "split_2": manifest_records.get(
                    relative2, {}
                ).get("split", row["split_2"]),
                "label_1": manifest_records.get(
                    relative1, {}
                ).get("label", row["label_1"]),
                "label_2": manifest_records.get(
                    relative2, {}
                ).get("label", row["label_2"]),
                "hamming_distance": int(row["hamming_distance"]),
                "comparison": (
                    "within_split" if same_split else "cross_split"
                ),
                **comparison,
            }
        )

        if index % 1000 == 0:
            print(f"Compared {index}/{len(selected)} candidate pairs...")

    results.sort(
        key=lambda row: (
            row["mean_absolute_difference"],
            row["changed_pixel_percentage"],
            row["image_path_1"],
            row["image_path_2"],
        )
    )

    fieldnames = [
        "image_path_1",
        "image_path_2",
        "split_1",
        "split_2",
        "label_1",
        "label_2",
        "hamming_distance",
        "comparison",
        "width",
        "height",
        "total_pixels",
        "changed_pixels",
        "changed_pixel_percentage",
        "mean_absolute_difference",
        "maximum_pixel_difference",
        "exact_pixel_equality",
        "sha256_equal",
    ]

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    cross_split_count = sum(
        row["comparison"] == "cross_split" for row in results
    )

    within_split_count = sum(
        row["comparison"] == "within_split" for row in results
    )

    summary = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "input_candidate_csv": str(CANDIDATES_CSV.relative_to(ROOT)),
        "manifest_records": len(manifest_records),
        "input_candidate_pairs": len(candidate_rows),
        "pairs_selected_for_pixel_comparison": len(selected),
        "max_phash_distance": MAX_PHASH_DISTANCE,
        "max_mean_absolute_difference": (
            MAX_MEAN_ABSOLUTE_DIFFERENCE
        ),
        "max_changed_pixel_percentage": (
            MAX_CHANGED_PIXEL_PERCENTAGE
        ),
        "missing_images": missing_images,
        "dimension_mismatches": dimension_mismatches,
        "comparison_errors": comparison_errors,
        "pixel_near_duplicate_candidates": len(results),
        "cross_split_candidates": cross_split_count,
        "within_split_candidates": within_split_count,
        "important_limitation": (
            "These are algorithmically selected candidates, not confirmed "
            "duplicates. The thresholds were chosen for strict screening "
            "and have not been validated as universal duplicate criteria. "
            "The audit only examines pairs selected by perceptual hashing; "
            "it may miss transformed duplicates with larger hash distances. "
            "Patient identity cannot be established from pixel similarity."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "-" * 65)
    print("AUDIT SUMMARY")
    print("-" * 65)
    print(json.dumps(summary, indent=2))

    print(f"\nCandidate CSV: {OUTPUT_CSV}")
    print(f"Summary JSON: {OUTPUT_JSON}")

    print("\nStrongest candidates:")

    for row in results[:20]:
        print(
            f"\nMean difference={row['mean_absolute_difference']:.6f} | "
            f"Changed={row['changed_pixel_percentage']:.6f}% | "
            f"{row['comparison']}"
        )
        print(f"  {row['image_path_1']} [{row['split_1']}]")
        print(f"  {row['image_path_2']} [{row['split_2']}]")

    if not results:
        print(
            "\nNo pairs met both strict screening criteria. "
            "This does not establish that no near-duplicates exist."
        )


if __name__ == "__main__":
    main()