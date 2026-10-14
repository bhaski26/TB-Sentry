from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageStat


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "raw" / "tbx11k" / "TBX11K" / "imgs"

IMAGE_1 = IMAGE_DIR / "sick" / "s1174.png"
IMAGE_2 = IMAGE_DIR / "sick" / "s1964.png"

OUTPUT_DIR = ROOT / "results" / "data_audit"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load both images as grayscale.
    with Image.open(IMAGE_1) as image:
        first = image.convert("L").copy()

    with Image.open(IMAGE_2) as image:
        second = image.convert("L").copy()

    print("=" * 60)
    print("NEAR-IDENTICAL IMAGE INSPECTION")
    print("=" * 60)

    print(f"\nImage 1: {IMAGE_1.relative_to(ROOT)}")
    print(f"Image 2: {IMAGE_2.relative_to(ROOT)}")

    print(f"\nImage 1 dimensions: {first.size}")
    print(f"Image 2 dimensions: {second.size}")

    if first.size != second.size:
        print("\nImages have different dimensions.")
        print("Direct pixel comparison cannot be performed.")
        return

    # Calculate the absolute pixel difference.
    difference = ImageChops.difference(first, second)

    statistics = ImageStat.Stat(difference)
    mean_absolute_difference = statistics.mean[0]

    # Calculate the number of identical and changed pixels.
    histogram = difference.histogram()
    total_pixels = first.width * first.height

    unchanged_pixels = histogram[0]
    changed_pixels = total_pixels - unchanged_pixels

    changed_percentage = (
        100.0 * changed_pixels / total_pixels
    )

    # Get the maximum absolute pixel difference.
    maximum_difference = difference.getextrema()[1]

    # Determine whether the decoded grayscale images are identical.
    exact_pixel_match = difference.getbbox() is None

    print("\n" + "-" * 60)
    print("PIXEL COMPARISON RESULTS")
    print("-" * 60)

    print(f"Total pixels: {total_pixels:,}")
    print(f"Unchanged pixels: {unchanged_pixels:,}")
    print(f"Changed pixels: {changed_pixels:,}")
    print(f"Changed pixel percentage: {changed_percentage:.6f}%")

    print(
        "Mean absolute pixel difference: "
        f"{mean_absolute_difference:.6f}"
    )

    print(
        f"Maximum pixel difference: {maximum_difference}"
    )

    print(f"Exact pixel equality: {exact_pixel_match}")

    # Save the original difference image.
    difference_path = (
        OUTPUT_DIR / "s1174_s1964_pixel_difference.png"
    )

    difference.save(difference_path)

    print(f"\nOriginal difference image saved to:\n{difference_path}")

    # Amplify the differences for easier visual inspection.
    enhanced = ImageEnhance.Contrast(difference).enhance(8)
    enhanced = ImageEnhance.Brightness(enhanced).enhance(12)

    enhanced_path = (
        OUTPUT_DIR / "s1174_s1964_pixel_difference_enhanced.png"
    )

    enhanced.save(enhanced_path)

    print(f"\nEnhanced difference image saved to:\n{enhanced_path}")

    # Save a side-by-side comparison of the original images.
    comparison = Image.new(
        "L",
        (first.width * 2, first.height),
    )

    comparison.paste(first, (0, 0))
    comparison.paste(second, (first.width, 0))

    comparison_path = (
        OUTPUT_DIR / "s1174_s1964_side_by_side.png"
    )

    comparison.save(comparison_path)

    print(f"\nSide-by-side image comparison saved to:\n{comparison_path}")

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)

    print(
        "\nInterpretation reminder: a very small pixel difference "
        "supports visual similarity but does not prove patient identity."
    )


if __name__ == "__main__":
    main()