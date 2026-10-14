from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageStat


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    ROOT / "data" / "raw" / "tbx11k" / "TBX11K" / "imgs"
)

IMAGE_1 = IMAGE_DIR / "sick" / "s2023.png"
IMAGE_2 = IMAGE_DIR / "sick" / "s3669.png"

OUTPUT_DIR = ROOT / "results" / "data_audit"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(IMAGE_1) as image:
        first = image.convert("L").copy()

    with Image.open(IMAGE_2) as image:
        second = image.convert("L").copy()

    print("Image 1:", IMAGE_1.relative_to(ROOT))
    print("Image 2:", IMAGE_2.relative_to(ROOT))

    print("Image 1 dimensions:", first.size)
    print("Image 2 dimensions:", second.size)

    if first.size != second.size:
        print("Dimensions differ; direct pixel comparison is unavailable.")
        return

    difference = ImageChops.difference(first, second)

    statistics = ImageStat.Stat(difference)
    mean_difference = statistics.mean[0]

    histogram = difference.histogram()
    total_pixels = first.width * first.height
    unchanged_pixels = histogram[0]
    changed_pixels = total_pixels - unchanged_pixels

    changed_percentage = 100 * changed_pixels / total_pixels

    print("\nPixel comparison")
    print("----------------")
    print("Total pixels:", total_pixels)
    print("Unchanged pixels:", unchanged_pixels)
    print("Changed pixels:", changed_pixels)
    print(f"Changed pixel percentage: {changed_percentage:.6f}%")
    print(f"Mean absolute pixel difference: {mean_difference:.6f}")
    print("Maximum pixel difference:", difference.getextrema()[1])
    print("Exact pixel equality:", difference.getbbox() is None)

    side_by_side = Image.new(
        "L",
        (first.width * 2, first.height),
    )

    side_by_side.paste(first, (0, 0))
    side_by_side.paste(second, (first.width, 0))

    side_by_side_path = (
        OUTPUT_DIR / "s2023_s3669_side_by_side.png"
    )

    side_by_side.save(side_by_side_path)

    enhanced_difference = ImageEnhance.Contrast(difference).enhance(8)
    enhanced_difference = ImageEnhance.Brightness(
        enhanced_difference
    ).enhance(12)

    difference_path = (
        OUTPUT_DIR / "s2023_s3669_pixel_difference_enhanced.png"
    )

    enhanced_difference.save(difference_path)

    print("\nSaved side-by-side image:", side_by_side_path)
    print("Saved enhanced difference image:", difference_path)


if __name__ == "__main__":
    main()