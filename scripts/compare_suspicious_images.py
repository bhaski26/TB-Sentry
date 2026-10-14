from pathlib import Path

from PIL import Image, ImageChops, ImageStat


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "raw" / "tbx11k" / "TBX11K" / "imgs"

PAIRS = [
    ("tb/tb1181.png", "health/h3904.png"),
    ("health/h0001.png", "health/h1781.png"),
    ("sick/s1174.png", "sick/s1964.png"),
]


def compare_images(path1, path2):
    with Image.open(path1) as image:
        image1 = image.convert("L").copy()

    with Image.open(path2) as image:
        image2 = image.convert("L").copy()

    print(f"\nPAIR: {path1.relative_to(IMAGE_DIR)}")
    print(f"   vs {path2.relative_to(IMAGE_DIR)}")

    print(f"Dimensions: {image1.size} vs {image2.size}")

    if image1.size != image2.size:
        print("Pixel comparison skipped: dimensions differ.")
        return

    difference = ImageChops.difference(image1, image2)

    mean_difference = ImageStat.Stat(difference).mean[0]
    exact_pixel_match = difference.getbbox() is None

    print(f"Mean absolute pixel difference: {mean_difference:.3f}")
    print(f"Exact pixel equality: {exact_pixel_match}")


def main():
    for first, second in PAIRS:
        path1 = IMAGE_DIR / first
        path2 = IMAGE_DIR / second

        if not path1.is_file() or not path2.is_file():
            print(f"\nERROR: Image not found: {path1} or {path2}")
            continue

        compare_images(path1, path2)


if __name__ == "__main__":
    main()