"""
Description:
Find images that are completely black or extremely dark.

The original images remain in dataset/incoming.
Dark candidates are copied to dataset/dark_candidates
for manual inspection.

Project: Smart Visual Alarm System
"""

from pathlib import Path
import shutil

from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INCOMING_DIR = PROJECT_ROOT / "dataset" / "incoming"
# per check
# DARK_CANDIDATES_DIR = PROJECT_ROOT / "dataset" / "dark_candidates"
DARK_CANDIDATES_DIR = PROJECT_ROOT / "dataset" / "rejected" / "dark"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Images with an average brightness below this value
# are considered potentially too dark.
DARK_THRESHOLD = 12


def calculate_average_brightness(image_path: Path) -> float:
    """
    Calculate the average brightness of an image.

    The image is converted to grayscale:
    0 means completely black;
    255 means completely white.
    """
    with Image.open(image_path) as image:
        grayscale_image = image.convert("L")
        statistics = ImageStat.Stat(grayscale_image)

    return statistics.mean[0]


def create_unique_destination(image_path: Path) -> Path:
    """
    Create a destination path without overwriting existing files.
    """
    destination = DARK_CANDIDATES_DIR / image_path.name
    counter = 1

    while destination.exists():
        destination = (
            DARK_CANDIDATES_DIR
            / f"{image_path.stem}_{counter}{image_path.suffix}"
        )
        counter += 1

    return destination


def main() -> None:
    """
    Find and copy extremely dark images for manual inspection.
    """
    if not INCOMING_DIR.exists():
        print(f"Incoming directory not found: {INCOMING_DIR}")
        return

    DARK_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in INCOMING_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    total_images = len(image_paths)
    dark_count = 0

    print("Starting dark image analysis...")
    print(f"Images found: {total_images}")
    print(f"Darkness threshold: {DARK_THRESHOLD}")
    print()

    for index, image_path in enumerate(image_paths, start=1):
        brightness = calculate_average_brightness(image_path)

        print(
            f"\rChecking image {index}/{total_images}",
            end="",
            flush=True,
        )

        if brightness < DARK_THRESHOLD:
            destination = create_unique_destination(image_path)

            # shutil.copy2(image_path, destination)
            shutil.move(str(image_path), str(destination))
            dark_count += 1

            print()
            print(
                f"Dark candidate: {image_path.name} "
                f"(brightness: {brightness:.2f})"
            )

    print()
    print()
    print("Dark image analysis completed.")
    print(f"Total images:    {total_images}")
    print(f"Dark candidates: {dark_count}")
    print()
    print(f"Review folder: {DARK_CANDIDATES_DIR}")


if __name__ == "__main__":
    main()