"""
Description:
Validate the train, validation, and test dataset structure.
The script checks image counts, unreadable files, and duplicate file hashes.
"""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" 

SPLITS = ["train", "val", "test"]
CLASSES = ["animal", "empty", "person"]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def get_image_paths(directory: Path) -> list[Path]:
    """Return supported image files contained in a directory."""

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def calculate_file_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    digest = sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_image(image_path: Path) -> bool:
    """
    Check whether an image can be opened and decoded.

    Returns:
        True if the image is valid, otherwise False.
    """

    try:
        with Image.open(image_path) as image:
            image.verify()

        return True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return False


def main() -> None:
    """Validate the prepared dataset."""

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    hashes: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    invalid_images: list[Path] = []

    total_images = 0

    print("\nDataset validation")
    print("=" * 70)

    for split_name in SPLITS:
        split_total = 0

        print(f"\n{split_name.upper()}")

        for class_name in CLASSES:
            class_directory = (
                DATASET_DIR / split_name / class_name
            )

            if not class_directory.exists():
                raise FileNotFoundError(
                    f"Missing directory: {class_directory}"
                )

            images = get_image_paths(class_directory)

            print(
                f"  {class_name:<10}: {len(images):>4} images"
            )

            split_total += len(images)
            total_images += len(images)

            for image_path in images:
                if not validate_image(image_path):
                    invalid_images.append(image_path)
                    continue

                file_hash = calculate_file_hash(image_path)

                hashes[file_hash].append(
                    (
                        split_name,
                        class_name,
                        image_path,
                    )
                )

        print(f"  {'TOTAL':<10}: {split_total:>4} images")

    duplicate_groups = [
        locations
        for locations in hashes.values()
        if len(locations) > 1
    ]

    cross_split_duplicates = []

    for locations in duplicate_groups:
        involved_splits = {
            split_name
            for split_name, _, _ in locations
        }

        if len(involved_splits) > 1:
            cross_split_duplicates.append(locations)

    print("\n" + "=" * 70)
    print(f"Total images: {total_images}")
    print(f"Invalid images: {len(invalid_images)}")
    print(
        "Duplicate groups across different splits: "
        f"{len(cross_split_duplicates)}"
    )

    if invalid_images:
        print("\nInvalid images:")

        for image_path in invalid_images:
            print(f"  {image_path}")

    if cross_split_duplicates:
        print("\nDuplicates found across train, val, or test:")

        for group_number, locations in enumerate(
            cross_split_duplicates,
            start=1,
        ):
            print(f"\n  Duplicate group {group_number}")

            for split_name, class_name, image_path in locations:
                print(
                    f"    {split_name:<5} "
                    f"{class_name:<10} "
                    f"{image_path.name}"
                )

    if not invalid_images and not cross_split_duplicates:
        print("\nDataset validation completed successfully.")
    else:
        print(
            "\nDataset validation completed with issues."
        )


if __name__ == "__main__":
    main()