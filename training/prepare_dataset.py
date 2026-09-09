"""
Description:
Create stratified train, validation, and test splits from the raw dataset.

Expected source structure:
dataset/raw/
├── animal/
├── empty/
└── person/

Generated structure:
dataset/training/
├── train/
│   ├── animal/
│   ├── empty/
│   └── person/
├── val/
│   ├── animal/
│   ├── empty/
│   └── person/
└── test/
    ├── animal/
    ├── empty/
    └── person/
"""

from pathlib import Path
import shutil # library for file operations
import random
import sys # library for system-specific parameters and functions

# -------------------------------------------
# Configurable parameters
# -------------------------------------------

# DIRECTORY ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "dataset" 
DESTINATION_DIR = PROJECT_ROOT / "dataset" 

CLASSES = ["empty", "person", "animal"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# MAXIMUM number of images per class to include in the dataset
MAX_IMAGES_PER_CLASS = 140

# Fixed seed for reproducible random splits.
RANDOM_SEED = 42


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# "move" removes files from dataset
# Use "copy" to preserve the original files.
OPERATION = "copy"

#-------------------------------------------
# UTILITY FUNCTIONS
#------------------------------------------

def validate_conf() -> None:
    """
    Validate the configuration parameters.
    Raises:
        ValueError: If the sum of TRAIN_RATIO, VAL_RATIO, and TEST_RATIO is not equal to 1.0.
    """
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not (0.99 <= total_ratio <= 1.01):  # Allowing a small margin for floating-point errors
        raise ValueError(
            f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0, but got {total_ratio}"
        )
    if OPERATION not in ["move", "copy"]:
        raise ValueError(f"OPERATION must be either 'move' or 'copy', but got '{OPERATION}'")

    if MAX_IMAGES_PER_CLASS <= 0:
        raise ValueError(f"MAX_IMAGES_PER_CLASS must be a positive integer, but got {MAX_IMAGES_PER_CLASS}")


def get_image_files(class_dir: Path) -> list[Path]:
    """
    Get a list of image paths from the specified class directory.
    Args:
        class_dir (Path): The directory containing images of a specific class.
    Returns:
        list[Path]: A list of Path objects for the images found in the class directory.
    """
    if not class_dir.exists():
        print(f"Class directory does not exist: {class_dir}")
        return []

    image_paths = [
        file_path for file_path in sorted(class_dir.iterdir())
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return image_paths

def calculate_split_sizes(total_images: int) -> tuple[int, int, int]:
    """
    Calculate the number of images for train, validation, and test splits.
    Args:
        total_images (int): Total number of images available for a class.
    Returns:
        tuple[int, int, int]: A tuple containing the number of images for train, validation, and test splits.
    """
    train_size = int(total_images * TRAIN_RATIO)
    val_size = int(total_images * VAL_RATIO)
    test_size = total_images - train_size - val_size  # Ensure all images are accounted for

    return train_size, val_size, test_size

def destination_is_not_empty() -> bool:
    """Return True when the destination already contains files."""

    if not DESTINATION_DIR.exists():
        return False

    return any(path.is_file() for path in DESTINATION_DIR.rglob("*"))


def create_destination_directories() -> None:
    """Create the train, validation, and test directory structure."""

    for split_name in ("train", "val", "test"):
        for class_name in CLASSES:
            destination = DESTINATION_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)


def transfer_file(source: Path, destination: Path) -> None:
    """Move or copy one file according to the selected operation."""

    if destination.exists():
        raise FileExistsError(
            f"Destination file already exists: {destination}"
        )

    if OPERATION == "move":
        shutil.move(str(source), str(destination))
    else:
        shutil.copy2(source, destination)


def transfer_split(
    files: list[Path],
    split_name: str,
    class_name: str,
) -> None:
    """Transfer a group of images to its destination split."""

    destination_directory = (
        DESTINATION_DIR / split_name / class_name
    )

    for source_file in files:
        destination_file = destination_directory / source_file.name
        transfer_file(source_file, destination_file)


# ---------------------------------------------------------------------------
# Main splitting procedure
# ---------------------------------------------------------------------------

def main() -> None:
    """Create balanced stratified dataset splits."""

    validate_conf()

    if destination_is_not_empty():
        print(
            "\nERROR: The destination directory already contains files:\n"
            f"  {DESTINATION_DIR}\n\n"
            "Remove or rename it before running this script again."
        )
        sys.exit(1)

    random_generator = random.Random(RANDOM_SEED)

    selected_images: dict[str, list[Path]] = {}

    print("\nDataset preparation")
    print("=" * 60)
    print(f"Source:      {SOURCE_DIR}")
    print(f"Destination: {DESTINATION_DIR}")
    print(f"Operation:   {OPERATION}")
    print(f"Seed:        {RANDOM_SEED}")
    print(f"Maximum:     {MAX_IMAGES_PER_CLASS} images per class")
    print("=" * 60)

    # Select images before moving anything.
    for class_name in CLASSES:
        class_directory = SOURCE_DIR / class_name
        images = get_image_files(class_directory)

        if not images:
            raise RuntimeError(
                f"No supported images found for class '{class_name}'."
            )

        random_generator.shuffle(images)

        selected_images[class_name] = images[
            :MAX_IMAGES_PER_CLASS
        ]

        print(
            f"{class_name:<10}: "
            f"{len(images):>6} available, "
            f"{len(selected_images[class_name]):>4} selected"
        )

    create_destination_directories()

    print("\nCreating splits")
    print("-" * 60)

    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in CLASSES:
        images = selected_images[class_name]

        train_size, val_size, test_size = calculate_split_sizes(
            len(images)
        )

        train_end = train_size
        val_end = train_size + val_size

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        transfer_split(train_files, "train", class_name)
        transfer_split(val_files, "val", class_name)
        transfer_split(test_files, "test", class_name)

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        print(
            f"{class_name:<10} | "
            f"train: {len(train_files):>3} | "
            f"val: {len(val_files):>3} | "
            f"test: {len(test_files):>3}"
        )

    print("-" * 60)
    print(
        f"{'TOTAL':<10} | "
        f"train: {total_train:>3} | "
        f"val: {total_val:>3} | "
        f"test: {total_test:>3}"
    )

    print("\nDataset successfully prepared.")
    print(f"Output directory: {DESTINATION_DIR}")

    if OPERATION == "move":
        print(
            "\nWarning: selected images were removed from dataset/raw."
        )


if __name__ == "__main__":
    main()