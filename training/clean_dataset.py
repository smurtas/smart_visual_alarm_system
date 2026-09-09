"""
Description: 
Validate the images stored in the incoming dataset folder.
If an image is corrupted or cannot be opened, it is moved to the rejected folder.
Valid images remain in the incoming folder.

Input:  dataset/incoming/
Output: dataset/rejected/corrupted/ for invalid images

Project: Smart Visual Alarm System
"""

from pathlib import Path
import shutil

from PIL import Image, UnidentifiedImageError

# project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INCOMING_DATASET_DIR = PROJECT_ROOT / "dataset" / "incoming"
REJECTED_DATASET_DIR = PROJECT_ROOT / "dataset" / "rejected"

# Dataset classes
# CLASSES = ["empty","person", "animal"]

# Supported image extensions
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def is_supported_image(image_path: Path) -> bool:
    """
    Ceck whether a file has a supported image extension.
    Args: 
    image_path: Path to the image to ckeck
    Returns:
    True if the image has a supported extension, False otherwise.
    """
    return image_path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_valid_image(image_path: Path) -> bool:
    """
    Check whether an image can be opened and is valid.
    Args:
    image_path: Path to the image to check
    Returns:
    True if the image is valid, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that the image is not corrupted
        return True
    except (UnidentifiedImageError, OSError):
        return False

def move_to_rejected(image_path: Path) -> Path:
    """
    Move an image to the rejected folder.
    Args:
    image_path: Path to the image to move
    class_name: Name of the class folder where the image was found
    Returns:
    destination path of the moved image in the rejected folder.
    """
    corrupted_dir = REJECTED_DATASET_DIR / "corrupted"
    corrupted_dir.mkdir(parents=True, exist_ok=True)

    destination_path = corrupted_dir / image_path.name

    counter = 1
    while destination_path.exists():
        destination_path = (
            corrupted_dir / f"{image_path.stem}_{counter}{image_path.suffix}"
        )
        counter += 1

    shutil.move(str(image_path), str(destination_path))
    return destination_path

"""
def process_class_folder(class_name: str) -> tuple[int, int]:
    ""
    Process a class folder by validating its images.
    Args:
    class_name: Name of the class folder to process
    Returns:
    A tuple containing
        - the number of valid images
        - number of rejected images.
""
    class_dir = INCOMING_DATASET_DIR / class_name

    if not class_dir.exists():
        print(f"Class directory {class_dir} not found. Skipping.")
        return 0, 0

    valid_count = 0
    rejected_count = 0

    for image_path in class_dir.iterdir():
        if not image_path.is_file() or not is_supported_image(image_path):
            continue  # Skip non-files and unsupported extensions

        if is_valid_image(image_path):
            valid_count += 1
        else:
            destination_path = move_to_rejected(image_path, class_name)
            rejected_count += 1

            print (f"Rejected:{image_path} "
                   f"-> {destination_path.relative_to(PROJECT_ROOT)} "
                   )
    return valid_count, rejected_count

"""

def main() -> None:
    """
    Main function to process all class folders in the raw dataset.
    """

    print("Starting dataset validation...")
    print(f"Incoming dataset directory: {INCOMING_DATASET_DIR}")
    print()

    total_valid = 0
    total_rejected = 0

    for image_path in INCOMING_DATASET_DIR.rglob("*"):
        if not image_path.is_file() or not is_supported_image(image_path):
            continue  # Skip non-files and unsupported extensions

        if is_valid_image(image_path):
            total_valid += 1
        else:
            destination_path = move_to_rejected(image_path)
            total_rejected += 1

            print (f"Rejected:{image_path} "
                   f"-> {destination_path.relative_to(PROJECT_ROOT)} "
                   )

    print()
    print(f"Total: {total_valid} valid,\n {total_rejected} rejected.")


if __name__ == "__main__":
    main()  