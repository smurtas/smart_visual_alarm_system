"""
Description:
Clean image filenames inside dataset/training.

The script:
- scans all subdirectories recursively;
- replaces spaces with underscores;
- removes commas;
- converts final counters such as "(6)" into "_6";
- preserves unique filenames;
- does not overwrite existing files.
"""

from pathlib import Path
import re


ROOT_DIR = Path("dataset/training")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def main() -> None:
    """Clean all supported image filenames inside ROOT_DIR."""

    if not ROOT_DIR.exists():
        raise FileNotFoundError(
            f"Root directory not found: {ROOT_DIR.resolve()}"
        )

    renamed_count = 0
    skipped_count = 0

    for image_path in ROOT_DIR.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        stem = image_path.stem
        extension = image_path.suffix.lower()

        # Convert a final duplicate counter:
        # "image (6)" -> "image_6"
        new_stem = re.sub(
            r"\s*\((\d+)\)$",
            r"_\1",
            stem,
        )

        # Remove commas.
        new_stem = new_stem.replace(",", "")

        # Replace spaces with underscores.
        new_stem = re.sub(
            r"\s+",
            "_",
            new_stem,
        )

        # Replace repeated underscores with one underscore.
        new_stem = re.sub(
            r"_+",
            "_",
            new_stem,
        )

        # Remove underscores from the beginning and end.
        new_stem = new_stem.strip("_")

        new_name = f"{new_stem}{extension}"
        new_path = image_path.with_name(new_name)

        if new_path == image_path:
            continue

        if new_path.exists():
            print(
                f"Skipped, destination already exists: {new_path}"
            )
            skipped_count += 1
            continue

        print(
            f"{image_path.name} -> {new_name}"
        )

        image_path.rename(new_path)
        renamed_count += 1

    print(f"\nRenamed images: {renamed_count}")
    print(f"Skipped images: {skipped_count}")


if __name__ == "__main__":
    main()