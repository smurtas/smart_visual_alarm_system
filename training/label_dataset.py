"""
Description:
Display one image fron the raw dataset.

Input:  dataset/raw/ (folder containing raw images)
Output:
    Images moved to:
    - dataset/raw/empty/
    - dataset/raw/person/
    - dataset/raw/animal/
    - dataset/rejected/manual/ (for images that cannot be classified)
    
Project: Smart Visual Alarm System
"""

from pathlib import Path
import shutil

from PIL import Image
import matplotlib.pyplot as plt

# Resolve the root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory currently containing all collected images
RAW_DATASET_DIR = PROJECT_ROOT / "dataset" / "incoming" 

# Destination directories for the three classes
EMPTY_DIR = PROJECT_ROOT / "dataset" / "raw" / "empty"
PERSON_DIR = PROJECT_ROOT / "dataset" / "raw" / "person"
ANIMAL_DIR = PROJECT_ROOT / "dataset" / "raw" / "animal"

# Directory for rejected images
REJECTED_DIR = PROJECT_ROOT / "dataset" / "rejected" / "manual"

# File extesions accepted by this script
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]    

def get_image_path () -> list[Path]:
    """
    Get the path of the first image in the raw dataset directory.
    Returns:
    A list of Path objects for the images found in the raw dataset directory.
    """


    if not RAW_DATASET_DIR.exists():
        print(f"Raw dataset directory does not exist: {RAW_DATASET_DIR}")
        return []
    image_paths = []
    for file_path in sorted(RAW_DATASET_DIR.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(file_path)

    return image_paths



def display_image(image_path: Path) -> None:
    """
    Display an image using matplotlib.
    Args:
    image_path: Path to the image to display
    """
    try:
        with Image.open(image_path) as img:

            # convert the image to RGB if it's not in that mode
            if img.mode != 'RGB':
                img = img.convert('RGB')

           # create a new Matplotlib window and display the image 
            plt.figure(figsize=(8, 8))

            # Display the image
            plt.imshow(img)
        
            plt.axis('off')  # Hide axes
            plt.title(f"Displaying: {image_path.name}")
            # Show the image without blocking the terminal input.
            plt.show(block=False)

            # Give Matplotlib enough time to render the window.
            plt.pause(0.1)
    except Exception as e:
        print(f"Error displaying image {image_path}: {e}")


def close_image_window() -> None:
    """
    Close the Matplotlib image window.
    """
    plt.close()

def get_destination_directory(class_name: str) -> Path | None:
    """
    Convert a keyboard input to a destination directory path.
    Args: choidce: Label selected by the user
    Returns:
    Destination directory assosiated with the selected label.
    None if the input is invalid.
    """

    # Map user input to destination directories
    destination_map = {
        "1": EMPTY_DIR,
        "2": PERSON_DIR,
        "3": ANIMAL_DIR,
        "r": REJECTED_DIR
    }
    return destination_map.get(class_name)

def build_destination_path(
    image_path: Path, destination_dir: Path
) -> Path:
    """
    Build a unique destination path for the image in the specified directory.
    Args:
    image_path: Path to the image to move
    destination_dir: Directory where the image will be moved
    Returns:
    A safe destination path for the image in the specified directory."""

    # Create the destination directory if it doesn't exist
    destination_dir.mkdir(parents=True, exist_ok=True)

    desstination_path = destination_dir / image_path.name
    counter = 1
    # Add a numerical suffix to the filename if a file with the same name already exists
    while desstination_path.exists():
        desstination_path = (
            destination_dir / f"{image_path.stem}_{counter}{image_path.suffix}"
        )
        counter += 1
    return desstination_path


def move_image(image_path: Path, destination_dir: Path) -> Path:
    """
    Move an image to the specified destination directory.
    Args:
    image_path: Original image path
    destination_dir: Directory where the image will be moved
    Returns:
    Final destination path 
    """

    # calculate a safe path that does not overwrite existing files
    destination_path = build_destination_path(image_path, destination_dir)

    # move the file from its current location to the destination directory
    shutil.move(str(image_path), str(destination_path))
    return destination_path

def main() -> None:
    """
    Run the manual image-labeling process.
    """

    print("Manual Image Labeling Tool")
    print(f"Searching for images in: {RAW_DATASET_DIR}")
    print()
    print("Commands:")
    print("1: Empty")
    print("2: Person")
    print("3: Animal")
    print("r: Rejected")
    print("q: Quit")
    print()

    image_path = get_image_path()

    if not image_path:
        print("No supported image files found.")
        return

    total_images = len(image_path)
    labeled_count = 0

    # Loop through each image in the raw dataset directory
    for index, image_path in enumerate(image_path, start=1):
        print(f"Processing image {index}/{total_images}: {image_path.name}")

        # Display the image to the user
        display_image(image_path)

        while True:
            # Ask the user to choose the correct label.
            choice = input(
                "Select label [1/2/3/r/q]: "
            ).strip().lower()

            # Stop the program without moving the current image.
            if choice == "q":
                close_image_window()
                print("Labeling stopped by the user.")
                print(f"Images labeled: {labeled_count}")
                return

            # Translate the selected command into a destination directory.
            destination_dir = get_destination_directory(choice)

            # Ask again if the command is not valid.
            if destination_dir is None:
                print("Invalid command. Use 1, 2, 3, r or q.")
                continue

            # The image is already inside the empty directory.
            # In this case, no physical move is required.
            if destination_dir == image_path.parent:
                print(f"Kept in empty: {image_path.name}")
            else:
                destination_path = move_image(
                    image_path,
                    destination_dir,
                )

                print(
                    f"Moved: {image_path.name} "
                    f"-> {destination_path.relative_to(PROJECT_ROOT)}"
                )

            labeled_count += 1
            close_image_window()
            print()
            break

    print("Labeling completed.")
    print(f"Images labeled: {labeled_count}")

if __name__ == "__main__":
    main()