"""
Description:
Display one original image and multiple augmented versions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from augmentation import train_transform


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANIMAL_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "training"
    / "train"
    / "animal"
)

IMAGE_PATH = next(
    (
        path
        for path in ANIMAL_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ),
    None,
)


def denormalize(tensor):
    """Undo ImageNet normalization for visualization."""

    mean = tensor.new_tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = tensor.new_tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    tensor = tensor * std + mean

    return tensor.clamp(0, 1)


def main() -> None:
    """Show multiple augmented versions of one image."""

    if IMAGE_PATH is None:
        raise FileNotFoundError(
            f"No images found in: {ANIMAL_DIR}"
        )

    print(f"Selected image: {IMAGE_PATH}")

    image = Image.open(IMAGE_PATH).convert("RGB")

    figure, axes = plt.subplots(
        3,
        4,
        figsize=(12, 9),
    )

    for axis in axes.flat:
        augmented_tensor = train_transform(image)
        augmented_tensor = denormalize(augmented_tensor)

        augmented_image = (
            augmented_tensor
            .permute(1, 2, 0)
            .numpy()
        )

        axis.imshow(augmented_image)
        axis.axis("off")

    figure.suptitle(
        f"Augmentation preview: {IMAGE_PATH.name}"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()