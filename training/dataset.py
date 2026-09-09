"""
Description:
Create PyTorch datasets and data loaders for training, validation, and test.
"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from augmentation import train_transform, evaluation_transform


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" 

BATCH_SIZE = 16
NUM_WORKERS = 0


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def get_datasets():
    """
    Create training, validation, and test datasets.

    Returns:
        tuple:
            train_dataset,
            val_dataset,
            test_dataset
    """

    train_directory = DATASET_DIR / "train"
    val_directory = DATASET_DIR / "val"
    test_directory = DATASET_DIR / "test"

    for directory in (
        train_directory,
        val_directory,
        test_directory,
    ):
        if not directory.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {directory}"
            )

    train_dataset = datasets.ImageFolder(
        root=train_directory,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=val_directory,
        transform=evaluation_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=test_directory,
        transform=evaluation_transform,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ---------------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------------

def get_dataloaders():
    """
    Create training, validation, and test data loaders.

    Returns:
        tuple:
            train_loader,
            val_loader,
            test_loader,
            class_names
    """

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = get_datasets()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    class_names = train_dataset.classes

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def main() -> None:
    """Test dataset and data loader creation."""

    (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    ) = get_dataloaders()

    print("\nDataset loaded successfully")
    print("=" * 50)

    print(f"Classes: {class_names}")
    print(
        f"Class mapping: "
        f"{train_loader.dataset.class_to_idx}"
    )

    print(
        f"Training images: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test images: "
        f"{len(test_loader.dataset)}"
    )

    images, labels = next(iter(train_loader))

    print("\nFirst training batch")
    print(f"Image tensor shape: {images.shape}")
    print(f"Label tensor shape: {labels.shape}")
    print(f"Labels: {labels}")


if __name__ == "__main__":
    main()