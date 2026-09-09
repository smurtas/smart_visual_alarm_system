"""
Description:
Evaluate the trained MCUNet-style model on the test dataset.

The script calculates:
- test loss;
- accuracy;
- precision, recall, and F1-score for each class;
- macro and weighted averages;
- confusion matrix;
- number of trainable parameters;
- model checkpoint size;
- average inference time.

The results are saved inside results/mcunet/.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch import nn

from dataset import get_dataloaders
from train_mcunet import MCUNet


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "mcunet"
MODEL_PATH = RESULTS_DIR / "best_mcunet.pt"

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR / "confusion_matrix_mcunet.png"
)

REPORT_PATH = (
    RESULTS_DIR / "classification_report_mcunet.txt"
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    """Select MPS, CUDA, or CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def count_trainable_parameters(model: nn.Module) -> int:
    """Count all trainable parameters in the model."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def calculate_model_size_megabytes(
    model_path: Path,
) -> float:
    """Return the checkpoint size in megabytes."""

    return model_path.stat().st_size / (1024 ** 2)


def synchronize_device(device: torch.device) -> None:
    """Synchronize the selected accelerator."""

    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    test_loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int], float]:
    """
    Evaluate the model and collect predictions.

    Returns:
        Average test loss.
        True labels.
        Predicted labels.
        Average inference time per image in milliseconds.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_inference_time = 0.0

    true_labels: list[int] = []
    predicted_labels: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            synchronize_device(device)

            start_time = time.perf_counter()

            logits = model(images)

            synchronize_device(device)

            end_time = time.perf_counter()

            total_inference_time += (
                end_time - start_time
            )

            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits.argmax(dim=1)

            true_labels.extend(
                labels.cpu().tolist()
            )

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

    average_loss = total_loss / total_samples

    average_inference_time_ms = (
        total_inference_time / total_samples
    ) * 1000

    return (
        average_loss,
        true_labels,
        predicted_labels,
        average_inference_time_ms,
    )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    true_labels: list[int],
    predicted_labels: list[int],
    class_names: list[str],
) -> None:
    """Create and save the confusion matrix."""

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        "MCUNet-style confusion matrix"
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ---------------------------------------------------------------------------
# Main procedure
# ---------------------------------------------------------------------------

def main() -> None:
    """Evaluate the best MCUNet checkpoint."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        _,
        _,
        test_loader,
        dataset_class_names,
    ) = get_dataloaders()

    device = select_device()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    class_names = checkpoint.get(
        "class_names",
        dataset_class_names,
    )

    dropout = checkpoint.get(
        "dropout",
        0.20,
    )

    image_size = checkpoint.get(
        "image_size",
        48,
    )

    sample_images, _ = next(
        iter(test_loader)
    )

    returned_size = sample_images.shape[2:]

    if returned_size != (
        image_size,
        image_size,
    ):
        raise ValueError(
            "Incorrect test image size. "
            f"Checkpoint expects {image_size} x {image_size}, "
            f"but the DataLoader returned "
            f"{returned_size[0]} x {returned_size[1]}."
        )

    model = MCUNet(
        num_classes=len(class_names),
        dropout=dropout,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    criterion = nn.CrossEntropyLoss()

    (
        test_loss,
        true_labels,
        predicted_labels,
        average_inference_time_ms,
    ) = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    trainable_parameters = (
        count_trainable_parameters(model)
    )

    model_size_mb = (
        calculate_model_size_megabytes(
            MODEL_PATH
        )
    )

    print("\nMCUNet-style evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Classes: {class_names}")
    print(f"Input size: {image_size} x {image_size}")

    print("\nOverall metrics")
    print("-" * 60)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )
    print(
        f"Checkpoint size: "
        f"{model_size_mb:.2f} MB"
    )
    print(
        "Average inference time per image: "
        f"{average_inference_time_ms:.3f} ms"
    )

    print("\nClassification report")
    print("-" * 60)
    print(report)

    report_content = (
        "MCUNet-style evaluation\n"
        + "=" * 60
        + "\n"
        + f"Test loss: {test_loss:.4f}\n"
        + f"Test accuracy: {accuracy:.4f}\n"
        + (
            "Trainable parameters: "
            f"{trainable_parameters:,}\n"
        )
        + (
            "Checkpoint size: "
            f"{model_size_mb:.2f} MB\n"
        )
        + (
            "Average inference time per image: "
            f"{average_inference_time_ms:.3f} ms\n\n"
        )
        + "Classification report\n"
        + "-" * 60
        + "\n"
        + report
    )

    REPORT_PATH.write_text(
        report_content,
        encoding="utf-8",
    )

    save_confusion_matrix(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names,
    )

    print("\nSaved results")
    print("-" * 60)
    print(f"Report: {REPORT_PATH}")
    print(
        f"Confusion matrix: "
        f"{CONFUSION_MATRIX_PATH}"
    )


if __name__ == "__main__":
    main()