"""
Description:
Train a simple multilayer perceptron baseline for image classification.
"""

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "mlp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "best_mlp_v2.pt"

IMAGE_SIZE = 48

EPOCHS = 100
LEARNING_RATE = 0.0002
PATIENCE = 12
WEIGHT_DECAY = 0.0005


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """Compact MLP baseline for 48 x 48 RGB images."""  

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        input_features = 3 * IMAGE_SIZE * IMAGE_SIZE # 3 channels, 48x48 pixels, changed from 96x96 to 48x48 for faster training

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(32, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model without updating its parameters."""

    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Train the MLP baseline."""

    (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    ) = get_dataloaders()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Classes: {class_names}")

    model = MLPClassifier(
        num_classes=len(class_names)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_accuracy:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": IMAGE_SIZE,
                },
                MODEL_PATH,
            )

            print(f"  Saved best model: {MODEL_PATH}")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping after {epoch} epochs."
            )
            break

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\nFinal test results")
    print("=" * 50)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()