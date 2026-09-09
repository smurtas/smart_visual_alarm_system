"""
Description:
Define, train, validate, and optionally test a compact TinyViT-style
model for three-class image classification.

The model is selected using validation loss. At the end of training,
the script asks the user whether to evaluate the final checkpoint on
the test set.

Classes:
- animal
- empty
- person

This is a lightweight Vision Transformer designed for the current
TinyML experiment. It is not the official Microsoft TinyViT model.
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

RESULTS_DIR = PROJECT_ROOT / "results" / "tinyvit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "best_tinyvit_V3.pt"

IMAGE_SIZE = 48
PATCH_SIZE = 6

EMBED_DIM = 48
NUM_HEADS = 3
NUM_LAYERS = 2
MLP_RATIO = 2
DROPOUT = 0.05

EPOCHS = 180
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0005

EARLY_STOPPING_PATIENCE = 25
SCHEDULER_PATIENCE = 8


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Convert an RGB image into a sequence of patch embeddings."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "IMAGE_SIZE must be divisible by PATCH_SIZE."
            )

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images into patch tokens."""

        patches = self.projection(images)

        # B x C x H x W -> B x C x N
        patches = patches.flatten(2)

        # B x C x N -> B x N x C
        patches = patches.transpose(1, 2)

        return patches


# ---------------------------------------------------------------------------
# Transformer encoder block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Lightweight Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()

        hidden_dim = embed_dim * mlp_ratio

        self.normalization_1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(dropout)

        self.normalization_2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the Transformer block output."""

        normalized_inputs = self.normalization_1(inputs)

        attention_output, _ = self.attention(
            normalized_inputs,
            normalized_inputs,
            normalized_inputs,
            need_weights=False,
        )

        inputs = (
            inputs
            + self.attention_dropout(attention_output)
        )

        feed_forward_output = self.feed_forward(
            self.normalization_2(inputs)
        )

        return inputs + feed_forward_output


# ---------------------------------------------------------------------------
# TinyViT-style model
# ---------------------------------------------------------------------------

class TinyViT(nn.Module):
    """Compact Vision Transformer for 48 x 48 RGB images."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        mlp_ratio: int = MLP_RATIO,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embedding.num_patches

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embed_dim,
            )
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_normalization = nn.LayerNorm(embed_dim)

        self.classifier = nn.Linear(
            embed_dim,
            num_classes,
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize model parameters."""

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.class_token,
            std=0.02,
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return classification logits."""

        patch_tokens = self.patch_embedding(images)

        batch_size = patch_tokens.size(0)

        class_tokens = self.class_token.expand(
            batch_size,
            -1,
            -1,
        )

        tokens = torch.cat(
            (class_tokens, patch_tokens),
            dim=1,
        )

        tokens = tokens + self.position_embedding
        tokens = self.embedding_dropout(tokens)

        tokens = self.transformer_blocks(tokens)
        tokens = self.final_normalization(tokens)

        class_output = tokens[:, 0]

        return self.classifier(class_output)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    """Select the best available computing device."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def ask_to_run_test() -> bool:
    """
    Ask whether the final checkpoint should be evaluated on the test set.

    Returns:
        True only when the user explicitly answers yes.
    """

    print("\nThe model configuration should now be considered final.")
    print(
        "Choose 'n' to modify the hyperparameters and repeat training "
        "without accessing the test set."
    )

    while True:
        answer = input(
            "\nProceed with the final test evaluation? [y/n]: "
        ).strip().lower()

        if answer in {"y", "yes", "s", "si", "sì"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Invalid answer. Enter 'y' or 'n'.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
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

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = logits.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


# ---------------------------------------------------------------------------
# Validation and test
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
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

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    class_names: list[str],
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
) -> None:
    """Save the best TinyViT checkpoint."""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "mlp_ratio": MLP_RATIO,
            "dropout": DROPOUT,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "trainable_parameters": count_parameters(model),
        },
        MODEL_PATH,
    )


# ---------------------------------------------------------------------------
# Main procedure
# ---------------------------------------------------------------------------

def main() -> None:
    """Train TinyViT and optionally evaluate it on the test set."""

    (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    ) = get_dataloaders()

    sample_images, _ = next(iter(train_loader))

    returned_height = sample_images.shape[2]
    returned_width = sample_images.shape[3]

    if (
        returned_height != IMAGE_SIZE
        or returned_width != IMAGE_SIZE
    ):
        raise ValueError(
            "Incorrect image size returned by the DataLoader. "
            f"Expected {IMAGE_SIZE} x {IMAGE_SIZE}, "
            f"but received {returned_height} x {returned_width}."
        )

    device = select_device()

    model = TinyViT(
        num_classes=len(class_names),
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
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-6,
    )

    print("\nTinyViT-style training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Input size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
    print(
        f"Number of patches: "
        f"{(IMAGE_SIZE // PATCH_SIZE) ** 2}"
    )
    print(f"Embedding dimension: {EMBED_DIM}")
    print(f"Attention heads: {NUM_HEADS}")
    print(f"Transformer blocks: {NUM_LAYERS}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Checkpoint: {MODEL_PATH}")
    print("=" * 70)

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(validation_loss)

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_accuracy:.4f} | "
            f"val loss: {validation_loss:.4f} | "
            f"val acc: {validation_accuracy:.4f} | "
            f"lr: {current_learning_rate:.7f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                class_names=class_names,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
            )

            print(f"  Saved best model: {MODEL_PATH}")

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"\nEarly stopping after {epoch} epochs."
            )
            break

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint was created: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model_size_mb = (
        MODEL_PATH.stat().st_size / (1024 ** 2)
    )

    print("\nBest validation checkpoint")
    print("=" * 70)
    print(f"Epoch: {checkpoint['epoch']}")
    print(
        f"Validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )
    print(
        f"Validation accuracy: "
        f"{checkpoint['validation_accuracy']:.4f}"
    )
    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )
    print(f"Checkpoint size: {model_size_mb:.2f} MB")
    print(f"Saved checkpoint: {MODEL_PATH}")

    # The test set is not accessed unless the user explicitly confirms.
    if not ask_to_run_test():
        print("\nTest evaluation skipped.")
        print(
            "Modify the hyperparameters at the top of this file, "
            "then run the training again."
        )
        return

    test_loss, test_accuracy = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nFinal test results")
    print("=" * 70)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )
    print(f"Checkpoint size: {model_size_mb:.2f} MB")
    print(f"Evaluated checkpoint: {MODEL_PATH}")


if __name__ == "__main__":
    main()