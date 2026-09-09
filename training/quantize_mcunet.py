"""
Description:
Load the trained PyTorch MCUNet checkpoint and quantize it directly
to the ESP-DL INT8 format for ESP32-S3.

This version bypasses ONNX and removes the Flatten/Reshape operation
from the deployment graph.
"""

from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from esp_ppq.api import espdl_quantize_torch


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_mcunet import MCUNet


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "mcunet"
    / "best_mcunet.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "mcunet"
    / "mcunet_int8.espdl"
)

CALIBRATION_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "calibration"
)

IMAGE_SIZE = 48

# ESP-DL supports batch size 1 on the device.
INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]

TARGET = "esp32s3"
DEVICE = "cpu"
NUM_OF_BITS = 8

BATCH_SIZE = 1
CALIBRATION_STEPS = 60

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Deployment model
# ---------------------------------------------------------------------------

class DeploymentMCUNet(nn.Module):
    """
    MCUNet deployment graph without Flatten/Reshape.

    The original feature extractor is preserved. Global average pooling
    is performed directly with torch.mean, producing a 2D tensor that
    can be passed to the final Linear layer.
    """

    def __init__(self, trained_model: MCUNet) -> None:
        super().__init__()

        self.features = trained_model.features

        linear_layers = [
            module
            for module in trained_model.modules()
            if isinstance(module, nn.Linear)
        ]

        if not linear_layers:
            raise RuntimeError(
                "No Linear classification layer found in MCUNet."
            )

        self.classifier = linear_layers[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)

        # Global average pooling without AdaptiveAvgPool + Flatten.
        features = torch.mean(
            features,
            dim=(2, 3),
        )

        return self.classifier(features)


# ---------------------------------------------------------------------------
# Calibration dataset
# ---------------------------------------------------------------------------

calibration_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


def calibration_collate_fn(batch):
    """Return only the batch of image tensors."""

    images, _ = zip(*batch)

    return torch.stack(images)


def create_calibration_loader() -> DataLoader:
    """Create the representative calibration dataset."""

    if not CALIBRATION_DIR.exists():
        raise FileNotFoundError(
            f"Calibration directory not found: {CALIBRATION_DIR}"
        )

    dataset = datasets.ImageFolder(
        root=CALIBRATION_DIR,
        transform=calibration_transform,
    )

    if len(dataset) < CALIBRATION_STEPS:
        raise RuntimeError(
            f"Expected at least {CALIBRATION_STEPS} images, "
            f"found {len(dataset)}."
        )

    return DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=calibration_collate_fn,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_deployment_model() -> tuple[nn.Module, dict]:
    """Load the trained checkpoint and build the deployment graph."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    class_names = checkpoint.get(
        "class_names",
        ["animal", "empty", "person"],
    )

    dropout = checkpoint.get(
        "dropout",
        0.20,
    )

    trained_model = MCUNet(
        num_classes=len(class_names),
        dropout=dropout,
    )

    trained_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    trained_model.eval()

    deployment_model = DeploymentMCUNet(
        trained_model
    )

    deployment_model.eval()

    return deployment_model, checkpoint


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_deployment_output(
    model: nn.Module,
) -> None:
    """Check that the deployment graph returns three logits."""

    sample_input = torch.randn(
        *INPUT_SHAPE,
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = model(sample_input)

    print(f"Deployment output shape: {tuple(output.shape)}")

    if tuple(output.shape) != (1, 3):
        raise RuntimeError(
            "Unexpected output shape. "
            f"Expected (1, 3), obtained {tuple(output.shape)}."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Quantize MCUNet directly from PyTorch."""

    model, checkpoint = load_deployment_model()

    verify_deployment_output(model)

    calibration_loader = create_calibration_loader()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove incomplete outputs from previous attempts.
    for suffix in (".espdl", ".info", ".json"):
        path = OUTPUT_PATH.with_suffix(suffix)

        if path.exists():
            path.unlink()

    print("\nDirect PyTorch to ESP-DL quantization")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Calibration images: {len(calibration_loader.dataset)}")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Target: {TARGET}")
    print(f"Precision: INT{NUM_OF_BITS}")
    print(
        "Classes: "
        f"{checkpoint.get('class_names', ['animal', 'empty', 'person'])}"
    )
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 70)

    espdl_quantize_torch(
        model=model,
        espdl_export_file=str(OUTPUT_PATH),
        calib_dataloader=calibration_loader,
        calib_steps=CALIBRATION_STEPS,
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=None,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )

    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            f"ESP-DL output was not created: {OUTPUT_PATH}"
        )

    print("\nQuantization completed successfully")
    print("-" * 70)
    print(f"ESP-DL model: {OUTPUT_PATH}")
    print(
        f"Model size: "
        f"{OUTPUT_PATH.stat().st_size / 1024:.2f} KB"
    )

    for suffix in (".info", ".json"):
        companion = OUTPUT_PATH.with_suffix(suffix)
        print(f"{suffix}: {companion.exists()} — {companion}")


if __name__ == "__main__":
    main()