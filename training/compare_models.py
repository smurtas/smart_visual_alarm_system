"""
Description:
Compare the final MLP, MCUNet-style, and TinyViT-style models.

The script:
- prints a formatted comparison table;
- identifies the best model for each metric;
- saves the results as CSV and TXT files;
- generates comparison charts.

The metrics are entered directly in MODEL_RESULTS.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "model_comparison.csv"
TEXT_PATH = OUTPUT_DIR / "model_comparison.txt"

ACCURACY_PLOT_PATH = OUTPUT_DIR / "accuracy_comparison.png"
LOSS_PLOT_PATH = OUTPUT_DIR / "loss_comparison.png"
F1_PLOT_PATH = OUTPUT_DIR / "macro_f1_comparison.png"
PARAMETERS_PLOT_PATH = OUTPUT_DIR / "parameters_comparison.png"
SIZE_PLOT_PATH = OUTPUT_DIR / "model_size_comparison.png"
INFERENCE_PLOT_PATH = OUTPUT_DIR / "inference_time_comparison.png"


# ---------------------------------------------------------------------------
# Final model results
# ---------------------------------------------------------------------------

MODEL_RESULTS = [
    {
        "model": "MLP",
        "test_accuracy": 0.8167,
        "test_loss": 0.4410,
        "macro_f1": 0.8257,
        "parameters": 889_411,
        "model_size_mb": 3.40,
        "inference_time_ms": 4.495,
    },
    {
        "model": "MCUNet",
        "test_accuracy": 0.9000,
        "test_loss": 0.3140,
        "macro_f1": 0.9027,
        "parameters": 58_563,
        "model_size_mb": 0.28,
        "inference_time_ms": 5.845,
    },
    {
        "model": "TinyViT",
        "test_accuracy": 0.7333,
        "test_loss": 0.6422,
        "macro_f1": 0.7398,
        "parameters": 46_563,
        "model_size_mb": 0.19,
        "inference_time_ms": 5.530,
    },
]


# ---------------------------------------------------------------------------
# Formatting utilities
# ---------------------------------------------------------------------------

def format_optional_float(
    value: float | None,
    decimals: int = 4,
) -> str:
    """Format a float or return N/A when the value is unavailable."""

    if value is None:
        return "N/A"

    return f"{value:.{decimals}f}"


def build_comparison_table() -> str:
    """Build the formatted model comparison table."""

    header = (
        f"{'Model':<12}"
        f"{'Accuracy':>12}"
        f"{'Loss':>10}"
        f"{'Macro F1':>12}"
        f"{'Parameters':>15}"
        f"{'Size MB':>12}"
        f"{'Time ms':>12}"
    )

    separator = "-" * len(header)

    lines = [
        "FINAL MODEL COMPARISON",
        "=" * len(header),
        header,
        separator,
    ]

    for result in MODEL_RESULTS:
        lines.append(
            f"{result['model']:<12}"
            f"{result['test_accuracy']:>12.4f}"
            f"{result['test_loss']:>10.4f}"
            f"{format_optional_float(result['macro_f1']):>12}"
            f"{result['parameters']:>15,}"
            f"{result['model_size_mb']:>12.2f}"
            f"{format_optional_float(result['inference_time_ms'], 3):>12}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Best-model analysis
# ---------------------------------------------------------------------------

def get_best_available(
    metric: str,
    highest_is_best: bool,
) -> dict | None:
    """Return the best model among entries with an available value."""

    available_results = [
        result
        for result in MODEL_RESULTS
        if result[metric] is not None
    ]

    if not available_results:
        return None

    selector = max if highest_is_best else min

    return selector(
        available_results,
        key=lambda result: result[metric],
    )


def build_best_results_summary() -> str:
    """Build a summary of the best model for every metric."""

    best_accuracy = get_best_available(
        metric="test_accuracy",
        highest_is_best=True,
    )

    best_loss = get_best_available(
        metric="test_loss",
        highest_is_best=False,
    )

    best_f1 = get_best_available(
        metric="macro_f1",
        highest_is_best=True,
    )

    fewest_parameters = get_best_available(
        metric="parameters",
        highest_is_best=False,
    )

    smallest_model = get_best_available(
        metric="model_size_mb",
        highest_is_best=False,
    )

    fastest_model = get_best_available(
        metric="inference_time_ms",
        highest_is_best=False,
    )

    lines = [
        "",
        "BEST RESULTS",
        "=" * 70,
        (
            f"Highest test accuracy: "
            f"{best_accuracy['model']} "
            f"({best_accuracy['test_accuracy']:.4f})"
        ),
        (
            f"Lowest test loss: "
            f"{best_loss['model']} "
            f"({best_loss['test_loss']:.4f})"
        ),
    ]

    if best_f1 is not None:
        lines.append(
            f"Highest available macro F1: "
            f"{best_f1['model']} "
            f"({best_f1['macro_f1']:.4f})"
        )

    lines.extend(
        [
            (
                f"Fewest parameters: "
                f"{fewest_parameters['model']} "
                f"({fewest_parameters['parameters']:,})"
            ),
            (
                f"Smallest checkpoint: "
                f"{smallest_model['model']} "
                f"({smallest_model['model_size_mb']:.2f} MB)"
            ),
        ]
    )

    if fastest_model is not None:
        lines.append(
            f"Fastest available inference time: "
            f"{fastest_model['model']} "
            f"({fastest_model['inference_time_ms']:.3f} ms)"
        )

    lines.extend(
        [
            "",
            "Selected deployment model: MCUNet",
            (
                "Reason: highest test accuracy with a compact "
                "parameter count and checkpoint size."
            ),
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def save_csv() -> None:
    """Save the comparison data as CSV."""

    field_names = [
        "model",
        "test_accuracy",
        "test_loss",
        "macro_f1",
        "parameters",
        "model_size_mb",
        "inference_time_ms",
    ]

    with CSV_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=field_names,
        )

        writer.writeheader()
        writer.writerows(MODEL_RESULTS)


def save_text_report(report: str) -> None:
    """Save the formatted comparison report."""

    TEXT_PATH.write_text(
        report,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def save_bar_plot(
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    decimals: int = 4,
) -> None:
    """Generate one bar plot using entries with available values."""

    available_results = [
        result
        for result in MODEL_RESULTS
        if result[metric] is not None
    ]

    if not available_results:
        print(f"Skipped plot: no values available for {metric}")
        return

    model_names = [
        result["model"]
        for result in available_results
    ]

    values = [
        float(result[metric])
        for result in available_results
    ]

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    bars = axis.bar(
        model_names,
        values,
    )

    axis.set_title(title)
    axis.set_ylabel(ylabel)

    maximum_value = max(values)

    if maximum_value > 0:
        axis.set_ylim(
            0,
            maximum_value * 1.18,
        )

    for bar, value in zip(
        bars,
        values,
        strict=True,
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def generate_plots() -> None:
    """Generate all model-comparison plots."""

    save_bar_plot(
        metric="test_accuracy",
        title="Test accuracy comparison",
        ylabel="Accuracy",
        output_path=ACCURACY_PLOT_PATH,
    )

    save_bar_plot(
        metric="test_loss",
        title="Test loss comparison",
        ylabel="Cross-entropy loss",
        output_path=LOSS_PLOT_PATH,
    )

    save_bar_plot(
        metric="macro_f1",
        title="Macro F1-score comparison",
        ylabel="Macro F1-score",
        output_path=F1_PLOT_PATH,
    )

    save_bar_plot(
        metric="parameters",
        title="Trainable parameter comparison",
        ylabel="Trainable parameters",
        output_path=PARAMETERS_PLOT_PATH,
        decimals=0,
    )

    save_bar_plot(
        metric="model_size_mb",
        title="Checkpoint size comparison",
        ylabel="Checkpoint size (MB)",
        output_path=SIZE_PLOT_PATH,
        decimals=2,
    )

    save_bar_plot(
        metric="inference_time_ms",
        title="Inference time comparison",
        ylabel="Time per image (ms)",
        output_path=INFERENCE_PLOT_PATH,
        decimals=3,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Create the complete model comparison."""

    table = build_comparison_table()
    summary = build_best_results_summary()

    complete_report = f"{table}\n{summary}"

    print()
    print(complete_report)

    save_csv()
    save_text_report(complete_report)
    generate_plots()

    print("\nSaved comparison outputs")
    print("=" * 70)
    print(f"CSV: {CSV_PATH}")
    print(f"Text report: {TEXT_PATH}")
    print(f"Accuracy plot: {ACCURACY_PLOT_PATH}")
    print(f"Loss plot: {LOSS_PLOT_PATH}")
    print(f"Macro F1 plot: {F1_PLOT_PATH}")
    print(f"Parameters plot: {PARAMETERS_PLOT_PATH}")
    print(f"Model size plot: {SIZE_PLOT_PATH}")
    print(f"Inference plot: {INFERENCE_PLOT_PATH}")


if __name__ == "__main__":
    main()