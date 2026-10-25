"""
plots.py — Training and evaluation visualization utilities.

Produces:
  - Training history: loss and accuracy curves.
  - Confusion matrix heatmap.
  - Per-class F1-score bar chart.
  - Experiment comparison table figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def plot_training_history(
    history: dict[str, list],
    figures_dir: Path,
    experiment_name: str = "default",
) -> None:
    """
    Plot and save training loss and accuracy curves.

    Args:
        history: Dict with keys train_loss, val_loss, train_acc, val_acc.
        figures_dir: Output directory.
        experiment_name: Used in filename and title.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training History — {experiment_name}", fontsize=12)

    # Loss
    ax_loss.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax_loss.plot(epochs, history["val_loss"], label="Val Loss", marker="o", markersize=3)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Accuracy
    ax_acc.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o", markersize=3)
    ax_acc.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", marker="o", markersize=3)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = figures_dir / f"training_history_{experiment_name}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved → {out_path}")


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    figures_dir: Path,
    experiment_name: str = "default",
    normalize: bool = True,
) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix (10×10).
        figures_dir: Output directory.
        experiment_name: Used in filename.
        normalize: If True, normalize rows to show per-class recall.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_arr = np.array(cm)

    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm_arr.astype(float) / row_sums
        fmt = ".2f"
        title_suffix = " (Normalized by row — Recall)"
    else:
        cm_display = cm_arr
        fmt = "d"
        title_suffix = " (Counts)"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=_CLASS_NAMES,
        yticklabels=_CLASS_NAMES,
        ax=ax,
        linewidths=0.3,
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(f"Confusion Matrix — {experiment_name}{title_suffix}", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    suffix = "normalized" if normalize else "counts"
    out_path = figures_dir / f"confusion_matrix_{experiment_name}_{suffix}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved → {out_path}")


def plot_per_class_f1(
    per_class_metrics: list[dict[str, Any]],
    figures_dir: Path,
    experiment_name: str = "default",
) -> None:
    """
    Plot per-class F1-score as a horizontal bar chart.

    Args:
        per_class_metrics: List of dicts with 'class' and 'f1' keys.
        figures_dir: Output directory.
        experiment_name: Used in filename.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    classes = [m["class"] for m in per_class_metrics]
    f1_scores = [m["f1"] for m in per_class_metrics]

    colors = ["#2196F3" if f1 >= 0.9 else "#FF9800" if f1 >= 0.8 else "#F44336" for f1 in f1_scores]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(classes, f1_scores, color=colors, alpha=0.85)
    ax.set_xlabel("F1-Score")
    ax.set_title(f"Per-Class F1-Score — {experiment_name}")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.9, color="green", linestyle="--", linewidth=1, alpha=0.6, label="0.90")
    ax.axvline(x=0.8, color="orange", linestyle="--", linewidth=1, alpha=0.6, label="0.80")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = figures_dir / f"per_class_f1_{experiment_name}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved → {out_path}")


def plot_experiment_comparison(
    experiments_summary: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    """
    Plot a grouped bar chart comparing key metrics across experiments.

    Args:
        experiments_summary: List of dicts, each with keys:
          name, val_acc, test_acc (optional), f1_weighted.
        figures_dir: Output directory.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not experiments_summary:
        return

    names = [e["name"] for e in experiments_summary]
    val_accs = [e.get("val_acc", 0) * 100 for e in experiments_summary]
    f1s = [e.get("f1_weighted", 0) * 100 for e in experiments_summary]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    ax.bar(x - width / 2, val_accs, width, label="Val Acc (%)", color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, f1s, width, label="F1 Weighted (%)", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("Experiment Comparison")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = figures_dir / "experiment_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved → {out_path}")
