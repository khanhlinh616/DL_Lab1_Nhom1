"""
error_analysis.py — Analysis of misclassified samples.

Finds misclassified examples, identifies most-confused class pairs,
and saves grids of wrong predictions with true/predicted labels and confidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fashionmnist.evaluation.metrics import gather_predictions


_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def find_misclassified(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> dict[str, Any]:
    """
    Identify misclassified indices and most-confused class pairs.

    Args:
        y_true: Ground-truth labels (N,).
        y_pred: Predicted labels (N,).
        y_probs: Softmax probabilities (N, 10).

    Returns:
        Dict with:
          - wrong_indices: indices of misclassified samples.
          - wrong_true: true labels at wrong_indices.
          - wrong_pred: predicted labels at wrong_indices.
          - wrong_conf: confidence of wrong prediction.
          - top_confused_pairs: list of (true_class, pred_class, count) sorted desc.
          - error_rate_per_class: per-class error rate.
    """
    wrong_mask = y_true != y_pred
    wrong_indices = np.where(wrong_mask)[0]
    wrong_true = y_true[wrong_mask]
    wrong_pred = y_pred[wrong_mask]
    wrong_conf = y_probs[wrong_mask, wrong_pred]

    # Most confused pairs
    pair_counts: dict[tuple[int, int], int] = {}
    for t, p in zip(wrong_true, wrong_pred):
        pair_counts[(int(t), int(p))] = pair_counts.get((int(t), int(p)), 0) + 1
    top_confused = sorted(pair_counts.items(), key=lambda x: -x[1])
    top_confused_pairs = [
        {
            "true_class": _CLASS_NAMES[t],
            "pred_class": _CLASS_NAMES[p],
            "count": cnt,
        }
        for (t, p), cnt in top_confused[:10]
    ]

    # Per-class error rate
    error_rate_per_class: dict[str, float] = {}
    for cls in range(10):
        cls_mask = y_true == cls
        if cls_mask.sum() == 0:
            error_rate_per_class[_CLASS_NAMES[cls]] = 0.0
        else:
            wrong_in_cls = ((y_true == cls) & (y_pred != cls)).sum()
            error_rate_per_class[_CLASS_NAMES[cls]] = round(
                float(wrong_in_cls / cls_mask.sum()), 4
            )

    return {
        "total_errors": int(wrong_mask.sum()),
        "total_samples": len(y_true),
        "error_rate": round(float(wrong_mask.mean()), 6),
        "wrong_indices": wrong_indices,
        "wrong_true": wrong_true,
        "wrong_pred": wrong_pred,
        "wrong_conf": wrong_conf,
        "top_confused_pairs": top_confused_pairs,
        "error_rate_per_class": error_rate_per_class,
    }


def plot_misclassified_grid(
    dataset: Any,
    wrong_indices: np.ndarray,
    wrong_true: np.ndarray,
    wrong_pred: np.ndarray,
    wrong_conf: np.ndarray,
    figures_dir: Path,
    n_samples: int = 25,
    mean: float = 0.2860,
    std: float = 0.3530,
) -> None:
    """
    Plot a grid of misclassified images with true/predicted labels and confidence.

    Images are de-normalized for display.

    Args:
        dataset: Dataset (test set) to retrieve images from.
        wrong_indices: Indices of wrong predictions.
        wrong_true: True labels at those indices.
        wrong_pred: Predicted labels at those indices.
        wrong_conf: Confidence of wrong predictions.
        figures_dir: Where to save the figure.
        n_samples: How many samples to show (max).
        mean, std: Normalization constants for de-normalizing display.
    """
    n = min(n_samples, len(wrong_indices))
    rng = np.random.default_rng(seed=42)
    chosen = rng.choice(len(wrong_indices), size=n, replace=False)

    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"Error Analysis — {n} Misclassified Samples", fontsize=12)

    for ax_idx, choice_idx in enumerate(chosen):
        ds_idx = int(wrong_indices[choice_idx])
        img_tensor, _ = dataset[ds_idx]           # (1, 28, 28) normalized
        img_np = img_tensor.squeeze(0).numpy()     # (28, 28)
        # De-normalize for display
        img_display = img_np * std + mean
        img_display = np.clip(img_display, 0.0, 1.0)

        true_name = _CLASS_NAMES[int(wrong_true[choice_idx])]
        pred_name = _CLASS_NAMES[int(wrong_pred[choice_idx])]
        conf = wrong_conf[choice_idx]

        ax = axes[ax_idx]
        ax.imshow(img_display, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"True: {true_name}\nPred: {pred_name} ({conf:.2%})",
            fontsize=7, color="red",
        )
        ax.axis("off")

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    out_path = figures_dir / "error_misclassified_grid.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[ErrorAnalysis] Saved → {out_path}")


def plot_correct_predictions_grid(
    dataset: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    figures_dir: Path,
    n_samples: int = 25,
    mean: float = 0.2860,
    std: float = 0.3530,
) -> None:
    """Plot a grid of correctly classified samples with confidence."""
    correct_mask = y_true == y_pred
    correct_indices = np.where(correct_mask)[0]
    correct_conf = y_probs[correct_mask, y_pred[correct_mask]]

    n = min(n_samples, len(correct_indices))
    rng = np.random.default_rng(seed=99)
    chosen = rng.choice(len(correct_indices), size=n, replace=False)

    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"Correct Predictions — {n} Samples", fontsize=12)

    for ax_idx, choice_idx in enumerate(chosen):
        ds_idx = int(correct_indices[choice_idx])
        img_tensor, _ = dataset[ds_idx]
        img_np = img_tensor.squeeze(0).numpy()
        img_display = np.clip(img_np * std + mean, 0.0, 1.0)
        pred_name = _CLASS_NAMES[int(y_pred[correct_indices[choice_idx]])]
        conf = correct_conf[choice_idx]

        ax = axes[ax_idx]
        ax.imshow(img_display, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"{pred_name}\n({conf:.2%})", fontsize=7, color="green")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    out_path = figures_dir / "correct_predictions_grid.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[ErrorAnalysis] Saved → {out_path}")
