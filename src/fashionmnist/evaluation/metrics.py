"""
metrics.py — Classification metrics for FashionMNIST.

Computes:
  - Accuracy
  - Per-class and aggregate precision, recall, F1-score
  - Confusion matrix
  - Full sklearn classification report

All metrics are computed from numpy arrays of true and predicted labels.
Gathering predictions from a DataLoader is also provided.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# Prediction gathering
# ---------------------------------------------------------------------------

@torch.no_grad()
def gather_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on all batches in loader and collect outputs.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader (no shuffle).
        device: Computation device.

    Returns:
        (all_labels, all_preds, all_probs)
        - all_labels: (N,) int array of ground-truth labels.
        - all_preds:  (N,) int array of predicted class indices.
        - all_probs:  (N, 10) float array of softmax probabilities.
    """
    model.eval()
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs, axis=0),
    )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground-truth labels (N,).
        y_pred: Predicted labels (N,).
        class_names: Optional list of class names.

    Returns:
        Dict with keys: accuracy, precision_macro, recall_macro, f1_macro,
        precision_weighted, recall_weighted, f1_weighted,
        per_class (list of per-class dicts), confusion_matrix (list of lists).
    """
    if class_names is None:
        class_names = _CLASS_NAMES

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = [
        {
            "class": class_names[i],
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": int((y_true == i).sum()),
        }
        for i in range(len(class_names))
    ]

    return {
        "accuracy": round(acc, 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return and print the sklearn classification report."""
    report = classification_report(
        y_true, y_pred,
        target_names=_CLASS_NAMES,
        digits=4,
    )
    print(report)
    return report


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Save metrics dict as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved → {output_path}")
