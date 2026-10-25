"""
dataset.py — FashionMNIST dataset loading.

Downloads (if needed) and returns torchvision FashionMNIST datasets
with raw transforms (ToTensor only) for EDA, or full transforms for training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T


_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_raw_dataset(
    data_root: str | Path,
    train: bool = True,
    download: bool = True,
) -> datasets.FashionMNIST:
    """
    Load FashionMNIST with minimal transform (ToTensor only, no normalization).

    Used during EDA to inspect original pixel distributions.

    Args:
        data_root: Directory where the dataset will be stored.
        train: If True, load the 60 000-sample train split.
        download: If True, download if not already present.

    Returns:
        FashionMNIST dataset instance.

    Raises:
        RuntimeError: If download fails or files are corrupted.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    return datasets.FashionMNIST(
        root=str(data_root),
        train=train,
        transform=T.ToTensor(),  # pixels in [0, 1]; shape (1, 28, 28)
        download=download,
    )


def load_dataset_with_transform(
    data_root: str | Path,
    train: bool,
    transform: T.Compose,
    download: bool = True,
) -> datasets.FashionMNIST:
    """
    Load FashionMNIST with a given torchvision transform.

    Args:
        data_root: Directory where the dataset is stored.
        train: If True, load train split; otherwise test split.
        transform: Composed transform to apply.
        download: Whether to download if missing.

    Returns:
        FashionMNIST dataset instance.
    """
    data_root = Path(data_root)
    return datasets.FashionMNIST(
        root=str(data_root),
        train=train,
        transform=transform,
        download=download,
    )


def get_class_names() -> list[str]:
    """Return the canonical 10-class label list (index = label integer)."""
    return list(_CLASS_NAMES)


def verify_dataset_integrity(ds: datasets.FashionMNIST) -> dict[str, Any]:
    """
    Run basic integrity checks on a FashionMNIST dataset.

    Checks:
      - Total sample count matches expected size.
      - All labels are integers in [0, 9].
      - No label is outside [0, 9].

    Args:
        ds: Dataset to check.

    Returns:
        Dict with keys: n_samples, label_min, label_max, invalid_label_count,
        expected_n_samples, passed.

    Raises:
        ValueError: If labels contain NaN-like objects (non-integer).
    """
    expected = 60_000 if ds.train else 10_000
    labels = ds.targets  # Tensor of shape (N,)

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    label_min = int(labels.min().item())
    label_max = int(labels.max().item())
    invalid_count = int(((labels < 0) | (labels > 9)).sum().item())

    result = {
        "n_samples": len(ds),
        "expected_n_samples": expected,
        "label_min": label_min,
        "label_max": label_max,
        "invalid_label_count": invalid_count,
        "passed": (len(ds) == expected) and (label_min == 0) and (label_max == 9) and (invalid_count == 0),
    }
    return result
