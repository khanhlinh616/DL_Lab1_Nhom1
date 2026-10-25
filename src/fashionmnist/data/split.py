"""
split.py — Train/validation splitting for FashionMNIST.

Strategy: stratified split on the 60 000-sample official train set.
The official 10 000-sample test set is always kept intact.

After splitting:
  - train indices: used with augmented transforms
  - val indices:   used with eval transforms
  - test set:      loaded separately, never used in model selection

Data leakage prevention:
  - Normalization constants are computed (or taken from config) based on the
    TRAINING portion only.
  - Validation and test use the same normalization constants but are never
    used to set those constants.
  - Test set is only touched for final model evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

from fashionmnist.data.transforms import get_eval_transform, get_train_transform


def stratified_split(
    full_train_ds: datasets.FashionMNIST,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Stratified split of indices from full_train_ds.

    Each class is split proportionally to preserve label distribution.

    Args:
        full_train_ds: Raw FashionMNIST train dataset.
        val_fraction: Fraction of samples to put in validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices).

    Raises:
        ValueError: If val_fraction is not in (0, 1).
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    labels = full_train_ds.targets
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    labels_np = labels.numpy()

    rng = np.random.default_rng(seed=seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for cls in range(10):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_fraction))
        val_indices.extend(cls_idx[:n_val].tolist())
        train_indices.extend(cls_idx[n_val:].tolist())

    return train_indices, val_indices


def build_dataloaders(
    cfg: dict[str, Any],
    full_train_ds: datasets.FashionMNIST,
    test_ds: datasets.FashionMNIST,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders.

    Steps:
      1. Compute stratified split indices.
      2. Create Subsets with appropriate transforms.
      3. Wrap in DataLoaders.

    Note: full_train_ds is used only for index computation; the actual
    Subsets are created from new dataset instances with correct transforms.

    Args:
        cfg: Merged configuration dictionary.
        full_train_ds: Raw FashionMNIST train dataset (any transform; used for label info).
        test_ds: Raw FashionMNIST test dataset (any transform; used for label info).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_root = cfg["paths"]["data_root"]
    seed = cfg["project"]["seed"]
    val_fraction = cfg["data"]["val_fraction"]
    batch_size = cfg["dataloader"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    train_indices, val_indices = stratified_split(full_train_ds, val_fraction, seed)

    # Load dataset instances with correct transforms
    train_transform = get_train_transform(cfg)
    eval_transform = get_eval_transform(cfg)

    train_full_with_aug = datasets.FashionMNIST(
        root=str(data_root), train=True, transform=train_transform, download=False
    )
    train_full_eval = datasets.FashionMNIST(
        root=str(data_root), train=True, transform=eval_transform, download=False
    )
    test_full = datasets.FashionMNIST(
        root=str(data_root), train=False, transform=eval_transform, download=False
    )

    train_subset = Subset(train_full_with_aug, train_indices)
    val_subset = Subset(train_full_eval, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"[Split] Train: {len(train_subset):,} | Val: {len(val_subset):,} | Test: {len(test_full):,}")
    return train_loader, val_loader, test_loader


def verify_no_overlap(
    train_indices: list[int],
    val_indices: list[int],
) -> bool:
    """
    Verify that train and validation index sets do not overlap.

    Returns:
        True if no overlap, raises AssertionError otherwise.
    """
    overlap = set(train_indices) & set(val_indices)
    assert len(overlap) == 0, f"Data leakage! {len(overlap)} overlapping samples found."
    return True
