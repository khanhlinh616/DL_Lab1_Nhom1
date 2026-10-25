"""
transforms.py — Torchvision transforms for FashionMNIST.

Provides separate transforms for train (with optional augmentation)
and for validation/test (deterministic only).

FashionMNIST pixel statistics (computed on the 60 000-sample training set):
    Mean ≈ 0.2860,  Std ≈ 0.3530
These values are stored in configs/default.yaml and passed via cfg.
"""
from __future__ import annotations

from typing import Any

import torchvision.transforms as T


def get_train_transform(cfg: dict[str, Any]) -> T.Compose:
    """
    Build training transform.

    When data.augment_train is True, adds:
      - RandomHorizontalFlip (p=0.5)
      - RandomCrop(28, padding=4)

    Always includes:
      - ToTensor()
      - Normalize(mean, std)
    """
    mean = cfg["data"]["normalize_mean"]
    std = cfg["data"]["normalize_std"]
    augment = cfg["data"].get("augment_train", True)

    ops: list = [T.ToTensor()]
    if augment:
        ops = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(28, padding=4),
        ] + ops
    ops.append(T.Normalize(mean=(mean,), std=(std,)))
    return T.Compose(ops)


def get_eval_transform(cfg: dict[str, Any]) -> T.Compose:
    """
    Build deterministic transform for validation and test sets.

    Applies:
      - ToTensor()
      - Normalize(mean, std)
    """
    mean = cfg["data"]["normalize_mean"]
    std = cfg["data"]["normalize_std"]
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(mean,), std=(std,)),
    ])
