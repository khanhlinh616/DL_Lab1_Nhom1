"""
reproducibility.py — Seed management and device selection.

Sets seeds for: Python random, numpy, torch CPU, torch CUDA (if available).
Note: Full bit-for-bit reproducibility is not guaranteed on GPU with
      certain CUDA operations (especially non-deterministic algorithms).
      CPU-only training is fully reproducible given same seed and environment.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms where available; may slow down GPU training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Select computation device.

    Args:
        prefer_gpu: If True, use CUDA if available; else always use CPU.

    Returns:
        torch.device instance.
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU.")
    return device


def reproducibility_info(seed: int) -> dict[str, str]:
    """Return a dict of environment info for reproducibility reporting."""
    import platform
    import sys
    return {
        "seed": str(seed),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    }
