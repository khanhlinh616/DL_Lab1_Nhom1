"""
checkpoint.py — Saving and loading model checkpoints.

A checkpoint stores:
  - model state_dict
  - optimizer state_dict
  - lr_scheduler state_dict (if any)
  - epoch number
  - best validation metric value
  - configuration used for training
  - model architecture name

This allows full resumption of training and exact reproduction of inference.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    best_metric: float,
    cfg: dict[str, Any],
    filename: str | None = None,
) -> Path:
    """
    Save a training checkpoint to disk.

    Args:
        checkpoint_dir: Directory to save checkpoint file.
        model: The model being trained.
        optimizer: Optimizer instance.
        scheduler: LR scheduler (or None).
        epoch: Current epoch number (1-indexed).
        best_metric: Best validation metric seen so far.
        cfg: Training configuration dict.
        filename: Override filename. Defaults to 'checkpoint_epoch_{epoch}.pt'.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch:03d}.pt"

    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "architecture": cfg.get("model", {}).get("architecture", "unknown"),
        "cfg_model": cfg.get("model", {}),
        "cfg_training": cfg.get("training", {}),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = checkpoint_dir / filename
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load a checkpoint from disk into model (and optionally optimizer/scheduler).

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: Model instance to load weights into.
        optimizer: Optimizer to restore state (optional).
        scheduler: LR scheduler to restore state (optional).
        device: Device to map tensors to. Defaults to CPU.

    Returns:
        The full checkpoint dict (for inspecting epoch, metric, etc.).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        KeyError: If expected keys are missing from checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cpu")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    required_keys = {"epoch", "model_state_dict", "best_metric"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {missing}")

    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return payload


def save_best_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    current_metric: float,
    best_metric: float,
    cfg: dict[str, Any],
) -> tuple[float, bool]:
    """
    Save model if current_metric exceeds best_metric (higher is better).

    Args:
        checkpoint_dir: Directory for saving.
        model: Current model.
        optimizer: Current optimizer.
        scheduler: LR scheduler or None.
        epoch: Current epoch.
        current_metric: Metric value at current epoch (e.g. val_accuracy).
        best_metric: Best metric seen so far.
        cfg: Config dict.

    Returns:
        (updated_best_metric, was_improved)
    """
    if current_metric > best_metric:
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=current_metric,
            cfg=cfg,
            filename="best_model.pt",
        )
        return current_metric, True
    return best_metric, False
