"""
trainer.py — Training and validation loops for FashionMNIST.

Implements:
  - One epoch of training (forward, loss, backward, optimizer step).
  - One epoch of validation (no grad, no weight update).
  - Full training loop with checkpointing, early stopping and history logging.
  - Optimizer and scheduler factory.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fashionmnist.training.checkpoint import save_best_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Optimizer and scheduler builders
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> optim.Optimizer:
    """
    Build optimizer from config.

    Supported: adam, sgd.
    """
    training_cfg = cfg["training"]
    name = training_cfg.get("optimizer", "adam").lower()
    lr = training_cfg["learning_rate"]
    wd = training_cfg.get("weight_decay", 0.0)

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: '{name}'. Choose from: adam, sgd.")


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: dict[str, Any],
) -> Any | None:
    """
    Build LR scheduler from config.

    Supported: cosine, step, null/None.
    """
    training_cfg = cfg["training"]
    name = training_cfg.get("lr_scheduler", None)
    epochs = training_cfg["epochs"]

    if name is None or str(name).lower() in ("null", "none", ""):
        return None
    elif name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    else:
        raise ValueError(f"Unknown lr_scheduler: '{name}'. Choose from: cosine, step, null.")


# ---------------------------------------------------------------------------
# Single-epoch loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Args:
        model: Model to train.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Computation device.

    Returns:
        (avg_loss, accuracy) over this epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one validation/test epoch without gradient updates.

    Args:
        model: Model to evaluate.
        loader: DataLoader (validation or test).
        criterion: Loss function.
        device: Computation device.

    Returns:
        (avg_loss, accuracy) over this epoch.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
    experiment_name: str = "default",
) -> dict[str, Any]:
    """
    Full training loop with validation, checkpointing and early stopping.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Configuration dict.
        device: Computation device.
        checkpoint_dir: Directory to save checkpoints.
        experiment_name: Label for logging.

    Returns:
        History dict with keys: train_loss, val_loss, train_acc, val_acc,
        best_val_acc, best_epoch, total_time_sec.
    """
    training_cfg = cfg["training"]
    epochs = training_cfg["epochs"]
    patience = training_cfg.get("early_stopping_patience", 5)
    save_every = training_cfg.get("save_every_n_epochs", 5)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    model = model.to(device)

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "lr": [],
    }
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    start_time = time.time()

    print(f"\n[Train] Starting '{experiment_name}' | Epochs={epochs} | Device={device}")
    print(f"[Train] Optimizer={training_cfg.get('optimizer')} | LR={training_cfg['learning_rate']} | "
          f"Scheduler={training_cfg.get('lr_scheduler', None)}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        ep_time = time.time() - ep_start
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
            f"LR={current_lr:.6f} | {ep_time:.1f}s"
        )

        # Best checkpoint
        _, improved = save_best_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            current_metric=val_acc,
            best_metric=best_val_acc,
            cfg=cfg,
        )
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            print(f"  [*] New best val_acc={best_val_acc:.4f} at epoch {epoch}")
        else:
            no_improve += 1

        # Periodic checkpoint
        if epoch % save_every == 0:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_val_acc,
                cfg=cfg,
            )

        # Early stopping
        if no_improve >= patience:
            print(f"\n[Train] Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"[Train] Done. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}. "
          f"Total time: {total_time:.1f}s")

    return {
        **history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_time_sec": round(total_time, 2),
        "epochs_run": len(history["train_loss"]),
    }
