"""
eda.py — Exploratory Data Analysis for FashionMNIST.

All EDA functions operate on the raw (pre-split) dataset.
Results are saved to artifacts/eda/ and artifacts/figures/.

Order of operations (enforced by the caller script):
  1. Load raw dataset.
  2. Call run_eda() — returns a stats dict and saves all artifacts.
  3. Only after EDA: split and build DataLoaders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets


_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# Main EDA entry point
# ---------------------------------------------------------------------------

def run_eda(
    train_ds: datasets.FashionMNIST,
    test_ds: datasets.FashionMNIST,
    eda_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    """
    Run full EDA on FashionMNIST raw datasets and save all artifacts.

    Args:
        train_ds: Raw train dataset (ToTensor transform, no normalization).
        test_ds: Raw test dataset (ToTensor transform, no normalization).
        eda_dir: Directory to save JSON statistics.
        figures_dir: Directory to save PNG figures.

    Returns:
        Dictionary of computed statistics (also saved as JSON).
    """
    eda_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[EDA] Computing dataset statistics...")
    stats = _compute_statistics(train_ds, test_ds)

    print("[EDA] Plotting raw class distribution before split...")
    _plot_raw_class_distribution(train_ds, figures_dir)

    print("[EDA] Plotting class distribution (Train vs Test)...")
    _plot_class_distribution(train_ds, test_ds, figures_dir)

    print("[EDA] Plotting post-split distribution (Train 50k / Val 10k / Test 10k)...")
    _plot_split_distribution(train_ds, test_ds, figures_dir)

    print("[EDA] Plotting sample grid...")
    _plot_sample_grid(train_ds, figures_dir)

    print("[EDA] Plotting pixel intensity histogram...")
    _plot_pixel_histogram(train_ds, figures_dir)

    print("[EDA] Plotting per-class mean images...")
    _plot_class_mean_images(train_ds, figures_dir)

    stats_path = eda_dir / "eda_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[EDA] Statistics saved -> {stats_path}")

    _print_eda_summary(stats)
    return stats


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_statistics(
    train_ds: datasets.FashionMNIST,
    test_ds: datasets.FashionMNIST,
) -> dict[str, Any]:
    """Compute overall and per-class pixel statistics."""
    train_labels = train_ds.targets  # shape (60000,)
    test_labels = test_ds.targets   # shape (10000,)

    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels)
    if not isinstance(test_labels, torch.Tensor):
        test_labels = torch.tensor(test_labels)

    # Pixel statistics — compute from all training images
    # FashionMNIST data is stored as uint8 numpy arrays internally
    train_data = train_ds.data.float() / 255.0  # (60000, 28, 28)
    pixel_mean = float(train_data.mean().item())
    pixel_std = float(train_data.std().item())
    pixel_min = float(train_data.min().item())
    pixel_max = float(train_data.max().item())

    # Per-class sample counts
    train_counts = {
        _CLASS_NAMES[i]: int((train_labels == i).sum().item())
        for i in range(10)
    }
    test_counts = {
        _CLASS_NAMES[i]: int((test_labels == i).sum().item())
        for i in range(10)
    }

    return {
        "dataset": "FashionMNIST",
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "image_shape": [1, 28, 28],
        "image_dtype": "uint8 (raw), float32 (after ToTensor)",
        "pixel_range_raw": [0, 255],
        "pixel_range_tensor": [0.0, 1.0],
        "pixel_mean_train": round(pixel_mean, 6),
        "pixel_std_train": round(pixel_std, 6),
        "pixel_min_train": round(pixel_min, 6),
        "pixel_max_train": round(pixel_max, 6),
        "num_classes": 10,
        "class_names": _CLASS_NAMES,
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "label_min_train": int(train_labels.min().item()),
        "label_max_train": int(train_labels.max().item()),
        "label_min_test": int(test_labels.min().item()),
        "label_max_test": int(test_labels.max().item()),
        "invalid_labels_train": int(((train_labels < 0) | (train_labels > 9)).sum().item()),
        "invalid_labels_test": int(((test_labels < 0) | (test_labels > 9)).sum().item()),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_raw_class_distribution(
    train_ds: datasets.FashionMNIST,
    figures_dir: Path,
) -> None:
    """Horizontal bar chart of raw dataset class distribution before any split (60,000 samples)."""
    train_labels = train_ds.targets
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels)

    counts = [(train_labels == i).sum().item() for i in range(10)]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    y = np.arange(10)
    bars = ax.barh(y, counts, color="#2b5c8f", alpha=0.85, height=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(_CLASS_NAMES, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Số lượng mẫu (samples)", fontsize=10)
    ax.set_title(f"FashionMNIST — Phân phối dữ liệu thô trước khi chia tập (Tổng: {total:,} mẫu Train thô)", fontsize=11, pad=10)
    ax.set_xlim(0, max(counts) * 1.25)

    for bar in bars:
        w = bar.get_width()
        pct = (w / total) * 100
        ax.text(w + 100, bar.get_y() + bar.get_height() / 2, f"{int(w):,} mẫu ({pct:.1f}%)",
                va="center", ha="left", fontsize=9, fontweight="bold", color="#1c3d5a")

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = figures_dir / "eda_raw_class_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved -> {out_path}")


def _plot_split_distribution(
    train_ds: datasets.FashionMNIST,
    test_ds: datasets.FashionMNIST,
    figures_dir: Path,
    val_fraction: float = 0.16666666666666666,
    seed: int = 42,
) -> None:
    """Grouped bar chart showing sample counts per class across Train (50k), Val (10k), and Test (10k)."""
    train_labels = train_ds.targets
    test_labels = test_ds.targets
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels)
    if not isinstance(test_labels, torch.Tensor):
        test_labels = torch.tensor(test_labels)

    labels_np = train_labels.numpy()
    rng = np.random.default_rng(seed=seed)
    train_split_counts = [0] * 10
    val_split_counts = [0] * 10
    test_counts = [(test_labels == i).sum().item() for i in range(10)]

    for cls in range(10):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_fraction))
        val_split_counts[cls] = n_val
        train_split_counts[cls] = len(cls_idx) - n_val

    x = np.arange(10)
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
    bars_train = ax.bar(x - width, train_split_counts, width, label="Tập Train sau chia (50.000 mẫu)", color="#2b5c8f", alpha=0.9)
    bars_val = ax.bar(x, val_split_counts, width, label="Tập Validation sau chia (10.000 mẫu)", color="#2a9d8f", alpha=0.9)
    bars_test = ax.bar(x + width, test_counts, width, label="Tập Test độc lập (10.000 mẫu)", color="#e76f51", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(_CLASS_NAMES, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Số lượng mẫu", fontsize=10)
    ax.set_title("FashionMNIST — Đối chiếu phân chia tập phân tầng (Stratified Split)", fontsize=11, pad=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.bar_label(bars_train, padding=2, fontsize=7)
    ax.bar_label(bars_val, padding=2, fontsize=7)
    ax.bar_label(bars_test, padding=2, fontsize=7)
    ax.set_ylim(0, max(train_split_counts) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = figures_dir / "dataset_split_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved -> {out_path}")


def _plot_class_distribution(
    train_ds: datasets.FashionMNIST,
    test_ds: datasets.FashionMNIST,
    figures_dir: Path,
) -> None:
    """Bar chart of sample counts per class for train and test sets."""
    train_labels = train_ds.targets
    test_labels = test_ds.targets
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels)
    if not isinstance(test_labels, torch.Tensor):
        test_labels = torch.tensor(test_labels)

    train_counts = [(train_labels == i).sum().item() for i in range(10)]
    test_counts = [(test_labels == i).sum().item() for i in range(10)]

    x = np.arange(10)
    width = 0.40

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_train = ax.bar(x - width / 2, train_counts, width, label="Train", color="#4C72B0", alpha=0.85)
    bars_test = ax.bar(x + width / 2, test_counts, width, label="Test", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(_CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of samples")
    ax.set_title("FashionMNIST — Class Distribution (Train vs Test)")
    ax.legend()
    ax.bar_label(bars_train, padding=2, fontsize=7)
    ax.bar_label(bars_test, padding=2, fontsize=7)
    ax.set_ylim(0, max(train_counts) * 1.15)

    plt.tight_layout()
    out_path = figures_dir / "eda_class_distribution.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved -> {out_path}")


def _plot_sample_grid(
    train_ds: datasets.FashionMNIST,
    figures_dir: Path,
    n_cols: int = 10,
    n_rows: int = 3,
) -> None:
    """Grid of sample images (one row per random sample across all classes)."""
    rng = np.random.default_rng(seed=42)
    labels = train_ds.targets
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8))
    fig.suptitle("FashionMNIST — Sample Images (3 rows × 10 classes)", fontsize=12)

    for row in range(n_rows):
        for col in range(n_cols):
            cls = col  # one column per class
            cls_indices = (labels == cls).nonzero(as_tuple=True)[0].numpy()
            idx = rng.choice(cls_indices)
            img_tensor, label = train_ds[int(idx)]  # (1, 28, 28)
            img_np = img_tensor.squeeze(0).numpy()  # (28, 28)

            ax = axes[row, col]
            ax.imshow(img_np, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row == 0:
                ax.set_title(_CLASS_NAMES[col], fontsize=7, pad=3)

    plt.tight_layout()
    out_path = figures_dir / "eda_sample_grid.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved -> {out_path}")


def _plot_pixel_histogram(
    train_ds: datasets.FashionMNIST,
    figures_dir: Path,
    n_bins: int = 50,
) -> None:
    """Pixel intensity histogram over all training images."""
    # Use raw uint8 data for histogram; convert to float [0,1]
    data = train_ds.data.float() / 255.0  # (60000, 28, 28)
    pixels = data.numpy().ravel()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pixels, bins=n_bins, color="#4C72B0", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Pixel intensity (0 = black, 1 = white)")
    ax.set_ylabel("Frequency")
    ax.set_title("FashionMNIST — Pixel Intensity Distribution (Train set)")
    mean_val = float(pixels.mean())
    std_val = float(pixels.std())
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_val:.4f}")
    ax.legend(fontsize=9)
    ax.text(0.72, 0.88, f"Std={std_val:.4f}", transform=ax.transAxes, fontsize=9)

    plt.tight_layout()
    out_path = figures_dir / "eda_pixel_histogram.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved -> {out_path}")


def _plot_class_mean_images(
    train_ds: datasets.FashionMNIST,
    figures_dir: Path,
) -> None:
    """Plot the mean image for each of the 10 classes."""
    data = train_ds.data.float() / 255.0   # (60000, 28, 28)
    labels = train_ds.targets               # (60000,)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    fig, axes = plt.subplots(1, 10, figsize=(14, 2))
    fig.suptitle("FashionMNIST — Per-Class Mean Image", fontsize=11)

    for cls in range(10):
        mask = labels == cls
        mean_img = data[mask].mean(dim=0).numpy()  # (28, 28)
        ax = axes[cls]
        ax.imshow(mean_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(_CLASS_NAMES[cls], fontsize=7, pad=2)
        ax.axis("off")

    plt.tight_layout()
    out_path = figures_dir / "eda_class_mean_images.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved -> {out_path}")


def _print_eda_summary(stats: dict[str, Any]) -> None:
    """Print a concise EDA summary to stdout."""
    print("\n" + "=" * 55)
    print("EDA SUMMARY — FashionMNIST")
    print("=" * 55)
    print(f"  Train samples : {stats['train_samples']:,}")
    print(f"  Test samples  : {stats['test_samples']:,}")
    print(f"  Image shape   : {stats['image_shape']} (C×H×W)")
    print(f"  Pixel mean    : {stats['pixel_mean_train']:.4f}")
    print(f"  Pixel std     : {stats['pixel_std_train']:.4f}")
    print(f"  Pixel range   : [{stats['pixel_min_train']:.4f}, {stats['pixel_max_train']:.4f}]")
    print(f"  Invalid labels (train): {stats['invalid_labels_train']}")
    print(f"  Invalid labels (test) : {stats['invalid_labels_test']}")
    print("\n  Train class distribution:")
    for cls_name, count in stats["train_class_counts"].items():
        bar = "#" * (count // 1000)
        print(f"    {cls_name:15s}: {count:5d}  {bar}")
    print("=" * 55 + "\n")
