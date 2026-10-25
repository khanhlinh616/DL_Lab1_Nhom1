"""
predict.py — Run inference on one image or a batch from the test set.

Loads best checkpoint, outputs predicted class name and confidence.

Usage:
  py -3 scripts/predict.py --experiment exp_cnn_small
  py -3 scripts/predict.py --experiment exp_cnn_small --n_samples 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from fashionmnist.config import load_config
from fashionmnist.data.dataset import load_raw_dataset, get_class_names
from fashionmnist.data.transforms import get_eval_transform
from fashionmnist.models.network import build_model
from fashionmnist.training.checkpoint import load_checkpoint
from fashionmnist.utils.reproducibility import set_seed, get_device
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a batch of images.

    Args:
        model: Trained model in eval mode.
        images: Tensor of shape (B, 1, 28, 28), normalized.
        device: Target device.

    Returns:
        (predicted_class_indices, confidence_scores) arrays of shape (B,).
    """
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
    return preds.cpu().numpy(), probs.max(dim=1).values.cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FashionMNIST inference")
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of random test samples to predict and display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment

    cfg = load_config(experiment_name=(None if experiment_name == "default" else experiment_name))
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = get_device(prefer_gpu=True)

    data_root = cfg["paths"]["data_root"]
    checkpoint_dir = cfg["paths"]["checkpoints_dir"] / experiment_name
    predictions_dir = cfg["paths"]["predictions_dir"]
    predictions_dir.mkdir(parents=True, exist_ok=True)
    mean = cfg["data"]["normalize_mean"]
    std = cfg["data"]["normalize_std"]
    class_names = get_class_names()

    # Load dataset
    eval_transform = get_eval_transform(cfg)
    try:
        test_ds = datasets.FashionMNIST(
            root=str(data_root), train=False, transform=eval_transform, download=False
        )
    except Exception as e:
        print(f"[ERROR] Dataset not found: {e}")
        sys.exit(1)

    # Load model
    best_ckpt_path = checkpoint_dir / "best_model.pt"
    if not best_ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {best_ckpt_path}")
        sys.exit(1)
    model = build_model(cfg)
    ckpt = load_checkpoint(best_ckpt_path, model, device=device)
    model = model.to(device)
    print(f"[Predict] Loaded epoch {ckpt['epoch']} (best_val_acc={ckpt['best_metric']:.4f})")

    # Pick random test samples
    rng = np.random.default_rng(seed=seed + 1)
    n = min(args.n_samples, len(test_ds))
    indices = rng.choice(len(test_ds), size=n, replace=False)

    images_list = []
    true_labels_list = []
    for idx in indices:
        img, label = test_ds[int(idx)]
        images_list.append(img)
        true_labels_list.append(label)

    images_tensor = torch.stack(images_list)  # (n, 1, 28, 28)
    preds, confidences = predict_batch(model, images_tensor, device)

    # Display grid
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"Predictions — {experiment_name}", fontsize=12)

    for i, (img_t, true_lbl, pred_lbl, conf) in enumerate(
        zip(images_tensor, true_labels_list, preds, confidences)
    ):
        img_np = img_t.squeeze(0).numpy()
        img_display = np.clip(img_np * std + mean, 0.0, 1.0)
        is_correct = int(true_lbl) == int(pred_lbl)
        color = "green" if is_correct else "red"

        ax = axes[i]
        ax.imshow(img_display, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"True: {class_names[int(true_lbl)]}\nPred: {class_names[int(pred_lbl)]} ({conf:.1%})",
            fontsize=6.5, color=color,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    out_path = predictions_dir / f"predictions_{experiment_name}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Predict] Saved predictions grid -> {out_path}")

    # Console summary
    correct = sum(int(t) == int(p) for t, p in zip(true_labels_list, preds))
    print(f"[Predict] Sample accuracy: {correct}/{n} = {correct/n:.1%}")


if __name__ == "__main__":
    main()
