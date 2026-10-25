"""
train.py — Train a FashionMNIST model according to config.

Supports running a named experiment from experiments.yaml or the default config.

Usage:
  py -3 scripts/train.py
  py -3 scripts/train.py --experiment baseline_mlp
  py -3 scripts/train.py --experiment exp_cnn_small
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from fashionmnist.config import load_config, get_class_names
from fashionmnist.data.dataset import load_raw_dataset
from fashionmnist.data.split import build_dataloaders, stratified_split, verify_no_overlap
from fashionmnist.models.network import build_model, count_parameters
from fashionmnist.training.trainer import train
from fashionmnist.utils.reproducibility import set_seed, get_device
from fashionmnist.utils.logging import save_experiment_log, print_experiment_summary
from fashionmnist.visualization.plots import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FashionMNIST classifier")
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment name from configs/experiments.yaml (default: use default config)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment or "default"

    print("=" * 60)
    print(f"FashionMNIST Training — Experiment: '{experiment_name}'")
    print("=" * 60)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(experiment_name=args.experiment)
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = get_device(prefer_gpu=True)

    data_root = cfg["paths"]["data_root"]
    checkpoint_dir = cfg["paths"]["checkpoints_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    experiments_dir = cfg["paths"]["experiments_dir"]

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[Data] Loading FashionMNIST...")
    try:
        full_train_ds = load_raw_dataset(data_root, train=True, download=True)
        test_ds_raw = load_raw_dataset(data_root, train=False, download=True)
    except Exception as e:
        print(f"[ERROR] Cannot load dataset: {e}")
        print("  → Run 'py -3 scripts/download_and_eda.py' first.")
        sys.exit(1)

    # Verify no leakage in split
    val_fraction = cfg["data"]["val_fraction"]
    train_idx, val_idx = stratified_split(full_train_ds, val_fraction, seed)
    verify_no_overlap(train_idx, val_idx)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, full_train_ds, test_ds_raw)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[Model] Building model...")
    model = build_model(cfg)
    n_params = count_parameters(model)
    arch = cfg["model"]["architecture"]
    print(f"  Architecture : {arch}")
    print(f"  Parameters   : {n_params:,}")

    # ── Training ──────────────────────────────────────────────────────────────
    exp_checkpoint_dir = checkpoint_dir / experiment_name
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        checkpoint_dir=exp_checkpoint_dir,
        experiment_name=experiment_name,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_history(history, figures_dir, experiment_name)

    # ── Log ───────────────────────────────────────────────────────────────────
    save_experiment_log(
        experiments_dir=experiments_dir,
        experiment_name=experiment_name,
        cfg=cfg,
        history=history,
        notes=f"Trained on CPU/GPU. Model: {arch}.",
    )

    print_experiment_summary(experiments_dir)
    print(f"\n[Done] Best checkpoint at: {exp_checkpoint_dir / 'best_model.pt'}")
    print(f"  Next step: py -3 scripts/evaluate.py --experiment {experiment_name}")


if __name__ == "__main__":
    main()
