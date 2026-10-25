"""
evaluate.py — Evaluate a trained model on the test set.

Loads the best checkpoint for an experiment and produces:
  - Accuracy, precision, recall, F1 (macro and weighted)
  - Per-class metrics
  - Confusion matrix (saved as figure)
  - Per-class F1 bar chart
  - Misclassified sample grid
  - Correct prediction grid
  - Full metrics JSON

IMPORTANT: Test set is only evaluated here, after model selection is final.
           Do NOT use test metrics to re-select model checkpoints.

Usage:
  py -3 scripts/evaluate.py --experiment exp_cnn_small
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from fashionmnist.config import load_config
from fashionmnist.data.dataset import load_raw_dataset
from fashionmnist.data.split import build_dataloaders
from fashionmnist.evaluation.error_analysis import (
    find_misclassified,
    plot_misclassified_grid,
    plot_correct_predictions_grid,
)
from fashionmnist.evaluation.metrics import (
    gather_predictions,
    compute_metrics,
    print_classification_report,
    save_metrics,
)
from fashionmnist.models.network import build_model
from fashionmnist.training.checkpoint import load_checkpoint
from fashionmnist.utils.logging import save_experiment_log, load_experiment_log
from fashionmnist.utils.reproducibility import set_seed, get_device
from fashionmnist.visualization.plots import (
    plot_confusion_matrix,
    plot_per_class_f1,
)
import torchvision.transforms as T
import torchvision.datasets as datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FashionMNIST model on test set")
    parser.add_argument(
        "--experiment", type=str, default="default",
        help="Experiment name (must match a trained experiment)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment

    print("=" * 60)
    print(f"FashionMNIST Evaluation — Experiment: '{experiment_name}'")
    print("=" * 60)

    cfg = load_config(experiment_name=(None if experiment_name == "default" else experiment_name))
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = get_device(prefer_gpu=True)

    data_root = cfg["paths"]["data_root"]
    checkpoint_dir = cfg["paths"]["checkpoints_dir"] / experiment_name
    figures_dir = cfg["paths"]["figures_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    predictions_dir = cfg["paths"]["predictions_dir"]
    experiments_dir = cfg["paths"]["experiments_dir"]
    mean = cfg["data"]["normalize_mean"]
    std = cfg["data"]["normalize_std"]

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n[Data] Loading FashionMNIST...")
    try:
        full_train_ds = load_raw_dataset(data_root, train=True, download=False)
        test_ds_raw = load_raw_dataset(data_root, train=False, download=False)
    except Exception as e:
        print(f"[ERROR] Dataset not found: {e}")
        print("  -> Run 'py -3 scripts/download_and_eda.py' first.")
        sys.exit(1)

    _, _, test_loader = build_dataloaders(cfg, full_train_ds, test_ds_raw)

    # Also load test set with eval transform for image display in error analysis
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(mean,), std=(std,)),
    ])
    test_display_ds = datasets.FashionMNIST(
        root=str(data_root), train=False, transform=eval_transform, download=False
    )

    # ── Load model + checkpoint ───────────────────────────────────────────────
    print(f"\n[Checkpoint] Loading from {checkpoint_dir / 'best_model.pt'}")
    model = build_model(cfg)
    best_ckpt_path = checkpoint_dir / "best_model.pt"

    if not best_ckpt_path.exists():
        print(f"[ERROR] Best checkpoint not found: {best_ckpt_path}")
        print("  -> Run training first: py -3 scripts/train.py --experiment {experiment_name}")
        sys.exit(1)

    ckpt = load_checkpoint(best_ckpt_path, model, device=device)
    model = model.to(device)
    print(f"  Loaded epoch {ckpt['epoch']} | Best val_acc={ckpt['best_metric']:.4f}")

    # ── Gather predictions ────────────────────────────────────────────────────
    print("\n[Eval] Gathering predictions on test set...")
    y_true, y_pred, y_probs = gather_predictions(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n[Eval] Classification Report:")
    report = print_classification_report(y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)       : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted)    : {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro)   : {metrics['recall_macro']:.4f}")

    # Save metrics
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, metrics_dir / f"test_metrics_{experiment_name}.json")

    # Save report text
    report_path = metrics_dir / f"classification_report_{experiment_name}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Eval] Report saved -> {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[Plots] Generating evaluation figures...")
    plot_confusion_matrix(metrics["confusion_matrix"], figures_dir, experiment_name, normalize=True)
    plot_confusion_matrix(metrics["confusion_matrix"], figures_dir, experiment_name, normalize=False)
    plot_per_class_f1(metrics["per_class"], figures_dir, experiment_name)

    # ── Error analysis ────────────────────────────────────────────────────────
    print("\n[ErrorAnalysis] Analyzing misclassifications...")
    error_info = find_misclassified(y_true, y_pred, y_probs)
    print(f"  Total errors: {error_info['total_errors']} / {error_info['total_samples']} "
          f"({error_info['error_rate']:.2%})")
    print("\n  Top confused class pairs:")
    for pair in error_info["top_confused_pairs"][:5]:
        print(f"    {pair['true_class']:15s} -> {pair['pred_class']:15s} : {pair['count']} errors")

    plot_misclassified_grid(
        dataset=test_display_ds,
        wrong_indices=error_info["wrong_indices"],
        wrong_true=error_info["wrong_true"],
        wrong_pred=error_info["wrong_pred"],
        wrong_conf=error_info["wrong_conf"],
        figures_dir=figures_dir,
        mean=mean, std=std,
    )
    plot_correct_predictions_grid(
        dataset=test_display_ds,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        figures_dir=figures_dir,
        mean=mean, std=std,
    )

    # ── Update experiment log ─────────────────────────────────────────────────
    try:
        existing_log = load_experiment_log(experiments_dir / f"{experiment_name}.json")
        history = existing_log.get("full_history", {})
        save_experiment_log(
            experiments_dir=experiments_dir,
            experiment_name=experiment_name,
            cfg=cfg,
            history={**history, **existing_log.get("training_summary", {})},
            metrics={**metrics, "test_accuracy": metrics["accuracy"]},
            notes=f"Evaluated on test set. Accuracy={metrics['accuracy']:.4f}",
        )
    except Exception:
        pass

    print(f"\n[Done] Evaluation complete for '{experiment_name}'.")
    print(f"  Figures : {figures_dir}")
    print(f"  Metrics : {metrics_dir}")


if __name__ == "__main__":
    main()
