"""
logging.py — Experiment logging utilities.

Saves experiment metadata (config, metrics, history, paths) to JSON.
Maintains a summary log of all experiments run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def save_experiment_log(
    experiments_dir: Path,
    experiment_name: str,
    cfg: dict[str, Any],
    history: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> Path:
    """
    Save a complete experiment log to JSON.

    Args:
        experiments_dir: Directory to store experiment logs.
        experiment_name: Unique name for this experiment run.
        cfg: Configuration dict used for this experiment.
        history: Training history dict.
        metrics: Evaluation metrics dict (or None if not yet evaluated).
        notes: Optional human-readable notes.

    Returns:
        Path to the saved log file.
    """
    experiments_dir.mkdir(parents=True, exist_ok=True)

    log: dict[str, Any] = {
        "experiment_name": experiment_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
        "config": {
            "model": cfg.get("model", {}),
            "training": cfg.get("training", {}),
            "dataloader": cfg.get("dataloader", {}),
            "data": {
                "val_fraction": cfg.get("data", {}).get("val_fraction"),
                "augment_train": cfg.get("data", {}).get("augment_train"),
                "normalize_mean": cfg.get("data", {}).get("normalize_mean"),
                "normalize_std": cfg.get("data", {}).get("normalize_std"),
            },
            "seed": cfg.get("project", {}).get("seed"),
        },
        "training_summary": {
            "epochs_run": history.get("epochs_run"),
            "best_val_acc": history.get("best_val_acc"),
            "best_epoch": history.get("best_epoch"),
            "total_time_sec": history.get("total_time_sec"),
            "final_train_loss": history["train_loss"][-1] if history.get("train_loss") else None,
            "final_val_loss": history["val_loss"][-1] if history.get("val_loss") else None,
        },
        "evaluation_metrics": metrics,
        "full_history": {
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", []),
            "train_acc": history.get("train_acc", []),
            "val_acc": history.get("val_acc", []),
        },
    }

    # Sanitize config paths (Path objects are not JSON-serializable)
    log = _sanitize_for_json(log)

    out_path = experiments_dir / f"{experiment_name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[Log] Experiment saved → {out_path}")
    return out_path


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def load_experiment_log(log_path: str | Path) -> dict[str, Any]:
    """Load a previously saved experiment log."""
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Experiment log not found: {log_path}")
    with log_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_experiment_summary(experiments_dir: Path) -> None:
    """Print a summary table of all completed experiments."""
    logs = sorted(experiments_dir.glob("*.json"))
    if not logs:
        print("[Log] No experiment logs found.")
        return

    print("\n" + "=" * 75)
    print(f"{'Experiment':<25} {'Arch':<15} {'Epochs':>6} {'BestValAcc':>11} {'Time(s)':>9}")
    print("-" * 75)
    for log_path in logs:
        try:
            log = load_experiment_log(log_path)
            name = log.get("experiment_name", log_path.stem)[:24]
            arch = log.get("config", {}).get("model", {}).get("architecture", "?")[:14]
            summary = log.get("training_summary", {})
            epochs = summary.get("epochs_run", "?")
            val_acc = summary.get("best_val_acc")
            t = summary.get("total_time_sec", "?")
            val_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
            print(f"{name:<25} {arch:<15} {str(epochs):>6} {val_str:>11} {str(t):>9}")
        except Exception as e:
            print(f"  [Warning] Could not read {log_path.name}: {e}")
    print("=" * 75 + "\n")
