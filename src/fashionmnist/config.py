"""
config.py — Centralized configuration loader.

Loads default.yaml and optionally merges an experiment config block.
All paths are resolved relative to the project root (parent of src/).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    """Walk upwards from this file until pyproject.toml is found."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: two levels up from src/fashionmnist/
    return here.parent.parent


PROJECT_ROOT: Path = _find_project_root()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: str | Path | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """
    Load and return the merged configuration dictionary.

    Args:
        config_path: Path to default YAML. Defaults to configs/default.yaml.
        experiment_name: If given, merge the matching block from experiments.yaml.

    Returns:
        Merged configuration dict with all paths resolved.

    Raises:
        FileNotFoundError: If config_path does not exist.
        KeyError: If experiment_name is not found in experiments.yaml.
        ValueError: If critical config keys are missing or invalid.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if experiment_name is not None:
        exp_path = PROJECT_ROOT / "configs" / "experiments.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiments config not found: {exp_path}")
        with exp_path.open("r", encoding="utf-8") as f:
            experiments = yaml.safe_load(f)
        if experiment_name not in experiments.get("experiments", {}):
            available = list(experiments.get("experiments", {}).keys())
            raise KeyError(
                f"Experiment '{experiment_name}' not found. Available: {available}"
            )
        cfg = _deep_merge(cfg, experiments["experiments"][experiment_name])

    cfg = _resolve_paths(cfg)
    _validate_config(cfg)
    return cfg


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve relative paths in the 'paths' section to absolute paths."""
    cfg = copy.deepcopy(cfg)
    paths = cfg.get("paths", {})
    for key, value in paths.items():
        if isinstance(value, str):
            resolved = (PROJECT_ROOT / value).resolve()
            paths[key] = resolved
    cfg["paths"] = paths
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Validate critical configuration values."""
    data_cfg = cfg.get("data", {})
    num_classes = data_cfg.get("num_classes")
    if num_classes != 10:
        raise ValueError(f"Expected num_classes=10, got {num_classes}")

    val_fraction = data_cfg.get("val_fraction", 0.0)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    training_cfg = cfg.get("training", {})
    lr = training_cfg.get("learning_rate", 0.0)
    if lr <= 0:
        raise ValueError(f"learning_rate must be positive, got {lr}")

    epochs = training_cfg.get("epochs", 0)
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")


def get_class_names(cfg: dict[str, Any]) -> list[str]:
    """Return ordered list of class names from config."""
    return cfg["data"]["class_names"]


def get_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Return resolved path dict."""
    return cfg["paths"]
