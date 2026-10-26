"""
download_and_eda.py — Step 1: Download FashionMNIST and run EDA.

Must be run before any training. Saves EDA statistics and figures to artifacts/.

Usage: py -3 scripts/download_and_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add src to path so package imports work without editable install
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import argparse
from fashionmnist.config import load_config
from fashionmnist.data.dataset import load_raw_dataset, verify_dataset_integrity
from fashionmnist.data.eda import run_eda
from fashionmnist.utils.reproducibility import set_seed, reproducibility_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FashionMNIST dataset and run Exploratory Data Analysis (EDA)"
    )
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)

    data_root = cfg["paths"]["data_root"]
    eda_dir = cfg["paths"]["eda_dir"]
    figures_dir = cfg["paths"]["figures_dir"]

    print("=" * 55)
    print("FashionMNIST - Download & EDA")
    print("=" * 55)

    info = reproducibility_info(seed)
    print(f"[Env] Python {info['python_version']} | PyTorch {info['torch_version']} | "
          f"NumPy {info['numpy_version']}")
    print(f"[Env] CUDA available: {info['cuda_available']}")

    # ── Step 1: Load (and download if needed) ────────────────────────────────
    print(f"\n[Data] Loading FashionMNIST from {data_root} ...")
    try:
        train_ds = load_raw_dataset(data_root, train=True, download=True)
        test_ds = load_raw_dataset(data_root, train=False, download=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to load/download FashionMNIST: {e}")
        print("  -> Check internet connectivity and retry.")
        print("  -> If files are partially downloaded, clear data/raw/ and retry.")
        sys.exit(1)

    print(f"[Data] Train: {len(train_ds):,} samples | Test: {len(test_ds):,} samples")

    # ── Step 2: Integrity check ───────────────────────────────────────────────
    print("\n[Integrity] Verifying datasets...")
    train_check = verify_dataset_integrity(train_ds)
    test_check = verify_dataset_integrity(test_ds)

    for split_name, check in [("Train", train_check), ("Test", test_check)]:
        status = "[PASS]" if check["passed"] else "[FAIL]"
        print(f"  {split_name}: {status} | "
              f"n={check['n_samples']:,} (expected {check['expected_n_samples']:,}) | "
              f"labels=[{check['label_min']},{check['label_max']}] | "
              f"invalid={check['invalid_label_count']}")

    if not (train_check["passed"] and test_check["passed"]):
        print("\n[WARNING] Integrity check failed. Inspect data before proceeding.")

    # ── Step 3: EDA ───────────────────────────────────────────────────────────
    print("\n[EDA] Running Exploratory Data Analysis...")
    stats = run_eda(
        train_ds=train_ds,
        test_ds=test_ds,
        eda_dir=eda_dir,
        figures_dir=figures_dir,
    )

    print("\n[Done] EDA complete.")
    print(f"  Statistics : {eda_dir / 'eda_statistics.json'}")
    print(f"  Figures    : {figures_dir}")
    print("\n  Next step  : py -3 scripts/train.py")


if __name__ == "__main__":
    main()
