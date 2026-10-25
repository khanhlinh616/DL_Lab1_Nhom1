"""
test_dataset.py — Tests for data loading and integrity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from fashionmnist.config import load_config
from fashionmnist.data.dataset import load_raw_dataset, get_class_names, verify_dataset_integrity
from fashionmnist.data.split import stratified_split, verify_no_overlap, build_dataloaders
from fashionmnist.data.transforms import get_train_transform, get_eval_transform


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def raw_train_ds(cfg):
    return load_raw_dataset(cfg["paths"]["data_root"], train=True, download=True)


@pytest.fixture(scope="module")
def raw_test_ds(cfg):
    return load_raw_dataset(cfg["paths"]["data_root"], train=False, download=True)


class TestDatasetLoading:
    def test_train_size(self, raw_train_ds):
        assert len(raw_train_ds) == 60_000, f"Expected 60000, got {len(raw_train_ds)}"

    def test_test_size(self, raw_test_ds):
        assert len(raw_test_ds) == 10_000, f"Expected 10000, got {len(raw_test_ds)}"

    def test_image_shape(self, raw_train_ds):
        img, label = raw_train_ds[0]
        assert img.shape == (1, 28, 28), f"Expected (1,28,28), got {img.shape}"

    def test_image_dtype(self, raw_train_ds):
        img, _ = raw_train_ds[0]
        assert img.dtype == torch.float32

    def test_image_range(self, raw_train_ds):
        img, _ = raw_train_ds[0]
        assert float(img.min()) >= 0.0
        assert float(img.max()) <= 1.0

    def test_labels_range(self, raw_train_ds):
        labels = raw_train_ds.targets
        assert int(labels.min()) == 0
        assert int(labels.max()) == 9

    def test_class_names_count(self):
        names = get_class_names()
        assert len(names) == 10

    def test_integrity_pass(self, raw_train_ds):
        result = verify_dataset_integrity(raw_train_ds)
        assert result["passed"] is True
        assert result["invalid_label_count"] == 0


class TestTransforms:
    def test_eval_transform_shape(self, cfg, raw_train_ds):
        transform = get_eval_transform(cfg)
        img_pil = raw_train_ds.data[0].numpy()
        from PIL import Image
        img_pil_obj = Image.fromarray(img_pil)
        result = transform(img_pil_obj)
        assert result.shape == (1, 28, 28)

    def test_eval_transform_dtype(self, cfg, raw_train_ds):
        transform = get_eval_transform(cfg)
        from PIL import Image
        img_pil = Image.fromarray(raw_train_ds.data[0].numpy())
        result = transform(img_pil)
        assert result.dtype == torch.float32

    def test_train_transform_shape(self, cfg, raw_train_ds):
        transform = get_train_transform(cfg)
        from PIL import Image
        img_pil = Image.fromarray(raw_train_ds.data[0].numpy())
        result = transform(img_pil)
        assert result.shape == (1, 28, 28)


class TestSplitting:
    def test_split_no_overlap(self, cfg, raw_train_ds):
        seed = cfg["project"]["seed"]
        val_fraction = cfg["data"]["val_fraction"]
        train_idx, val_idx = stratified_split(raw_train_ds, val_fraction, seed)
        assert verify_no_overlap(train_idx, val_idx) is True

    def test_split_sizes(self, cfg, raw_train_ds):
        seed = cfg["project"]["seed"]
        val_fraction = cfg["data"]["val_fraction"]
        train_idx, val_idx = stratified_split(raw_train_ds, val_fraction, seed)
        assert len(train_idx) + len(val_idx) == len(raw_train_ds)
        assert len(val_idx) > 0
        assert len(train_idx) > len(val_idx)

    def test_split_label_balance(self, cfg, raw_train_ds):
        """Each class should appear in both splits."""
        seed = cfg["project"]["seed"]
        val_fraction = cfg["data"]["val_fraction"]
        train_idx, val_idx = stratified_split(raw_train_ds, val_fraction, seed)
        all_labels = raw_train_ds.targets.numpy()
        train_classes = set(all_labels[train_idx])
        val_classes = set(all_labels[val_idx])
        assert train_classes == set(range(10))
        assert val_classes == set(range(10))

    def test_dataloaders_build(self, cfg, raw_train_ds, raw_test_ds):
        train_ld, val_ld, test_ld = build_dataloaders(cfg, raw_train_ds, raw_test_ds)
        assert len(train_ld) > 0
        assert len(val_ld) > 0
        assert len(test_ld) > 0

    def test_dataloader_batch_shape(self, cfg, raw_train_ds, raw_test_ds):
        train_ld, _, _ = build_dataloaders(cfg, raw_train_ds, raw_test_ds)
        imgs, labels = next(iter(train_ld))
        assert imgs.ndim == 4
        assert imgs.shape[1] == 1
        assert imgs.shape[2] == 28
        assert imgs.shape[3] == 28
        assert labels.ndim == 1
        assert labels.dtype == torch.int64

    def test_labels_in_range_from_loader(self, cfg, raw_train_ds, raw_test_ds):
        train_ld, _, _ = build_dataloaders(cfg, raw_train_ds, raw_test_ds)
        imgs, labels = next(iter(train_ld))
        assert int(labels.min()) >= 0
        assert int(labels.max()) <= 9
