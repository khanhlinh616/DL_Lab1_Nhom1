"""
test_training.py — Tests for training loop and evaluation steps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from fashionmnist.config import load_config
from fashionmnist.models.network import SimpleCNN, BaselineMLP
from fashionmnist.training.trainer import (
    train_one_epoch,
    evaluate_one_epoch,
    build_optimizer,
    build_scheduler,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_tiny_loader(n_samples: int = 32, batch_size: int = 8) -> DataLoader:
    """Create a DataLoader with random (image, label) pairs."""
    images = torch.randn(n_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (n_samples,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


class TestOptimizerAndScheduler:
    def test_build_adam(self, cfg):
        model = SimpleCNN()
        cfg_copy = dict(cfg)
        cfg_copy["training"] = {**cfg["training"], "optimizer": "adam", "learning_rate": 0.001, "weight_decay": 0.0}
        opt = build_optimizer(model, cfg_copy)
        assert isinstance(opt, torch.optim.Adam)

    def test_build_sgd(self, cfg):
        model = SimpleCNN()
        cfg_copy = dict(cfg)
        cfg_copy["training"] = {**cfg["training"], "optimizer": "sgd", "learning_rate": 0.01, "weight_decay": 0.0}
        opt = build_optimizer(model, cfg_copy)
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_optimizer_raises(self, cfg):
        model = SimpleCNN()
        cfg_copy = dict(cfg)
        cfg_copy["training"] = {**cfg["training"], "optimizer": "rmsprop"}
        with pytest.raises(ValueError):
            build_optimizer(model, cfg_copy)

    def test_cosine_scheduler(self, cfg):
        model = SimpleCNN()
        cfg_copy = dict(cfg)
        cfg_copy["training"] = {**cfg["training"], "optimizer": "adam", "learning_rate": 0.001,
                                "weight_decay": 0.0, "lr_scheduler": "cosine", "epochs": 10}
        opt = build_optimizer(model, cfg_copy)
        sch = build_scheduler(opt, cfg_copy)
        assert sch is not None

    def test_no_scheduler(self, cfg):
        model = SimpleCNN()
        cfg_copy = dict(cfg)
        cfg_copy["training"] = {**cfg["training"], "optimizer": "adam", "learning_rate": 0.001,
                                "weight_decay": 0.0, "lr_scheduler": None, "epochs": 5}
        opt = build_optimizer(model, cfg_copy)
        sch = build_scheduler(opt, cfg_copy)
        assert sch is None


class TestTrainingLoop:
    def test_train_one_epoch_returns_finite(self):
        model = SimpleCNN()
        loader = make_tiny_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0.0 <= acc <= 1.0
        assert loss != float("inf")
        assert loss == loss  # not NaN

    def test_evaluate_one_epoch_no_grad(self):
        """Verify model parameters do not change during evaluation."""
        model = SimpleCNN()
        loader = make_tiny_loader()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        params_before = [p.clone() for p in model.parameters()]
        loss, acc = evaluate_one_epoch(model, loader, criterion, device)
        params_after = list(model.parameters())

        for before, after in zip(params_before, params_after):
            assert torch.allclose(before, after), "Parameters changed during evaluation!"

    def test_train_updates_parameters(self):
        """A single training step should change model parameters."""
        model = SimpleCNN()
        loader = make_tiny_loader(n_samples=8, batch_size=8)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        params_before = [p.clone().detach() for p in model.parameters()]
        train_one_epoch(model, loader, criterion, optimizer, device)
        params_after = list(model.parameters())

        changed = any(
            not torch.allclose(b, a.detach())
            for b, a in zip(params_before, params_after)
        )
        assert changed, "Parameters did not change after one training epoch!"

    def test_model_train_mode_during_training(self):
        """Model should be in train() mode during train_one_epoch."""
        model = SimpleCNN()
        loader = make_tiny_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        # After train_one_epoch, model stays in train mode
        train_one_epoch(model, loader, criterion, optimizer, device)
        assert model.training is True

    def test_model_eval_mode_during_evaluation(self):
        """Model should be in eval() mode during evaluate_one_epoch."""
        model = SimpleCNN()
        loader = make_tiny_loader()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        evaluate_one_epoch(model, loader, criterion, device)
        assert model.training is False

    def test_loss_is_finite_and_positive(self):
        model = BaselineMLP()
        loader = make_tiny_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        loss, _ = train_one_epoch(model, loader, criterion, optimizer, device)
        assert loss > 0
        assert loss != float("inf")
        assert loss == loss  # not NaN
