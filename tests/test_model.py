"""
test_model.py — Tests for model architectures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from fashionmnist.config import load_config
from fashionmnist.models.network import BaselineMLP, SimpleCNN, build_model, count_parameters


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def dummy_batch():
    """A small batch of random 1×28×28 images."""
    return torch.randn(8, 1, 28, 28)


class TestBaselineMLP:
    def test_output_shape(self, dummy_batch):
        model = BaselineMLP(hidden_units=128, dropout=0.0)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (8, 10), f"Expected (8,10), got {out.shape}"

    def test_no_softmax_in_output(self, dummy_batch):
        """Outputs should be raw logits — not probabilities summing to 1."""
        model = BaselineMLP()
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        row_sums = torch.softmax(out, dim=1).sum(dim=1)
        # raw logit row sums should NOT be ~1.0
        assert not torch.allclose(out.sum(dim=1), torch.ones(8), atol=0.1)

    def test_has_trainable_params(self):
        model = BaselineMLP()
        assert count_parameters(model) > 0


class TestSimpleCNN:
    def test_output_shape(self, dummy_batch):
        model = SimpleCNN()
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (8, 10), f"Expected (8,10), got {out.shape}"

    def test_single_sample(self):
        model = SimpleCNN()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, 28, 28))
        assert out.shape == (1, 10)

    def test_output_is_finite(self, dummy_batch):
        model = SimpleCNN()
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert torch.isfinite(out).all()

    def test_parameter_count(self):
        model = SimpleCNN(hidden_channels=[32, 64], fc_hidden=256)
        n = count_parameters(model)
        assert n > 10_000, f"Model seems too small: {n} params"

    def test_custom_channels(self, dummy_batch):
        model = SimpleCNN(hidden_channels=[16, 32], fc_hidden=128, dropout=0.1)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (8, 10)


class TestBuildModel:
    def test_build_baseline_mlp(self, cfg):
        cfg_copy = dict(cfg)
        cfg_copy["model"] = {"architecture": "BaselineMLP", "hidden_units": 128, "dropout": 0.0}
        model = build_model(cfg_copy)
        assert isinstance(model, BaselineMLP)

    def test_build_simple_cnn(self, cfg):
        model = build_model(cfg)
        assert isinstance(model, SimpleCNN)

    def test_build_unknown_raises(self, cfg):
        cfg_copy = dict(cfg)
        cfg_copy["model"] = {"architecture": "UnknownArch"}
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model(cfg_copy)
