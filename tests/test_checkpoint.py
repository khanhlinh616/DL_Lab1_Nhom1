"""
test_checkpoint.py — Tests for checkpoint save and load.
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from fashionmnist.config import load_config
from fashionmnist.models.network import SimpleCNN
from fashionmnist.training.checkpoint import save_checkpoint, load_checkpoint, save_best_checkpoint


@pytest.fixture(scope="module")
def cfg():
    return load_config()


class TestCheckpoint:
    def test_save_and_load_state_dict(self, cfg, tmp_path):
        """Saved and reloaded model produces identical output."""
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        dummy_input = torch.randn(4, 1, 28, 28)
        model.eval()
        with torch.no_grad():
            output_before = model(dummy_input).clone()

        out_path = save_checkpoint(
            checkpoint_dir=tmp_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            best_metric=0.85,
            cfg=cfg,
        )
        assert out_path.exists()

        # Load into fresh model
        model2 = SimpleCNN()
        payload = load_checkpoint(out_path, model2, device=device)
        model2.eval()
        with torch.no_grad():
            output_after = model2(dummy_input)

        assert torch.allclose(output_before, output_after, atol=1e-5), \
            "Output differs after save/load!"

    def test_checkpoint_required_keys(self, cfg, tmp_path):
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters())
        out_path = save_checkpoint(tmp_path, model, optimizer, None, 3, 0.9, cfg)

        payload = torch.load(out_path, map_location="cpu", weights_only=False)
        for key in ("epoch", "model_state_dict", "best_metric", "optimizer_state_dict"):
            assert key in payload, f"Missing key: {key}"

    def test_load_epoch_matches(self, cfg, tmp_path):
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters())
        save_checkpoint(tmp_path, model, optimizer, None, epoch=7, best_metric=0.92, cfg=cfg)
        out_path = tmp_path / "checkpoint_epoch_007.pt"

        payload = load_checkpoint(out_path, SimpleCNN(), device=torch.device("cpu"))
        assert payload["epoch"] == 7

    def test_load_missing_file_raises(self, cfg, tmp_path):
        fake_path = tmp_path / "nonexistent.pt"
        with pytest.raises(FileNotFoundError):
            load_checkpoint(fake_path, SimpleCNN())

    def test_best_checkpoint_updates_on_improvement(self, cfg, tmp_path):
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters())

        new_best, improved = save_best_checkpoint(
            checkpoint_dir=tmp_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            current_metric=0.80,
            best_metric=0.70,
            cfg=cfg,
        )
        assert improved is True
        assert new_best == 0.80
        assert (tmp_path / "best_model.pt").exists()

    def test_best_checkpoint_no_update_on_regression(self, cfg, tmp_path):
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters())

        new_best, improved = save_best_checkpoint(
            checkpoint_dir=tmp_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=2,
            current_metric=0.60,
            best_metric=0.80,
            cfg=cfg,
        )
        assert improved is False
        assert new_best == 0.80

    def test_optimizer_state_restored(self, cfg, tmp_path):
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Do a dummy optimizer step to populate state
        loss = model(torch.randn(2, 1, 28, 28)).sum()
        loss.backward()
        optimizer.step()

        out_path = save_checkpoint(tmp_path, model, optimizer, None, epoch=1, best_metric=0.5, cfg=cfg,
                                   filename="opt_test.pt")

        model2 = SimpleCNN()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        load_checkpoint(out_path, model2, optimizer=optimizer2)

        # Check optimizer state has entries
        assert len(optimizer2.state) > 0
